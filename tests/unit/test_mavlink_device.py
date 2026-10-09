"""Device resolution for MAVLink: same flight controller, any USB port.

The by-id name udev assigns encodes the physical USB topology, so a Pixhawk
moved to another socket appears under a different path. These tests pin the
resolution order and the reconnect behaviour that depends on it.
"""
import glob as _glob
import os

import pytest

from drone.mavlink.connection import (
    MavlinkConnection,
    _strip_usb_topology,
    resolve_device,
)

# captured before monkeypatching, otherwise the stub would call itself
_REAL_GLOB = _glob.glob

BY_ID = "usb-3D_Robotics_PX4_FMU_v2.x_0-if00"
BY_ID_MOVED = "usb-3D_Robotics_PX4_FMU_v2.x_1-if00"


@pytest.fixture
def byid(tmp_path, monkeypatch):
    """A fake /dev/serial/by-id directory plus a fake ttyACM/ttyUSB glob root."""
    d = tmp_path / "by-id"
    d.mkdir()
    monkeypatch.setattr("drone.mavlink.connection.glob.glob",
                        lambda pattern: _fake_glob(pattern, d))
    return d


def _fake_glob(pattern, d):
    """Stand in for glob.glob, honouring the real by-id dir and /dev tty*.

    Patterns are normalised because os.path.join uses ``\\`` on Windows, which
    would otherwise miss the ``/dev/serial/by-id/`` prefix check.
    """
    p = pattern.replace("\\", "/")
    if p.startswith("/dev/serial/by-id/"):
        return sorted(str(x) for x in d.glob("*"))
    if p.startswith("/dev/tty"):
        return []          # no loose tty nodes in these tests
    return _REAL_GLOB(pattern)


def _touch(directory, name):
    p = directory / name
    p.write_text("x")
    return p


# ------------------------------------------------------- topology stripping
def test_strip_usb_topology_removes_port_and_interface():
    assert _strip_usb_topology(BY_ID) == "usb-3D_Robotics_PX4_FMU_v2.x"
    assert _strip_usb_topology(BY_ID_MOVED) == "usb-3D_Robotics_PX4_FMU_v2.x"


def test_strip_usb_topology_keeps_device_names_without_a_port():
    # A plain ttyACM-style name has no topology suffix to remove.
    assert _strip_usb_topology("/dev/ttyACM0") == "ttyACM0"


# ------------------------------------------------------------- resolution
def test_configured_path_wins_when_present(tmp_path):
    dev = tmp_path / "ttyACM0"
    dev.write_text("x")
    assert resolve_device(str(dev)) == str(dev)


def test_resolves_same_controller_after_port_change(byid):
    """Configured for port 0, actually enumerated on port 1 -> still resolves."""
    found = _touch(byid, BY_ID_MOVED)
    resolved = resolve_device("/dev/serial/by-id/" + BY_ID)
    assert resolved == str(found)


def test_falls_back_to_any_flight_controller_node(byid):
    """Unconfigured/unknown name still finds a Pixhawk by its USB signature."""
    found = _touch(byid, BY_ID)
    resolved = resolve_device("/dev/serial/by-id/usb-SomethingElse_0-if00")
    assert resolved == str(found)


def test_returns_none_when_nothing_plugged_in(byid):
    assert resolve_device("/dev/serial/by-id/" + BY_ID) is None


def test_udp_and_tcp_are_passed_through():
    assert resolve_device("udp:0.0.0.0:14550") == "udp:0.0.0.0:14550"
    assert resolve_device("tcp:127.0.0.1:5760") == "tcp:127.0.0.1:5760"


def test_wrong_path_falls_back_to_flight_controller(byid):
    found = _touch(byid, BY_ID)
    assert resolve_device("/dev/ttyAMA0", by_id_dir="/dev/serial/by-id") == str(found)


# -------------------------------------------------------------- reconnect
def test_connect_without_device_fails_cleanly():
    """No flight controller attached must not raise, just report False."""
    conn = MavlinkConnection("/dev/serial/by-id/" + BY_ID)
    conn._last_resolved = None
    assert conn.connect() is False
    assert conn.master is None


def test_reconnect_closes_stale_handle(monkeypatch):
    """A stale serial handle is dropped so the next connect can re-resolve."""
    conn = MavlinkConnection("/dev/serial/by-id/" + BY_ID)

    class FakeMaster:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    master = FakeMaster()
    conn._master = master
    conn._close_master()
    assert master.closed is True
    assert conn.master is None


def test_maintain_reconnects_after_heartbeat_loss(monkeypatch):
    """Heartbeat loss must clear the master so a replugged FC is re-opened."""
    import asyncio

    conn = MavlinkConnection("udp:0.0.0.0:14550", heartbeat_timeout_s=0.0)
    connects = []

    def fake_connect():
        connects.append(1)
        conn._master = object()
        return True

    monkeypatch.setattr(conn, "connect", fake_connect)

    async def run():
        # one iteration: master is None -> connect; then simulate the loop
        # noticing a lost heartbeat and dropping the handle.
        for _ in range(3):
            if conn.master is None:
                conn.connect()
            await asyncio.sleep(0.01)
            if conn.connected and _ == 2:
                conn.connected = True
                conn.last_heartbeat = 0.0
                if time_since(conn) > conn.heartbeat_timeout_s:
                    conn._close_master()
        await asyncio.sleep(0.01)

    def time_since(c):
        import time as _t
        return _t.time() - c.last_heartbeat

    asyncio.run(run())
    assert connects, "maintain loop must attempt (re)connection when no master"
    assert conn.master is not None


def test_strip_topology_is_stable_across_ports():
    """Both port spellings must collapse to the same prefix."""
    a = _strip_usb_topology("/dev/serial/by-id/" + BY_ID)
    b = _strip_usb_topology("/dev/serial/by-id/" + BY_ID_MOVED)
    assert a == b
    assert os.path.basename(a)
