"""MAVLink connection with reconnect. Supports serial/udp/tcp. Never hard-codes device."""
from __future__ import annotations

import asyncio
import glob
import logging
import os
import time
from collections import deque
from typing import Any

log = logging.getLogger("drone.mavlink")

# Substrings that identify a flight controller's USB-serial node. Checked in
# order against /dev/serial/by-id entries when the configured device is absent.
_FC_MARKERS = ("3D_Robotics", "PX4", "PX4_FMU", "ArduPilot", "APM", "FMU")


def _strip_usb_topology(name: str) -> str:
    """Drop the USB topology suffix from a by-id name.

    udev encodes the physical port in by-id names, e.g.
    ``usb-3D_Robotics_PX4_FMU_v2.x_0-if00``: the ``_0`` is the USB port and
    ``if00`` the interface. The same controller in another socket becomes
    ``..._1-if00``, so both parts must go for the prefix to be port-stable.
    """
    base = os.path.basename(name)
    if "-if" in base:
        base = base.split("-if", 1)[0]
    head, sep, tail = base.rpartition("_")
    if sep and tail.isdigit() and head:
        base = head
    return base


def resolve_device(configured: str, by_id_dir: str = "/dev/serial/by-id") -> str | None:
    """Find the device to open for MAVLink, tolerating USB port changes.

    Resolution order:
      1. the configured path verbatim, when it exists;
      2. any by-id entry sharing the configured name's stable prefix (same
         flight controller, different USB port);
      3. any by-id entry that looks like a flight controller;
      4. ``/dev/ttyACM*`` then ``/dev/ttyUSB*``.

    Returns ``None`` when nothing is plugged in. Never raises.
    """
    if configured.startswith(("udp:", "tcp:", "serial:")):
        return configured

    if configured and os.path.exists(configured):
        return configured

    if configured.startswith(by_id_dir.rstrip("/") + "/"):
        prefix = _strip_usb_topology(configured)
        matches = sorted(glob.glob(os.path.join(by_id_dir, prefix + "*")))
        if matches:
            return matches[0]

    try:
        entries = sorted(glob.glob(os.path.join(by_id_dir, "*")))
    except OSError:
        entries = []
    for entry in entries:
        if any(marker.lower() in os.path.basename(entry).lower() for marker in _FC_MARKERS):
            return entry

    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


class MavlinkConnection:
    def __init__(self, connection_string: str, baud: int = 115200,
                 heartbeat_timeout_s: float = 5.0, reconnect_delay_s: float = 2.0):
        self.connection_string = connection_string
        self.baud = baud
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.reconnect_delay_s = reconnect_delay_s
        self._master: Any = None
        self.connected = False
        self.last_heartbeat = 0.0
        self._cmd_seq = 0
        self.px4_status: deque = deque(maxlen=20)  # (ts, text) — PX4 STATUSTEXT history

    @property
    def resolved_device(self) -> str | None:
        """The device currently in use, or the one that would be opened."""
        return getattr(self._master, "port", None) or self._last_resolved

    _last_resolved: str | None = None

    def connect(self) -> bool:
        from pymavlink import mavutil
        device = resolve_device(self.connection_string)
        if device is None:
            log.warning("MAVLink: no flight controller found (configured %s)", self.connection_string)
            self._last_resolved = None
            return False
        if device != self.connection_string:
            log.info("MAVLink: using %s (configured %s)", device, self.connection_string)
        self._last_resolved = device
        try:
            if device.startswith(("udp:", "tcp:")):
                self._master = mavutil.mavlink_connection(device)
            else:
                self._master = mavutil.mavlink_connection(device, baud=self.baud)
            self._install_status_logger()
            log.info(f"MAVLink connecting: {device}")
            return True
        except Exception as e:
            log.error(f"MAVLink connect failed {device}: {e}")
            self._master = None
            return False

    def _close_master(self) -> None:
        """Drop the serial handle so the next connect re-resolves the device.

        Without this a flight controller replugged into another USB port leaves
        us holding a stale file descriptor and reconnect can never succeed.
        """
        master, self._master = self._master, None
        self.connected = False
        if master is None:
            return
        for attr in ("close",):
            closer = getattr(master, attr, None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing a dead handle is best effort
                    pass

    def _install_status_logger(self):
        """Permanently log every PX4 STATUSTEXT (prearm denials, estimator warnings...).

        Installed on the raw connection so it fires no matter which loop drains
        the message off the wire, and survives outside command wait windows.
        """
        master = self._master
        hooks = getattr(master, "message_hooks", None) if master is not None else None
        if hooks is None:
            return

        def hook(*args):
            m = args[-1]  # pymavlink calls hook(self, msg)
            try:
                if m.get_type() != "STATUSTEXT":
                    return
                text = str(m.text).rstrip("\x00")
            except Exception:
                return
            if not text:
                return
            self.px4_status.append((time.time(), text))
            log.warning(f"PX4: {text}")

        hooks.append(hook)

    def wait_heartbeat(self, timeout: float = 10.0) -> bool:
        try:
            m = self._master.wait_heartbeat(timeout=timeout)
            if m is not None:
                self.connected = True
                self.last_heartbeat = time.time()
                return True
        except Exception as e:
            log.error(f"heartbeat wait failed: {e}")
        return False

    async def maintain(self):
        """Background reconnect loop. Never issues duplicate dangerous commands.

        Re-resolves the device on every connect, so moving the flight controller
        to another USB port (or replugging it) recovers without a restart.
        """
        while True:
            if self._master is None:
                self.connect()
                await asyncio.sleep(self.reconnect_delay_s)
                continue
            try:
                msg = self._master.recv_match(blocking=False)
                if msg is not None and msg.get_type() == "HEARTBEAT":
                    self.last_heartbeat = time.time()
                    if not self.connected:
                        self.connected = True
                        log.info("MAVLink heartbeat restored - resynchronizing telemetry")
                if self.connected and time.time() - self.last_heartbeat > self.heartbeat_timeout_s:
                    self._close_master()
                    log.warning("MAVLINK_RECONNECTING: heartbeat lost")
            except Exception as e:
                log.error(f"MAVLink loop error: {e}")
                self._close_master()
            await asyncio.sleep(0.2)

    @property
    def master(self): return self._master
    @property
    def target_system(self): return getattr(self._master, "target_system", 1) if self._master else 1
    @property
    def target_component(self): return getattr(self._master, "target_component", 1) if self._master else 1
