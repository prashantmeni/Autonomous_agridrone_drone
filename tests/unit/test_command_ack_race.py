"""A fast flight-controller reply must not be missed.

Every command (arm, takeoff, land, RTL, parameter writes) waits for an ACK. The
ACK hook used to be installed *after* the command was sent, so a controller that
answers within a few milliseconds had its reply discarded and the caller
reported a timeout for a command that actually succeeded — takeoff and RTL
would then be retried or reported as failed while the aircraft was already
acting on them.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone.mavlink import commands as C  # noqa: E402
from drone.mavlink import parameters as P  # noqa: E402


class FastMaster:
    """Answers synchronously, inside the send call, like a real FC would."""

    def __init__(self, master_hook_owner=None):
        self.message_hooks = []
        self.sends = []

    class _Mav:
        def __init__(self, owner):
            self.owner = owner

        def command_long_send(self, *a):
            self.owner.sends.append(("command", a[2]))
            self.owner._reply("COMMAND_ACK", command=a[2], result=0)

        def param_set_send(self, *a):
            name = a[2].decode().rstrip("\x00")
            self.owner.sends.append(("param_set", name))
            self.owner._reply("PARAM_VALUE", param_id=name, param_value=a[3])

        def param_request_read_send(self, *a):
            name = a[2].decode().rstrip("\x00")
            self.owner.sends.append(("param_read", name))
            self.owner._reply("PARAM_VALUE", param_id=name, param_value=12.5)

    def __init__(self):
        self.message_hooks = []
        self.sends = []
        self.mav = FastMaster._Mav(self)

    def _reply(self, mtype, **fields):
        msg = SimpleNamespace(_type=mtype, **fields)
        msg.get_type = lambda: mtype
        for hook in list(self.message_hooks):
            hook(self, msg)


def conn_with(master):
    return SimpleNamespace(master=master, connected=True, target_system=1,
                           target_component=1, px4_status=[])


def test_immediate_command_ack_is_caught():
    master = FastMaster()
    ok, detail = C._cmd(conn_with(master), 400, 1, 0, 0, 0, 0, 0, 0, timeout=2.0)
    assert ok, f"fast ACK was missed: {detail}"


def test_command_does_not_wait_for_a_timeout():
    master = FastMaster()
    t0 = time.time()
    C._cmd(conn_with(master), 22, 0, 0, 0, 0, 0, 0, 15.0, timeout=5.0)
    assert time.time() - t0 < 1.0, "command blocked on the timeout despite an instant ACK"


def test_rejected_command_still_reports_the_result():
    """MAV_RESULT 2 = DENIED; the caller must hear why, not 'ok'."""
    master = FastMaster()
    master.mav.command_long_send = lambda *a: master._reply(
        "COMMAND_ACK", command=a[2], result=2)
    ok, detail = C._cmd(conn_with(master), 400, 1, 0, 0, 0, 0, 0, 0, timeout=2.0)
    assert ok is False
    assert "DENIED" in detail


def test_immediate_param_set_ack_is_caught():
    master = FastMaster()
    assert P.set_param(conn_with(master), "RTL_RETURN_ALT", 42.0, timeout=2.0) is True
    assert ("param_set", "RTL_RETURN_ALT") in master.sends


def test_immediate_param_read_reply_is_caught():
    master = FastMaster()
    value = P.read_param(conn_with(master), "RTL_RETURN_ALT", timeout=2.0)
    assert value == 12.5


def test_wait_for_without_send_still_waits_and_times_out():
    master = FastMaster()
    t0 = time.time()
    found, _text, timed_out = C.wait_for(conn_with(master), "COMMAND_ACK",
                                         lambda m: True, timeout=0.3)
    assert found is None and timed_out
    assert time.time() - t0 >= 0.25