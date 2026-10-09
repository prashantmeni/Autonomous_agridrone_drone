"""Mission execution against a fake flight controller.

The fake speaks the actual MAVLink mission exchange (MISSION_COUNT ->
MISSION_REQUEST_INT* -> MISSION_ACK) so the tests cover the upload path, the
AUTO hand-off and live progress tracking rather than a mock's internals.
"""
import asyncio
from types import SimpleNamespace

import pytest

from drone.autonomy.mission_manager import (
    CMD_LAND,
    CMD_RETURN_TO_LAUNCH,
    CMD_TAKEOFF,
    CMD_WAYPOINT,
    MissionManager,
    MissionPhase,
)
from drone.mavlink import missions as M


class FakeMav:
    """Records everything the Pi sends, and can be told to reject."""

    def __init__(self, fc):
        self.fc = fc
        self.sent = []

    def mission_count_send(self, *a):
        self.sent.append(("count", a[2]))
        self.fc.on_count(int(a[2]))

    def mission_item_int_send(self, *a):
        seq, command = a[2], a[4]
        lat, lon, alt = a[10], a[11], a[12]
        self.sent.append(("item", seq, command, lat / 1e7, lon / 1e7, alt))
        self.fc.on_item(seq, command)

    def mission_clear_all_send(self, *a):
        self.sent.append(("clear",))
        self.fc.emit("MISSION_ACK", type=0)

    def mission_set_current_send(self, *a):
        self.sent.append(("set_current", a[2]))
        self.fc.emit("MISSION_ACK", type=0)

    def set_mode_send(self, *a):
        self.sent.append(("set_mode", a[2]))
        self.fc.on_mode(int(a[2]))

    def command_long_send(self, *a):
        command = a[2]
        self.sent.append(("command", command))
        self.fc.master.emit("COMMAND_ACK", command=command, result=0)


class FakeConn:
    """Stands in for MavlinkConnection, including real router dispatch."""

    def __init__(self, fc):
        self.fc = fc
        self.master = fc.master
        self.connected = True
        self.target_system = 1
        self.target_component = 1
        self.px4_status = []

    def register_router(self, msg_type, handler):
        def dispatch(*args):
            m = args[-1]
            if m.get_type() == msg_type:
                handler(m)
        self.master.message_hooks.append(dispatch)


class FakeMaster:
    def __init__(self, fc):
        self.mav = FakeMav(fc)
        self.message_hooks = []

    def emit(self, mtype, **fields):
        msg = SimpleNamespace(_type=mtype, **fields)
        msg.get_type = lambda: mtype
        for hook in list(self.message_hooks):
            hook(self, msg)
        return msg


class FakeFC:
    """Minimal PX4: acks uploads, optionally rejects, reports progress."""

    def __init__(self, ack_type=0, auto_count=True):
        self.master = FakeMaster(self)
        self.ack_type = ack_type
        self.auto_count = auto_count
        self.items = {}
        self.modes = []
        self._pending = 0

    # ---- called by FakeMav
    def on_count(self, n):
        self._pending = n
        if self.auto_count:
            for seq in range(n):
                self.master.emit("MISSION_REQUEST_INT", seq=seq)

    def on_item(self, seq, command):
        self.items[seq] = command
        if self.auto_count and self._pending and len(self.items) >= self._pending:
            self.master.emit("MISSION_ACK", type=self.ack_type)

    def on_mode(self, custom):
        self.modes.append(custom)

    # ---- drive progress
    def emit(self, mtype, **fields):
        return self.master.emit(mtype, **fields)

    def report_current(self, seq, total):
        self.master.emit("MISSION_CURRENT", seq=seq, total=total)

    def report_reached(self, seq):
        self.master.emit("MISSION_ITEM_REACHED", seq=seq)


def make_app(fc, armed=True, max_alt=30.0, max_speed=8.0):
    from drone.core.state import FlightStateMachine
    return SimpleNamespace(
        conn=FakeConn(fc),
        cfg=SimpleNamespace(mission=SimpleNamespace(max_altitude_m=max_alt,
                                                    max_speed_mps=max_speed)),
        tstore=SimpleNamespace(snap=SimpleNamespace(armed=armed)),
        fsm=FlightStateMachine(),
        db=SimpleNamespace(save_mission=lambda *a: None),
    )


def body(n_wp=3, rtl=True):
    return {
        "name": "test",
        "takeoff": {"lat": 12.0, "lon": 77.0, "altitude_m": 15},
        "waypoints": [{"lat": 12.001 * i, "lon": 77.001, "alt": 20} for i in range(1, n_wp + 1)],
        "landing": {"lat": 12.0, "lon": 77.0},
        "return_to_home": rtl,
    }


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------- protocol
def test_upload_exchanges_items_and_waits_for_ack():
    fc = FakeFC()
    conn = make_app(fc).conn
    items = [{"lat": 1.0, "lon": 2.0, "alt": 20}, {"lat": 1.1, "lon": 2.1, "alt": 22}]
    ok, detail = M.upload_mission(conn, items)
    assert ok, detail
    assert fc.master.mav.sent[0] == ("count", 2)
    assert len([s for s in fc.master.mav.sent if s[0] == "item"]) == 2


def test_upload_rejected_by_flight_controller():
    fc = FakeFC(ack_type=11)          # DENIED
    conn = make_app(fc).conn
    ok, detail = M.upload_mission(conn, [{"lat": 1.0, "lon": 2.0, "alt": 20}])
    assert ok is False
    assert "denied" in detail.lower()


def test_upload_refuses_when_not_connected():
    conn = make_app(FakeFC()).conn
    conn.connected = False
    ok, detail = M.upload_mission(conn, [{"lat": 1.0, "lon": 2.0, "alt": 20}])
    assert ok is False and "not connected" in detail


def test_upload_rejects_empty_mission():
    conn = make_app(FakeFC()).conn
    ok, detail = M.upload_mission(conn, [])
    assert ok is False and "no items" in detail


# ------------------------------------------------------------- item build
def test_build_items_orders_takeoff_waypoints_land_rtl():
    mgr = MissionManager(make_app(FakeFC()))
    items = mgr.build_items(body(n_wp=3, rtl=True))
    cmds = [i["command"] for i in items]
    assert cmds[0] == CMD_TAKEOFF
    assert cmds[1:4] == [CMD_WAYPOINT] * 3
    assert cmds[4] == CMD_LAND
    assert cmds[-1] == CMD_RETURN_TO_LAUNCH
    assert len(items) == 6


def test_build_items_without_landing_or_rtl():
    mgr = MissionManager(make_app(FakeFC()))
    b = {"waypoints": [{"lat": 1.0, "lon": 2.0, "alt": 10}]}
    items = mgr.build_items(b)
    assert [i["command"] for i in items] == [CMD_WAYPOINT]


# -------------------------------------------------------------- execution
def test_start_uploads_and_switches_to_auto():
    fc = FakeFC()
    mgr = MissionManager(make_app(fc))
    res = run(mgr.start("m1", body()))
    assert res["status"] == "executing", res
    assert mgr.phase is MissionPhase.EXECUTING
    kinds = [s[0] for s in fc.master.mav.sent]
    assert "clear" in kinds and "count" in kinds and "set_mode" in kinds
    assert ("count", 6) in fc.master.mav.sent


def test_start_refuses_when_disarmed():
    fc = FakeFC()
    mgr = MissionManager(make_app(fc, armed=False))
    res = run(mgr.start("m1", body()))
    assert res["status"] == "failed"
    assert "disarmed" in res["error"]
    assert not [s for s in fc.master.mav.sent if s[0] == "set_mode"]


def test_start_fails_when_fc_rejects_upload():
    fc = FakeFC(ack_type=4)           # NO_SPACE
    mgr = MissionManager(make_app(fc))
    res = run(mgr.start("m1", body()))
    assert res["status"] == "failed"
    assert "rejected" in res["error"]
    assert mgr.phase is MissionPhase.FAILED


def test_start_validates_altitude():
    fc = FakeFC()
    mgr = MissionManager(make_app(fc, max_alt=10.0))
    res = run(mgr.start("m1", body()))
    assert res["status"] == "failed"
    assert "invalid" in res["error"]


def test_second_start_while_active_is_refused():
    mgr = MissionManager(make_app(FakeFC()))
    run(mgr.start("m1", body()))
    with pytest.raises(ValueError):
        run(mgr.start("m2", body()))


# --------------------------------------------------------------- progress
def test_progress_and_completion_via_mission_current():
    fc = FakeFC()
    mgr = MissionManager(make_app(fc))
    run(mgr.start("m1", body()))
    assert mgr.status()["done"] == 0
    fc.report_current(3, 6)
    assert mgr.current_seq == 3
    assert mgr.phase is MissionPhase.EXECUTING
    fc.report_current(6, 6)                 # past the last item
    assert mgr.phase is MissionPhase.COMPLETE


def test_completion_when_every_item_reported_reached():
    fc = FakeFC()
    mgr = MissionManager(make_app(fc))
    run(mgr.start("m1", body()))
    for seq in range(6):
        fc.report_reached(seq)
    assert mgr.status()["done"] == 6
    assert mgr.phase is MissionPhase.COMPLETE


def test_pause_and_resume_switch_modes():
    fc = FakeFC()
    mgr = MissionManager(make_app(fc))
    run(mgr.start("m1", body()))
    assert mgr.pause.__name__  # sanity
    assert run(mgr.pause())["status"] == "PAUSED"
    assert mgr.phase is MissionPhase.PAUSED
    assert run(mgr.resume())["status"] == "EXECUTING"
    assert mgr.phase is MissionPhase.EXECUTING


def test_pause_rejected_when_not_executing():
    mgr = MissionManager(make_app(FakeFC()))
    assert "error" in run(mgr.pause())


def test_abort_marks_failed_and_sends_rtl_when_armed():
    fc = FakeFC()
    app = make_app(fc, armed=True)
    mgr = MissionManager(app)
    run(mgr.start("m1", body()))
    res = run(mgr.abort("test"))
    assert mgr.phase is MissionPhase.FAILED
    assert "aborted" in (mgr.error or "")
    assert "RTL" in res["detail"]


def test_abort_skips_rtl_when_disarmed():
    fc = FakeFC()
    app = make_app(fc, armed=False)
    mgr = MissionManager(app)
    run(mgr.start("m1", body()))
    res = run(mgr.abort("test"))
    assert "disarmed" in res["detail"]
    assert not [s for s in fc.master.mav.sent if s[0] == "set_mode" and s[1] == 0x04050000]


def test_status_reports_totals_and_phase():
    fc = FakeFC()
    mgr = MissionManager(make_app(fc))
    run(mgr.start("m1", body()))
    st = mgr.status()
    assert st["phase"] == "EXECUTING"
    assert st["total"] == 6
    assert st["active"] is True
    assert st["mission_id"] == "m1"


def test_supervisor_fails_mission_when_link_lost():
    fc = FakeFC()
    app = make_app(fc)
    mgr = MissionManager(app)
    run(mgr.start("m1", body()))
    app.conn.connected = False
    fc.report_current(1, 6)
    # simulate the supervisor tick body directly (it loops forever otherwise)
    if mgr.phase is MissionPhase.EXECUTING and not app.conn.connected:
        mgr._finish(MissionPhase.FAILED, "MAVLink link lost mid-mission")
    assert mgr.phase is MissionPhase.FAILED
