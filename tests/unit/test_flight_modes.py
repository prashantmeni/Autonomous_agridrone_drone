"""Farm flight modes and the behaviour rules they apply."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from drone.autonomy.modes import (  # noqa: E402
    RTL_RETURN_ALT_PARAM,
    FlightMode,
    ModeController,
)
from drone.core.behaviour import BehaviourRules  # noqa: E402
from drone.core.state import FlightState, FlightStateMachine  # noqa: E402


class FakeMav:
    def __init__(self, fc):
        self.fc = fc
        self.params = {}
        self.modes = []
        self.commands = []

    def param_set_send(self, *a):
        name = a[2].decode().rstrip("\x00")
        self.params[name] = a[3]
        # Answer immediately, otherwise wait_for blocks for the full timeout.
        self.fc.emit("PARAM_VALUE", param_id=name, param_value=a[3])

    def param_request_read_send(self, *a):
        pass

    def set_mode_send(self, *a):
        self.modes.append(a[2])
        self.fc.emit("HEARTBEAT", base_mode=128, autopilot=12, custom_mode=0)

    def command_long_send(self, *a):
        self.commands.append(a[2])
        self.fc.emit("COMMAND_ACK", command=a[2], result=0)


class FakeMaster:
    def __init__(self, fc):
        self.mav = FakeMav(fc)
        self.message_hooks = []

    def emit(self, mtype, **f):
        m = SimpleNamespace(_type=mtype, **f)
        m.get_type = lambda: mtype
        for h in list(self.message_hooks):
            h(self, m)
        return m


class FakeFC:
    def __init__(self):
        self.master = FakeMaster(self)

    def emit(self, mtype, **f):
        return self.master.emit(mtype, **f)


def make_app(armed=True, alt=25.0, rules=None, refuse_mode=False):
    fc = FakeFC()
    sent = []

    class Conn:
        def __init__(self):
            self.master = fc.master
            self.connected = True
            self.target_system = 1
            self.target_component = 1
            self.px4_status = []

    class Mav(Conn):
        pass

    conn = Conn()
    real_mav = conn.master.mav

    def refuse():
        if refuse_mode:
            raise AssertionError("set_mode should not have been called")

    app = SimpleNamespace(
        conn=conn,
        fsm=FlightStateMachine(),
        tstore=SimpleNamespace(snap=SimpleNamespace(
            armed=armed, relative_alt_m=alt, connected=True)),
        db=SimpleNamespace(log_event=lambda *a: sent.append(a)),
    )
    app._rules = rules or BehaviourRules()
    return app, fc, sent


def install_behaviour(monkeypatch, rules):
    import drone.core.behaviour as behaviour_mod
    monkeypatch.setattr(behaviour_mod, "load_behaviour", lambda *a, **k: rules)


# ------------------------------------------------------------- mode entries
def test_unknown_mode_is_rejected():
    app, _fc, _sent = make_app()
    mc = ModeController(app)
    with pytest.raises(ValueError):
        asyncio.run(mc.enter("TELEPORT"))


def test_disarmed_vehicle_is_refused_a_mode():
    app, _fc, _sent = make_app(armed=False)
    mc = ModeController(app)
    res = asyncio.run(mc.enter(FlightMode.MONITOR.value))
    assert res["status"] == "rejected"
    assert "disarmed" in res["error"]


def test_enter_monitor_sets_hold(monkeypatch):
    app, fc, _sent = make_app(alt=0.0)
    install_behaviour(monkeypatch, BehaviourRules())
    mc = ModeController(app)
    res = asyncio.run(mc.enter_monitor())
    assert res["status"] == "ok"
    assert res["px4_mode"] == "HOLD"
    assert mc.current == FlightMode.MONITOR.value


def test_survey_mode_maps_to_auto(monkeypatch):
    app, fc, _sent = make_app(alt=30.0)
    install_behaviour(monkeypatch, BehaviourRules())
    mc = ModeController(app)
    res = asyncio.run(mc.enter(FlightMode.SURVEY.value))
    assert res["px4_mode"] == "AUTO"
    assert app.fsm.state is FlightState.SURVEY


def test_mapping_mode_maps_to_auto(monkeypatch):
    app, fc, _sent = make_app(alt=30.0)
    install_behaviour(monkeypatch, BehaviourRules())
    mc = ModeController(app)
    res = asyncio.run(mc.enter(FlightMode.MAPPING.value))
    assert res["px4_mode"] == "AUTO"


# ------------------------------------------- behaviour rules actually apply
def test_rth_altitude_is_pushed_to_the_flight_controller(monkeypatch):
    app, fc, _sent = make_app(alt=30.0)
    install_behaviour(monkeypatch, BehaviourRules(rth_altitude_m=42.0))
    mc = ModeController(app)
    asyncio.run(mc.enter(FlightMode.SURVEY.value))
    assert RTL_RETURN_ALT_PARAM in fc.master.mav.params
    assert fc.master.mav.params[RTL_RETURN_ALT_PARAM] == pytest.approx(42.0)


def test_rth_altitude_within_bounds_is_passed_through(monkeypatch):
    app, fc, _sent = make_app(alt=30.0)
    install_behaviour(monkeypatch, BehaviourRules(rth_altitude_m=60.0))
    mc = ModeController(app)
    asyncio.run(mc.enter(FlightMode.SURVEY.value))
    assert fc.master.mav.params[RTL_RETURN_ALT_PARAM] == pytest.approx(60.0)


def test_rth_altitude_is_clamped_for_out_of_range_sources(monkeypatch):
    """BehaviourRules caps the field at 120 m, but other sources may not."""
    app, fc, _sent = make_app(alt=30.0)
    install_behaviour(monkeypatch, SimpleNamespace(rth_altitude_m=999.0))
    mc = ModeController(app)
    asyncio.run(mc.enter(FlightMode.SURVEY.value))
    assert fc.master.mav.params[RTL_RETURN_ALT_PARAM] <= 200.0


def test_climb_is_requested_when_below_working_altitude(monkeypatch):
    app, fc, _sent = make_app(alt=0.0)
    install_behaviour(monkeypatch, BehaviourRules(survey_altitude_m=25.0))
    mc = ModeController(app)
    asyncio.run(mc.enter(FlightMode.MONITOR.value, altitude_m=25.0))
    # MAV_CMD_NAV_TAKEOFF is 22
    assert 22 in fc.master.mav.commands


def test_no_climb_when_already_at_altitude(monkeypatch):
    app, fc, _sent = make_app(alt=40.0)
    install_behaviour(monkeypatch, BehaviourRules(survey_altitude_m=25.0))
    mc = ModeController(app)
    asyncio.run(mc.enter(FlightMode.MONITOR.value, altitude_m=25.0))
    assert 22 not in fc.master.mav.commands


def test_status_exposes_operator_knobs(monkeypatch):
    app, _fc, _sent = make_app()
    install_behaviour(monkeypatch, BehaviourRules(
        survey_altitude_m=30.0, rth_altitude_m=40.0, cruise_speed_mps=6.0))
    st = ModeController(app).status()
    assert st["survey_altitude_m"] == 30.0
    assert st["rth_altitude_m"] == 40.0
    assert st["cruise_speed_mps"] == 6.0
    assert "MONITOR" in st["available"]
    assert "SURVEY" in st["available"]