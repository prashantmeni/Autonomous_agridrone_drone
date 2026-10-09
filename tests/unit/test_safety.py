"""Preflight gate and in-flight failsafe behaviour.

These tests pin the two things that were previously fake: preflight must not
report PASS for anything it has not measured, and the failsafe must not command
the aircraft without sustained evidence (and never while disarmed).
"""
import asyncio
from types import SimpleNamespace

from drone.core.config import SafetyConfig
from drone.mavlink.telemetry import TelemetrySnapshot, TelemetryStore
from drone.safety.failsafe_supervisor import FailsafeSupervisor
from drone.safety.flight_safety import preflight_check


def make_app(snap: TelemetrySnapshot, *, boundaries=None, services=None,
             camera_enabled=True, obstacle_enabled=False, safety=None):
    class DB:
        def __init__(self):
            self.rows = boundaries or []
            self.events = []

        def log_event(self, *a):
            self.events.append(a)

        @property
        def con(self):
            outer = self

            class Cursor:
                def execute(self, sql):
                    return self

                def fetchall(self):
                    return outer.rows
            return Cursor()

    cfg = SimpleNamespace(
        safety=safety or SafetyConfig(),
        camera=SimpleNamespace(enabled=camera_enabled),
        obstacle_avoidance=SimpleNamespace(enabled=obstacle_enabled),
    )
    app = SimpleNamespace(
        tstore=SimpleNamespace(snap=snap),
        cfg=cfg,
        db=DB(),
    )
    if services is not None:
        import drone.runtime as runtime
        runtime.get_services = lambda _app: services
    return app


def snap(**kw) -> TelemetrySnapshot:
    s = TelemetrySnapshot()
    s.connected = True
    s.gps_fix = 3
    s.satellites = 12
    s.hdop = 0.8
    s.battery_remaining_pct = 90
    s.ekf_flags = 0
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class FakeCam:
    def __init__(self, status="READY"):
        self._status = status

    def status(self):
        return {"status": self._status}


class FakeProx:
    def __init__(self, available=True):
        self._available = available

    def status(self):
        return {"available": self._available}


# ------------------------------------------------------------- ekf parsing
def test_ekf_unknown_until_estimator_status_seen():
    s = TelemetrySnapshot()
    assert s.ekf_known is False
    assert s.ekf_estimate_ok is False
    s.ekf_flags = 0
    assert s.ekf_known is True and s.ekf_estimate_ok is True


def test_estimator_error_flags_fail_the_estimate():
    s = TelemetrySnapshot()
    s.ekf_flags = 0x08            # POS_ERROR
    assert s.ekf_estimate_ok is False
    assert "POS_ERROR" in s.ekf_issues


def test_gps_only_flags_do_not_fail_the_estimate():
    s = TelemetrySnapshot()
    s.ekf_flags = 0x100           # GPS_NOFIX: a GPS problem, not a filter fault
    assert s.ekf_estimate_ok is True


def test_estimator_status_message_updates_snapshot():
    store = TelemetryStore()
    msg = SimpleNamespace(flags=0x04, vel_ratio=1.2, hpos_ratio=0.9)
    msg.get_type = lambda: "ESTIMATOR_STATUS"
    store.update_from_msg(msg)
    assert store.snap.ekf_flags == 0x04
    assert store.snap.ekf_vel_ratio == 1.2
    assert "VELOCITY_ERROR" in store.snap.ekf_issues


# ---------------------------------------------------------------- preflight
def test_healthy_aircraft_is_ready():
    r = preflight_check(make_app(snap(), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["pixhawk"] == "PASS"
    assert r["checks"]["gps"] == "PASS"
    assert r["checks"]["battery"] == "PASS"
    assert r["checks"]["ekf"] == "PASS"
    assert r["ready"] is True


def test_disconnected_flight_controller_blocks_arming():
    r = preflight_check(make_app(snap(connected=False), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["pixhawk"] == "FAIL"
    assert r["ready"] is False


def test_missing_battery_is_unknown_not_pass():
    r = preflight_check(make_app(snap(battery_remaining_pct=-1), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["battery"] == "UNKNOWN"
    assert r["ready"] is False, "an unmeasured battery must not read as ready"


def test_missing_estimator_status_is_unknown_not_pass():
    r = preflight_check(make_app(snap(ekf_flags=-1), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["ekf"] == "UNKNOWN"
    assert r["ready"] is False


def test_ekf_position_error_blocks_arming():
    r = preflight_check(make_app(snap(ekf_flags=0x08), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["ekf"] == "FAIL"
    assert r["ready"] is False
    assert "POS_ERROR" in r["ekf_issues"]


def test_weak_gps_blocks_arming_when_required():
    r = preflight_check(make_app(snap(gps_fix=2), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["gps"] == "FAIL"


def test_low_satellite_count_blocks_arming():
    r = preflight_check(make_app(snap(satellites=4), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["gps"] == "FAIL"


def test_poor_hdop_blocks_arming():
    r = preflight_check(make_app(snap(hdop=4.0), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["gps"] == "FAIL"


def test_obstacle_avoidance_enabled_without_sensor_is_unknown():
    """Never claim obstacle detection PASS just because it is switched on."""
    r = preflight_check(make_app(snap(), obstacle_enabled=True,
                                 services=SimpleNamespace(camera=FakeCam(),
                                                         proximity=FakeProx(False))))
    assert r["checks"]["obstacle_detection"] == "UNKNOWN"


def test_obstacle_avoidance_disabled_is_skipped():
    r = preflight_check(make_app(snap(), obstacle_enabled=False,
                                 services=SimpleNamespace(camera=FakeCam(),
                                                         proximity=FakeProx())))
    assert r["checks"]["obstacle_detection"] == "SKIPPED"


def test_camera_unavailable_is_reported():
    r = preflight_check(make_app(snap(), services=SimpleNamespace(
        camera=FakeCam("CAMERA_UNAVAILABLE"), proximity=FakeProx())))
    assert r["checks"]["camera"] == "WARNING"


def test_geofence_outside_blocks_when_required():
    ring = [[[77.0, 12.0], [77.1, 12.0], [77.1, 12.1], [77.0, 12.1], [77.0, 12.0]]]
    rows = [("{\"type\":\"Polygon\",\"coordinates\":%s}" % ring,)]
    safety = SafetyConfig(geofence_required=True)
    r = preflight_check(make_app(
        snap(lat=20.0, lon=20.0), boundaries=rows, safety=safety,
        services=SimpleNamespace(camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["geofence"] == "FAIL"
    assert r["ready"] is False


def test_geofence_inside_passes_when_required():
    ring = [[[77.0, 12.0], [77.1, 12.0], [77.1, 12.1], [77.0, 12.1], [77.0, 12.0]]]
    rows = [("{\"type\":\"Polygon\",\"coordinates\":%s}" % ring,)]
    safety = SafetyConfig(geofence_required=True)
    r = preflight_check(make_app(
        snap(lat=12.05, lon=77.05), boundaries=rows, safety=safety,
        services=SimpleNamespace(camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["geofence"] == "PASS"


def test_geofence_skipped_when_not_required():
    r = preflight_check(make_app(snap(), services=SimpleNamespace(
        camera=FakeCam(), proximity=FakeProx())))
    assert r["checks"]["geofence"] == "SKIPPED"


# ----------------------------------------------------------------- failsafe
def failsafe_app(battery, *, armed=True, alt=30.0, connected=True):
    s = snap(battery_remaining_pct=battery, armed=armed, relative_alt_m=alt,
             connected=connected)
    sent = []

    class Master:
        def __init__(self):
            self.mav = self
            self.message_hooks = []

        def emit(self, mtype, **f):
            m = SimpleNamespace(_type=mtype, **f)
            m.get_type = lambda: mtype
            for h in list(self.message_hooks):
                h(self, m)
            return m

        def command_long_send(self, *a):
            sent.append(("command", a[2]))
            self.emit("COMMAND_ACK", command=a[2], result=0)

        def set_mode_send(self, *a):
            sent.append(("mode", a[2]))

    master = Master()

    class Conn:
        def __init__(self):
            self.master = master
            self.connected = connected
            self.target_system = 1
            self.target_component = 1
            self.px4_status = []

    app = make_app(s)
    app.conn = Conn()
    app.cfg.safety = SafetyConfig(rtl_battery_percent=25, land_battery_percent=15)
    return app, sent


def tick_n(app, n=1):
    fs = FailsafeSupervisor(app)
    for _ in range(n):
        asyncio.run(fs.tick())
    return fs


def test_critical_battery_commands_land_after_hysteresis():
    app, sent = failsafe_app(10)
    tick_n(app, 1)
    assert sent == [], "a single sample must not command the aircraft"
    tick_n(app, 2)
    assert any(s == ("command", 21) for s in sent), sent


def test_low_battery_commands_rtl_after_hysteresis():
    app, sent = failsafe_app(20)
    tick_n(app, 4)
    assert any(s == ("command", 20) for s in sent), sent


def test_healthy_battery_commands_nothing():
    app, sent = failsafe_app(80)
    tick_n(app, 10)
    assert sent == []


def test_no_command_while_disarmed():
    app, sent = failsafe_app(5, armed=False, alt=0.0)
    tick_n(app, 6)
    assert sent == [], "grounded aircraft must never be commanded"


def test_no_command_below_minimum_altitude():
    app, sent = failsafe_app(5, armed=True, alt=0.2)
    tick_n(app, 6)
    assert sent == []


def test_repeated_critical_battery_is_rate_limited():
    app, sent = failsafe_app(5)
    fs = FailsafeSupervisor(app)
    for _ in range(12):
        asyncio.run(fs.tick())
    lands = [s for s in sent if s == ("command", 21)]
    assert len(lands) == 1, f"expected one landing command, got {len(lands)}"


def test_battery_recovery_resets_the_streak():
    app, sent = failsafe_app(10)
    fs = FailsafeSupervisor(app)
    asyncio.run(fs.tick())
    app.tstore.snap.battery_remaining_pct = 90
    asyncio.run(fs.tick())
    app.tstore.snap.battery_remaining_pct = 10
    asyncio.run(fs.tick())
    assert not [s for s in sent if s == ("command", 21)], sent


def test_link_loss_is_reported_not_commanded():
    app, sent = failsafe_app(80, connected=False)
    fs = FailsafeSupervisor(app)
    report = asyncio.run(fs.tick())
    assert "MAVLINK_LOST" in report["advice"]
    assert sent == [], "link loss is the flight controller's failsafe, not ours"


def test_ekf_error_is_reported():
    app, _sent = failsafe_app(80)
    app.tstore.snap.ekf_flags = 0x08
    report = asyncio.run(FailsafeSupervisor(app).tick())
    assert "EKF_ESTIMATE_ERROR" in report["advice"]
    assert report["ekf_ok"] is False
