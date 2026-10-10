"""Obstacle avoidance: ultrasonic driver, braking discipline, camera hazards."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from drone.core.behaviour import BehaviourRules  # noqa: E402
from drone.perception.avoidance import (  # noqa: E402
    BRAKE_COOLDOWN_S,
    AvoidanceSupervisor,
    ObstacleLevel,
)
from drone.perception.hazard_detection import HazardDetector  # noqa: E402
from drone.perception.proximity import ProximityRig  # noqa: E402
from drone.perception.ultrasonic import (  # noqa: E402
    MAX_RANGE_M,
    MIN_RANGE_M,
    US_PER_CM,
    HcSr04,
    SimulatedUltrasonic,
)


# ------------------------------------------------------------------ sensor
def test_simulated_sensor_reads_a_known_distance():
    backend = SimulatedUltrasonic(distance_m=1.5)
    sensor = HcSr04(23, 24, backend=backend, samples=1)
    assert sensor.measure_m() == pytest.approx(1.5, abs=0.01)


def test_pulse_width_converts_to_metres():
    backend = SimulatedUltrasonic(distance_m=2.0)
    assert backend.pulse_width_us(24) == pytest.approx(2.0 * 100 * US_PER_CM)


def test_no_echo_returns_none_rather_than_zero():
    class SilentBackend(SimulatedUltrasonic):
        def pulse_width_us(self, pin, timeout_s=0.04):
            return 0.0

    sensor = HcSr04(23, 24, backend=SilentBackend(), samples=2)
    assert sensor.measure_m() is None
    assert sensor.last_error == "no echo"


def test_out_of_range_readings_are_discarded():
    class SpikyBackend(SimulatedUltrasonic):
        def __init__(self):
            super().__init__(1.0)
            self.n = 0

        def pulse_width_us(self, pin, timeout_s=0.04):
            self.n += 1
            # A single impossible spike, then two good samples.
            if self.n == 1:
                return 9_000_000.0        # ~1.5 km: reflection artefact
            return 1.0 * 100 * US_PER_CM

    sensor = HcSr04(23, 24, backend=SpikyBackend(), samples=3)
    value = sensor.measure_m()
    assert value == pytest.approx(1.0, abs=0.01), "median must reject the spike"


def test_implausible_range_bounds_are_respected():
    backend = SimulatedUltrasonic(distance_m=MAX_RANGE_M * 10)
    sensor = HcSr04(23, 24, backend=backend, samples=2)
    assert sensor.measure_m() is None

    backend2 = SimulatedUltrasonic(distance_m=MIN_RANGE_M / 10)
    sensor2 = HcSr04(23, 24, backend=backend2, samples=2)
    assert sensor2.measure_m() is None


def test_median_smooths_disagreeing_samples():
    class MixedBackend(SimulatedUltrasonic):
        def __init__(self):
            super().__init__(1.0)
            self.seq = [0.5, 1.5, 1.0]

        def pulse_width_us(self, pin, timeout_s=0.04):
            d = self.seq.pop(0) if self.seq else 1.0
            return d * 100 * US_PER_CM

    sensor = HcSr04(23, 24, backend=MixedBackend(), samples=3)
    assert sensor.measure_m() == pytest.approx(1.0, abs=0.01)


def test_sensor_writes_configures_the_pins():
    backend = SimulatedUltrasonic(1.0)
    sensor = HcSr04(23, 24, backend=backend, samples=1)
    sensor.measure_m()
    assert backend.pins[23] == "out"
    assert backend.pins[24] == "in"
    assert (23, True) in backend.writes and (23, False) in backend.writes


# ------------------------------------------------------------ proximity rig
def test_rig_builds_ultrasonic_from_config():
    """A real ultrasonic needs GPIO; without it the rig degrades honestly."""
    from drone.core.config import ObstacleConfig
    cfg = ObstacleConfig(enabled=True, sensors={
        "front": {"type": "ultrasonic", "trig_pin": 23, "echo_pin": 24},
    })
    rig = ProximityRig(cfg)
    # On a machine with no GPIO library the sensor cannot be created, and the
    # rig must report that rather than invent a distance.
    assert rig.sensors["front"] is None


def test_rig_builds_a_simulated_sensor_for_bench_testing():
    from drone.core.config import ObstacleConfig
    cfg = ObstacleConfig(enabled=True, sensors={
        "front": {"type": "simulated", "distance_m": 1.2},
    })
    rig = ProximityRig(cfg)
    assert rig.sensors["front"] is not None
    snap = SimpleNamespace(relative_alt_m=10.0)
    st = rig.status(snap)
    assert st["available"] is True
    assert st["distances_m"]["front"] == pytest.approx(1.2, abs=0.05)


def test_rig_reports_unavailable_with_no_sensors():
    from drone.core.config import ObstacleConfig
    rig = ProximityRig(ObstacleConfig(enabled=True))
    snap = SimpleNamespace(relative_alt_m=10.0)
    st = rig.status(snap)
    assert st["available"] is False
    assert st["sensor_status"] == "OBSTACLE_SENSOR_UNAVAILABLE"


def test_rig_survives_a_broken_sensor_config():
    from drone.core.config import ObstacleConfig
    cfg = ObstacleConfig(enabled=True, sensors={"front": {"type": "nonsense"}})
    rig = ProximityRig(cfg)
    assert rig.sensors["front"] is None


# ---------------------------------------------------------------- avoidance
class FakeProx:
    def __init__(self, distances):
        self._d = distances

    def status(self, snap):
        return {"distances_m": self._d, "available": bool(self._d)}


def avoid_app(distances, *, armed=True, alt=20.0, brake_enabled=True):
    import drone.runtime as runtime

    sent = {"modes": [], "events": []}

    class Master:
        def __init__(self):
            self.message_hooks = []
            self.mav = self

        def set_mode_send(self, *a):
            sent["modes"].append(a[2])

    master = Master()
    app = SimpleNamespace(
        tstore=SimpleNamespace(snap=SimpleNamespace(armed=armed, relative_alt_m=alt)),
        conn=SimpleNamespace(master=master, connected=True, target_system=1,
                             target_component=1, px4_status=[]),
        db=SimpleNamespace(log_event=lambda *a: sent["events"].append(a)),
    )
    runtime.get_services = lambda _app: SimpleNamespace(proximity=FakeProx(distances))
    import drone.core.behaviour as behaviour_mod
    behaviour_mod.load_behaviour = lambda *a, **k: BehaviourRules(
        obstacle_brake_distance_m=3.0)
    app._sent = sent
    return app


def tick(app, n=1):
    sup = AvoidanceSupervisor(app)
    for _ in range(n):
        asyncio.run(sup.tick())
    return sup


def test_no_sensors_means_clear_and_no_command():
    app = avoid_app({})
    sup = tick(app, 6)
    assert sup.level == ObstacleLevel.CLEAR
    assert app._sent["modes"] == []


def test_close_front_sensor_triggers_a_brake_hold():
    app = avoid_app({"front": 1.0})
    sup = tick(app, 4)
    assert sup.level == ObstacleLevel.BRAKE
    assert app._sent["modes"], "a brake should switch the FC to HOLD"


def test_single_close_reading_does_not_brake():
    """Hysteresis: one noisy sample must not command the aircraft."""
    app = avoid_app({"front": 0.5})
    tick(app, 1)
    assert app._sent["modes"] == []


def test_brake_is_rate_limited():
    app = avoid_app({"front": 0.4})
    sup = AvoidanceSupervisor(app)
    for _ in range(20):
        asyncio.run(sup.tick())
    assert len(app._sent["modes"]) <= 2, app._sent["modes"]


def test_no_brake_while_disarmed():
    app = avoid_app({"front": 0.3}, armed=False, alt=0.0)
    tick(app, 6)
    assert app._sent["modes"] == []


def test_warning_level_between_thresholds():
    app = avoid_app({"front": 5.0})
    sup = tick(app, 4)
    assert sup.level == ObstacleLevel.WARNING
    assert app._sent["modes"] == [], "a warning must not command"


def test_level_clears_after_sustained_clear_readings():
    app = avoid_app({"front": 0.5})
    sup = AvoidanceSupervisor(app)
    for _ in range(4):
        asyncio.run(sup.tick())
    assert sup.level == ObstacleLevel.BRAKE
    import drone.runtime as runtime
    runtime.get_services = lambda _app: SimpleNamespace(proximity=FakeProx({"front": 30.0}))
    for _ in range(6):
        asyncio.run(sup.tick())
    assert sup.level == ObstacleLevel.CLEAR


def test_evaluate_level_uses_the_closest_forward_sensor():
    sup = AvoidanceSupervisor(SimpleNamespace())
    assert sup.evaluate_level({"front": 10.0, "left": 1.0}, 3.0, 6.0) == ObstacleLevel.BRAKE
    assert sup.evaluate_level({"front": 10.0, "left": 5.0}, 3.0, 6.0) == ObstacleLevel.WARNING
    assert sup.evaluate_level({"front": 20.0}, 3.0, 6.0) == ObstacleLevel.CLEAR
    assert sup.evaluate_level({}, 3.0, 6.0) == ObstacleLevel.CLEAR


def test_cooldown_constant_is_sane():
    assert BRAKE_COOLDOWN_S >= 5.0


# ------------------------------------------------------------ camera hazard
def test_uniform_frame_is_clear():
    det = HazardDetector()
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    res = det.detect(frame)
    assert res.hazard is False


def test_none_frame_is_handled():
    res = HazardDetector().detect(None)
    assert res.hazard is False and "no frame" in res.reason


def test_moving_object_in_forward_view_is_flagged():
    det = HazardDetector(motion_threshold=0.02)
    quiet = np.full((240, 320, 3), 60, dtype=np.uint8)
    det.detect(quiet)                      # establish a background frame
    moved = quiet.copy()
    moved[80:200, 100:220] = 235          # something appears in the forward ROI
    res = det.detect(moved)
    assert res.hazard is True
    assert res.kind in ("motion", "person")


def test_static_frame_after_a_movement_settles():
    det = HazardDetector(motion_threshold=0.02)
    a = np.full((240, 320, 3), 60, dtype=np.uint8)
    det.detect(a)
    b = a.copy()
    b[80:200, 100:220] = 235
    det.detect(b)
    res = det.detect(b)                    # unchanged frame again: no new motion
    assert res.motion == pytest.approx(0.0, abs=1e-6)


def test_result_serialises_for_the_websocket():
    det = HazardDetector()
    res = det.detect(np.full((240, 320, 3), 100, dtype=np.uint8))
    payload = res.to_dict()
    assert set(payload) >= {"hazard", "kind", "confidence", "motion", "reason"}