"""Proximity sensor rig for obstacle avoidance.

Directional rangefinders are optional hardware: serial sensors (e.g. TFmini
style, centimetre lines) can be bound per direction via behaviour config.
With no sensors fitted every distance reports null and the state is
UNKNOWN / OBSTACLE_SENSOR_UNAVAILABLE — never invented numbers.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("drone.perception.proximity")

DIRECTIONS = ("front", "rear", "left", "right", "down")


class SerialRangeSensor:
    """Line-oriented serial rangefinder; expects a centimetre value per line."""

    def __init__(self, port: str, baud: int = 115200):
        self.port, self.baud = port, baud
        self._ser = None
        self._lock = threading.Lock()

    def _open(self):
        if self._ser is not None:
            return self._ser
        try:
            import serial
            self._ser = serial.Serial(self.port, self.baud, timeout=0.15)
            log.info("proximity sensor on %s", self.port)
        except Exception as e:
            log.warning("proximity sensor %s unavailable: %s", self.port, e)
            self._ser = None
        return self._ser

    def get_distance_m(self) -> float | None:
        with self._lock:
            ser = self._open()
            if ser is None:
                return None
            try:
                raw = ser.readline().decode("ascii", errors="ignore").strip()
                if not raw:
                    return None
                value = float("".join(c for c in raw if c.isdigit() or c == "."))
                if value <= 0:
                    return None
                return round(value / 100.0, 2)  # cm -> m
            except Exception:
                return None


class ProximityRig:
    def __init__(self, cfg, sensor_ports: dict[str, str] | None = None):
        from ..perception.obstacle_detection import ObstacleDetector

        self.cfg = cfg
        self.detector = ObstacleDetector(
            warning=cfg.warning_distance_m,
            critical=cfg.critical_distance_m,
            emergency=cfg.emergency_distance_m,
        )
        self.sensors: dict[str, SerialRangeSensor | None] = {d: None for d in DIRECTIONS}
        for direction, port in (sensor_ports or {}).items():
            if direction in self.sensors and port:
                self.sensors[direction] = SerialRangeSensor(str(port))

    def _distance(self, direction: str, snap) -> float | None:
        if direction == "down":
            # downward clearance is real telemetry (height above ground)
            alt = snap.relative_alt_m
            return round(alt, 2) if alt and alt > 0 else 0.0
        sensor = self.sensors.get(direction)
        if sensor is None:
            return None
        return sensor.get_distance_m()

    def status(self, snap) -> dict:
        distances = {d: self._distance(d, snap) for d in DIRECTIONS}
        forward = [distances[d] for d in ("front", "rear", "left", "right") if distances[d] is not None]

        if not self.cfg.enabled:
            state, evaluate_status = "DISABLED", "OBSTACLE_AVOIDANCE_DISABLED"
            evaluated = {"state": "UNKNOWN", "status": evaluate_status}
            minimum = None
        elif not forward:
            evaluated = self.detector.evaluate(None)
            state, minimum = evaluated["state"], None
        else:
            minimum = min(forward)
            evaluated = self.detector.evaluate(minimum)
            state = evaluated["state"]

        return {
            "enabled": bool(self.cfg.enabled),
            "available": len(forward) > 0,
            "sensor_status": "READY" if forward else "OBSTACLE_SENSOR_UNAVAILABLE",
            "state": state,
            "distances_m": distances,
            "min_distance_m": minimum,
            "thresholds_m": {
                "warning": self.cfg.warning_distance_m,
                "critical": self.cfg.critical_distance_m,
                "emergency": self.cfg.emergency_distance_m,
            },
        }
