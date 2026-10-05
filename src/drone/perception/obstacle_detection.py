"""Obstacle framework: sensor-agnostic risk states."""
from __future__ import annotations
from enum import Enum

class ObstacleState(str, Enum):
    CLEAR = "CLEAR"; WARNING = "WARNING"; OBSTACLE_DETECTED = "OBSTACLE_DETECTED"
    PATH_BLOCKED = "PATH_BLOCKED"; EMERGENCY_STOP = "EMERGENCY_STOP"

class ObstacleSensor:
    def get_distance_m(self) -> float | None: raise NotImplementedError
    @property
    def status(self) -> str: return "UNAVAILABLE"

class ObstacleDetector:
    def __init__(self, warning=5.0, critical=2.0, emergency=1.0, sensor: ObstacleSensor | None = None):
        self.w, self.c, self.e = warning, critical, emergency
        self.sensor = sensor
    def evaluate(self, distance: float | None = None) -> dict:
        d = distance if distance is not None else (self.sensor.get_distance_m() if self.sensor else None)
        if d is None: return {"state": "UNKNOWN", "status": "OBSTACLE_SENSOR_UNAVAILABLE"}
        if d <= self.e: return {"state": ObstacleState.EMERGENCY_STOP.value, "distance_m": d}
        if d <= self.c: return {"state": ObstacleState.PATH_BLOCKED.value, "distance_m": d}
        if d <= self.w: return {"state": ObstacleState.WARNING.value, "distance_m": d}
        return {"state": ObstacleState.CLEAR.value, "distance_m": d}
