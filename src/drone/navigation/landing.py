"""Precision landing controller (sensor-agnostic). Vision loss aborts, never continues silently."""
from __future__ import annotations
import time

class LandingController:
    def __init__(self, max_descent: float = 0.8, lost_timeout_s: float = 2.0):
        self.max_descent = max_descent; self.lost_timeout = lost_timeout_s
        self._last_seen = 0.0; self.aborted = False

    def update(self, offset: tuple[float,float] | None, confidence: float) -> dict:
        if offset is None or confidence <= 0:
            lost = time.time() - self._last_seen
            if self._last_seen and lost > self.lost_timeout:
                self.aborted = True
                return {"action": "ABORT", "reason": "MARKER_LOST_TIMEOUT", "fallback": "GPS_LANDING"}
            return {"action": "HOLD", "reason": "MARKER_NOT_VISIBLE"}
        self._last_seen = time.time()
        return {"action": "CORRECT", "dx": offset[0], "dy": offset[1], "descent_mps": self.max_descent}
