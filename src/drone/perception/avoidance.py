"""Obstacle avoidance supervisor: rangefinders + forward-camera hazards.

Two inputs, deliberately different in authority:

* **Rangefinders** (ultrasonic) drive braking. When clearance falls below the
  operator's brake distance the drone stops advancing and holds. Commands are
  gated on being armed and airborne, use hysteresis so one noisy reading cannot
  bounce the aircraft, and are rate limited.
* **Camera hazards** (a person or vehicle in the forward view) are *reported*
  only. Alerting and logging a hazard is reliable at this frame rate; steering
  from a 5 FPS camera feed is not, so the camera never commands the aircraft.

PX4 keeps its own distance-sensor failsafes; this layer is the companion
computer's additional reaction and never overrides an active autopilot failsafe.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..telemetry import publisher as pub

log = logging.getLogger("drone.perception.avoidance")

# Consecutive samples before acting / clearing.
TRIGGER_N = 3
CLEAR_N = 5
# Seconds between two brake commands.
BRAKE_COOLDOWN_S = 15.0
AIRBORNE_MIN_ALT_M = 1.0


class ObstacleLevel:
    CLEAR = "CLEAR"
    WARNING = "WARNING"
    BRAKE = "BRAKE"


class AvoidanceSupervisor:
    def __init__(self, app):
        self.app = app
        self.level = ObstacleLevel.CLEAR
        self.last_brake: float = 0.0
        self._warn_streak = 0
        self._clear_streak = 0
        self.last_report: dict | None = None
        self.hazard: dict | None = None

    # ------------------------------------------------------------- helpers
    def _behaviour(self):
        from ..core.behaviour import load_behaviour
        return load_behaviour()

    def _airborne(self) -> bool:
        snap = self.app.tstore.snap
        return bool(getattr(snap, "armed", False)) and \
            float(getattr(snap, "relative_alt_m", 0.0) or 0.0) > AIRBORNE_MIN_ALT_M

    def clearances(self) -> dict:
        """Forward-facing clearances from the rangefinder rig, if fitted."""
        try:
            from ..runtime import get_services
            status = get_services(self.app).proximity.status(self.app.tstore.snap)
            return dict(status.get("distances_m") or {})
        except Exception:  # noqa: BLE001 - no rig, or it failed: report nothing
            return {}

    @staticmethod
    def _usable(v) -> bool:
        return isinstance(v, (int, float)) and v >= 0

    def evaluate_level(self, distances: dict, brake_m: float, warn_m: float) -> str:
        """Closest forward-facing clearance decides the level."""
        forward = [distances.get(d) for d in ("front", "left", "right", "rear")]
        forward = [float(v) for v in forward if self._usable(v)]
        if not forward:
            return ObstacleLevel.CLEAR
        closest = min(forward)
        if closest <= brake_m:
            return ObstacleLevel.BRAKE
        if closest <= warn_m:
            return ObstacleLevel.WARNING
        return ObstacleLevel.CLEAR

    # -------------------------------------------------------------- actions
    async def _brake(self, reason: str, closest: float) -> None:
        from ..mavlink import commands as C

        now = time.time()
        if now - self.last_brake < BRAKE_COOLDOWN_S:
            return
        self.last_brake = now
        log.warning("OBSTACLE BRAKE: %s (closest %.2f m)", reason, closest)
        try:
            self.app.db.log_event("WARNING", "obstacle", "BRAKE",
                                  {"reason": reason, "closest_m": round(closest, 2)})
        except Exception:  # noqa: BLE001
            pass
        if not self._airborne():
            log.info("obstacle brake not commanded: vehicle is not airborne")
            return
        try:
            # HOLD stops forward motion without descending or landing.
            C.set_mode(self.app.conn, "HOLD")
            await pub.publisher.broadcast({"type": "obstacle", "data": {
                "action": "BRAKE", "reason": reason, "closest_m": round(closest, 2)}})
        except Exception as e:  # noqa: BLE001
            log.error("obstacle brake failed: %s", e)

    async def set_hazard(self, hazard: dict | None) -> None:
        """Record a camera hazard. Reported and logged, never commanded."""
        if hazard == self.hazard:
            return
        self.hazard = hazard
        try:
            if hazard:
                self.app.db.log_event("WARNING", "camera", "HAZARD", hazard)
            await pub.publisher.broadcast({"type": "hazard", "data": hazard or {"clear": True}})
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------- tick
    async def tick(self) -> dict:
        rules = self._behaviour()
        brake_m = float(getattr(rules, "obstacle_brake_distance_m", 3.0) or 3.0)
        warn_m = brake_m * 2.0
        distances = self.clearances()
        level = self.evaluate_level(distances, brake_m, warn_m)
        forward = [float(v) for v in
                   (distances.get(d) for d in ("front", "left", "right", "rear"))
                   if self._usable(v)]
        closest = min(forward) if forward else None

        if level == ObstacleLevel.BRAKE:
            self._warn_streak += 1
            self._clear_streak = 0
            if self._warn_streak >= TRIGGER_N:
                self.level = ObstacleLevel.BRAKE
                await self._brake("clearance below brake distance", closest or 0.0)
        elif level == ObstacleLevel.WARNING:
            self._warn_streak += 1
            self._clear_streak = 0
            if self._warn_streak >= TRIGGER_N:
                self.level = ObstacleLevel.WARNING
        else:
            self._clear_streak += 1
            if self._clear_streak >= CLEAR_N and self.level != ObstacleLevel.CLEAR:
                self.level = ObstacleLevel.CLEAR
                self._warn_streak = 0
                log.info("obstacle avoidance released (%s)", level)

        report = {
            "level": self.level,
            "clearances_m": distances,
            "closest_m": round(closest, 2) if closest is not None else None,
            "brake_distance_m": brake_m,
            "warn_distance_m": round(warn_m, 2),
            "airborne": self._airborne(),
            "hazard": self.hazard,
        }
        if report != self.last_report:
            self.last_report = report
            try:
                await pub.publisher.broadcast({"type": "obstacle_status", "data": report})
            except Exception:  # noqa: BLE001
                pass
        return report

    async def run(self, interval_s: float = 0.4) -> None:
        while True:
            try:
                await self.tick()
            except Exception as e:  # noqa: BLE001 - must never die in flight
                log.error("avoidance supervisor: %s", e)
            await asyncio.sleep(interval_s)