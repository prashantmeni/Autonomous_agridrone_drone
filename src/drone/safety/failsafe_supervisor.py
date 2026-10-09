"""Failsafe supervisor: watch battery/link/GPS and act while airborne.

The flight controller keeps its own failsafes and stays authoritative — PX4
handles datalink loss and low battery on its own. This loop adds the companion
computer's view: it applies hysteresis so a single noisy sample cannot command
the aircraft, refuses to act on the ground, rate-limits every action, and
broadcasts what it decided.

Actions are deliberately conservative: land at critical battery, return home at
low battery, and only ever report (never command) link or GPS problems.
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..telemetry import publisher as pub

log = logging.getLogger("drone.safety.failsafe")

# Consecutive evaluations before an action is committed.
TRIGGER_COUNT = {"RTL": 3, "LAND": 2}
# Minimum seconds between two commands of the same kind.
COOLDOWN_S = {"RTL": 60.0, "LAND": 120.0}
# Below this relative altitude we treat the vehicle as grounded.
AIRBORNE_MIN_ALT_M = 1.0


class FailsafeSupervisor:
    def __init__(self, app):
        self.app = app
        self._streaks: dict[str, int] = {}
        self._last_action: dict[str, float] = {}
        self.last_report: dict | None = None

    # ------------------------------------------------------------ helpers
    def _airborne(self) -> bool:
        snap = self.app.tstore.snap
        return bool(getattr(snap, "armed", False)) and \
            float(getattr(snap, "relative_alt_m", 0.0) or 0.0) > AIRBORNE_MIN_ALT_M

    def _battery_pct(self) -> int:
        raw = self.app.tstore.snap.battery_remaining_pct
        return int(raw) if raw is not None and raw >= 0 else -1

    def _armed(self) -> bool:
        return bool(getattr(self.app.tstore.snap, "armed", False))

    def _commit(self, action: str) -> bool:
        """Record an action if it is not rate limited. Returns True if allowed."""
        now = time.time()
        last = self._last_action.get(action, 0.0)
        if now - last < COOLDOWN_S.get(action, 60.0):
            return False
        self._last_action[action] = now
        self._streaks[action] = 0
        return True

    def _streak(self, action: str, present: bool) -> int:
        if present:
            self._streaks[action] = self._streaks.get(action, 0) + 1
        else:
            self._streaks[action] = 0
        return self._streaks[action]

    async def _command(self, action: str, reason: str) -> None:
        from ..mavlink import commands as C

        if not self._commit(action):
            log.info("failsafe %s suppressed (cooldown): %s", action, reason)
            return
        log.warning("FAILSAFE %s: %s", action, reason)
        try:
            self.app.db.log_event("WARNING", "failsafe", action, {"reason": reason})
        except Exception:  # noqa: BLE001
            pass
        if not self._armed():
            return
        try:
            if action == "LAND":
                await asyncio.to_thread(C.land, self.app.conn)
            elif action == "RTL":
                await asyncio.to_thread(C.rtl, self.app.conn)
            await pub.publisher.broadcast(
                {"type": "failsafe", "data": {"action": action, "reason": reason}})
        except Exception as e:  # noqa: BLE001
            log.error("failsafe %s failed to send: %s", action, e)

    async def _report(self, payload: dict) -> None:
        if payload != self.last_report:
            self.last_report = payload
            try:
                await pub.publisher.broadcast({"type": "failsafe", "data": payload})
            except Exception:  # noqa: BLE001
                pass

    # --------------------------------------------------------------- tick
    async def tick(self) -> dict:
        snap = self.app.tstore.snap
        cfg = self.app.cfg.safety
        battery = self._battery_pct()
        airborne = self._airborne()

        land = 0 <= battery < int(getattr(cfg, "land_battery_percent", 15))
        rtl = (not land) and 0 <= battery < int(getattr(cfg, "rtl_battery_percent", 25))

        if land and airborne and self._streak("LAND", True) >= TRIGGER_COUNT["LAND"]:
            await self._command("LAND", f"battery {battery}% at or below "
                                       f"{getattr(cfg, 'land_battery_percent', 15)}%")
        elif rtl and airborne and self._streak("RTL", True) >= TRIGGER_COUNT["RTL"]:
            await self._command("RTL", f"battery {battery}% below "
                                       f"{getattr(cfg, 'rtl_battery_percent', 25)}%")
        else:
            self._streak("LAND", land)
            self._streak("RTL", rtl)

        # Reported, never commanded: the flight controller owns these.
        report = {
            "airborne": airborne,
            "armed": self._armed(),
            "battery_pct": battery,
            "connected": bool(getattr(snap, "connected", False)),
            "gps_fix": int(getattr(snap, "gps_fix", 0) or 0),
            "ekf_ok": bool(getattr(snap, "ekf_estimate_ok", False)),
            "ekf_issues": list(getattr(snap, "ekf_issues", []) or []),
            "advice": [],
        }
        if not snap.connected:
            report["advice"].append("MAVLINK_LOST")
        if cfg.gps_required and (snap.gps_fix or 0) < 3:
            report["advice"].append("GPS_DEGRADED")
        if snap.ekf_known and not snap.ekf_estimate_ok:
            report["advice"].append("EKF_ESTIMATE_ERROR")
        if 0 <= battery < int(getattr(cfg, "rtl_battery_percent", 25)):
            report["advice"].append("BATTERY_LOW")
        if land:
            report["advice"].append("BATTERY_CRITICAL")
        await self._report(report)
        return report

    async def run(self, interval_s: float = 2.0) -> None:
        while True:
            try:
                await self.tick()
            except Exception as e:  # noqa: BLE001 - failsafe must never die
                log.error("failsafe supervisor: %s", e)
            await asyncio.sleep(interval_s)
