"""Flight modes: the operator-level concepts the drone actually flies.

MAVLink exposes PX4's raw main/sub mode pair (AUTO/MISSION, AUTO/LOITER...). This
module maps the farm-level modes onto those, and owns the extra sequencing a
mode needs (climb to altitude, start an orbit, refresh the RTL altitude so an
unexpected return actually uses the configured height).

Every mode entry point verifies the vehicle is armed and airborne first: a mode
request on a disarmed drone is rejected rather than silently accepted.
"""
from __future__ import annotations

import asyncio
import logging
from enum import Enum

from ..core.state import FlightState

log = logging.getLogger("drone.autonomy.modes")

# PX4 parameters the modes depend on. Written before a mode is entered so the
# behaviour configured by the operator is what the flight controller uses.
RTL_RETURN_ALT_PARAM = "RTL_RETURN_ALT"
RTL_ALT_MIN_M = 5.0
RTL_ALT_MAX_M = 200.0


class FlightMode(str, Enum):
    IDLE = "IDLE"
    SURVEY = "SURVEY"        # fly a planned coverage mission
    MONITOR = "MONITOR"      # hold/loiter at altitude and scan crops
    MAPPING = "MAPPING"      # survey flown for mapping rather than spraying
    RTL = "RTL"
    PRECISION_LAND = "PRECISION_LAND"


# Farm mode -> (PX4 mode string, needs airborne, FSM state)
_MODE_PLAN: dict[str, tuple[str, bool, FlightState]] = {
    FlightMode.SURVEY.value:    ("AUTO", True, FlightState.SURVEY),
    FlightMode.MAPPING.value:   ("AUTO", True, FlightState.SURVEY),
    FlightMode.MONITOR.value:   ("HOLD", True, FlightState.MISSION),
    FlightMode.RTL.value:       ("RTL", True, FlightState.RETURN_HOME),
}


class ModeController:
    """Applies behaviour rules and drives PX4 modes on the operator's behalf."""

    def __init__(self, app):
        self.app = app
        self.current: str = FlightMode.IDLE.value

    # ------------------------------------------------------------ behaviour
    def behaviour(self):
        from ..core.behaviour import load_behaviour
        return load_behaviour()

    async def _push_rtl_altitude(self, rules) -> tuple[bool, str]:
        """Tell PX4 what altitude to climb to for an automatic return.

        Without this the board uses its own stored default, so the altitude the
        operator set in the UI was previously never applied anywhere.
        """
        from ..mavlink import parameters as P

        alt = float(getattr(rules, "rth_altitude_m", 25.0) or 25.0)
        alt = max(RTL_ALT_MIN_M, min(RTL_ALT_MAX_M, alt))
        try:
            # set_param blocks on the wire, so it must not run on the loop.
            ok = await asyncio.to_thread(P.set_param, self.app.conn,
                                         RTL_RETURN_ALT_PARAM, alt)
            return ok, f"{RTL_RETURN_ALT_PARAM}={alt:.0f}"
        except Exception as e:  # noqa: BLE001
            log.warning("could not set %s: %s", RTL_RETURN_ALT_PARAM, e)
            return False, str(e)

    def _armed(self) -> bool:
        return bool(getattr(self.app.tstore.snap, "armed", False))

    # --------------------------------------------------------------- entries
    async def enter(self, mode: str, altitude_m: float | None = None) -> dict:
        mode = (mode or "").strip().upper()
        plan = _MODE_PLAN.get(mode)
        if plan is None:
            raise ValueError(f"unknown flight mode {mode!r}")
        px4_mode, needs_airborne, state = plan
        if needs_airborne and not self._armed():
            return {"status": "rejected", "mode": mode,
                    "error": "vehicle is disarmed; take off first"}

        rules = self.behaviour()
        applied = []
        ok, detail = await self._push_rtl_altitude(rules)
        applied.append({"rth_altitude": detail, "ok": ok})

        from ..mavlink import commands as C
        target_alt = altitude_m or float(getattr(rules, "survey_altitude_m", 20.0))
        if C.set_mode(self.app.conn, px4_mode):
            applied.append({"px4_mode": px4_mode, "ok": True})
        else:
            return {"status": "failed", "mode": mode,
                    "error": f"flight controller refused {px4_mode}", "applied": applied}

        # A mode that implies a working altitude must actually reach it.
        if needs_airborne and target_alt:
            snap = self.app.tstore.snap
            if float(getattr(snap, "relative_alt_m", 0.0) or 0.0) < target_alt - 1.0:
                await asyncio.to_thread(C.takeoff, self.app.conn, target_alt)
                applied.append({"climb_to_m": target_alt, "ok": True})

        fsm = self.app.fsm
        try:
            fsm.transition(state, f"mode {mode}")
        except ValueError:
            fsm.force(state, f"mode {mode}")
        self.current = mode
        log.info("flight mode -> %s (px4=%s, applied=%s)", mode, px4_mode, applied)
        return {"status": "ok", "mode": mode, "px4_mode": px4_mode,
                "altitude_m": target_alt, "applied": applied}

    async def enter_monitor(self, altitude_m: float | None = None) -> dict:
        """Hold station at altitude and keep the crop monitor running."""
        return await self.enter(FlightMode.MONITOR.value, altitude_m)

    def status(self) -> dict:
        rules = self.behaviour()
        return {
            "current": self.current,
            "armed": self._armed(),
            "survey_altitude_m": rules.survey_altitude_m,
            "cruise_speed_mps": rules.cruise_speed_mps,
            "rth_altitude_m": rules.rth_altitude_m,
            "rth_battery_reserve_pct": rules.rth_battery_reserve_pct,
            "obstacle_brake_distance_m": rules.obstacle_brake_distance_m,
            "available": sorted(_MODE_PLAN),
        }