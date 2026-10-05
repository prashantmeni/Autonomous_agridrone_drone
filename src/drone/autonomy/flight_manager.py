"""High-level flight orchestration with preflight gate. Never blindly commands flight."""
from __future__ import annotations
import asyncio, logging
from ..core.state import FlightState
log = logging.getLogger("drone.autonomy.flight")

class FlightManager:
    def __init__(self, app):
        self.app = app

    async def takeoff(self, alt_m: float) -> dict:
        from ..safety.flight_safety import preflight_check
        report = preflight_check(self.app)
        if not report["ready"]:
            raise RuntimeError(f"preflight FAIL: {report}")
        from ..mavlink import commands as C
        fsm = self.app.fsm
        try: fsm.transition(FlightState.ARMING, "takeoff requested")
        except ValueError: fsm.force(FlightState.ARMING, "takeoff requested")
        ok, why = await asyncio.to_thread(C.arm, self.app.conn)
        if not ok: fsm.force(FlightState.DISARMED, "arm failed"); raise RuntimeError(f"arm failed: {why}")
        fsm.transition(FlightState.TAKEOFF, f"takeoff to {alt_m}m")
        ok, why = await asyncio.to_thread(C.takeoff, self.app.conn, alt_m)
        if not ok:
            # never leave the vehicle armed after a rejected takeoff
            d_ok, d_why = await asyncio.to_thread(C.disarm, self.app.conn)
            if d_ok:
                fsm.force(FlightState.DISARMED, f"takeoff failed: {why}")
            else:
                fsm.force(FlightState.EMERGENCY, f"takeoff failed ({why}) and disarm failed ({d_why})")
            msg = f"takeoff failed: {why}"
            if not d_ok:
                msg += f"; disarm also failed: {d_why}"
            raise RuntimeError(msg)
        return {"status": "takeoff commanded", "alt_m": alt_m}

    async def rtl(self): 
        from ..mavlink import commands as C
        if not self.app.tstore.snap.armed:
            self.app.fsm.force(FlightState.DISARMED, "RTL ignored - vehicle disarmed")
            return {"ok": True, "skipped": "disarmed"}
        self.app.fsm.force(FlightState.RETURN_HOME, "RTL requested")
        return {"ok": await asyncio.to_thread(C.rtl, self.app.conn)}

    async def land(self):
        from ..mavlink import commands as C
        if not self.app.tstore.snap.armed:
            self.app.fsm.force(FlightState.DISARMED, "land ignored - vehicle disarmed")
            return {"ok": True, "skipped": "disarmed"}
        self.app.fsm.force(FlightState.LANDING, "land requested")
        return {"ok": await asyncio.to_thread(C.land, self.app.conn)}
