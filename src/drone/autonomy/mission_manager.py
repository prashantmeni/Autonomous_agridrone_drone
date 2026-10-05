"""Mission manager: create/validate/start/abort. PX4 flies, Pi supervises."""
from __future__ import annotations
import logging, uuid, asyncio
from ..core.state import FlightStateMachine, FlightState
log = logging.getLogger("drone.autonomy.mission")

class MissionManager:
    def __init__(self, app):
        self.app = app
        self.active_id: str | None = None
        self.progress: dict = {"done": 0, "total": 0}

    async def start(self, mission_id: str, body: dict) -> dict:
        from .waypoint_manager import validate_mission
        ok, errs = validate_mission(body, self.app.cfg.mission.max_altitude_m, self.app.cfg.mission.max_speed_mps)
        if not ok: raise ValueError(f"mission invalid: {errs}")
        self.app.db.save_mission(mission_id or str(uuid.uuid4()), body.get("name", mission_id), body)
        self.active_id = mission_id
        wps = body.get("waypoints", [])
        self.progress = {"done": 0, "total": len(wps)}
        try: self.app.fsm.transition(FlightState.MISSION, f"mission {mission_id} start")
        except ValueError: self.app.fsm.force(FlightState.MISSION, f"mission {mission_id} start")
        log.info(f"mission {mission_id} started with {len(wps)} waypoints")
        return {"mission_id": mission_id, "status": "started"}

    async def abort(self, reason: str = "user") -> dict:
        if self.app.fsm.state not in (FlightState.DISARMED, FlightState.LANDED):
            try: self.app.fsm.transition(FlightState.ABORT, reason)
            except ValueError: self.app.fsm.force(FlightState.ABORT, reason)
        from ..mavlink import commands as C
        await asyncio.to_thread(C.rtl, self.app.conn)
        return {"status": "aborting", "reason": reason}
