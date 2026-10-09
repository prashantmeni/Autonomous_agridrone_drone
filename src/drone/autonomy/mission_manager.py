"""Mission execution: upload to the flight controller and supervise progress.

PX4 flies the mission; the Pi supervises it. This manager owns the whole
protocol exchange — clear, upload, arm-off, AUTO, then track MISSION_CURRENT
and MISSION_ITEM_REACHED until the mission completes — and reports honest
phase/progress. A phase only becomes EXECUTING after the flight controller has
acknowledged the upload and accepted AUTO, so nothing here can claim the drone
is flying a mission it has not accepted.
"""
from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

from ..core.state import FlightState
from ..mavlink import missions as M
from ..telemetry import publisher as pub
from .waypoint_manager import validate_mission

log = logging.getLogger("drone.autonomy.mission")

# MAV_CMD ids used when building the item list.
CMD_WAYPOINT = 16
CMD_TAKEOFF = 22
CMD_LAND = 21
CMD_RETURN_TO_LAUNCH = 20


class MissionPhase(str, Enum):
    IDLE = "IDLE"
    VALIDATING = "VALIDATING"
    UPLOADING = "UPLOADING"
    EXECUTING = "EXECUTING"
    PAUSED = "PAUSED"
    ABORTING = "ABORTING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class MissionManager:
    """Single owner of mission state for the process lifetime."""

    def __init__(self, app):
        self.app = app
        self.phase = MissionPhase.IDLE
        self.mission_id: str | None = None
        self.name: str | None = None
        self.total = 0
        self.current_seq = 0
        self.reached: set[int] = set()
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self._routed = False

    # ------------------------------------------------------------- status
    @property
    def active(self) -> bool:
        return self.phase in (MissionPhase.EXECUTING, MissionPhase.PAUSED,
                              MissionPhase.ABORTING)

    @property
    def done_count(self) -> int:
        return len(self.reached)

    def status(self) -> dict:
        return {
            "phase": self.phase.value,
            "mission_id": self.mission_id,
            "name": self.name,
            "total": self.total,
            "done": self.done_count,
            "current_seq": self.current_seq,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "active": self.active,
            "elapsed_s": (round(time.time() - self.started_at, 1)
                          if self.started_at and self.active else None),
        }

    # ------------------------------------------------------- mavlink wiring
    def _install_routers(self) -> None:
        if self._routed:
            return
        self.app.conn.register_router("MISSION_CURRENT", self._on_mission_current)
        self.app.conn.register_router("MISSION_ITEM_REACHED", self._on_item_reached)
        self._routed = True

    def _on_mission_current(self, msg) -> None:
        # Fires from the connection hook, i.e. on whichever loop drained the
        # message, so this must stay synchronous and cheap.
        try:
            self.current_seq = int(msg.seq)
            self.total = int(msg.total) or self.total
        except Exception as e:  # noqa: BLE001 - never let a router kill the link
            log.warning("MISSION_CURRENT parse failed: %s", e)
            return
        if self.active and self.total and self.current_seq >= self.total:
            self._finish(MissionPhase.COMPLETE, "all mission items reached")

    def _on_item_reached(self, msg) -> None:
        try:
            self.reached.add(int(msg.seq))
        except Exception as e:  # noqa: BLE001
            log.warning("MISSION_ITEM_REACHED parse failed: %s", e)
        if self.active and self.total and len(self.reached) >= self.total:
            self._finish(MissionPhase.COMPLETE, "all mission items reached")

    def _finish(self, phase: MissionPhase, detail: str) -> None:
        if not self.active:
            return
        self.phase = phase
        self.finished_at = time.time()
        log.info("mission %s -> %s (%s)", self.mission_id, phase.value, detail)

    # --------------------------------------------------------- item build
    def build_items(self, body: dict) -> list[dict]:
        """Compose the full item list: takeoff, waypoints, land, optional RTL."""
        items: list[dict] = []
        takeoff = body.get("takeoff") or {}
        wp_list = body.get("waypoints") or []

        if takeoff:
            ref = wp_list[0] if wp_list else {"lat": 0.0, "lon": 0.0}
            items.append({
                "lat": float(takeoff.get("lat", ref["lat"])),
                "lon": float(takeoff.get("lon", ref["lon"])),
                "alt": float(takeoff.get("altitude_m", takeoff.get("alt", 0)) or 0),
                "command": CMD_TAKEOFF,
                "hold_s": float(takeoff.get("hold_s", 3) or 0),
            })
        for w in wp_list:
            items.append({
                "lat": float(w["lat"]),
                "lon": float(w["lon"]),
                "alt": float(w.get("alt", w.get("altitude_m", 0)) or 0),
                "command": CMD_WAYPOINT,
                "hold_s": float(w.get("hold_s", 0) or 0),
            })
        landing = body.get("landing") or {}
        if landing:
            last = wp_list[-1] if wp_list else {"lat": 0.0, "lon": 0.0}
            items.append({
                "lat": float(landing.get("lat", last["lat"])),
                "lon": float(landing.get("lon", last["lon"])),
                "alt": float(landing.get("altitude_m", 0) or 0),
                "command": CMD_LAND,
            })
        if body.get("return_to_home"):
            items.append({"lat": 0.0, "lon": 0.0, "alt": 0.0,
                          "command": CMD_RETURN_TO_LAUNCH})
        return items

    # ------------------------------------------------------------ commands
    async def start(self, mission_id: str, body: dict) -> dict:
        if self.active:
            raise ValueError(f"a mission is already {self.phase.value}")
        conn = self.app.conn

        self.phase = MissionPhase.VALIDATING
        self.error = None
        ok, errs = validate_mission(body, self.app.cfg.mission.max_altitude_m,
                                     self.app.cfg.mission.max_speed_mps)
        if not ok:
            self.phase = MissionPhase.FAILED
            self.error = f"mission invalid: {errs}"
            return {"status": "failed", "error": self.error}

        items = self.build_items(body)
        if not items:
            self.phase = MissionPhase.FAILED
            self.error = "mission has no items"
            return {"status": "failed", "error": self.error}

        snap = self.app.tstore.snap
        if not getattr(snap, "armed", False):
            self.phase = MissionPhase.FAILED
            self.error = "vehicle is disarmed; take off before starting a mission"
            return {"status": "failed", "error": self.error}

        self._install_routers()
        self.mission_id = mission_id
        self.name = body.get("name") or mission_id
        self.total = len(items)
        self.current_seq = 0
        self.reached = set()
        self.started_at = time.time()
        self.finished_at = None

        self.phase = MissionPhase.UPLOADING
        cleared, detail = await asyncio.to_thread(M.clear_mission, conn)
        if not cleared:
            self.phase = MissionPhase.FAILED
            self.error = f"could not clear FC mission: {detail}"
            return {"status": "failed", "error": self.error}

        uploaded, detail = await asyncio.to_thread(M.upload_mission, conn, items)
        if not uploaded:
            self.phase = MissionPhase.FAILED
            self.error = f"upload rejected by flight controller: {detail}"
            return {"status": "failed", "error": self.error}

        started, detail = await asyncio.to_thread(M.start_mission, conn)
        if not started:
            self.phase = MissionPhase.FAILED
            self.error = f"flight controller refused AUTO: {detail}"
            return {"status": "failed", "error": self.error}

        self.phase = MissionPhase.EXECUTING
        target = FlightState.SURVEY if body.get("survey") else FlightState.MISSION
        fsm = self.app.fsm
        try:
            fsm.transition(target, f"mission {mission_id} start")
        except ValueError:
            fsm.force(target, f"mission {mission_id} start")
        log.info("mission %s uploaded (%d items) and executing", mission_id, len(items))
        return {"status": "executing", "mission_id": mission_id,
                "total": len(items), "phase": self.phase.value}

    async def pause(self) -> dict:
        if self.phase is not MissionPhase.EXECUTING:
            return {"status": self.phase.value, "error": "mission is not executing"}
        ok, detail = await asyncio.to_thread(M.pause_mission, self.app.conn)
        if ok:
            self.phase = MissionPhase.PAUSED
        return {"status": self.phase.value, "detail": detail}

    async def resume(self) -> dict:
        if self.phase is not MissionPhase.PAUSED:
            return {"status": self.phase.value, "error": "mission is not paused"}
        ok, detail = await asyncio.to_thread(M.resume_mission, self.app.conn)
        if ok:
            self.phase = MissionPhase.EXECUTING
        return {"status": self.phase.value, "detail": detail}

    async def abort(self, reason: str = "user") -> dict:
        from ..mavlink import commands as C
        fsm = self.app.fsm
        if self.phase is not MissionPhase.IDLE and \
                fsm.state not in (FlightState.DISARMED, FlightState.LANDED):
            try:
                fsm.transition(FlightState.ABORT, reason)
            except ValueError:
                fsm.force(FlightState.ABORT, reason)
        was_active = self.active
        if was_active:
            self.phase = MissionPhase.ABORTING
        # RTL only means something while armed; a disarmed drone is on the ground.
        if getattr(self.app.tstore.snap, "armed", False):
            await asyncio.to_thread(C.rtl, self.app.conn)
            detail = "RTL sent"
        else:
            detail = "vehicle disarmed, no RTL needed"
        if was_active:
            self.phase = MissionPhase.FAILED if reason != "completed" else MissionPhase.COMPLETE
            self.error = f"aborted: {reason}"
            self.finished_at = time.time()
        log.info("mission abort requested (%s): %s", reason, detail)
        return {"status": self.phase.value, "reason": reason, "detail": detail}

    # ---------------------------------------------------------- supervisor
    async def run(self) -> None:
        """Background supervisor: notice link loss and completion promptly."""
        last = None
        while True:
            try:
                if self.phase is MissionPhase.EXECUTING and \
                        not self.app.conn.connected:
                    self._finish(MissionPhase.FAILED, "MAVLink link lost mid-mission")
                st = self.status()
                if st != last:
                    last = st
                    await pub.publisher.broadcast({"type": "mission_status", "data": st})
            except Exception as e:  # noqa: BLE001 - supervisor must not die
                log.error("mission supervisor: %s", e)
            await asyncio.sleep(0.5)
