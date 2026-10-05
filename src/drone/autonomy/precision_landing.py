"""Closed-loop precision landing: ArUco vision servoing onto the charging pad.

Runs as an asyncio task against the live camera. Marker loss falls back to a
plain GPS landing instead of continuing blind. Never runs without a real
MAVLink link and a real camera.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

log = logging.getLogger("drone.autonomy.precision_landing")


class PrecisionLandingTask:
    def __init__(self):
        self.active = False
        self.status: dict = self._idle_status()

    @staticmethod
    def _idle_status() -> dict:
        return {
            "active": False,
            "phase": "IDLE",
            "marker_found": False,
            "offset_x": None,
            "offset_y": None,
            "confidence": 0.0,
            "decision": None,
            "altitude_m": None,
            "fallback": None,
            "message": "precision landing idle",
            "started_at": None,
            "updated_at": time.time(),
        }

    def _set(self, **kw) -> None:
        self.status.update(kw)
        self.status["updated_at"] = time.time()

    async def start(self, app, camera) -> dict:
        if self.active:
            return self.status
        if not app.conn.connected or app.conn.master is None:
            raise RuntimeError("MAVLink not connected")
        frame = camera.frame()
        cam_status = camera.status()
        if not cam_status.get("available") or frame is None:
            raise RuntimeError("camera unavailable — precision landing needs a live feed")

        self.active = True
        self._set(active=True, phase="SEARCH", fallback=None,
                  message="scanning for landing pad marker", started_at=time.time())
        asyncio.create_task(self._run(app, camera))
        return self.status

    async def _run(self, app, camera) -> None:
        from ..core.state import FlightState
        from ..mavlink import commands as C
        from ..navigation.coordinate_transform import offset_latlon
        from ..navigation.landing import LandingController
        from ..perception.landing_marker import ArucoDetector

        detector = ArucoDetector()
        controller = LandingController(
            max_descent=app.cfg.precision_landing.max_descent_speed_mps,
            lost_timeout_s=app.cfg.precision_landing.marker_lost_timeout_s,
        )
        if not detector.available:
            self._set(active=False, phase="FAILED", fallback=None,
                      message="DETECTOR_UNAVAILABLE — ArUco support missing in OpenCV")
            self.active = False
            return

        try:
            while self.active:
                snap = app.tstore.snap
                if not snap.armed and app.fsm.state == FlightState.LANDING:
                    app.fsm.force(FlightState.LANDED, "precision landing touchdown")
                    self._set(active=False, phase="TOUCHDOWN", message="landed on pad")
                    break
                if app.fsm.state not in (FlightState.PRECISION_LANDING, FlightState.LANDING):
                    self._set(active=False, phase="ABORTED", message="operator took over")
                    break
                if not app.conn.connected:
                    self._set(active=False, phase="FAILED", fallback="WAIT_RECONNECT",
                              message="link lost during precision landing")
                    break

                frame = camera.frame()
                det = detector.detect(frame)
                found = bool(det.get("found"))
                offset = (det.get("offset_x"), det.get("offset_y")) if found else None
                decision = controller.update(offset, det.get("confidence", 0.0) or 0.0)
                action = decision.get("action")

                self._set(
                    marker_found=found,
                    offset_x=round(offset[0], 3) if offset else None,
                    offset_y=round(offset[1], 3) if offset else None,
                    confidence=det.get("confidence", 0.0),
                    decision=action,
                    altitude_m=round(snap.relative_alt_m, 2),
                )

                if action == "ABORT":
                    log.warning("precision landing abort: %s", decision.get("reason"))
                    await asyncio.to_thread(C.land, app.conn)
                    app.fsm.force(FlightState.LANDING,
                                  f"fallback GPS landing: {decision.get('reason')}")
                    self._set(phase="FALLBACK_GPS_LANDING", fallback="GPS_LANDING",
                              message=decision.get("reason", "marker lost"))
                    continue

                if action == "CORRECT" and snap.lat is not None and snap.lon is not None:
                    ox, oy = offset  # normalized -0.5..0.5
                    k = max(0.4, (snap.relative_alt_m or 1.0) * 1.1)
                    forward = max(-1.5, min(1.5, -oy * k * 2))
                    right = max(-1.5, min(1.5, ox * k * 2))
                    h = math.radians(snap.heading_deg or 0.0)
                    north = forward * math.cos(h) + right * (-math.sin(h))
                    east = forward * math.sin(h) + right * math.cos(h)
                    tlat, tlon = offset_latlon(snap.lat, snap.lon, east, north)

                    aligned = abs(ox) < 0.06 and abs(oy) < 0.06
                    if aligned and snap.relative_alt_m > 0.6:
                        alt_amsl = max(0.0, snap.alt_m - 0.6)
                        self._set(phase="DESCEND", message="aligned — descending on pad")
                    else:
                        alt_amsl = snap.alt_m
                        self._set(phase="ALIGN", message="servoing toward pad marker")
                    try:
                        await asyncio.to_thread(C.goto, app.conn, tlat, tlon, alt_amsl)
                    except Exception as e:
                        log.warning("goto during precision landing failed: %s", e)

                    if snap.relative_alt_m < 0.45:
                        await asyncio.to_thread(C.land, app.conn)
                        app.fsm.force(FlightState.LANDING, "final touchdown on marker")
                        self._set(phase="TOUCHDOWN", message="pad reached — landing")
                else:
                    self._set(phase="HOLD" if found else "SEARCH",
                              message=decision.get("reason", "holding"))

                await asyncio.sleep(0.2)
        except Exception as e:
            log.error("precision landing loop error: %s", e)
            self._set(phase="FAILED", message=str(e))
        finally:
            self.active = False
            self._set(active=False)

    def get(self) -> dict:
        return dict(self.status)
