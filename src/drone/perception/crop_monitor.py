"""In-flight crop monitoring: periodically classify live camera frames.

Runs as a background task while `disease_detection.enabled` is true. Every
`inference_interval_s` it pulls the latest camera frame through
`crop_ai.analyze` and:

  - stores reliable, above-threshold detections in the DB with GPS,
  - broadcasts the result on the WebSocket as ``{"type": "crop_ai"}``.

The loop is a no-op when the camera has no frame and failures are logged then
throttled, so the loop can never take the drone app down.
"""
from __future__ import annotations

import asyncio
import logging

from ..perception import crop_ai
from ..runtime import get_services
from ..telemetry import publisher as pub

log = logging.getLogger("drone.perception.crop_monitor")


class CropAIMonitor:
    def __init__(self, app):
        self.app = app
        self.cfg = app.cfg.disease_detection

    async def run(self) -> None:
        interval = max(0.5, self.cfg.inference_interval_s or 2.0)
        while True:
            try:
                result = await self._analyze()
            except Exception as e:  # noqa: BLE001 - a monitor must never die
                log.error("crop monitor analysis failed: %s", e)
                result = None
            if result:
                await self._handle(result)
            await asyncio.sleep(interval)

    async def _analyze(self) -> dict | None:
        svc = get_services(self.app)
        svc.start()
        frame = svc.camera.frame()
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        return await asyncio.to_thread(
            crop_ai.analyze,
            frame,
            self.cfg.model_path,
            self.cfg.confidence_threshold,
            source="camera",
            tiled=True,
        )

    async def _handle(self, result: dict) -> None:
        reliable = bool(result.get("reliable"))
        conf = float(result.get("confidence") or 0.0)
        if reliable and conf >= self.cfg.confidence_threshold:
            self._store(result, conf)
        await pub.publisher.broadcast({"type": "crop_ai", "data": result})

    def _store(self, result: dict, confidence: float) -> None:
        snap = self.app.tstore.snap
        self.app.db.add_detection(
            result.get("crop") or "unknown",
            result.get("disease") or "unclassified",
            float(confidence),
            getattr(snap, "lat", None) or 0.0,
            getattr(snap, "lon", None) or 0.0,
        )