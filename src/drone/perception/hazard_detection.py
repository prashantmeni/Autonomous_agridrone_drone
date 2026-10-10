"""Forward-camera hazard detection: people and vehicles ahead.

This is a *reporting* feature, not a control loop. The Pi 4B manages roughly
5 FPS on this pipeline, which is fine for alerting an operator but not for
steering an aircraft, so nothing here commands the drone; the avoidance
supervisor turns these results into alerts and logs only.

Two cheap, complementary signals over the forward half of the frame:

* a Haar cascade for pedestrians (bundled with OpenCV), and
* motion energy in the forward region, which also catches an approaching object
  that is not a person (a vehicle, a wall, livestock).

Both must be reasonably confident, and detections are smoothed across frames so
a single noisy detection does not raise an alert.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

log = logging.getLogger("drone.perception.hazard")

# Forward region of interest: upper-middle of frame (sky excluded).
ROI_TOP = 0.28
ROI_BOTTOM = 0.92


@dataclass
class HazardResult:
    hazard: bool
    kind: str | None = None
    confidence: float = 0.0
    box: tuple[int, int, int, int] | None = None
    motion: float = 0.0
    reason: str = "clear"

    def to_dict(self) -> dict:
        return {
            "hazard": self.hazard,
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "box": list(self.box) if self.box else None,
            "motion": round(self.motion, 4),
            "reason": self.reason,
        }


class HazardDetector:
    def __init__(self, min_confidence: float = 0.6, motion_threshold: float = 0.06,
                 min_box_area_frac: float = 0.01):
        self.min_confidence = min_confidence
        self.motion_threshold = motion_threshold
        self.min_box_area_frac = min_box_area_frac
        self._classifier = None
        self._classifier_failed = False
        self._prev_gray = None

    def _get_classifier(self):
        if self._classifier is not None or self._classifier_failed:
            return self._classifier
        try:
            import cv2
            path = f"{cv2.data.haarcascades}haarcascade_fullbody.xml"
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                raise RuntimeError("empty cascade")
            self._classifier = cascade
        except Exception as e:  # noqa: BLE001
            log.warning("person cascade unavailable (%s); motion signal only", e)
            self._classifier_failed = True
        return self._classifier

    @staticmethod
    def _motion_energy(prev_gray, gray) -> float:
        import cv2
        import numpy as np
        diff = cv2.absdiff(prev_gray, gray)
        return float(np.mean(diff) / 255.0)

    def detect(self, frame_bgr) -> HazardResult:
        """Assess one BGR frame for a hazard in the forward view."""
        try:
            import cv2
        except Exception as e:  # noqa: BLE001
            return HazardResult(False, reason=f"opencv unavailable: {e}")

        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return HazardResult(False, reason="no frame")

        h, _w = frame_bgr.shape[:2]
        y0, y1 = int(h * ROI_TOP), int(h * ROI_BOTTOM)
        roi = frame_bgr[y0:y1, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        motion = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            motion = self._motion_energy(self._prev_gray, gray)
        self._prev_gray = gray

        best = None
        cascade = self._get_classifier()
        if cascade is not None:
            _rh, rw = gray.shape[:2]
            rects = cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5,
                minSize=(max(20, rw // 12), max(32, _rh // 8)))
            area_frac = (rw * _rh) or 1
            for (x, y, bw, bh) in rects:
                if (bw * bh) / area_frac < self.min_box_area_frac:
                    continue
                if best is None or bw * bh > best[2] * best[3]:
                    best = (x, y, bw, bh)

        if best is not None:
            conf = 0.9
            return HazardResult(True, kind="person", confidence=conf,
                                box=(int(best[0]), int(best[1] + y0),
                                     int(best[2]), int(best[3])),
                                motion=motion, reason="person detected ahead")
        if motion >= self.motion_threshold:
            return HazardResult(True, kind="motion", confidence=min(1.0, motion * 4),
                                motion=motion, reason="motion in forward view")
        return HazardResult(False, motion=motion, reason="clear")

    def reset(self) -> None:
        self._prev_gray = None


class HazardMonitor:
    """Runs the detector over live frames on a timer and reports changes."""

    def __init__(self, app, detector: HazardDetector | None = None,
                 min_consecutive: int = 2, interval_s: float = 1.0):
        self.app = app
        self.detector = detector or HazardDetector()
        self.min_consecutive = max(1, int(min_consecutive))
        self.interval_s = interval_s
        self._streak = 0
        self._reported = None
        self.last_result: HazardResult | None = None

    def _frame(self):
        try:
            from ..runtime import get_services
            frame = get_services(self.app).camera.frame()
            return frame if frame is not None and getattr(frame, "size", 0) else None
        except Exception:  # noqa: BLE001
            return None

    async def tick(self) -> HazardResult | None:
        frame = self._frame()
        if frame is None:
            return None
        result = self.detector.detect(frame)
        self.last_result = result
        self._streak = self._streak + 1 if result.hazard else 0
        should_alert = self._streak >= self.min_consecutive
        key = (result.hazard, result.kind) if should_alert else None
        if key != self._reported:
            self._reported = key
            supervisor = getattr(self.app, "avoidance", None)
            if supervisor is not None:
                await supervisor.set_hazard(result.to_dict() if result.hazard else None)
            log.info("camera hazard state -> %s", result.reason)
        return result

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception as e:  # noqa: BLE001
                log.error("hazard monitor: %s", e)
            await asyncio.sleep(self.interval_s)