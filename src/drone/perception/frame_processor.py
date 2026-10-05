"""Frame skipping / scaling for Pi performance."""
from __future__ import annotations
import time
class FrameProcessor:
    def __init__(self, interval_s: float = 2.0, scale: float = 0.5):
        self.interval = interval_s; self.scale = scale; self._last = 0.0
    def should_process(self) -> bool:
        now = time.time()
        if now - self._last >= self.interval:
            self._last = now; return True
        return False
    def downscale(self, frame):
        if frame is None: return None
        try:
            import cv2
            h, w = frame.shape[:2]
            return cv2.resize(frame, (int(w*self.scale), int(h*self.scale)))
        except Exception: return frame
