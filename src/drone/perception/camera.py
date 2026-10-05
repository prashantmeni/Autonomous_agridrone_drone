"""Camera abstraction. Returns CAMERA_UNAVAILABLE when no device. Never crashes app."""
from __future__ import annotations
import logging
log = logging.getLogger("drone.perception.camera")

class CameraInterface:
    def open(self) -> bool: raise NotImplementedError
    def read(self): raise NotImplementedError
    def close(self): pass
    @property
    def status(self) -> str: return "UNAVAILABLE"

class OpenCVCamera(CameraInterface):
    def __init__(self, index=0, width=1280, height=720, fps=15):
        self.index, self.w, self.h, self.fps = index, width, height, fps
        self._cap = None
    def open(self) -> bool:
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.index)
            self._cap.set(3, self.w); self._cap.set(4, self.h)
            ok = self._cap.isOpened()
            log.info(f"camera open: {ok}")
            return ok
        except Exception as e:
            log.warning(f"camera unavailable: {e}"); return False
    def read(self):
        if self._cap is None: return None
        ok, f = self._cap.read()
        return f if ok else None
    @property
    def status(self): return "READY" if self._cap and self._cap.isOpened() else "UNAVAILABLE"

class CameraManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend: CameraInterface = OpenCVCamera(width=cfg.width, height=cfg.height, fps=cfg.fps)
        self.available = False
    def start(self) -> str:
        if not self.cfg.enabled: return "UNAVAILABLE"
        self.available = self.backend.open()
        return "READY" if self.available else "CAMERA_UNAVAILABLE"
