"""ArUco/AprilTag landing marker abstraction. Simulation-safe, honest about availability."""
from __future__ import annotations
class LandingDetector:
    def detect(self, frame): raise NotImplementedError

class ArucoDetector(LandingDetector):
    def __init__(self): self.available = self._probe()
    def _probe(self):
        try: import cv2; return hasattr(cv2, "aruco")
        except Exception: return False
    def detect(self, frame):
        if not self.available or frame is None:
            return {"found": False, "status": "DETECTOR_UNAVAILABLE"}
        try:
            import cv2
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            corners, ids, _ = cv2.aruco.detectMarkers(frame, cv2.aruco.DetectorParameters(), d) if hasattr(cv2.aruco, "DetectorParameters") else cv2.aruco.detectMarkers(frame, d)
            if ids is None or len(ids) == 0: return {"found": False, "confidence": 0.0}
            cx = float(sum(c[0][:, 0].mean() for c in corners) / len(corners))
            cy = float(sum(c[0][:, 1].mean() for c in corners) / len(corners))
            h, w = frame.shape[:2]
            return {"found": True, "offset_x": (cx - w/2)/w, "offset_y": (cy - h/2)/h, "confidence": 0.9}
        except Exception as e:
            return {"found": False, "status": f"ERROR:{e}"}
