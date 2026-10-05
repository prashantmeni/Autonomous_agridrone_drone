"""Live camera service: frame capture, MJPEG streaming, on-disk video recording.

Honest by design: when no camera device exists every call reports
CAMERA_UNAVAILABLE instead of inventing frames. Recording writes real
MP4 files under data/recordings.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("drone.perception.stream")


class _V4L2BayerCapture:
    """Raw capture for 10-bit Bayer V4L2 sensors (e.g. Raspberry Pi OV5647).

    OpenCV cannot decode those natively: the buffer is pulled with
    CONVERT_RGB disabled and demosaiced here. Falls back upstream when the
    device does not negotiate the GB10 format (plain USB webcams).
    """

    def __init__(self, cv2, dev: str, size):
        import numpy as np

        self._cv2 = cv2
        self._np = np
        self.w, self.h = size
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GB10"))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self._cap = cap
        self._bayer = int(cap.get(cv2.CAP_PROP_FOURCC)) == int(
            cv2.VideoWriter_fourcc(*"GB10")
        )

    def negotiated(self) -> bool:
        return self._bayer and self._cap.isOpened()

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def read(self):
        ok, raw = self._cap.read()
        if not ok or raw is None:
            return False, None
        buf = raw.reshape(-1)
        need = self.w * self.h * 2
        if buf.size < need:
            return False, None
        raw16 = self._np.frombuffer(buf[:need].tobytes(), dtype=self._np.uint16).reshape(
            self.h, self.w
        )
        bgr16 = self._cv2.cvtColor(raw16, self._cv2.COLOR_BayerGB2BGR)
        return True, (bgr16 >> 2).astype(self._np.uint8)

    def release(self) -> None:
        self._cap.release()


class CameraService:
    def __init__(self, cfg, record_dir: str = "data/recordings"):
        self.cfg = cfg
        self.record_dir = Path(record_dir)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._cap = None
        self._frame = None          # latest raw BGR frame (ndarray)
        self._jpeg: bytes | None = None
        self._frame_ts = 0.0
        self._status = "NOT_STARTED"
        self._size = (int(cfg.width), int(cfg.height))
        self._fps = max(1, int(cfg.fps))
        self._writer = None
        self._recording = False
        self._recording_path: str | None = None
        self._recording_started = 0.0
        self._frames_written = 0

    # ------------------------------------------------------------------ start
    def start(self) -> str:
        if not self.cfg.enabled:
            self._status = "UNAVAILABLE"
            return self._status
        with self._lock:
            if self._running:
                return self._status
            self._running = True
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()
        time.sleep(0.4)
        return self._status

    # Candidate capture geometries, most preferred first. The configured
    # resolution is tried first, then common sensor fallbacks: CSI sensors
    # such as the IMX219/OV5647 refuse modes they cannot produce and the
    # kernel aborts the whole pipeline with -EINVAL rather than rescaling.
    _FALLBACK_SIZES = ((1280, 720), (1024, 768), (960, 540), (800, 600), (640, 480), (320, 240))

    # Only sensor capture nodes are probed. The bcm2835-isp / rkisp nodes
    # (video10+, video21+) are pipeline stages with no source of their own:
    # opening one just stalls on select() until it times out.
    _DEFAULT_DEVICE = "/dev/video0"

    def _open_capture(self, cv2):
        """Open the camera, negotiating a geometry the sensor actually supports.

        Returns an opened capture, or None when no geometry yields frames. The
        granted size is written back to self._size so status and recording
        agree with the real frames.
        """
        dev = getattr(self.cfg, "device", "") or self._DEFAULT_DEVICE

        # Preferred order: configured geometry first, then fallbacks.
        want = (int(self.cfg.width), int(self.cfg.height))
        sizes = [want] + [s for s in self._FALLBACK_SIZES if s != want]

        for w, h in sizes:
            cap = self._try_geometry(cv2, dev, w, h)
            if cap is not None:
                self._size = self._negotiated_size(cv2, cap, w, h)
                log.info("camera opened on %s at %sx%s", dev, *self._size)
                return cap
        return None

    # Bound how long one geometry may stall before we move on. V4L2 nodes
    # that cannot stream block in select() for their full timeout, so without
    # this a multi-geometry sweep takes tens of seconds.
    _PROBE_TIMEOUT_S = 2.5

    def _try_geometry(self, cv2, dev: str, w: int, h: int):
        """Try raw-Bayer then plain capture at one geometry; None if no frames."""
        try:
            bayer = _V4L2BayerCapture(cv2, dev, (w, h))
            if bayer.negotiated():
                ok, _ = bayer.read()
                if ok:
                    log.info("raw bayer capture active at %sx%s", w, h)
                    return bayer
            bayer.release()
        except Exception as e:
            log.debug("raw bayer capture unavailable on %s at %sx%s: %s", dev, w, h, e)

        cap = None
        try:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            t0 = time.time()
            while time.time() - t0 < self._PROBE_TIMEOUT_S:
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
        except Exception as e:
            log.debug("capture failed on %s at %sx%s: %s", dev, w, h, e)
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        return None

    def _negotiated_size(self, cv2, cap, fallback_w: int, fallback_h: int) -> tuple[int, int]:
        """Read back the geometry the driver actually granted."""
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            return fallback_w, fallback_h
        return (w, h) if w > 0 and h > 0 else (fallback_w, fallback_h)

    def _run(self) -> None:
        try:
            import cv2
        except Exception as e:
            log.warning("opencv unavailable: %s", e)
            with self._lock:
                self._status = "CAMERA_UNAVAILABLE"
                self._running = False
            return

        try:
            cap = self._open_capture(cv2)
            if cap is None:
                log.warning("no camera geometry produced frames (sensor may be absent)")
                with self._lock:
                    self._status = "CAMERA_UNAVAILABLE"
                    self._running = False
                return
            self._cap = cap
            with self._lock:
                self._status = "READY"
            log.info("camera stream started at %sx%s", *self._size)
            interval = 1.0 / self._fps
            while True:
                with self._lock:
                    if not self._running:
                        break
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    with self._lock:
                        if self._status == "READY":
                            self._status = "CAMERA_UNAVAILABLE"
                    time.sleep(0.5)
                    continue
                okk, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not okk:
                    continue
                jpeg = bytes(buf)
                with self._lock:
                    self._frame = frame
                    self._jpeg = jpeg
                    self._frame_ts = time.time()
                    self._status = "READY"
                    if self._recording and self._writer is not None:
                        try:
                            self._writer.write(frame)
                            self._frames_written += 1
                        except Exception as e:
                            log.error("record write failed: %s", e)
                            self._recording = False
                dt = time.time() - t0
                if dt < interval:
                    time.sleep(max(0.0, interval - dt))
        except Exception as e:
            log.error("camera capture loop crashed: %s", e)
            with self._lock:
                self._status = "CAMERA_UNAVAILABLE"
                self._running = False
        finally:
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass

    # ----------------------------------------------------------------- status
    def status(self) -> dict:
        with self._lock:
            age = time.time() - self._frame_ts if self._frame_ts else None
            return {
                "status": self._status,
                "available": self._status == "READY",
                "width": self._size[0],
                "height": self._size[1],
                "fps": self._fps,
                "frame_age_s": round(age, 2) if age is not None else None,
                "recording": self._recording,
                "recording_path": self._recording_path,
                "recording_s": round(time.time() - self._recording_started, 1)
                if self._recording and self._recording_started
                else 0.0,
                "frames_written": self._frames_written,
            }

    # ---------------------------------------------------------------- streams
    def _await_first_frame(self, timeout_s: float) -> bool:
        """Block until the capture thread produces a frame or the wait expires.

        Probing several geometries can take longer than one HTTP request, so
        callers get a bounded wait instead of an immediate miss.
        """
        deadline = time.time() + max(0.1, timeout_s)
        while time.time() < deadline:
            with self._lock:
                if self._jpeg is not None and self._status == "READY":
                    return True
                if self._status in ("CAMERA_UNAVAILABLE", "UNAVAILABLE") and not self._running:
                    return False
            time.sleep(0.1)
        with self._lock:
            return self._jpeg is not None

    def snapshot(self, wait_s: float = 12.0) -> bytes | None:
        if self.status()["status"] == "NOT_STARTED":
            self.start()
        self._await_first_frame(wait_s)
        with self._lock:
            return self._jpeg

    def frame(self):
        if self.status()["status"] == "NOT_STARTED":
            self.start()
        with self._lock:
            return self._frame

    def mjpeg(self):
        """Yield multipart JPEG frames. Shows a real NO SIGNAL slate when unavailable."""
        if self.status()["status"] == "NOT_STARTED":
            self.start()
        try:
            import cv2
        except Exception:
            while True:
                time.sleep(1.0)
                return

        last_sent = 0.0
        while True:
            with self._lock:
                jpeg = self._jpeg
                st = self._status
                ts = self._frame_ts
            if jpeg and st == "READY" and ts != last_sent:
                last_sent = ts
                yield jpeg
            else:
                if st != "READY":
                    last_sent = 0.0
                    import numpy as np
                    img = np.zeros((720, 1280, 3), dtype="uint8")
                    cv2.putText(img, "NO CAMERA SIGNAL", (300, 340), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (60, 70, 90), 2, cv2.LINE_AA)
                    cv2.putText(img, st, (390, 400), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (45, 55, 70), 2, cv2.LINE_AA)
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ok:
                        yield bytes(buf)
                        time.sleep(1.0)
                        continue
                time.sleep(0.05)

    # --------------------------------------------------------------- recording
    def start_recording(self) -> dict:
        if self.status()["status"] == "NOT_STARTED":
            self.start()
        with self._lock:
            if self._recording:
                return {"recording": True, "path": self._recording_path}
            if self._status != "READY" or self._frame is None:
                return {"recording": False, "error": "CAMERA_UNAVAILABLE"}
            try:
                import cv2
                self.record_dir.mkdir(parents=True, exist_ok=True)
                path = self.record_dir / f"flight_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                h, w = self._frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(self._fps), (w, h)
                )
                if not writer.isOpened():
                    return {"recording": False, "error": "RECORDER_UNAVAILABLE"}
                self._writer = writer
                self._recording_path = str(path)
                self._recording_started = time.time()
                self._frames_written = 0
                self._recording = True
                return {"recording": True, "path": self._recording_path}
            except Exception as e:
                log.error("recorder start failed: %s", e)
                return {"recording": False, "error": f"RECORDER_ERROR:{e}"}

    def stop_recording(self) -> dict:
        with self._lock:
            if not self._recording:
                return {"recording": False}
            self._recording = False
            writer, self._writer = self._writer, None
            path = self._recording_path
            frames = self._frames_written
            dur = time.time() - self._recording_started
        try:
            if writer is not None:
                writer.release()
        except Exception as e:
            log.error("recorder stop failed: %s", e)
        log.info("recording stopped: %s (%s frames, %.1fs)", path, frames, dur)
        return {"recording": False, "path": path, "frames": frames, "duration_s": round(dur, 1)}
