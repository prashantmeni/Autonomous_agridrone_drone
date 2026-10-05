"""Camera abstraction.

The inference engine never opens a camera. Anything that can yield a numpy
frame satisfies the same interface, so swapping USB webcam -> Pi CSI camera ->
video file -> still image touches only this module.

Backends:
  usb      - OpenCV VideoCapture on a device index (default)
  file     - a recorded video file
  picamera2- Raspberry Pi CSI camera, when picamera2 is installed
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CameraConfig


class CameraUnavailableError(RuntimeError):
    """Raised when no camera backend can be opened."""


class Camera:
    """Context manager yielding BGR frames."""

    def __init__(self, backend: str = "usb", *, device: int = 0,
                 width: int = 640, height: int = 480, fps: int = 15,
                 path: str | None = None):
        self.backend = backend
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.path = path
        self._capture = None
        self._picam = None

    @classmethod
    def from_config(cls, cfg: CameraConfig, backend: str = "usb",
                    path: str | None = None) -> "Camera":
        return cls(
            backend=backend,
            device=cfg.device,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            path=path,
        )

    # ------------------------------------------------------------------- open
    def open(self) -> "Camera":
        if self.backend == "file":
            self._open_file()
        elif self.backend == "picamera2":
            self._open_picamera2()
        else:
            self._open_usb()
        return self

    def _open_usb(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailableError(
                f"could not open camera device {self.device}. "
                "Check the connection, or use --backend file / picamera2."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture = cap

    def _open_file(self) -> None:
        import cv2

        if not self.path or not Path(self.path).is_file():
            raise CameraUnavailableError(f"video file not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise CameraUnavailableError(f"could not open video file: {self.path}")
        self._capture = cap

    def _open_picamera2(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as e:
            raise CameraUnavailableError(
                "picamera2 is not installed. On Raspberry Pi OS: "
                "sudo apt install -y python3-picamera2"
            ) from e
        try:
            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
        except Exception as e:
            raise CameraUnavailableError(f"could not open CSI camera: {e}") from e
        self._picam = picam

    # ------------------------------------------------------------------- read
    def read(self) -> np.ndarray | None:
        """Next BGR frame, or None at end of stream."""
        if self._picam is not None:
            # picamera2 gives RGB; convert so downstream stays BGR-consistent.
            frame = self._picam.capture_array()
            if frame is None:
                return None
            import cv2

            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._capture is None:
            raise CameraUnavailableError("camera not opened; call open() first")
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

    # ------------------------------------------------------------- context mgr
    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()

    @property
    def is_open(self) -> bool:
        return self._capture is not None or self._picam is not None