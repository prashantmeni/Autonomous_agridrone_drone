"""CSI camera backend selection in CameraService.

Covers the picamera2 branch wiring only: dispatch on `cfg.type`, graceful
unavailability when the optional package is missing, and a stubbed end-to-end
capture loop without hardware.
"""
import sys
from types import SimpleNamespace

import numpy as np

from drone.perception.stream import CameraService, _camera_kind, _Picamera2Capture


def cam_cfg(**kw):
    base = dict(enabled=True, type="auto", width=1280, height=720, fps=15, device="")
    base.update(kw)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------- dispatch
def test_camera_kind_parses_type():
    assert _camera_kind(cam_cfg(type="picamera2")) == "picamera2"
    assert _camera_kind(cam_cfg(type="CSI")) == "csi"
    assert _camera_kind(cam_cfg()) == "auto"
    assert _camera_kind(SimpleNamespace()) == "auto"


def test_picamera2_missing_is_graceful(monkeypatch):
    """No picamera2 package -> CAMERA_UNAVAILABLE, never a crash or import."""
    def fake_import(name, *a, **k):
        if name == "picamera2":
            raise ImportError("no picamera2")
        return __import__(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    svc = CameraService(cam_cfg(type="picamera2"), record_dir="data/recordings")
    cap = svc._open_capture(cv2=None)  # the picamera2 branch never touches cv2
    assert cap is None


def test_picamera2_branch_uses_libcamera_not_v4l2(monkeypatch):
    """type=picamera2 must pick _Picamera2Capture and never call the V4L2 sweep."""
    called = {"cv2": False}

    def fake_import(name, *a, **k):
        if name == "picamera2":
            return sys.modules["tests_fake_picamera2"]
        return __import__(name, *a, **k)

    class FakeCam:
        def __init__(self):
            self.conf = None

        def create_video_configuration(self, main, controls):
            self.conf = dict(main=main, controls=controls)
            return self.conf

        def configure(self, conf):
            pass

        def start(self):
            pass

        def capture_array(self):
            return np.zeros((480, 640, 3), np.uint8)

        def stop(self):
            pass

        def close(self):
            pass

    cam = FakeCam()
    fake = SimpleNamespace(Picamera2=lambda: cam)
    monkeypatch.setitem(sys.modules, "tests_fake_picamera2", fake)
    monkeypatch.setattr("builtins.__import__", fake_import)

    svc = CameraService(cam_cfg(type="picamera2", width=640, height=480))
    cap = svc._open_capture(cv2=None)
    assert isinstance(cap, _Picamera2Capture)
    ok, frame = cap.read()
    assert ok and frame.shape == (480, 640, 3)
    assert cam.conf is not None  # configured through libcamera path
    cap.release()


def test_auto_type_keeps_v4l2_sweep(monkeypatch):
    """type=auto must route to the V4L2/Bayer sweep (device probe), not picamera2."""
    calls = []

    def fake_import(name, *a, **k):
        if name == "picamera2":
            raise ImportError("must not be imported for auto")
        return __import__(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    svc = CameraService(cam_cfg(type="auto"), record_dir="data/recordings")

    # The sweep calls cv2.VideoCapture; a stub that always fails to open means
    # "no camera" — and crucially no picamera2 import was attempted.
    class FakeCV2:
        CAP_V4L2 = 0
        CAP_PROP_FOURCC = 1
        CAP_PROP_CONVERT_RGB = 2
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_BUFFERSIZE = 5

        @staticmethod
        def VideoWriter_fourcc(*a):
            return 0

        @staticmethod
        def VideoCapture(*a, **k):
            calls.append("VideoCapture")
            return SimpleNamespace(
                isOpened=lambda: False,
                release=lambda: None,
                set=lambda *a, **k: False,
                get=lambda *a, **k: 0,
            )

    cap = svc._open_capture(cv2=FakeCV2)
    assert cap is None
    assert "VideoCapture" in calls