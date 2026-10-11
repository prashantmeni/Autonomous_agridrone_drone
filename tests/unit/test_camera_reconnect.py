"""Auto-reconnect: a camera fitted (or reseated) later must come up on its own.

A CSI module on the Pi's unicam is only probed once at boot, so before this
change a camera absent at startup left the capture thread dead until someone
rebooted the Pi, and a camera unplugged mid-run flipped the status to
CAMERA_UNAVAILABLE without ever reopening the device.

These tests pin the recovery behaviour without hardware: a fake capture that
only starts yielding frames after N probes.
"""
import threading
import time
from types import SimpleNamespace

import numpy as np

from drone.perception.stream import CameraService


def cam_cfg(**kw):
    base = dict(enabled=True, type="auto", width=640, height=480, fps=15, device="")
    base.update(kw)
    return SimpleNamespace(**base)


class FakeCap:
    """Stands in for a capture handle that fails until it is 'plugged in'."""

    def __init__(self):
        self.released = False

    def isOpened(self):
        return not self.released

    def read(self):
        if self.released:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_reconnect_when_absent_at_startup():
    """No camera at boot -> the thread must stay alive and pick one up later."""
    svc = CameraService(cam_cfg(), record_dir="data/recordings")
    svc._RECONNECT_MIN_S = 0.01
    svc._RECONNECT_MAX_S = 0.05

    probes = {"n": 0}

    def open_capture(cv2):
        probes["n"] += 1
        # Absent for the first three probes, then "connected".
        return FakeCap() if probes["n"] > 3 else None

    svc._open_capture = open_capture
    svc.start()

    deadline = time.time() + 5
    while time.time() < deadline:
        if svc.status()["status"] == "READY":
            break
        time.sleep(0.05)
    st = svc.status()

    svc.stop()
    assert st["status"] == "READY", f"camera never recovered: {st}"
    assert probes["n"] > 3, "should have retried the probe"
    assert st["reconnects"] == 1, st
    assert st["retrying"] is False


def test_status_reports_retrying_while_waiting():
    """While probing, status must say so instead of looking dead."""
    svc = CameraService(cam_cfg(), record_dir="data/recordings")
    svc._RECONNECT_MIN_S = 0.05
    svc._RECONNECT_MAX_S = 0.1
    svc._open_capture = lambda cv2: None

    svc.start()
    deadline = time.time() + 3
    while time.time() < deadline:
        if svc.status()["retrying"]:
            break
        time.sleep(0.05)
    st = svc.status()
    svc.stop()

    assert st["retrying"] is True, st
    assert st["available"] is False


def test_reconnect_after_midrun_disconnect():
    """Unplugging mid-run must reopen the device, not just flag it dead."""
    svc = CameraService(cam_cfg(), record_dir="data/recordings")
    svc._RECONNECT_MIN_S = 0.01
    svc._RECONNECT_MAX_S = 0.05
    svc._READ_FAIL_LIMIT = 3

    caps = []

    def open_capture(cv2):
        cap = FakeCap()
        caps.append(cap)
        return cap

    svc._open_capture = open_capture
    svc.start()

    deadline = time.time() + 5
    while time.time() < deadline and svc.status()["status"] != "READY":
        time.sleep(0.05)
    assert svc.status()["status"] == "READY"

    # Simulate an unplug: the live handle starts failing.
    caps[0].released = True

    deadline = time.time() + 5
    while time.time() < deadline:
        st = svc.status()
        if len(caps) > 1 and st["status"] == "READY":
            break
        time.sleep(0.05)
    st = svc.status()
    svc.stop()

    assert len(caps) > 1, "should have reopened the camera"
    assert st["available"] is True


def test_stop_ends_the_reconnect_loop():
    """stop() must actually terminate a thread that is waiting for a camera."""
    svc = CameraService(cam_cfg(), record_dir="data/recordings")
    svc._RECONNECT_MIN_S = 0.2
    svc._RECONNECT_MAX_S = 0.4
    svc._open_capture = lambda cv2: None

    svc.start()
    deadline = time.time() + 3
    while time.time() < deadline and not svc.status()["retrying"]:
        time.sleep(0.05)

    th = svc._thread
    svc.stop()
    th.join(timeout=5)
    assert not th.is_alive(), "reconnect thread did not stop"
    assert svc.status()["retrying"] is False