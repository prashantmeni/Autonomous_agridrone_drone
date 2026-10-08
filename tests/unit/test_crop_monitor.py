"""Crop monitor loop: periodic in-flight classification, storage, broadcast.

The monitor drives the real bounding loop in `crop_monitor.py` but the camera,
engine and adapter are swapped for fakes here, so these tests run without a
camera device or model file.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from drone.core.config import AppConfig, DiseaseConfig
from drone.perception import crop_monitor
from drone.telemetry import publisher as pub


class FakeDB:
    def __init__(self):
        self.rows = []

    def add_detection(self, crop, disease, conf, lat, lon, image=""):
        self.rows.append((crop, disease, conf, lat, lon, image))


class FakeCam:
    def __init__(self, frame):
        self._frame = frame

    def frame(self):
        return self._frame


class FakeServices:
    def __init__(self, frame):
        self.camera = FakeCam(frame)

    def start(self):
        pass


class FakeApp:
    def __init__(self, frame=None, interval=2.0):
        if frame is None:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.cfg = AppConfig(disease_detection=DiseaseConfig(
            enabled=True, model_path="models/synthetic.onnx",
            confidence_threshold=0.5, inference_interval_s=interval))
        self.db = FakeDB()
        self.tstore = SimpleNamespace(snap=SimpleNamespace(lat=12.34, lon=56.78))
        self.camera_frame = frame


def run(coro):
    return asyncio.run(coro)


def detection(reliable=True, confidence=0.9):
    return {
        "reliable": reliable,
        "confidence": confidence,
        "crop": "Tomato",
        "disease": "Bacterial spot",
        "class_name": "Tomato___Bacterial_spot",
        "status": "OK" if reliable else "rejected",
    }


# ---------------------------------------------------------------- analysis
def test_analyze_pulls_camera_frame_through_adapter(monkeypatch):
    app = FakeApp()
    frame = np.full((64, 64, 3), 120, np.uint8)

    def fake_analyze(f, path, threshold, source="camera"):
        assert f is frame
        assert path == app.cfg.disease_detection.model_path
        return detection()

    monkeypatch.setattr(crop_monitor, "get_services",
                        lambda a: FakeServices(frame))
    monkeypatch.setattr(crop_monitor.crop_ai, "analyze", fake_analyze)
    out = run(crop_monitor.CropAIMonitor(app)._analyze())
    assert out["disease"] == "Bacterial spot"


def test_analyze_returns_none_without_frame(monkeypatch):
    monkeypatch.setattr(crop_monitor, "get_services",
                        lambda a: FakeServices(None))
    out = run(crop_monitor.CropAIMonitor(FakeApp())._analyze())
    assert out is None


# ----------------------------------------------------------------- storage
def test_handle_stores_reliable_above_threshold(monkeypatch):
    app = FakeApp()
    monitor = crop_monitor.CropAIMonitor(app)
    run(monitor._handle(detection(confidence=0.9)))
    assert len(app.db.rows) == 1
    crop, disease, conf, lat, lon, _image = app.db.rows[0]
    assert (crop, disease, conf, lat, lon) == ("Tomato", "Bacterial spot", 0.9, 12.34, 56.78)


def test_handle_skips_below_threshold(monkeypatch):
    app = FakeApp()
    run(crop_monitor.CropAIMonitor(app)._handle(detection(confidence=0.3)))
    assert app.db.rows == []


def test_handle_skips_unreliable(monkeypatch):
    app = FakeApp()
    run(crop_monitor.CropAIMonitor(app)._handle(detection(reliable=False, confidence=0.9)))
    assert app.db.rows == []


def test_handle_broadcasts_every_result():
    app = FakeApp()
    q = pub.publisher.subscribe()
    try:
        run(crop_monitor.CropAIMonitor(app)._handle(detection(confidence=0.9)))
        msg = q.get_nowait()
    finally:
        pub.publisher.unsubscribe(q)
    assert msg["type"] == "crop_ai"
    assert msg["data"]["disease"] == "Bacterial spot"


# --------------------------------------------------------------------- loop
def test_run_loops_then_stores_and_broadcasts(monkeypatch):
    app = FakeApp(interval=0.01)
    calls = {"n": 0}

    async def fake_analyze(self):
        calls["n"] += 1
        return detection(confidence=0.9) if calls["n"] == 2 else None

    monkeypatch.setattr(crop_monitor.CropAIMonitor, "_analyze", fake_analyze)
    q = pub.publisher.subscribe()
    monitor = crop_monitor.CropAIMonitor(app)

    async def driver():
        task = asyncio.create_task(monitor.run())
        deadline = asyncio.get_event_loop().time() + 5.0
        while calls["n"] < 3 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
        if not task.done():
            task.cancel()
        await asyncio.sleep(0)

    try:
        run(driver())
        assert calls["n"] >= 3, "loop kept ticking"
        assert len(app.db.rows) == 1
        try:
            msg = q.get_nowait()
        except asyncio.QueueEmpty:
            msg = None
        assert msg is not None and msg["type"] == "crop_ai"
    finally:
        pub.publisher.unsubscribe(q)


def test_run_survives_analyze_errors(monkeypatch):
    app = FakeApp(interval=0.01)
    calls = {"n": 0}

    async def fake_analyze(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return detection() if calls["n"] == 2 else None

    monkeypatch.setattr(crop_monitor.CropAIMonitor, "_analyze", fake_analyze)
    monitor = crop_monitor.CropAIMonitor(app)

    async def driver():
        task = asyncio.create_task(monitor.run())
        deadline = asyncio.get_event_loop().time() + 5.0
        while calls["n"] < 3 and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0)

    run(driver())
    assert calls["n"] >= 3
    assert len(app.db.rows) == 1