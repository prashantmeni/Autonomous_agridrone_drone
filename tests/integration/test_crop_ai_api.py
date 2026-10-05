"""HTTP contract for the Crop AI endpoints the dashboard renders.

These lock the response shape the frontend depends on, so a backend refactor
cannot silently break the Crop AI tab.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from drone.main import build_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    cfg, dapp, api = build_app(str(ROOT / "config" / "development.yaml"))
    return TestClient(api)


def textured(seed: int = 0, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 45, (size, size, 3)), 0, 255).astype(np.uint8)


def jpeg_bytes(seed: int = 0) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", textured(seed))
    assert ok
    return buf.tobytes()


# --------------------------------------------------------------- model status
def test_disease_model_endpoint_200(client):
    r = client.get("/api/disease/model")
    assert r.status_code == 200
    assert "status" in r.json()


def test_crop_ai_model_alias_200(client):
    r = client.get("/api/crop-ai/model")
    assert r.status_code == 200
    body = r.json()
    # Fields the dashboard's Plant Health page reads.
    assert "confidence_thresholds" in body
    assert "module_available" in body


def test_crop_ai_alias_reports_module_availability(client):
    assert client.get("/api/crop-ai/model").json()["module_available"] is True


# -------------------------------------------------------------------- uploads
def test_upload_rejects_unsupported_extension(client):
    r = client.post("/api/disease/model",
                    files={"file": ("m.zip", b"x", "application/zip")})
    assert r.status_code == 400


@pytest.mark.parametrize("name", ["m.pt", "m.pth", "m.h5", "m.keras"])
def test_upload_rejects_frameworks_needing_heavy_deps(client, name):
    r = client.post("/api/disease/model",
                    files={"file": (name, b"x", "application/octet-stream")})
    assert r.status_code == 400


def test_upload_rejects_corrupt_model(client):
    """A rejected candidate must not replace the active model."""
    r = client.post("/api/disease/model",
                    files={"file": ("broken.onnx", b"definitely not onnx",
                                    "application/octet-stream")})
    assert r.status_code == 400
    assert r.json()["detail"]["error_code"] == "MODEL_INVALID"


def test_upload_rejects_empty_file(client):
    r = client.post("/api/disease/model",
                    files={"file": ("empty.onnx", b"", "application/octet-stream")})
    assert r.status_code == 400


def test_labels_require_their_model_first(client):
    r = client.post("/api/disease/model",
                    files={"file": ("orphan.onnx.txt", b"a\nb\n", "text/plain")})
    assert r.status_code == 400
    assert "Upload the model file first" in r.json()["detail"]


def test_rejected_model_leaves_no_file_behind(client):
    models = ROOT / "models"
    client.post("/api/disease/model",
                files={"file": ("leftover.onnx", b"junk",
                                "application/octet-stream")})
    assert not (models / "leftover.onnx").exists()


# ------------------------------------------------------------------- analysis
def test_analyze_rejects_undecodable_upload(client):
    r = client.post("/api/disease/analyze",
                    files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["error_code"] == "IMAGE_INVALID"
    assert body["reliable"] is False
    assert body["confidence_level"] == "NONE"


def test_analyze_without_model_is_503_not_500(client):
    r = client.post("/api/disease/analyze",
                    files={"file": ("leaf.jpg", jpeg_bytes(1), "image/jpeg")})
    assert r.status_code == 503


def test_analyze_does_not_leak_a_traceback(client):
    r = client.post("/api/disease/analyze",
                    files={"file": ("leaf.jpg", jpeg_bytes(2), "image/jpeg")})
    assert "Traceback" not in r.text
    assert "File \"" not in r.text


def test_no_disease_endpoint_references_removed_modules(client):
    """The legacy perception trio must be gone from the running app."""
    import drone.api.server as server_mod

    src = Path(server_mod.__file__).read_text(encoding="utf-8")
    for banned in ("plant_disease", "model_activation"):
        assert banned not in src, banned
