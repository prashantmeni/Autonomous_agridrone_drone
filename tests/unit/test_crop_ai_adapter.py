"""Adapter contract between the drone backend and ai_disease_detection.

Covers the wiring only: module availability, model status, quality thresholds,
and the shape of analyze() output. Inference numerics are tested inside the AI
module's own suite.
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

from drone.perception import crop_ai  # noqa: E402


def textured(seed: int = 0, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 45, (size, size, 3)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- availability
def test_module_is_available_in_this_tree():
    """The adapter must resolve the real package, not silently degrade."""
    assert crop_ai.available(), crop_ai.unavailable_reason()


def test_unavailable_reason_is_none_when_healthy():
    assert crop_ai.unavailable_reason() is None


def test_labels_load_from_the_detection_module():
    assert len(crop_ai.load_labels()) == 38


# --------------------------------------------------------------- model status
def test_status_when_disabled():
    s = crop_ai.model_status("models/x.onnx", False, 0.6)
    assert s["status"] == "MODEL_NOT_AVAILABLE"
    assert s["enabled"] is False


def test_status_when_path_empty():
    s = crop_ai.model_status("", True, 0.6)
    assert s["status"] == "MODEL_NOT_AVAILABLE"
    assert s["model_name"] is None
    assert s["model_path"] == ""


def test_status_when_file_missing():
    s = crop_ai.model_status("models/does_not_exist.onnx", True, 0.6)
    assert s["status"] == "MODEL_NOT_FOUND"
    assert "does not exist" in s["detail"]


def test_status_reports_thirty_eight_labels():
    """A healthy system knows its label count even before a model is loaded."""
    assert crop_ai.model_status("", False, 0.6)["labels"] == 0


# ------------------------------------------------------------------ thresholds
def test_quality_thresholds_are_exposed():
    t = crop_ai.quality_thresholds()
    assert t["min_brightness"] < t["max_brightness"]
    assert t["min_blur_score"] > 0


# --------------------------------------------------------------- quality gate
def test_check_image_accepts_a_good_frame():
    assert crop_ai.check_image(textured(1))["valid"] is True


def test_check_image_rejects_a_black_frame():
    r = crop_ai.check_image(np.zeros((224, 224, 3), dtype=np.uint8))
    assert r["valid"] is False
    assert r["error_code"] in ("IMAGE_TOO_DARK", "IMAGE_TOO_BRIGHT")


def test_check_image_rejects_an_undersized_frame():
    r = crop_ai.check_image(textured(2, size=8))
    assert r["error_code"] == "IMAGE_TOO_SMALL"


# ------------------------------------------------------------------- uploads
def test_max_upload_bytes_is_bounded():
    assert 0 < crop_ai.MAX_UPLOAD_BYTES <= 1024 * 1024 * 1024


@pytest.mark.parametrize("name,expected", [
    ("model.onnx.txt", "model.onnx"),
    ("model.onnx.json", "model.onnx"),
    ("model.onnx.labels", "model.onnx"),
    ("model.onnx", "model.onnx"),
])
def test_label_suffix_stripping(name, expected):
    assert crop_ai._strip_label_suffix(name) == expected


def test_model_dir_exists_or_is_creatable():
    d = crop_ai.model_dir()
    assert d.name == "models"


# ------------------------------------------------------- no-flight guarantee
def test_adapter_does_not_import_flight_control():
    """A disease module must never reach flight control."""
    src = (ROOT / "src" / "drone" / "perception" / "crop_ai.py").read_text(
        encoding="utf-8")
    for banned in ("pymavlink", "mavutil", "px4", "ardupilot", "Vehicle",
                   "set_mode", "arm", "disarm"):
        assert banned.lower() not in src.lower(), banned


def test_detection_module_has_no_flight_control_imports():
    for path in (ROOT / "ai_disease_detection" / "ai_disease").glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for banned in ("pymavlink", "mavutil", "ardupilot", "px4"):
            assert banned not in text, f"{path.name}: {banned}"
