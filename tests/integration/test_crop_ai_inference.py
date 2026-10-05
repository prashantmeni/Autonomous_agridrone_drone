"""End-to-end adapter inference against a real ONNX Runtime session.

Builds a tiny synthetic classifier graph so the whole backend -> adapter ->
AI-module path runs without a trained model. Proves the response shape the
dashboard renders, and that the quality gate blocks inference before the model.

The model here is a fixed lookup table, not a trained network: it verifies
wiring and JSON shape only, never accuracy.
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

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnx  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

from drone.perception import crop_ai  # noqa: E402

NUM_CLASSES = 38
# One strong class (index 0), the rest near zero: a peaked softmax-like output.
SCORES = np.full(NUM_CLASSES, 0.002, dtype=np.float32)
SCORES[0] = 0.93


@pytest.fixture(scope="module")
def synthetic_model(tmp_path_factory) -> Path:
    """A valid (1,3,224,224) -> (1,38) ONNX graph with constant output."""
    path = tmp_path_factory.mktemp("models") / "synthetic.onnx"
    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 224, 224])
    out = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, NUM_CLASSES])
    const = helper.make_tensor("w", TensorProto.FLOAT, [NUM_CLASSES],
                               SCORES.tolist())
    node = helper.make_node("Identity", ["w"], ["output"], name="const_out")
    graph = helper.make_graph([node], "synthetic", [inp], [out], [const])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def textured(seed: int = 0, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 45, (size, size, 3)), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ happy path
def test_analyze_returns_dashboard_shape(synthetic_model):
    out = crop_ai.analyze(textured(1), str(synthetic_model), 0.60,
                          source="upload", image_path="leaf.jpg")
    # Fields PlantHealth.tsx reads.
    for key in ("status", "success", "crop", "disease", "confidence",
                "confidence_level", "reliable", "image_quality", "model",
                "timing", "source"):
        assert key in out, key
    assert out["success"] is True
    assert out["reliable"] is True
    assert out["status"] == "OK"
    assert out["confidence"] > 0.9
    assert out["confidence_level"] == "HIGH"
    assert out["source"] == "upload"


def test_timing_block_is_populated(synthetic_model):
    out = crop_ai.analyze(textured(2), str(synthetic_model), 0.60)
    assert set(out["timing"]) == {"preprocess_ms", "inference_ms",
                                  "postprocess_ms", "total_ms"}
    assert out["timing"]["inference_ms"] > 0.0
    assert out["timing"]["total_ms"] >= out["timing"]["inference_ms"]


def test_model_block_names_the_active_model(synthetic_model):
    out = crop_ai.analyze(textured(3), str(synthetic_model), 0.60)
    assert out["model"]["name"] == "synthetic.onnx"


def test_quality_block_present_on_success(synthetic_model):
    out = crop_ai.analyze(textured(4), str(synthetic_model), 0.60)
    assert out["image_quality"]["valid"] is True
    assert out["image_quality"]["metrics"]["brightness"] > 0


def test_argmax_maps_to_first_label(synthetic_model):
    out = crop_ai.analyze(textured(5), str(synthetic_model), 0.60)
    assert out["class_id"] == 0
    assert out["label"] == out["disease"]


# --------------------------------------------------------------- quality gate
def test_black_frame_is_rejected_without_a_disease(synthetic_model):
    out = crop_ai.analyze(np.zeros((224, 224, 3), np.uint8),
                          str(synthetic_model), 0.60, source="camera")
    assert out["success"] is False
    assert out["reliable"] is False
    assert out["status"] != "OK"
    assert out["confidence_level"] == "NONE"
    assert out["disease"] is None or out["disease"] == "Unknown"


def test_rejection_carries_actionable_text(synthetic_model):
    out = crop_ai.analyze(np.zeros((224, 224, 3), np.uint8),
                          str(synthetic_model), 0.60)
    assert out["message"]
    assert out["suggestion"]


def test_rejection_skips_the_model_call(synthetic_model):
    """The gate runs before inference, so timing must show zero inference."""
    out = crop_ai.analyze(np.zeros((224, 224, 3), np.uint8),
                          str(synthetic_model), 0.60)
    assert out["timing"]["inference_ms"] == 0.0


def test_undersized_frame_rejected(synthetic_model):
    out = crop_ai.analyze(textured(6, 16), str(synthetic_model), 0.60)
    assert out["status"] == "IMAGE_TOO_SMALL"


def test_flat_grey_frame_rejected(synthetic_model):
    out = crop_ai.analyze(np.full((224, 224, 3), 120, np.uint8),
                          str(synthetic_model), 0.60)
    assert out["success"] is False
    assert out["confidence_level"] == "NONE"


# ------------------------------------------------------------ confidence band
@pytest.mark.parametrize("threshold,expected", [
    (0.10, "HIGH"),    # 0.93 clears the 0.80 high cut-off
    (0.60, "HIGH"),
])
def test_band_is_high_when_peaked(synthetic_model, threshold, expected):
    out = crop_ai.analyze(textured(7), str(synthetic_model), threshold)
    assert out["confidence_level"] == expected


def test_band_uses_configured_cutoffs(synthetic_model):
    """Bands must come from the config the model endpoint advertises."""
    out = crop_ai.analyze(textured(11), str(synthetic_model), 0.10,
                          high_threshold=0.99, medium_threshold=0.95)
    # 0.93 is below both, so it lands in the lowest band rather than HIGH.
    assert out["confidence_level"] == "LOW"


def test_band_is_none_below_store_threshold(synthetic_model):
    out = crop_ai.analyze(textured(8), str(synthetic_model), 0.99)
    assert out["confidence_level"] == "NONE"
    assert out["success"] is False


def test_error_code_and_marked_always_present(synthetic_model):
    """The UI branches on these, so they must exist on every response."""
    for frame in (textured(12), np.zeros((224, 224, 3), np.uint8)):
        out = crop_ai.analyze(frame, str(synthetic_model), 0.60)
        assert "error_code" in out
        assert "marked" in out
        assert out["marked"] is False


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(Exception) as exc:
        crop_ai.analyze(textured(9), str(tmp_path / "nope.onnx"), 0.60)
    assert "not found" in str(exc.value).lower()
