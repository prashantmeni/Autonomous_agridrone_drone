"""The adapter's JSON must contain every field the dashboard renders.

The frontend is TypeScript, so a missing field is a silent `undefined` at
runtime rather than a compile error. This test is the contract check: it parses
the fields PlantHealth.tsx actually reads and asserts the adapter emits them.
"""
from __future__ import annotations

import re
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

PAGE = ROOT / "dashboard" / "src" / "pages" / "PlantHealth.tsx"
# analysis.<field> / analysis?.<field>, plus nested image_quality./timing. reads.
ACCESS = re.compile(
    r"analysis\??\.(image_quality|timing|detail|note|marked|label)"
    r"(?:\??\.(valid|error_code|metrics|total_ms|inference_ms))?"
    r"|analysis\??\.([a-z_]+)"
)


@pytest.fixture(scope="module")
def synthetic_model(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "contract.onnx"
    inp = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 224, 224])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 38])
    scores = np.full(38, 0.002, dtype=np.float32)
    scores[0] = 0.93
    const = helper.make_tensor("w", TensorProto.FLOAT, [38], scores.tolist())
    node = helper.make_node("Identity", ["w"], ["output"], name="const_out")
    graph = helper.make_graph([node], "c", [inp], [out], [const])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 45, (224, 224, 3)), 0, 255).astype(np.uint8)


def frontend_fields() -> set[str]:
    src = PAGE.read_text(encoding="utf-8")
    fields: set[str] = set()
    for match in ACCESS.finditer(src):
        for group in match.groups():
            if group:
                fields.add(group)
    return fields


def test_frontend_page_exists():
    assert PAGE.is_file()


def test_frontend_reads_fields():
    """Guard against the regex silently matching nothing."""
    fields = frontend_fields()
    assert {"confidence", "disease", "reliable", "status"} <= fields


@pytest.mark.parametrize("frame_name", ["good", "rejected"])
def test_adapter_emits_every_field_the_frontend_reads(synthetic_model, frame_name):
    frame = textured(1) if frame_name == "good" else np.zeros((224, 224, 3),
                                                             np.uint8)
    body = crop_ai.analyze(frame, str(synthetic_model), 0.60, source="camera")

    missing = {f for f in frontend_fields() if f not in body}
    assert not missing, f"dashboard reads fields the adapter omits: {missing}"


def test_nested_quality_and_timing_blocks_are_present(synthetic_model):
    body = crop_ai.analyze(textured(2), str(synthetic_model), 0.60)
    assert isinstance(body["image_quality"], dict)
    assert "valid" in body["image_quality"]
    assert isinstance(body["timing"], dict)
    for key in ("total_ms", "inference_ms"):
        assert key in body["timing"]


def test_rejected_analysis_also_exposes_frontend_fields(synthetic_model):
    """The UI renders its error branch from the same response object."""
    body = crop_ai.analyze(np.zeros((224, 224, 3), np.uint8),
                           str(synthetic_model), 0.60)
    for key in ("status", "success", "reliable", "confidence_level",
                "message", "suggestion", "confidence"):
        assert key in body, key
