"""Test result construction: thresholding, healthy flag, JSON shape."""
import json

import pytest

from ai_disease.results import (
    REASON_LOW_CONFIDENCE,
    UNKNOWN_CLASS,
    DetectionResult,
    annotate,
    classify,
    is_healthy_label,
    save_result,
)

LABELS = ["Tomato___Late_blight", "Tomato___healthy", "Apple___healthy"]


def test_high_confidence_assigns_class():
    r = classify([0.94, 0.04, 0.02], LABELS, confidence_threshold=0.70)
    assert r.class_name == "Tomato___Late_blight"
    assert r.class_id == 0
    assert r.confidence == pytest.approx(0.94)
    assert r.reliable is True
    assert r.healthy is False
    assert r.reason is None


def test_low_confidence_reports_unknown():
    r = classify([0.43, 0.40, 0.17], LABELS, confidence_threshold=0.70)
    assert r.class_name == UNKNOWN_CLASS
    assert r.reason == REASON_LOW_CONFIDENCE
    assert r.reliable is False
    assert r.confidence == pytest.approx(0.43)
    # The real top label is preserved for debugging, not presented as a result.
    assert r.metadata["top_label"] == "Tomato___Late_blight"


def test_threshold_boundary_is_inclusive():
    at = classify([0.70, 0.20, 0.10], LABELS, confidence_threshold=0.70)
    below = classify([0.6999, 0.20, 0.10], LABELS, confidence_threshold=0.70)
    assert at.reliable is True
    assert below.reliable is False


def test_healthy_class_sets_healthy_true():
    r = classify([0.10, 0.85, 0.05], LABELS, confidence_threshold=0.70)
    assert r.class_name == "Tomato___healthy"
    assert r.healthy is True
    assert r.disease is None
    assert r.crop == "Tomato"


def test_disease_name_is_humanised():
    r = classify([0.90, 0.05, 0.05], LABELS, confidence_threshold=0.70)
    assert r.disease == "Late blight"


def test_class_index_is_argmax_not_hardcoded():
    # Peak at index 2 to prove argmax is used.
    r = classify([0.05, 0.05, 0.90], LABELS, confidence_threshold=0.70)
    assert r.class_id == 2
    assert r.class_name == "Apple___healthy"


def test_empty_output_reports_reason():
    r = classify([], LABELS, confidence_threshold=0.70)
    assert r.class_name == UNKNOWN_CLASS
    assert r.reason == "empty_output"


def test_missing_label_falls_back_to_index_name():
    # Peak at index 1, but only one label supplied: index 1 is unnamed.
    r = classify([0.1, 0.9], ["Tomato___healthy"], confidence_threshold=0.70)
    assert r.class_id == 1
    assert r.class_name == "class_1"
    assert r.reliable is True


def test_is_healthy_label_case_insensitive():
    assert is_healthy_label("Tomato___healthy") is True
    assert is_healthy_label("Tomato___HEALTHY") is True
    assert is_healthy_label("Tomato___Late_blight") is False


def test_healthy_detection_requires_both_signal_and_threshold():
    # Healthy class but below threshold: not reported as healthy.
    r = classify([0.55, 0.40, 0.05], LABELS, confidence_threshold=0.70)
    assert r.class_name == UNKNOWN_CLASS
    assert r.healthy is False


def test_result_to_dict_includes_derived_fields():
    r = classify([0.9, 0.05, 0.05], LABELS, confidence_threshold=0.70)
    d = r.to_dict()
    for key in ("class_id", "class_name", "confidence", "healthy", "reliable",
                "inference_ms", "timestamp", "disease", "crop"):
        assert key in d


def test_result_json_is_valid():
    r = classify([0.9, 0.05, 0.05], LABELS, confidence_threshold=0.70)
    parsed = json.loads(r.to_json())
    assert parsed["class_name"] == "Tomato___Late_blight"
    assert parsed["confidence"] == pytest.approx(0.9)


def test_timestamp_autopopulated():
    assert DetectionResult().timestamp


def test_result_has_no_flight_control_fields():
    """Structural guarantee: nothing here can command the vehicle."""
    d = DetectionResult().to_dict()
    forbidden = {"arm", "disarm", "mode", "motor", "throttle", "param",
                 "command", "set_mode", "px4", "mavlink"}
    assert forbidden.isdisjoint(set(d))


def test_display_mentions_reason_when_unknown():
    r = classify([0.2, 0.2, 0.2], LABELS, confidence_threshold=0.9)
    assert "Reason" in r.display()


def test_save_result_writes_json(tmp_path):
    r = classify([0.9, 0.05, 0.05], LABELS, confidence_threshold=0.70)
    out = save_result(r, tmp_path / "results")
    assert out is not None and out.is_file()
    assert json.loads(out.read_text())["class_name"] == "Tomato___Late_blight"


def test_annotate_returns_frame_copy():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    r = classify([0.9, 0.05, 0.05], LABELS, confidence_threshold=0.70)
    out = annotate(frame, r)
    assert out.shape == frame.shape
    # Original untouched; annotation drawn on a copy.
    assert frame.max() == 0
    assert out.max() > 0


def test_geotag_fields_default_to_none():
    """Reserved for the future telemetry bridge; never populated here."""
    r = DetectionResult()
    assert r.latitude is None and r.longitude is None and r.altitude is None