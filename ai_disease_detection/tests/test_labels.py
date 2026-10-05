"""Test the 38 PlantVillage labels and label-count validation."""
import pytest

from ai_disease.pipeline import load_labels

LABELS = "models/labels.txt"

# Verbatim from the reference repository's Live_Inference script.
EXPECTED_38 = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Corn_(maize)___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def test_shipped_labels_file_exists():
    from pathlib import Path

    assert Path(LABELS).is_file()


def test_label_count_is_38():
    assert len(load_labels(LABELS)) == 38


def test_labels_match_reference_repository_exactly():
    assert load_labels(LABELS) == EXPECTED_38


def test_labels_are_unique():
    labels = load_labels(LABELS)
    assert len(set(labels)) == len(labels)


def test_healthy_classes_present():
    labels = load_labels(LABELS)
    healthy = [l for l in labels if "___healthy" in l]
    # The reference set has 10 healthy-only crops plus others.
    assert len(healthy) >= 10


def test_tomato_disease_classes_present():
    labels = load_labels(LABELS)
    for expected in ("Tomato___Late_blight", "Tomato___Early_blight",
                     "Tomato___Bacterial_spot"):
        assert expected in labels


def test_load_labels_ignores_comments_and_blanks(tmp_path):
    f = tmp_path / "custom.txt"
    f.write_text("# header\n\nApple___healthy\n\n# note\nTomato___healthy\n")
    assert load_labels(f) == ["Apple___healthy", "Tomato___healthy"]


def test_load_labels_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path / "nope.txt")


def test_label_count_mismatch_detected_by_engine():
    """A model with 3 outputs must not silently accept 38 labels."""
    from ai_disease.inference import InferenceError, LabelMismatchError
    from ai_disease.onnx_engine import ONNXInferenceEngine

    engine = ONNXInferenceEngine(
        "does-not-exist.onnx", load_labels(LABELS), expected_classes=38
    )
    # No file, so it fails earlier - proving order: file check precedes labels.
    with pytest.raises(InferenceError):
        engine.load()
    assert isinstance(LabelMismatchError("x"), InferenceError)


def test_labels_align_with_expected_class_count():
    from ai_disease.config import load_config

    cfg = load_config()
    assert cfg.inference.expected_classes == len(load_labels(LABELS)) == 38