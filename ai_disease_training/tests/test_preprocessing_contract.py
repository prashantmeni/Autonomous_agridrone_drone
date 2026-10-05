"""Regression guards against double normalisation.

Keras EfficientNet graphs contain their own `Rescaling(1/255)` stem plus an
ImageNet `Normalization`, and expect raw [0, 255] pixels. The production module
(ai_disease_detection) divides by 255 and sends [0, 1].

The correct arrangement, which these tests lock in:

  training / evaluation   feed raw [0, 255]  (no external rescaling)
  exported ONNX graph      accepts [0, 1] NCHW, prefixed with Rescaling(255)

A second external divide anywhere in the training path would scale pixels twice
and silently desynchronise training from inference. These tests fail if that
returns.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN = ROOT / "train.py"
EVALUATE = ROOT / "evaluate.py"
EXPORT = ROOT / "export_onnx.py"
VALIDATE = ROOT / "validate_onnx.py"
CONFIG = ROOT / "config.yaml"
DETECTION = ROOT.parent / "ai_disease_detection"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def calls_to(path: Path, attr: str):
    tree = ast.parse(source(path))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == attr]


# ------------------------------------------- no external scaling in training
def test_train_has_no_rescaling_1_over_255():
    """train.py must not scale pixels before EfficientNet."""
    assert "Rescaling(1.0 / 255.0)" not in source(TRAIN)
    assert "1.0 / 255.0" not in source(TRAIN)
    assert "1/255.0" not in source(TRAIN)


def test_evaluate_has_no_rescaling_1_over_255():
    assert "Rescaling(1.0 / 255.0)" not in source(EVALUATE)
    assert "1.0 / 255.0" not in source(EVALUATE)


def test_train_never_maps_a_rescale_layer_over_datasets():
    """A dataset .map() applying a Rescaling is the double-scale bug."""
    train_src = source(TRAIN)
    assert not re.search(r"\.map\(\s*lambda[^)]*rescale", train_src)
    assert "dataset.map(lambda x, y: (rescale(x), y))" not in train_src
    assert "train_ds.map(lambda x, y: (rescale(x), y))" not in train_src


def test_evaluate_never_maps_a_rescale_layer_over_dataset():
    eval_src = source(EVALUATE)
    assert not re.search(r"\.map\(\s*lambda[^)]*rescale", eval_src)


def test_no_255_division_applied_to_pixels_in_training_path():
    """Any `/ 255.0` in train/evaluate would be a second scale."""
    for path in (TRAIN, EVALUATE):
        for line_no, line in enumerate(source(path).splitlines(), 1):
            code = line.split("#", 1)[0]
            if "/ 255.0" in code or "/255.0" in code:
                # Division in an assertion message or metric is harmless, but
                # any arithmetic on an image tensor is not.
                assert "assert" in line or "print" in line, (
                    f"{path.name}:{line_no} divides by 255 in the training path: "
                    f"{line.strip()}"
                )


# ----------------------------------------- export restores [0,1] correctly
def test_export_prefixes_graph_with_rescaling_255():
    """The ONNX input must stay [0,1] while the backbone sees [0,255]."""
    export_src = source(EXPORT)
    assert "Rescaling(255.0)" in export_src, \
        "export must undo the production /255 before EfficientNet's own stem"
    # It must be applied to the input, before the transpose.
    assert re.search(
        r"inp\s*=\s*tf\.keras\.Input[\s\S]*?Rescaling\(255\.0\)\(inp\)", export_src
    )
    assert "tf.transpose" in export_src


def test_export_does_not_add_a_second_division():
    export_src = source(EXPORT)
    assert "Rescaling(1.0 / 255.0)" not in export_src


def test_export_docstring_states_the_contract():
    export_src = source(EXPORT)
    assert "[0, 1]" in export_src
    assert "255" in export_src
    # The stale "already divided by 255" claim must be gone.
    assert "already divided by 255" not in export_src


def test_train_docstring_explains_the_contract():
    train_src = source(TRAIN)
    assert "PREPROCESSING CONTRACT" in train_src
    assert "Rescaling(255)" in train_src


# ----------------------------------------------------- config must agree
def test_config_disables_external_rescale():
    text = source(CONFIG)
    assert "apply_in_graph: false" in text, \
        "apply_in_graph must be false; Keras EfficientNet rescales internally"
    assert 'type: "divide_255"' in text, \
        "the production contract stays divide_255"


def test_config_documents_why():
    text = source(CONFIG)
    assert "Rescaling(1/255)" in text
    assert "Rescaling(255)" in text


# ----------------------------- production module must stay the [0,1] contract
def test_production_preprocessing_still_divides_by_255():
    pre = DETECTION / "ai_disease" / "preprocessing.py"
    assert pre.is_file(), "production preprocessing.py missing"
    text = pre.read_text(encoding="utf-8")
    assert "x / 255.0" in text, "production must keep divide_255"
    # And must not have gained a *255 that would undo it.
    assert "x * 255" not in text
    assert "x * 255.0" not in text


def test_production_preprocessing_emits_unit_range():
    pre = (DETECTION / "ai_disease" / "preprocessing.py").read_text(encoding="utf-8")
    assert 'kind == "divide_255"' in pre
    assert "return x / 255.0" in pre


def test_production_config_still_declares_divide_255():
    cfg = (DETECTION / "config.yaml").read_text(encoding="utf-8")
    assert 'type: divide_255' in cfg or 'type: "divide_255"' in cfg


# ------------------------------------------------- validation script agrees
def test_validate_onnx_builds_both_representations():
    text = source(VALIDATE)
    assert "raw255" in text and "unit01" in text
    # Keras gets NHWC [0,255]; ONNX gets NCHW [0,1].
    assert "keras_model.predict(raw255" in text
    assert "onnx_in.name: unit01" in text


def test_validate_onnx_does_not_keras_divide_by_255():
    text = source(VALIDATE)
    assert "rgb.astype(np.float32) / 255.0" not in text
    # The only /255 must be the explicit [0,1] build for the ONNX input.
    assert "rgb / 255.0" in text


# -------------------------------------------- end-to-end transform equality
def test_transform_chain_is_mathematically_consistent():
    """raw [0,255] must reach EfficientNet identically on both paths.

    training  : pixels [0,255] -> Keras stem (1/255, Normalization) -> backbone
    inference : pixels [0,1] -> Rescaling(255) -> Keras stem -> backbone

    Both sequences are x / 255, so the backbone input is identical.
    """
    x = 255.0                      # a saturated pixel
    training = x * (1.0 / 255.0)   # Keras stem only
    inference = (x / 255.0) * 255.0 * (1.0 / 255.0)  # production, graph prefix, stem
    assert training == pytest.approx(inference)
    assert training == pytest.approx(1.0)


def test_a_double_divide_would_be_detected():
    """Guard: the bug we fixed would produce a value 255x too small."""
    x = 255.0
    correct = x * (1.0 / 255.0)            # 1.0
    double = x * (1.0 / 255.0) * (1.0 / 255.0)   # 0.0039
    assert double == pytest.approx(correct / 255.0)
    assert not math_is_close(double, correct)
    # Sanity: the ONNX path (production /255, then graph *255, then stem)
    via_onnx = (x / 255.0) * 255.0 * (1.0 / 255.0)
    assert math_is_close(via_onnx, correct)


def math_is_close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


# ------------------------------------------- backbone migration expectations
def test_config_backbone_is_efficientnetb0():
    assert 'backbone: "EfficientNetB0"' in source(CONFIG)


def test_train_backbone_registry_covers_b0_to_b7():
    text = source(TRAIN)
    for b in range(8):
        assert f'"EfficientNetB{b}"' in text, f"B{b} missing from BACKBONES"
    assert "BACKBONES" in text


def test_train_reads_backbone_from_config_not_hardcoded():
    text = source(TRAIN)
    assert 'm["backbone"]' in text, "backbone must come from config"
    assert "getattr(applications, BACKBONES[backbone_name])" in text
    # No hardcoded constructor call left.
    assert "tf.keras.applications.EfficientNetB7(" not in text
    assert "tf.keras.applications.EfficientNetB0(" not in text


def test_head_architecture_preserved():
    text = source(TRAIN)
    assert "GlobalAveragePooling2D" in text
    assert 'm["num_classes"]' in text
    assert 'activation=m["activation"]' in text
    assert 'm["include_top"]' in text
    assert 'm["weights"]' in text


def test_input_shape_preserved():
    text = source(TRAIN)
    assert "m[\"input_height\"], m[\"input_width\"], 3" in text
    cfg = source(CONFIG)
    assert "input_height: 224" in cfg
    assert "input_width: 224" in cfg
    assert "num_classes: 38" in cfg
    assert "include_top: false" in cfg
    assert 'weights: "imagenet"' in cfg
    assert "freeze_backbone: true" in cfg
    assert 'activation: "softmax"' in cfg


def test_labels_file_untouched():
    labels = DETECTION / "models" / "labels.txt"
    assert labels.is_file()
    lines = [s.strip() for s in labels.read_text(encoding="utf-8").splitlines()
             if s.strip()]
    assert len(lines) == 38
    assert "Tomato___healthy" in lines
    assert "Potato___Early_blight" in lines


def test_production_inference_untouched_by_this_change():
    """These production files must not gain any EfficientNet awareness."""
    for name in ("onnx_engine.py", "preprocessing.py", "results.py",
                 "inference.py", "pipeline.py"):
        path = DETECTION / "ai_disease" / name
        text = path.read_text(encoding="utf-8").lower()
        assert "efficientnet" not in text, f"{name} mentions EfficientNet"
        assert "rescaling" not in text, f"{name} mentions Rescaling"


def test_no_flight_control_introduced():
    banned = ("pymavlink", "mavutil", "ardupilot", "mavproxy", "rclpy", "rospy")
    offenders = []
    for path in ROOT.glob("*.py"):
        text = source(path).lower()
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, offenders
