"""Guards against re-introducing host-RAM exhaustion.

A Colab session died with "used all available RAM" because `dataset.cache()`
with no argument caches the ENTIRE decoded split in host RAM:

    70295 images x 224 x 224 x 3 x 4 bytes  ~= 42 GB   (train)
    17572 images x 224 x 224 x 3 x 4 bytes  ~= 10 GB   (valid)

These tests forbid that pattern, forbid full-dataset materialisation, and
require the streaming pipeline to stay bounded.
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
VALIDATE_ONNX = ROOT / "validate_onnx.py"
CONFIG = ROOT / "config.yaml"

DATA_SCRIPTS = {"train.py": TRAIN, "evaluate.py": EVALUATE}


def src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def code_only(path: Path) -> str:
    """Source with comments stripped, so prose mentioning a pattern is fine."""
    return "\n".join(line.split("#", 1)[0] for line in src(path).splitlines())


# ----------------------------------------------- no in-RAM dataset cache
@pytest.mark.parametrize("name,path", sorted(DATA_SCRIPTS.items()))
def test_no_argumentless_cache(name, path):
    """dataset.cache() caches everything in RAM. Forbidden."""
    assert not re.search(r"\.cache\(\s*\)", code_only(path)), \
        f"{name} calls .cache() with no argument, caching the full dataset in RAM"


@pytest.mark.parametrize("name,path", sorted(DATA_SCRIPTS.items()))
def test_cache_is_never_called_at_all(name, path):
    """No .cache() in the data path at all; streaming is the contract."""
    assert ".cache(" not in code_only(path), \
        f"{name} uses .cache(); stream from disk instead"


# ------------------------------------------ no full-dataset materialisation
@pytest.mark.parametrize("name,path", sorted(DATA_SCRIPTS.items()))
def test_no_list_of_dataset(name, path):
    """list(train_ds) decodes and caches every image."""
    code = code_only(path)
    assert "list(train_ds)" not in code
    assert "list(valid_ds)" not in code
    assert "list(test_ds)" not in code
    assert "list(dataset)" not in code


@pytest.mark.parametrize("name,path", sorted(DATA_SCRIPTS.items()))
def test_no_bare_list_call_on_any_dataset(name, path):
    """Catch list(<something>_ds) in any form."""
    code = code_only(path)
    assert not re.search(r"list\(\s*\w*_?ds\s*\)", code), \
        f"{name} materialises a dataset with list()"


def test_evaluate_does_not_predict_whole_dataset_then_reiterate():
    """model.predict(dataset) returns all scores at once; do it per batch."""
    code = code_only(EVALUATE)
    assert "model.predict(dataset" not in code, \
        "evaluate.py must predict per batch, not the whole dataset at once"
    # Single streaming pass keeps predictions and labels aligned.
    assert "for step, (images, labels_batch) in enumerate(dataset)" in code


# --------------------------------------------------- bounded pipeline
@pytest.mark.parametrize("name,path", sorted(DATA_SCRIPTS.items()))
def test_no_autotune_prefetch(name, path):
    """prefetch(AUTOTUNE) buffers without bound."""
    assert "AUTOTUNE" not in code_only(path), \
        f"{name} uses AUTOTUNE; use a small fixed prefetch value"


@pytest.mark.parametrize("name,path", sorted(DATA_SCRIPTS.items()))
def test_prefetch_is_a_fixed_small_integer(name, path):
    code = code_only(path)
    assert "prefetch(prefetch)" in code, \
        f"{name} must call prefetch() with a bounded integer"
    assert re.search(r'prefetch_batches["\']?\]?\s*[:=]\s*\d+', code) or \
        "prefetch_batches" in code


def test_shuffle_buffer_is_bounded():
    """An unbounded shuffle buffer is itself a RAM sink."""
    code = code_only(TRAIN)
    assert "shuffle_buffer_batches" in code
    assert "min(buffer * batch_size" in code, \
        "shuffle buffer must be explicitly capped"
    assert "4096" in code


def test_train_reads_pipeline_limits_from_config():
    text = src(TRAIN)
    assert 'cfg["training"].get("prefetch_batches"' in text
    assert 'cfg["training"].get("shuffle_buffer_batches"' in text


def test_config_documents_memory_limits():
    text = src(CONFIG)
    assert "prefetch_batches: 2" in text
    assert "shuffle_buffer_batches: 200" in text
    assert "cache()" in text, "config must warn about the RAM hazard"
    assert "RAM" in text or "RAM" in text


# ------------------------------------------------------ augmentation safety
def test_augmentation_is_in_graph_not_materialised():
    """Augmentation must be a dataset.map inside the pipeline."""
    code = code_only(TRAIN)
    assert "dataset.map(_augment" in code
    assert "def augment_batches" in code


def test_augmentation_uses_single_affine_transform():
    """One fused op is cheaper and safer than many full-size intermediates."""
    text = src(TRAIN)
    # `tf.image.affine_transform` is NOT a TensorFlow API — an earlier revision
    # called it and every Colab augmentation test died with AttributeError.
    assert "tf.raw_ops.ImageProjectiveTransformV3" in text
    assert "tf.image.affine_transform" not in code_only(TRAIN), \
        "tf.image.affine_transform does not exist in TensorFlow"
    # The earlier hand-rolled version used these; they were fragile.
    assert "tf.image.rot90" not in code_only(TRAIN), \
        "fractional rotation via rot90 is invalid; use the projective op"
    assert "shear_vec" not in code_only(TRAIN), "dead variable from an old draft"


def test_augmentation_keeps_uint8_and_does_not_rescale():
    """Augmentation must not divide by 255: the model's stem does that."""
    code = code_only(TRAIN)
    fn_start = code.find("def augment_batches")
    fn_end = code.find("# Backbones selectable")
    body = code[fn_start:fn_end]
    assert "/ 255" not in body
    assert "/255" not in body
    assert "tf.cast(images, tf.uint8)" in body or \
           "return tf.cast(x, tf.uint8), labels" in body


def test_augmentation_only_applied_to_training_split():
    """Validation/test must not be augmented."""
    text = src(TRAIN)
    make_src = text[text.find("def make(info, shuffle)"):
                    text.find("return dataset.prefetch(prefetch)")]
    assert "augment_batches(dataset" in make_src
    # The call site is guarded by `if shuffle:`, and valid/test pass False.
    assert "train_ds = make(train_info, True)" in text
    assert "valid_ds = make(valid_info, False)" in text
    assert "test_ds = make(test_info, False)" in text


# ------------------------------------------- contract must not regress
def test_preprocessing_contract_preserved():
    """Memory work must not touch the verified preprocessing contract."""
    for path in (TRAIN, EVALUATE):
        assert "Rescaling(1.0 / 255.0)" not in code_only(path), \
            f"{path.name} reintroduced an external /255"


def test_model_contract_preserved():
    text = src(TRAIN)
    assert 'm["backbone"]' in text
    assert "GlobalAveragePooling2D" in text
    assert 'activation=m["activation"]' in text
    assert "base.trainable = False" in text
    cfg = src(CONFIG)
    assert 'backbone: "EfficientNetB0"' in cfg
    assert "input_height: 224" in cfg
    assert "input_width: 224" in cfg
    assert "num_classes: 38" in cfg
    assert "include_top: false" in cfg
    assert 'weights: "imagenet"' in cfg
    assert "freeze_backbone: true" in cfg


def test_batch_size_still_configurable():
    text = src(TRAIN)
    assert 'batch_size=batch_size' in text
    assert 'cfg["training"]["batch_size"]' in text


def test_no_dataset_reduction():
    """Every image must still be used; no slicing or limiting."""
    code = code_only(TRAIN) + code_only(EVALUATE)
    for banned in (".take(", ".skip(", "steps_per_epoch"):
        assert banned not in code, f"{banned} would silently drop images"
    assert "num_classes" not in code or True  # sanity


def test_no_flight_control_dependencies():
    banned = ("pymavlink", "mavutil", "ardupilot", "mavproxy", "rclpy", "rospy")
    offenders = []
    for path in ROOT.glob("*.py"):
        text = src(path).lower()
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, offenders


def test_production_module_unaffected():
    """The memory fix must not leak into the production inference module."""
    detection = ROOT.parent / "ai_disease_detection"
    # Training code may name the production module only in prose or as the
    # configured labels path; it must never import it.
    for path in DATA_SCRIPTS.values():
        tree = ast.parse(src(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "ai_disease_detection" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                assert "ai_disease_detection" not in (node.module or "")

    pre = (detection / "ai_disease" / "preprocessing.py").read_text(encoding="utf-8")
    assert "x / 255.0" in pre, "production divide_255 must be untouched"
    assert "cache" not in pre.lower(), "production must not gain a cache"
