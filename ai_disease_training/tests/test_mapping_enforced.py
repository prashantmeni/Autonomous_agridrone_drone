"""Verifies the label-mapping guarantee in train.py and evaluate.py.

Uses AST inspection rather than importing those modules, because they require
TensorFlow at import time. This keeps the check runnable on any machine while
still asserting on the real source.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    build_label_mapping,
    class_names_to_index_list,
    load_config,
    load_production_labels,
    normalize_class_name,
)

CONFIG = ROOT / "config.yaml"


def function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found in {path.name}")


def call_kwargs(func_src: str, func_name: str) -> str:
    """Return the source of one call inside a function body."""
    tree = ast.parse(func_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            target = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if target == func_name:
                return ast.unparse(node)
    raise AssertionError(f"no call to {func_name} in the function")


# ------------------------------------------------------------- mapping math
def test_index_i_means_labels_i_when_folders_are_reversed():
    """Core guarantee, independent of how the filesystem sorts."""
    labels = load_production_labels(load_config(CONFIG))
    mapping, missing, unexpected = build_label_mapping(list(reversed(labels)),
                                                        labels)
    assert not missing and not unexpected
    ordered = class_names_to_index_list(mapping)
    for i, folder in enumerate(ordered):
        assert mapping[folder] == i
        assert normalize_class_name(folder) == normalize_class_name(labels[i])


@pytest.mark.parametrize("rotation", [1, 7, 19, 37])
def test_mapping_survives_arbitrary_rotation(rotation):
    labels = load_production_labels(load_config(CONFIG))
    rotated = labels[rotation:] + labels[:rotation]
    mapping, missing, unexpected = build_label_mapping(rotated, labels)
    assert not missing and not unexpected
    ordered = class_names_to_index_list(mapping)
    for i, folder in enumerate(ordered):
        assert mapping[folder] == i


def test_mapping_survives_reversed_dataset():
    labels = load_production_labels(load_config(CONFIG))
    mapping, _, _ = build_label_mapping(list(reversed(labels)), labels)
    assert class_names_to_index_list(mapping) == labels


def test_flag_does_not_change_mapping():
    """require_exact_label_order only gates validation, never the mapping."""
    labels = load_production_labels(load_config(CONFIG))
    reordered = list(reversed(labels))
    results = []
    for flag in (True, False):
        cfg = load_config(CONFIG)
        cfg["dataset"]["require_exact_label_order"] = flag
        m, missing, unexpected = build_label_mapping(reordered, labels)
        assert not missing and not unexpected
        results.append(class_names_to_index_list(m))
    assert results[0] == results[1] == labels


# ---------------------------------------------------- train.py enforcement
def test_build_datasets_derives_class_names_from_mapping():
    src = function_source(ROOT / "train.py", "build_datasets")
    assert "sorted(mapping.items()" in src
    assert "key=lambda kv: kv[1]" in src
    call = call_kwargs(src, "image_dataset_from_directory")
    assert "class_names=class_names" in call


def test_build_datasets_shuffles_train_only():
    src = function_source(ROOT / "train.py", "build_datasets")
    assert "make(train_info, True)" in src
    assert "make(valid_info, False)" in src
    assert "make(test_info, False)" in src


def test_build_datasets_verifies_split_folders_match_train():
    src = function_source(ROOT / "train.py", "build_datasets")
    assert "does not match the train split" in src
    assert "missing from" in src and "unexpected in" in src


def test_build_datasets_rejects_wrong_class_count():
    src = function_source(ROOT / "train.py", "build_datasets")
    assert "mapping covers" in src
    assert "expected" in src


def test_train_main_cross_checks_saved_mapping():
    src = function_source(ROOT / "train.py", "main")
    assert "label_mapping" in src
    assert "no longer matches" in src
    assert "Re-run validate_dataset.py" in src


def test_train_main_writes_mapping_before_training():
    src = function_source(ROOT / "train.py", "main")
    assert "save_json" in src
    assert "production_labels" in src


def test_train_main_uses_output_dir_helper():
    src = function_source(ROOT / "train.py", "main")
    assert "output_dir(cfg)" in src


# --------------------------------------------------- evaluate.py safeguards
def test_evaluate_refuses_silent_valid_substitution():
    src = function_source(ROOT / "evaluate.py", "build_eval_dataset")
    assert "Refusing to silently substitute the validation split" in src
    assert "not a real" in src


def test_evaluate_flags_non_test_split_in_report():
    src = function_source(ROOT / "evaluate.py", "main")
    assert "MEASURED ON THE VALIDATION SPLIT" in src
    assert "is_true_test_set" in src


def test_evaluate_defaults_to_test_split():
    tree = ast.parse((ROOT / "evaluate.py").read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "add_argument":
            continue
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if "--split" not in flags:
            continue
        found = True
        kwargs = {k.arg: k.value for k in node.keywords}
        assert getattr(kwargs.get("default"), "value", None) == "test"
        choices = getattr(kwargs.get("choices"), "elts", [])
        assert "test" in [getattr(e, "value", None) for e in choices]
    assert found, "--split argument not found in evaluate.py"


# ------------------------------------------------------------- flight safety
def test_no_flight_control_references_in_training_pipeline():
    banned = ("pymavlink", "mavutil", "ardupilot", "mavproxy", "rclpy", "rospy",
              "vehicle.arm", "set_mode")
    offenders = []
    for path in ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, offenders


def test_production_module_untouched_by_training_pipeline():
    """Training code may not import the production module.

    Reading labels.txt (a plain data file) is required and allowed; importing
    src.* from the inference package is not, since that would couple the
    training pipeline to runtime code.
    """
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "ai_disease_detection" not in alias.name, \
                        f"{path.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "ai_disease_detection" not in module, \
                    f"{path.name} imports from {module}"


def test_training_reads_production_labels_as_data_only():
    """The only permitted coupling: reading labels.txt via a config path."""
    cfg_text = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "ai_disease_detection/models/labels.txt" in cfg_text


def test_no_model_exists_before_training():
    out = ROOT / "output"
    for name in ("model2.h5", "disease_model.onnx", "model.keras"):
        assert not (out / name).is_file(), f"{name} must not exist before training"
    prod = ROOT.parent / "ai_disease_detection" / "models" / "disease_model.onnx"
    assert not prod.is_file(), "production model must not exist before training"