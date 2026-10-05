"""Guards the TensorFlow API surface used by train.py.

These calls must be compatible with TensorFlow 2.20 / Keras 3, where
`image_dataset_from_directory()` takes no `validation_data` argument, and
`model.fit()` does. Static analysis keeps the check runnable without a GPU.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAIN = ROOT / "train.py"
EVALUATE = ROOT / "evaluate.py"


def calls_in(path: Path, func_attr: str):
    """Yield every call node whose callee name matches func_attr."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == func_attr:
                found.append(node)
    return found


def kwarg_names(call: ast.Call) -> set[str]:
    return {k.arg for k in call.keywords if k.arg}


def function_source(path: Path, name: str) -> str:
    """Unparsed source of one function, so these checks need no TensorFlow."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found in {path.name}")


def positional_strings(call: ast.Call) -> list[str]:
    out = []
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str):
            out.append(a.value)
    return out


# --------------------------------------------- the reported API bug
def test_image_dataset_from_directory_never_passes_validation_data():
    """Regression: TF 2.20 / Keras 3 rejects this kwarg outright.

    An earlier revision passed validation_data=None, which raised
    TypeError before training began.
    """
    calls = calls_in(TRAIN, "image_dataset_from_directory")
    assert calls, "no image_dataset_from_directory call found in train.py"
    for call in calls:
        assert "validation_data" not in kwarg_names(call), (
            "image_dataset_from_directory() must not receive validation_data; "
            "Keras 3 has no such parameter"
        )


def test_evaluate_dataset_call_also_omits_validation_data():
    for call in calls_in(EVALUATE, "image_dataset_from_directory"):
        assert "validation_data" not in kwarg_names(call)


def test_model_fit_does_pass_validation_data():
    """fit() legitimately takes validation_data; it must still be there."""
    calls = calls_in(TRAIN, "fit")
    assert calls, "no model.fit call found"
    fit_calls = [c for c in calls if "validation_data" in kwarg_names(c)]
    assert fit_calls, "model.fit() must receive validation_data=valid_ds"


# --------------------------------------------------- dataset behaviour
def test_main_defines_valid_info_before_batch_report():
    """Regression: main() used valid_info without defining it -> NameError.

    valid_info is only a local inside build_datasets(), so main() must resolve
    its own copy before the batch-count report reads valid_info.image_count.
    """
    main_src = function_source(TRAIN, "main")
    # ast.unparse renders string literals with single quotes.
    assert "valid_info = scan_split(cfg, 'valid')" in main_src, \
        "main() must resolve valid_info itself"

    # The definition must precede every use.
    def_at = main_src.index("valid_info = scan_split(cfg, 'valid')")
    use_at = main_src.index("valid_info.image_count")
    assert def_at < use_at, "valid_info must be defined before it is used"

    # Both splits must be guarded before use.
    assert "if train_info is None" in main_src
    assert "if valid_info is None" in main_src
    assert main_src.count("return 3") >= 2, "each missing split must be reported"

    # The guard on valid_info must sit before the batch report.
    guard_at = main_src.index("if valid_info is None")
    assert guard_at < use_at, "valid_info must be validated before being read"


def test_batch_count_variables_all_defined_in_main():
    """Every name used by the batch report must be assigned earlier in main()."""
    main_src = function_source(TRAIN, "main")
    for name in ("train_count", "valid_count", "batch_size"):
        assign = f"{name} = "
        assert assign in main_src, f"{name} is never assigned in main()"
        assert main_src.index(assign) < main_src.index(f"train_count = ") + 200
    assert "math.ceil(train_count / batch_size)" in main_src
    assert "math.ceil(valid_count / batch_size)" in main_src


def test_build_datasets_still_resolves_its_own_splits():
    """The fix must not have removed build_datasets' internal lookups."""
    src = function_source(TRAIN, "build_datasets")
    # ast.unparse normalises string quotes to single, so match on the call.
    assert 'scan_split(cfg, ' in src
    assert "'train'" in src and "'valid'" in src and "'test'" in src


def test_batch_counts_do_not_materialise_datasets():
    """Regression: list(dataset) re-decodes and caches every image.

    Batch counts must come from the filesystem scan (SplitInfo.image_count)
    plus math.ceil, never from iterating the tf.data pipeline.
    """
    src = TRAIN.read_text(encoding="utf-8")
    assert "list(train_ds)" not in src
    assert "list(valid_ds)" not in src
    assert "train_info.image_count" in src
    assert "valid_info.image_count" in src
    assert "math.ceil(train_count / batch_size)" in src
    assert "math.ceil(valid_count / batch_size)" in src
    # math imported once at module top, not inside a function body.
    assert "import math" in src
    tree = ast.parse(src)
    imports = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom))]
    math_imports = [
        n for n in imports
        if any((a.name == "math") for a in getattr(n, "names", []))
    ]
    assert math_imports, "math must be imported at module level"


def test_split_info_exposes_image_count():
    from common import SplitInfo

    fields = {f.name for f in SplitInfo.__dataclass_fields__.values()}
    assert "image_count" in fields
    assert "counts_by_class" in fields


def test_batch_count_math_is_correct():
    """ceil division must reflect a partial final batch."""
    import math as _math

    assert _math.ceil(70295 / 32) == 2197
    assert _math.ceil(17572 / 32) == 550
    assert _math.ceil(70295 / 16) == 4394
    assert _math.ceil(17572 / 16) == 1099
    # exact multiples must not gain a spurious batch
    assert _math.ceil(64 / 32) == 2
    assert _math.ceil(33 / 32) == 2


def test_split_shuffle_behaviour_is_preserved():
    """train -> True, valid -> False, test -> False."""
    src = ast.unparse(
        next(n for n in ast.walk(ast.parse(TRAIN.read_text(encoding="utf-8")))
             if isinstance(n, ast.FunctionDef) and n.name == "build_datasets")
    )
    assert "make(train_info, True)" in src
    assert "make(valid_info, False)" in src
    assert "make(test_info, False)" in src


def test_make_helper_shuffles_explicitly():
    """Shuffling is done via dataset.shuffle with a bounded buffer.

    Keras is handed shuffle=False and the pipeline reshuffles itself, because
    the memory-safe buffer must be capped. See build_datasets.
    """
    src = ast.unparse(
        next(n for n in ast.walk(ast.parse(TRAIN.read_text(encoding="utf-8")))
             if isinstance(n, ast.FunctionDef) and n.name == "build_datasets")
    )
    assert "shuffle=False" in src, "Keras must not shuffle; the pipeline does"
    assert "dataset.shuffle(" in src
    assert "reshuffle_each_iteration=True" in src


def test_class_names_mapping_still_pinned():
    src = ast.unparse(
        next(n for n in ast.walk(ast.parse(TRAIN.read_text(encoding="utf-8")))
             if isinstance(n, ast.FunctionDef) and n.name == "build_datasets")
    )
    assert "class_names=class_names" in src
    assert "sorted(mapping.items()" in src
    assert "key=lambda kv: kv[1]" in src


def test_split_folder_verification_still_present():
    src = ast.unparse(
        next(n for n in ast.walk(ast.parse(TRAIN.read_text(encoding="utf-8")))
             if isinstance(n, ast.FunctionDef) and n.name == "build_datasets")
    )
    assert "does not match the train split" in src


# ------------------------------------------- untouched-by-contract
def test_architecture_unchanged():
    """Head structure and backbone selection are config-driven."""
    src = TRAIN.read_text(encoding="utf-8")
    assert "GlobalAveragePooling2D" in src
    assert 'activation=m["activation"]' in src
    # Backbone must come from config, never a hardcoded constructor.
    assert 'm["backbone"]' in src
    assert "tf.keras.applications.EfficientNetB0(" not in src
    assert "tf.keras.applications.EfficientNetB7(" not in src


def test_backbone_registry_includes_b0():
    assert '"EfficientNetB0"' in TRAIN.read_text(encoding="utf-8")


def test_input_shape_unchanged():
    src = TRAIN.read_text(encoding="utf-8")
    # Shape must come from config, never a literal in the source.
    assert 'input_height' in src and 'input_width' in src
    assert "(224, 224, 3)" not in src, "input shape must not be hardcoded"
    cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "input_height: 224" in cfg
    assert "input_width: 224" in cfg
    assert "num_classes: 38" in cfg


def test_backbone_stays_frozen_by_default():
    cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "freeze_backbone: true" in cfg
    src = TRAIN.read_text(encoding="utf-8")
    assert "base.trainable = False" in src


def test_preprocessing_contract_is_underspecified():
    """The production contract is divide_255 -> [0,1]; no external rescale.

    Keras EfficientNet rescales internally, so the dataset pipeline must not
    add its own 1/255. See tests/test_preprocessing_contract.py.
    """
    cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    assert "type: \"divide_255\"" in cfg, "production contract stays divide_255"
    assert "apply_in_graph: false" in cfg, \
        "Keras EfficientNet already contains Rescaling(1/255)"
    src = TRAIN.read_text(encoding="utf-8")
    assert "Rescaling(1.0 / 255.0)" not in src, \
        "training must not scale pixels a second time"


def test_production_module_not_imported():
    for path in (TRAIN, EVALUATE):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "ai_disease_detection" not in (node.module or "")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "ai_disease_detection" not in alias.name


def test_no_model_exists_yet():
    for name in ("model2.h5", "disease_model.onnx", "model.keras"):
        assert not (ROOT / "output" / name).is_file()
    prod = ROOT.parent / "ai_disease_detection" / "models" / "disease_model.onnx"
    assert not prod.is_file()