"""Test the dataset-validation and label-mapping logic.

Runs without TensorFlow: the pieces most likely to go silently wrong are class
ordering and path resolution, not the model.
"""
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
    resolve,
    scan_split,
    set_seed,
    strip_private_keys,
)

yaml = pytest.importorskip("yaml")

CONFIG = ROOT / "config.yaml"


def make_dataset(tmp_path: Path, classes: list[str], per_class: int = 2) -> Path:
    """Create a fake dataset tree so validation can be exercised offline."""
    for split in ("train", "valid"):
        for name in classes:
            folder = tmp_path / split / name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(per_class):
                (folder / f"{i}.jpg").write_bytes(b"fake")
    return tmp_path


@pytest.fixture()
def cfg():
    return load_config(CONFIG)


# ------------------------------------------------------------------- config
def test_config_loads():
    c = load_config(CONFIG)
    assert c["model"]["num_classes"] == 38
    assert c["model"]["backbone"] == "EfficientNetB0"
    assert c["model"]["input_height"] == 224
    assert c["model"]["input_width"] == 224
    assert c["preprocessing"]["type"] == "divide_255"


def test_config_arch_matches_reference():
    """Architecture shape is unchanged; only the backbone was migrated to B0."""
    c = load_config(CONFIG)
    assert c["model"]["include_top"] is False
    assert c["model"]["activation"] == "softmax"
    assert c["model"]["freeze_backbone"] is True
    assert c["training"]["loss"] == "categorical_crossentropy"
    assert c["model"]["weights"] == "imagenet"


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config(ROOT / "no_such_config.yaml")


def test_resolve_is_relative_to_config():
    c = load_config(CONFIG)
    p = resolve(c, "data/thing")
    assert p.is_absolute()
    assert p.name == "thing"


def test_resolve_keeps_absolute():
    c = load_config(CONFIG)
    assert resolve(c, "/opt/x").name == "x"
    assert resolve(c, "/opt/x").is_absolute()


def test_strip_private_keys():
    c = load_config(CONFIG)
    assert "_base_dir" not in strip_private_keys(c)
    assert "model" in strip_private_keys(c)


# ------------------------------------------------------------------ labels
def test_production_labels_are_38(cfg):
    labels = load_production_labels(cfg)
    assert len(labels) == 38
    assert len(set(labels)) == 38


def test_production_labels_match_detection_module(cfg):
    """Must be the same file the inference module reads."""
    detection_labels = (
        ROOT.parent / "ai_disease_detection" / "models" / "labels.txt"
    )
    assert detection_labels.is_file(), "production labels.txt missing"
    on_disk = [
        s.strip() for s in detection_labels.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]
    assert load_production_labels(cfg) == on_disk


def test_production_labels_contain_known_classes(cfg):
    labels = load_production_labels(cfg)
    for expected in ("Tomato___Late_blight", "Tomato___healthy",
                     "Potato___Early_blight", "Apple___Apple_scab"):
        assert expected in labels


def test_healthy_classes_present(cfg):
    labels = load_production_labels(cfg)
    assert sum(1 for l in labels if l.endswith("___healthy")) >= 10


# ------------------------------------------------- class name normalisation
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Tomato___Late_blight", "tomato late blight"),
        ("Tomato Late Blight", "tomato late blight"),
        ("Pepper,_bell___Bacterial_spot", "pepper bell bacterial spot"),
        ("Pepper bell Bacterial spot", "pepper bell bacterial spot"),
        ("Corn_(maize)___Common_rust", "corn (maize) common rust"),
        ("Cherry_(including_sour)___healthy", "cherry (including sour) healthy"),
    ],
)
def test_normalize_class_name(raw, expected):
    assert normalize_class_name(raw) == expected


def test_normalization_is_idempotent():
    once = normalize_class_name("Tomato___Late_blight")
    assert normalize_class_name(once) == once


# ----------------------------------------------------------- label mapping
def test_mapping_is_identity_when_names_match(cfg):
    labels = load_production_labels(cfg)
    mapping, missing, unexpected = build_label_mapping(list(labels), labels)
    assert missing == []
    assert unexpected == []
    assert mapping == {name: i for i, name in enumerate(labels)}
    assert class_names_to_index_list(mapping) == labels


def test_mapping_detects_reordered_folders(cfg):
    labels = load_production_labels(cfg)
    shuffled = list(reversed(labels))
    mapping, missing, unexpected = build_label_mapping(shuffled, labels)
    assert missing == [] and unexpected == []
    # Ordering must still come out matching labels.txt.
    assert class_names_to_index_list(mapping) == labels


def test_mapping_detects_spacing_differences(cfg):
    """Renamed folders still map onto the correct production indices."""
    labels = load_production_labels(cfg)
    spaced = [l.replace("___", " ").replace("_", " ") for l in labels]
    mapping, missing, unexpected = build_label_mapping(spaced, labels)
    assert missing == []
    assert unexpected == []
    # Ordering is by production index, so index i keeps its disease meaning
    # even though the folder name differs.
    ordered = class_names_to_index_list(mapping)
    assert len(ordered) == 38
    for i, name in enumerate(ordered):
        assert mapping[name] == i
        assert normalize_class_name(name) == normalize_class_name(labels[i])


def test_mapping_reports_missing_classes(cfg):
    labels = load_production_labels(cfg)
    partial = labels[:-3]
    mapping, missing, unexpected = build_label_mapping(partial, labels)
    assert len(missing) == 3
    assert unexpected == []


def test_mapping_reports_unexpected_classes(cfg):
    labels = load_production_labels(cfg)
    extra = list(labels) + ["Tomato___Imaginary_disease"]
    mapping, missing, unexpected = build_label_mapping(extra, labels)
    assert missing == []
    assert unexpected == ["Tomato___Imaginary_disease"]


def test_mapping_indices_are_contiguous(cfg):
    labels = load_production_labels(cfg)
    mapping, _, _ = build_label_mapping(list(reversed(labels)), labels)
    assert sorted(mapping.values()) == list(range(38))


# ----------------------------------------------------------------- scanning
def test_scan_split_counts_images(cfg, tmp_path):
    labels = load_production_labels(cfg)
    root = make_dataset(tmp_path, labels, per_class=3)
    c = dict(cfg)
    c["dataset"] = dict(cfg["dataset"])
    c["dataset"]["root"] = str(root)

    info = scan_split(c, "train")
    assert info is not None
    assert len(info.class_dirs) == 38
    assert info.image_count == 38 * 3


def test_scan_split_missing_returns_none(cfg, tmp_path):
    c = dict(cfg)
    c["dataset"] = dict(cfg["dataset"])
    c["dataset"]["root"] = str(tmp_path / "nope")
    assert scan_split(c, "train") is None


def test_scan_split_accepts_validation_alias(cfg, tmp_path):
    labels = load_production_labels(cfg)
    root = make_dataset(tmp_path, labels, per_class=1)
    (root / "valid").rename(root / "validation")
    c = dict(cfg)
    c["dataset"] = dict(cfg["dataset"])
    c["dataset"]["root"] = str(root)
    assert scan_split(c, "valid") is not None


def test_scan_split_ignores_non_image_files(cfg, tmp_path):
    labels = load_production_labels(cfg)
    root = make_dataset(tmp_path, labels, per_class=1)
    (root / "train" / labels[0] / "notes.txt").write_text("ignore me")
    c = dict(cfg)
    c["dataset"] = dict(cfg["dataset"])
    c["dataset"]["root"] = str(root)
    info = scan_split(c, "train")
    assert info.image_count == 38  # the .txt must not be counted


def test_set_seed_is_repeatable():
    import random

    set_seed(42)
    a = [random.random() for _ in range(3)]
    set_seed(42)
    b = [random.random() for _ in range(3)]
    assert a == b


# --------------------------------------------------- script-level behaviour
def _write_config(tmp_path: Path, overrides: dict) -> Path:
    """Copy config.yaml into tmp_path with overrides applied."""
    import yaml as y

    raw = load_config(CONFIG)
    raw.pop("_config_path", None)
    raw.pop("_base_dir", None)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        raw.setdefault(section, {})[key] = value
    out = tmp_path / "cfg.yaml"
    out.write_text(y.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out


def test_validate_dataset_fails_without_dataset(tmp_path):
    """Missing dataset must exit 2, verified against an isolated config.

    Cannot use the real config: a dataset may genuinely exist where the tests
    run (Colab), so point dataset.root at a guaranteed-absent path instead.
    """
    import validate_dataset

    missing = tmp_path / "definitely_absent_dataset_root"
    assert not missing.exists()
    cfg_path = _write_config(tmp_path, {"dataset.root": str(missing)})

    assert validate_dataset.main(["--config", str(cfg_path)]) == 2


def test_validate_dataset_fails_when_train_split_absent(tmp_path):
    """A root with only the labels present must still exit 2."""
    import validate_dataset

    empty = tmp_path / "empty_root"
    empty.mkdir()
    cfg_path = _write_config(tmp_path, {"dataset.root": str(empty)})

    assert validate_dataset.main(["--config", str(cfg_path)]) == 2


def _order_mismatched_folders(labels: list[str]) -> list[str]:
    """38 folder names that map correctly but sort differently.

    scan_split() returns sorted() folders, so simply reversing the input list
    changes nothing. Using a space variant of one label makes it sort before
    the "___" entries (0x20 < 0x5F) while normalising to the same class.
    """
    folders = list(labels)
    for i, name in enumerate(folders):
        if name == "Tomato___healthy":
            folders[i] = "Tomato healthy"
            break
    return folders


def test_order_mismatch_fixture_really_mismatches():
    """Guard the premise: these folders must not sort into production order."""
    labels = load_production_labels(load_config(CONFIG))
    folders = _order_mismatched_folders(labels)
    assert len(folders) == 38
    assert sorted(folders) != sorted(labels)
    # Still all mappable, so only the folder *name* differs; every class still
    # resolves to its production index.
    mapping, missing, unexpected = build_label_mapping(folders, labels)
    assert not missing and not unexpected
    ordered = class_names_to_index_list(mapping)
    assert len(ordered) == 38
    for i, folder in enumerate(ordered):
        assert mapping[folder] == i
        assert normalize_class_name(folder) == normalize_class_name(labels[i])


def test_validate_dataset_returns_5_when_strict_order_required(tmp_path):
    """With require_exact_label_order=true and mismatched order, exit 5."""
    import validate_dataset

    labels = load_production_labels(load_config(CONFIG))
    root = make_dataset(tmp_path, _order_mismatched_folders(labels), per_class=1)
    cfg_path = _write_config(tmp_path, {
        "dataset.root": str(root),
        "dataset.require_exact_label_order": True,
    })

    assert validate_dataset.main(["--config", str(cfg_path)]) == 5


def test_validate_dataset_passes_mismatched_when_not_strict(tmp_path):
    """The real dataset case: differing order + strict=false must pass."""
    import validate_dataset

    labels = load_production_labels(load_config(CONFIG))
    root = make_dataset(tmp_path, _order_mismatched_folders(labels), per_class=1)
    cfg_path = _write_config(tmp_path, {
        "dataset.root": str(root),
        "dataset.require_exact_label_order": False,
    })

    assert validate_dataset.main(["--config", str(cfg_path)]) == 0


def test_shipped_config_disables_strict_label_order():
    """The real dataset sorts differently, so the shipped config must be lax."""
    cfg = load_config(CONFIG)
    assert cfg["dataset"]["require_exact_label_order"] is False, \
        "config must accept the explicit mapping for this dataset"


def test_validate_dataset_passes_with_complete_dataset(cfg, tmp_path, capsys):
    import validate_dataset

    labels = load_production_labels(cfg)
    root = make_dataset(tmp_path, labels, per_class=2)
    cfg_path = _write_config(tmp_path, {"dataset.root": str(root)})

    rc = validate_dataset.main(["--config", str(cfg_path)])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "VALIDATION PASSED" in out
    assert "Number of classes: 38" in out


def test_validate_dataset_rejects_wrong_class_count(cfg, tmp_path):
    import validate_dataset

    labels = load_production_labels(cfg)
    root = make_dataset(tmp_path, labels[:-2], per_class=1)  # only 36 classes
    cfg_path = _write_config(tmp_path, {"dataset.root": str(root)})

    assert validate_dataset.main(["--config", str(cfg_path)]) == 3


def test_validate_dataset_rejects_extra_class_directory(cfg, tmp_path):
    """An extra folder makes the count 39, caught before anything else."""
    import validate_dataset

    labels = load_production_labels(cfg) + ["Tomato___Not_A_Real_Disease"]
    root = make_dataset(tmp_path, labels, per_class=1)
    cfg_path = _write_config(tmp_path, {"dataset.root": str(root)})

    assert validate_dataset.main(["--config", str(cfg_path)]) == 3


def test_validate_dataset_rejects_unexpected_class(cfg, tmp_path):
    """38 folders where one is unlabelled -> unexpected-class failure."""
    import validate_dataset

    labels = list(load_production_labels(cfg))
    # Replace the last real class with an unknown one, keeping 38 folders.
    folders = labels[:-1] + ["Tomato___Not_A_Real_Disease"]
    root = make_dataset(tmp_path, folders, per_class=1)
    cfg_path = _write_config(tmp_path, {"dataset.root": str(root)})

    assert validate_dataset.main(["--config", str(cfg_path)]) == 4


# ------------------------------------------------------ flight isolation
def test_training_scripts_have_no_flight_control_imports():
    banned = ("pymavlink", "mavutil", "ardupilot", "mavproxy", "rclpy", "rospy")
    offenders = []
    for path in ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, offenders


def test_training_requirements_exclude_flight_packages():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for banned in ("pymavlink", "ardupilot", "mavproxy", "torch", "yolov5"):
        assert f"\n{banned}" not in text


def test_no_model_file_is_present():
    """The pipeline must not ship fabricated weights."""
    out = ROOT / "output"
    for name in ("model2.h5", "disease_model.onnx", "model.keras"):
        assert not (out / name).is_file(), f"{name} must not exist before training"
    prod = ROOT.parent / "ai_disease_detection" / "models" / "disease_model.onnx"
    assert not prod.is_file(), "production model must not exist before training"