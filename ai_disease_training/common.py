"""Shared helpers for the training pipeline.

Loaded by every script so config parsing, dataset discovery, and label mapping
behave identically during validation, training, export, and comparison.

Deliberately does NOT import TensorFlow: validate_dataset.py must run on any
machine, including one that only has onnxruntime.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = TRAINING_DIR / "config.yaml"


# --------------------------------------------------------------------- config
def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config is not a mapping: {path}")
    cfg["_config_path"] = str(path)
    cfg["_base_dir"] = str(path.parent)
    return cfg


def resolve(cfg: dict, relative: str | Path) -> Path:
    """Resolve a config-relative path.

    Tries, in order: as-is (relative to cwd), relative to the config file, and
    relative to the package root. The package-root fallback keeps paths working
    when a config copy is placed elsewhere - e.g. in a Colab working directory.
    """
    p = Path(relative)
    if p.is_absolute():
        return p

    base = Path(cfg["_base_dir"])
    candidates = [Path.cwd() / p, base / p, TRAINING_DIR / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Nothing exists yet (e.g. an output file about to be created): use the
    # config-relative location, which is the documented behaviour.
    return base / p


# --------------------------------------------------------------------- labels
def load_production_labels(cfg: dict) -> list[str]:
    """The 38 labels, read from the production module.

    This is the single source of truth for class ordering. The training run
    must emit output indices in exactly this order.
    """
    path = resolve(cfg, cfg["dataset"]["labels_file"])
    if not path.is_file():
        raise FileNotFoundError(
            f"production labels file not found: {path}\n"
            "It must contain exactly the 38 class names, one per line."
        )
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            labels.append(s)
    if len(labels) != cfg["model"]["num_classes"]:
        raise ValueError(
            f"{path} has {len(labels)} labels but the model expects "
            f"{cfg['model']['num_classes']}"
        )
    if len(set(labels)) != len(labels):
        raise ValueError(f"{path} contains duplicate labels")
    return labels


def normalize_class_name(name: str) -> str:
    """Canonical form used when comparing dataset folders to production labels.

    PlantVillage folders sometimes use spaces where the label list uses
    underscores, so "Pepper bell Bacterial spot" and "Pepper,_bell___Bacterial_spot"
    must be recognised as the same class.
    """
    text = str(name).strip()
    text = text.replace("___", "__")       # unify the triple separator
    text = text.replace("_", " ")
    text = text.replace(",", " ")          # Pepper,_bell -> Pepper bell
    text = " ".join(text.split())
    return text.casefold()


def build_label_mapping(
    dataset_classes: list[str],
    production_labels: list[str],
) -> tuple[dict[str, int], list[str], list[str]]:
    """Map dataset folder names onto production label indices.

    Returns (mapping, missing, unexpected) where mapping is
    ``{dataset_folder_name: production_index}``.
    """
    production_by_norm = {normalize_class_name(l): i
                          for i, l in enumerate(production_labels)}
    seen: set[str] = set()

    mapping: dict[str, int] = {}
    unexpected: list[str] = []
    for folder in dataset_classes:
        key = normalize_class_name(folder)
        if key in production_by_norm:
            mapping[folder] = production_by_norm[key]
            seen.add(key)
        else:
            unexpected.append(folder)

    missing = [l for l in production_labels
               if normalize_class_name(l) not in seen]
    return mapping, missing, unexpected


def class_names_to_index_list(mapping: dict[str, int]) -> list[str]:
    """Dataset class names ordered so index i == production index i."""
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


# -------------------------------------------------------------------- dataset
@dataclass
class SplitInfo:
    name: str
    root: Path
    class_dirs: list[str]
    image_count: int
    counts_by_class: dict[str, int]


def find_split_dir(cfg: dict, kind: str) -> Path | None:
    """Locate a dataset split, tolerating valid/validation naming differences."""
    ds = cfg["dataset"]
    root = resolve(cfg, ds["root"])
    if kind == "train":
        return root / ds["train_dir"] if (root / ds["train_dir"]).is_dir() else None
    if kind == "valid":
        for candidate in ds.get("valid_dir_alternatives") or [ds["valid_dir"]]:
            path = root / candidate
            if path.is_dir():
                return path
        return None
    if kind == "test":
        path = root / ds["test_dir"]
        return path if path.is_dir() else None
    raise ValueError(f"unknown split kind: {kind}")


def scan_split(cfg: dict, kind: str) -> SplitInfo | None:
    """Enumerate class folders and count images in one split."""
    root = find_split_dir(cfg, kind)
    if root is None:
        return None
    extensions = {e.lower() for e in cfg["dataset"]["image_extensions"]}

    class_dirs = sorted(d.name for d in root.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
    counts: dict[str, int] = {}
    total = 0
    for name in class_dirs:
        folder = root / name
        n = sum(1 for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in extensions)
        counts[name] = n
        total += n
    return SplitInfo(kind, root, class_dirs, total, counts)


def iter_images(folder: Path, extensions) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    )


# ---------------------------------------------------------------------- misc
def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def output_dir(cfg: dict) -> Path:
    # Always anchored to the config/package location, never to an existing
    # cwd-relative match, so outputs land in one predictable place.
    raw = Path(cfg["output"]["dir"])
    path = raw if raw.is_absolute() else TRAINING_DIR / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_path(cfg: dict, key: str) -> Path:
    return output_dir(cfg) / cfg["output"][key]


def strip_private_keys(cfg: dict) -> dict:
    """Config minus the underscore-prefixed runtime keys, for saving."""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}