"""Verify the dataset matches the production labels before any training.

Fails loudly on a class-count mismatch or an unmappable class, because a
silent ordering difference would make every prediction wrong while looking
perfectly healthy.

Run:
    python validate_dataset.py
    python validate_dataset.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_label_mapping,
    class_names_to_index_list,
    load_config,
    load_production_labels,
    output_path,
    save_json,
    scan_split,
    strip_private_keys,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate the PlantVillage dataset.")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--save-mapping", action="store_true",
                   help="write output/label_mapping.json for later steps")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    expected_n = cfg["model"]["num_classes"]
    print("=" * 68)
    print("DATASET VALIDATION")
    print("=" * 68)

    # ---------------------------------------------------- production labels
    try:
        labels = load_production_labels(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"FATAL: {e}")
        return 2
    print(f"\nProduction labels ({len(labels)}): {cfg['dataset']['labels_file']}")

    # --------------------------------------------------------- dataset splits
    print(f"\nDataset root: {cfg['dataset']['root']}")
    splits = {}
    for kind in ("train", "valid", "test"):
        info = scan_split(cfg, kind)
        splits[kind] = info
        if info is None:
            note = "required" if kind in ("train", "valid") else "optional"
            print(f"  {kind:6}: NOT FOUND ({note})")
            if kind in ("train", "valid"):
                print(f"FATAL: the '{kind}' split is required but was not found.")
                return 2
        else:
            print(f"  {kind:6}: {info.image_count:>7} images  "
                  f"in {len(info.class_dirs)} class folders")

    train = splits["train"]
    valid = splits["valid"]

    if train.image_count == 0:
        print("FATAL: the train split contains no images.")
        return 2
    if valid.image_count == 0:
        print("FATAL: the validation split contains no images.")
        return 2

    # ---------------------------------------------------------- class counts
    print(f"\n{'=' * 68}")
    print(f"Number of classes: {len(train.class_dirs)} (expected {expected_n})")
    print(f"{'=' * 68}")

    if len(train.class_dirs) != expected_n:
        print(f"FATAL: train split has {len(train.class_dirs)} classes, "
              f"expected {expected_n}.")
        print("       Every class in the production labels must exist in the dataset.")
        return 3

    if len(set(train.class_dirs)) != len(train.class_dirs):
        print("FATAL: duplicate class folder names in the train split.")
        return 3

    for kind in ("valid", "test"):
        info = splits[kind]
        if info and len(info.class_dirs) != expected_n:
            print(f"FATAL: {kind} split has {len(info.class_dirs)} classes, "
                  f"expected {expected_n}.")
            return 3

    # ------------------------------------------------------ label comparison
    mapping, missing, unexpected = build_label_mapping(train.class_dirs, labels)

    print("\nMissing classes (in labels.txt, absent from dataset):")
    print("  " + (", ".join(missing) if missing else "none"))

    print("\nUnexpected classes (in dataset, not in labels.txt):")
    print("  " + (", ".join(unexpected) if unexpected else "none"))

    if unexpected:
        print("\nFATAL: the dataset contains classes the production module cannot "
              "label.")
        print("       Refusing to continue; index alignment would be unsafe.")
        return 4
    if missing:
        print("\nFATAL: the dataset is missing classes the production module "
              "expects.")
        print("       Refusing to train; the head would have untrained outputs.")
        return 4

    # ------------------------------------------------------- ordering report
    ordered = class_names_to_index_list(mapping)
    exact_order = ordered == list(labels)

    print("\nClass-to-index mapping (dataset folder -> production index):")
    for i, name in enumerate(ordered):
        print(f"  {i:>2}: {name}")

    print(f"\nOrdering identical to labels.txt: {'YES' if exact_order else 'NO'}")
    if not exact_order:
        print("\nThe dataset sorts differently from labels.txt. Training will use")
        print("the explicit mapping below, so index i always means the same")
        print("disease as index i in the production module.")
        for i, (a, b) in enumerate(zip(ordered, labels)):
            if a != b:
                print(f"  index {i}: dataset={a}  production={b}")

    strict = cfg["dataset"].get("require_exact_label_order", False)
    if not exact_order:
        print("\nThis is handled by the explicit mapping above: train.py passes")
        print("class_names in mapping order, so output index i means labels.txt[i]")
        print("no matter how the filesystem sorts. No folder renaming required.")
    if strict and not exact_order:
        print("\nFATAL: require_exact_label_order is true in config.yaml but the")
        print("       orderings differ. Set it to false to proceed with the")
        print("       explicit mapping, or rename the dataset folders.")
        return 5

    # ------------------------------------------------------------- imbalance
    print("\nClass balance (train split):")
    counts = sorted(train.counts_by_class.items(), key=lambda kv: kv[1])
    lo = counts[0]
    hi = counts[-1]
    print(f"  min: {lo[0]} = {lo[1]}")
    print(f"  max: {hi[0]} = {hi[1]}")
    ratio = (hi[1] / lo[1]) if lo[1] else 0.0
    print(f"  ratio max/min: {ratio:.2f}")
    if ratio > 5:
        print("  NOTE: strong imbalance; consider class_weight in train.py")

    if args.save_mapping:
        path = output_path(cfg, "label_mapping")
        save_json({
            "num_classes": expected_n,
            "exact_order_match": exact_order,
            "mapping": mapping,
            "ordered_dataset_classes": ordered,
            "production_labels": labels,
        }, path)
        print(f"\nSaved mapping: {path}")

    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())