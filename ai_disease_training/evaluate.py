"""Evaluate the trained model and record real metrics.

Computes accuracy, precision, recall, F1, and a confusion matrix on the held-out
test split. Writes both machine-readable and human-readable output.

These numbers describe ONLY the model trained by this pipeline. They are not,
and must not be presented as, metrics for the reference repository's author.

Usage:
    python evaluate.py
    python evaluate.py --split valid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_label_mapping,
    load_config,
    load_production_labels,
    output_dir,
    save_json,
    scan_split,
    set_seed,
    strip_private_keys,
)

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:  # pragma: no cover
    print("TensorFlow and NumPy are required for evaluation.\n"
          "  pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the trained classifier.")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None, help="path to the trained Keras model")
    p.add_argument("--split", default="test", choices=["test", "valid"])
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args(argv)


def load_model(path: Path):
    """Load the trained model, tolerating .h5 and .keras formats."""
    if not path.is_file():
        print(f"FATAL: trained model not found: {path}\n"
              "       Run train.py first.", file=sys.stderr)
        raise SystemExit(2)
    try:
        return tf.keras.models.load_model(str(path), compile=False)
    except Exception as e:
        # Newer Keras refuses some legacy .h5 files with compile=True metadata.
        print(f"  standard load failed ({e}); retrying without compile metadata")
        try:
            return tf.keras.models.load_model(str(path))
        except Exception as e2:
            print(f"FATAL: could not load {path}: {e2}", file=sys.stderr)
            raise SystemExit(2)


def build_eval_dataset(cfg: dict, split: str):
    ds = cfg["dataset"]
    info = scan_split(cfg, split)
    if info is None:
        print(f"FATAL: '{split}' split not found. Test metrics need a test set.",
              file=sys.stderr)
        print("  Refusing to silently substitute the validation split: a model",
              file=sys.stderr)
        print("  selected using validation data has already seen it, so any",
              file=sys.stderr)
        print("  accuracy measured there would be optimistic and not a real",
              file=sys.stderr)
        print("  test result. Supply a test/ directory, or pass --split valid",
              file=sys.stderr)
        print("  deliberately if you accept a validation-set figure.",
              file=sys.stderr)
        raise SystemExit(3)

    labels = load_production_labels(cfg)
    mapping, missing, unexpected = build_label_mapping(info.class_dirs, labels)
    if missing or unexpected:
        print("FATAL: class mapping is not safe for evaluation.", file=sys.stderr)
        raise SystemExit(3)
    ordered = [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(info.root),
        image_size=(h, w),
        batch_size=cfg["evaluation"].get("batch_size", 32),
        label_mode="categorical",
        class_names=ordered,
        shuffle=False,
    )
    # No external Rescaling(1/255) here: Keras EfficientNet graphs contain their
    # own Rescaling(1/255) stem plus ImageNet Normalization and expect raw
    # [0, 255] pixels. Applying a second divide would desynchronise evaluation
    # from training. Must mirror train.py exactly.
    #
    # No .cache(): it caches the entire decoded split in RAM, which OOMs on a
    # large test set. Stream from disk with a bounded prefetch instead.
    prefetch = int(cfg["evaluation"].get("prefetch_batches", 2))
    dataset = dataset.prefetch(prefetch)
    return dataset, labels, info


def confusion_matrix(y_true, y_pred, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf(cm: np.ndarray):
    """Per-class precision, recall, F1 and support from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp),
                   where=denom > 0)
    return precision, recall, f1, actual.astype(int)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.batch_size:
        cfg["evaluation"]["batch_size"] = args.batch_size
    set_seed(cfg["training"]["seed"])

    out = output_dir(cfg)
    model_path = Path(args.model) if args.model else out / cfg["output"]["model_h5"]
    model = load_model(model_path)
    print(f"Loaded model: {model_path}")
    model.summary()

    dataset, labels, split_info = build_eval_dataset(cfg, args.split)
    n = len(labels)

    print(f"\nEvaluating on the '{args.split}' split: "
          f"{split_info.image_count} images, {n} classes")

    # Single streaming pass: predictions and labels are collected per batch, so
    # they stay aligned and nothing large is held beyond one batch of scores.
    # (num images x 38 float32 is ~2.7 MB at this split size; acceptable.)
    y_pred_list = []
    y_true_list = []
    for step, (images, labels_batch) in enumerate(dataset):
        probs = model.predict(images, verbose=0)
        y_pred_list.append(np.argmax(probs, axis=1))
        y_true_list.append(np.argmax(labels_batch.numpy(), axis=1))
        if step % 50 == 0:
            print(f"  batch {step} ...", flush=True)

    y_pred = (np.concatenate(y_pred_list) if y_pred_list
              else np.array([], dtype=int))
    y_true = (np.concatenate(y_true_list) if y_true_list
              else np.array([], dtype=int))

    if y_true.shape[0] != y_pred.shape[0]:
        print(f"FATAL: prediction/label count mismatch "
              f"({y_pred.shape[0]} vs {y_true.shape[0]})")
        return 4

    cm = confusion_matrix(y_true, y_pred, n)
    precision, recall, f1, support = prf(cm)

    accuracy = float(np.mean(y_true == y_pred))
    macro_p = float(precision.mean())
    macro_r = float(recall.mean())
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0

    # Top-k accuracy, cheap and informative for a 38-class problem.
    k = min(cfg["evaluation"].get("top_k", 5), n)
    topk_hits = sum(1 for i in range(len(y_true))
                    if y_true[i] in np.argsort(-probs[i])[:k])
    top_k_accuracy = topk_hits / len(y_true) if len(y_true) else 0.0

    lines = []
    lines.append("=" * 74)
    lines.append("CROP DISEASE CLASSIFIER - EVALUATION")
    lines.append("=" * 74)
    lines.append(f"Model      : {model_path.name}")
    lines.append(f"Split      : {args.split} ({split_info.image_count} images)")
    lines.append(f"Classes    : {n}")
    lines.append("")
    lines.append("These metrics describe ONLY the model trained by this repository's")
    lines.append("pipeline. They are not metrics for any other party.")
    lines.append("")
    if args.split != "test":
        lines.append("*** WARNING: MEASURED ON THE VALIDATION SPLIT ***")
        lines.append("Validation data was used for early stopping and checkpoint")
        lines.append("selection, so these figures are optimistic and are NOT a test")
        lines.append("result. Do not report them as held-out accuracy.")
        lines.append("")
    lines.append(f"Top-1 accuracy : {accuracy:.4f}")
    lines.append(f"Top-{k} accuracy  : {top_k_accuracy:.4f}")
    lines.append(f"Macro precision: {macro_p:.4f}")
    lines.append(f"Macro recall   : {macro_r:.4f}")
    lines.append(f"Macro F1       : {macro_f1:.4f}")
    lines.append(f"Weighted F1    : {weighted_f1:.4f}")
    lines.append("")
    lines.append("-" * 74)
    lines.append(f"{'class':<52}{'P':>7}{'R':>7}{'F1':>7}{'n':>6}")
    lines.append("-" * 74)
    for i, label in enumerate(labels):
        lines.append(f"{label:<52}{precision[i]:>7.3f}{recall[i]:>7.3f}"
                     f"{f1[i]:>7.3f}{support[i]:>6d}")
    lines.append("-" * 74)
    lines.append("")
    lines.append("Confusion matrix (rows = truth, cols = predicted):")
    for i in range(n):
        row = " ".join(f"{cm[i][j]:>5}" for j in range(n))
        lines.append(f"  [{i:>2}] {labels[i]:<48}{row}")

    text = "\n".join(lines)
    eval_path = out / cfg["output"]["evaluation"]
    eval_path.write_text(text + "\n", encoding="utf-8")

    metrics = {
        "model": model_path.name,
        "split": args.split,
        "is_true_test_set": args.split == "test",
        "num_images": int(len(y_true)),
        "num_classes": n,
        "top1_accuracy": accuracy,
        f"top{k}_accuracy": top_k_accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(n)
        },
        "confusion_matrix": cm.tolist(),
        "class_index_order": labels,
    }
    save_json(metrics, out / cfg["output"]["metrics"])

    print("\n" + text)
    print(f"\nSaved: {eval_path}")
    print(f"Saved: {out / cfg['output']['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())