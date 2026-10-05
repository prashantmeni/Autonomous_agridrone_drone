"""Compare the Keras model against the exported ONNX model.

Runs the same images through both and compares top-1 class, top-1 probability,
and the top-5 distribution. Exits non-zero on any disagreement beyond the
configured tolerance, so a bad conversion can never be declared valid.

Usage:
    python validate_onnx.py
    python validate_onnx.py --num-images 25 --max-top1-diff 0.005
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
    scan_split,
    set_seed,
)

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    print("NumPy is required.", file=sys.stderr)
    raise SystemExit(2)

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow is needed to produce Keras predictions.\n"
          "       Run this on the Colab/GPU machine that holds model2.h5,\n"
          "       not on the Raspberry Pi.", file=sys.stderr)
    raise SystemExit(2)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required to run the ONNX model.", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Validate ONNX against Keras.")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None, help="trained Keras model")
    p.add_argument("--onnx", default=None, help="exported ONNX model")
    p.add_argument("--split", default="test", choices=["test", "valid"])
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--max-top1-diff", type=float, default=None,
                   help="tolerance for top-1 probability difference")
    return p.parse_args(argv)


def load_images(cfg: dict, split: str, limit: int, labels: list[str]):
    """Load images for the Keras-vs-ONNX comparison.

    Returns both representations of the same pixels: `raw255` in NHWC
    [0, 255] for the Keras model, and `unit01` in NCHW [0, 1] for the ONNX
    graph, which is exactly what the production module emits.
    """
    import cv2

    ds = cfg["dataset"]
    info = scan_split(cfg, split)
    if info is None:
        print(f"FATAL: '{split}' split not found.", file=sys.stderr)
        raise SystemExit(3)

    mapping, missing, unexpected = build_label_mapping(info.class_dirs, labels)
    if missing or unexpected:
        print("FATAL: class mapping is not safe.", file=sys.stderr)
        raise SystemExit(3)
    ordered = [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    index_of = {name: i for i, name in enumerate(ordered)}

    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]
    exts = {e.lower() for e in ds["image_extensions"]}

    samples: list[np.ndarray] = []
    truths: list[int] = []
    # Walk classes round-robin so a limited sample still spans many classes.
    per_class: list[list[Path]] = []
    for name in ordered:
        folder = info.root / name
        files = sorted(f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in exts)
        per_class.append(files)

    cursor = [0] * len(ordered)
    while len(samples) < limit:
        added = False
        for ci, files in enumerate(per_class):
            if cursor[ci] < len(files) and len(samples) < limit:
                samples.append(files[cursor[ci]])
                truths.append(index_of[ordered[ci]])
                cursor[ci] += 1
                added = True
        if not added:
            break

    if not samples:
        print("FATAL: no images found to compare.", file=sys.stderr)
        raise SystemExit(3)

    raw255 = np.zeros((len(samples), h, w, 3), dtype=np.float32)
    unit01 = np.zeros((len(samples), 3, h, w), dtype=np.float32)
    kept_truth: list[int] = []
    for i, path in enumerate(samples):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # Keras backbone expects raw [0, 255]: it contains its own 1/255 stem.
        raw255[i] = rgb
        # The exported graph input is [0, 1] NCHW; the graph restores [0, 255]
        # via its own Rescaling(255) prefix. This is what production sends.
        unit01[i] = np.transpose(rgb / 255.0, (2, 0, 1))
        kept_truth.append(truths[i])

    return raw255, unit01, np.array(kept_truth), samples


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    n_img = args.num_images or cfg["validation"].get("num_images", 10)
    top_k = cfg["validation"].get("top_k", 5)
    max_diff = args.max_top1_diff if args.max_top1_diff is not None else \
        cfg["validation"].get("max_top1_diff", 0.01)

    out = output_dir(cfg)
    keras_path = Path(args.model) if args.model else out / cfg["output"]["model_h5"]
    onnx_path = Path(args.onnx) if args.onnx else out / cfg["output"]["onnx"]

    for p in (keras_path, onnx_path):
        if not p.is_file():
            print(f"FATAL: {p} not found.", file=sys.stderr)
            raise SystemExit(2)

    labels = load_production_labels(cfg)
    n = len(labels)

    # ------------------------------------------------------------- load both
    keras_model = None
    for compile_flag in (False, True):
        try:
            keras_model = tf.keras.models.load_model(str(keras_path),
                                                     compile=compile_flag)
            break
        except Exception:
            continue
    if keras_model is None:
        print(f"FATAL: could not load Keras model {keras_path}", file=sys.stderr)
        raise SystemExit(2)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_in = sess.get_inputs()[0]
    onnx_out = sess.get_outputs()[0]
    print(f"Keras : {keras_path.name}")
    print(f"ONNX  : {onnx_path.name}")
    print(f"ONNX input  {onnx_in.name} {onnx_in.shape}")
    print(f"ONNX output {onnx_out.name} {onnx_out.shape}")
    print(f"\nComparing {n_img} image(s), top-{top_k}, tolerance {max_diff}\n")

    raw255, unit01, truths, paths = load_images(cfg, args.split, n_img, labels)

    # Identical pixels, two shapes: NHWC [0,255] for Keras (its own stem does
    # the 1/255), NCHW [0,1] for ONNX (the export prefixes Rescaling(255) so it
    # restores [0,255]). Both models therefore see the same effective range.
    keras_probs = np.asarray(keras_model.predict(raw255, verbose=0))
    onnx_probs = np.asarray(
        sess.run([onnx_out.name], {onnx_in.name: unit01})[0])

    if keras_probs.shape[1] != n or onnx_probs.shape[1] != n:
        print(f"FATAL: expected {n} outputs, got keras={keras_probs.shape} "
              f"onnx={onnx_probs.shape}")
        return 4

    print("=" * 96)
    print(f"{'image':<34}{'top1 keras':<22}{'top1 onnx':<22}{'p_k':>8}{'p_o':>8}{'diff':>9}")
    print("=" * 96)

    same_top1 = 0
    worst_diff = 0.0
    worst_vec = 0.0
    failures: list[str] = []

    for i, path in enumerate(paths):
        k_idx = int(np.argmax(keras_probs[i]))
        o_idx = int(np.argmax(onnx_probs[i]))
        k_p = float(keras_probs[i][k_idx])
        o_p = float(onnx_probs[i][o_idx])
        diff = abs(k_p - o_p)
        vec = float(np.max(np.abs(keras_probs[i] - onnx_probs[i])))
        worst_diff = max(worst_diff, diff)
        worst_vec = max(worst_vec, vec)
        if k_idx == o_idx:
            same_top1 += 1

        name = path.name[:32]
        flag = ""
        if k_idx != o_idx:
            flag = "  <-- CLASS MISMATCH"
            failures.append(f"{path.name}: keras={labels[k_idx]} "
                            f"onnx={labels[o_idx]}")
        elif diff > max_diff:
            flag = "  <-- TOLERANCE"
            failures.append(f"{path.name}: top-1 diff {diff:.6f} > {max_diff}")
        print(f"{name:<34}{labels[k_idx][:20]:<22}{labels[o_idx][:20]:<22}"
              f"{k_p:>8.4f}{o_p:>8.4f}{diff:>9.2e}{flag}")

    n_cmp = len(paths)
    print("\n" + "=" * 96)
    print(f"Top-1 agreement : {same_top1}/{n_cmp} "
          f"({100.0 * same_top1 / n_cmp:.1f}%)")
    print(f"Worst top-1 prob diff : {worst_diff:.3e}")
    print(f"Worst element-wise diff: {worst_vec:.3e}")

    # Top-5 set overlap, reported but not fatal.
    overlaps = []
    for i in range(n_cmp):
        k5 = set(np.argsort(-keras_probs[i])[:top_k].tolist())
        o5 = set(np.argsort(-onnx_probs[i])[:top_k].tolist())
        overlaps.append(len(k5 & o5) / top_k)
    print(f"Mean top-{top_k} overlap    : {np.mean(overlaps):.3f}")

    if failures:
        print("\nVALIDATION FAILED")
        for f in failures:
            print(f"  {f}")
        print("\nDo NOT deploy this ONNX model. Investigate the conversion or")
        print("preprocessing before continuing.")
        return 5

    print("\nVALIDATION PASSED")
    print("Keras and ONNX agree within tolerance.")
    print(f"\nDeploy: copy {onnx_path.name} to")
    print("  ~/agridrone/ai_disease_detection/models/disease_model.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())