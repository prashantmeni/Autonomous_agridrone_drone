"""Export the trained Keras model to ONNX for Raspberry Pi inference.

Input contract, matching the production engine exactly:

    input : float32 [-1, 3, 224, 224]  values in [0, 1]
    output: float32 [-1, 38]           softmax probabilities

Preprocessing
-------------
The production module divides pixels by 255 and sends [0, 1]. Keras
EfficientNet graphs contain their own `Rescaling(1/255)` stem plus ImageNet
`Normalization` and expect raw [0, 255]. wrap_nchw() therefore prefixes the
graph with `Rescaling(255)` so the ONNX input stays [0, 1] while the backbone
still receives the range its ImageNet weights were trained on.

Training (train.py) and evaluation (evaluate.py) feed raw [0, 255] for exactly
the same reason, and add no external rescaling of their own.

NCHW vs NHWC
-------------
Keras models are NHWC internally. Two options exist:

  export_nchw: true   Wrap the Keras graph so ONNX receives NCHW directly.
                      No transpose in the runtime; the graph carries one.
  export_nchw: false  Export the raw NHWC graph and let ONNXInferenceEngine
                      transpose (it detects NHWC via the input shape).

Both are implemented. The exported shape is printed and verified, because a
silent layout mismatch would produce garbage predictions.

Usage:
    python export_onnx.py
    python export_onnx.py --opset 13
    python export_onnx.py --export-nchw false
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, output_dir  # noqa: E402

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:  # pragma: no cover
    print("TensorFlow and NumPy are required for export.\n"
          "  pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Export the trained model to ONNX.")
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=None, help="path to the trained Keras model")
    p.add_argument("--opset", type=int, default=None)
    p.add_argument("--export-nchw", dest="nchw", action="store_true", default=None,
                   help="force NCHW input (default from config)")
    p.add_argument("--export-nhwc", dest="nchw", action="store_false",
                   help="force NHWC input")
    p.add_argument("--output", default=None, help="path for the .onnx file")
    return p.parse_args(argv)


def load_model(path: Path):
    if not path.is_file():
        print(f"FATAL: trained model not found: {path}\n"
              "       Run train.py first.", file=sys.stderr)
        raise SystemExit(2)
    for compile_flag in (False, True):
        try:
            return tf.keras.models.load_model(str(path), compile=compile_flag)
        except Exception:
            continue
    print(f"FATAL: could not load {path}", file=sys.stderr)
    raise SystemExit(2)


def wrap_nchw(model, height: int, width: int):
    """Expose the production NCHW [0,1] contract in front of the Keras graph.

    Chain:
        input (None, 3, H, W) float32, values in [0, 1]   <- production sends this
        -> Rescaling(255)                                 [0, 1] -> [0, 255]
        -> transpose                                      NCHW   -> NHWC
        -> Keras EfficientNet (own stem: Rescaling(1/255) + Normalization)
        -> softmax

    Why Rescaling(255) is required: Keras EfficientNet graphs contain their own
    `Rescaling(1/255)` stem and ImageNet `Normalization`, and expect raw
    [0, 255] pixels. ai_disease_detection divides by 255 and sends [0, 1], so
    the graph must undo that division or every pixel arrives ~255x too small.

    This keeps preprocessing entirely outside the production module: its
    `divide_255` and [0, 1] output stay exactly as they are.
    """
    inp = tf.keras.Input(shape=(3, height, width), name="input_nchw", dtype=tf.float32)
    # [0, 1] -> [0, 255] so the backbone sees the range it was trained on.
    x = tf.keras.layers.Rescaling(255.0)(inp)
    # NCHW -> NHWC. A Permute layer, not tf.transpose: tf.transpose cannot
    # take a symbolic KerasTensor (Keras 3 raises "A KerasTensor cannot be
    # used as input to a TensorFlow function"), and Permute works on both
    # Keras 2 and 3 and exports to ONNX as a plain Transpose.
    x = tf.keras.layers.Permute((2, 3, 1))(x)
    out = model(x, training=False)
    wrapped = tf.keras.Model(inp, out, name="crop_disease_nchw")
    return wrapped


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    opset = args.opset or cfg["export"].get("opset", 13)
    export_nchw = args.nchw if args.nchw is not None else cfg["export"].get(
        "export_nchw", True)

    h, w = cfg["model"]["input_height"], cfg["model"]["input_width"]
    n = cfg["model"]["num_classes"]

    out = output_dir(cfg)
    model_path = Path(args.model) if args.model else out / cfg["output"]["model_h5"]
    keras_model = load_model(model_path)
    print(f"Loaded Keras model: {model_path}")

    try:
        import tf2onnx
    except ImportError:
        print("FATAL: tf2onnx is required.\n"
              "  pip install tf2onnx>=1.15.0", file=sys.stderr)
        raise SystemExit(2)

    target = keras_model
    if export_nchw:
        target = wrap_nchw(keras_model, h, w)
        target.summary()
        print("\nWrapped for NCHW input.")
    else:
        print("\nExporting NHWC input; the runtime engine will transpose.")

    onnx_path = Path(args.output) if args.output else out / cfg["output"]["onnx"]
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Channels-first throughout, which is what the production engine supplies.
    input_signature = [tf.TensorSpec(
        shape=(None, 3, h, w) if export_nchw else (None, h, w, 3),
        dtype=tf.float32, name="input")]
    output_signature = [tf.TensorSpec(
        shape=(None, n), dtype=tf.float32, name="output")]

    print(f"\nConverting to ONNX (opset {opset})...")
    try:
        model_proto, _ = tf2onnx.convert.from_keras(
            target,
            input_signature=input_signature,
            output_signature=output_signature,
            opset=opset,
        )
    except Exception as e:
        print(f"FATAL: conversion failed: {e}", file=sys.stderr)
        return 4

    tf2onnx.utils.save_model(model_proto, str(onnx_path))
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved ONNX model: {onnx_path}  ({size_mb:.1f} MB)")

    # ------------------------------------------------------- shape check
    print("\nVerifying the exported graph signature...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed here; shape check skipped.")
        print("Run validate_onnx.py on a machine with onnxruntime.")
        return 0

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"  input : {inp.name} {inp.shape} {inp.type}")
    print(f"  output: {out.name} {out.shape} {out.type}")

    problems = []
    if len(inp.shape) != 4:
        problems.append("input is not 4D")
    if export_nchw:
        if not (isinstance(inp.shape[1], int) and inp.shape[1] == 3):
            problems.append(
                f"expected NCHW with channels at axis 1, got {inp.shape}")
        expected_w = inp.shape[3] if len(inp.shape) == 4 else None
    else:
        if not (isinstance(inp.shape[3], int) and inp.shape[3] == 3):
            problems.append(
                f"expected NHWC with channels at axis 3, got {inp.shape}")
    if isinstance(out.shape[-1], int) and out.shape[-1] != n:
        problems.append(f"output width {out.shape[-1]} != {n}")

    if problems:
        print("\nFATAL: exported signature does not match the contract:")
        for p in problems:
            print(f"  - {p}")
        return 5

    # A dummy run proves the graph executes and returns 38 sane probabilities.
    probe = np.zeros((1, 3, h, w) if export_nchw else (1, h, w, 3),
                     dtype=np.float32)
    result = sess.run([out.name], {inp.name: probe})[0]
    print(f"  smoke test output shape: {result.shape}, "
          f"sum={float(result.sum()):.4f}")
    if result.shape != (1, n):
        print(f"FATAL: smoke test returned {result.shape}, expected (1, {n})")
        return 5

    print("\nEXPORT OK")
    print(f"Next: python validate_onnx.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())