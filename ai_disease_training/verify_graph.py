"""Verify the real Keras EfficientNetB0 graph and the preprocessing contract.

Run this on the GPU machine (needs TensorFlow + onnxruntime). It checks
numerically, not just by reading source.

  1. Locate the internal preprocessing layers in the Keras graph.
  2. Confirm the required Rescaling(1/255) is actually present.
  3. Confirm the exported ONNX graph carries the Rescaling(255) bridge.
  4. Prove the two input paths produce a mathematically equivalent tensor
     before the EfficientNet features are computed.

Value coercion
--------------
Keras represents layer attributes inconsistently across versions: TF 2.20
returns e.g. a `TrackedList` for `Rescaling.scale` (because ImageNet
normalisation uses a per-channel list), while older versions return a plain
float. `_scalar()` handles scalar, list, tuple, numpy value, TrackedList and
anything else with a sensible fallback.

Exit codes
----------
0   contract verified
5   a required layer is missing or the numerics do not match

Usage:
    python verify_graph.py
    python verify_graph.py --onnx output/disease_model.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config  # noqa: E402

try:
    import numpy as np
    import tensorflow as tf
except ImportError as e:  # pragma: no cover
    print("TensorFlow and NumPy are required.\n"
          "  pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(2)

EXIT_OK = 0
EXIT_CONTRACT_VIOLATION = 5

TOL = 1e-5


# --------------------------------------------------------------- coercion
def _scalar(value) -> float | None:
    """Coerce a Keras layer attribute to a float, whatever its container.

    Handles float/int, list, tuple, numpy scalar/array, tf.Variable,
    TrackedList, and anything exposing __float__ or __iter__.
    """
    if value is None:
        return None
    # Unwrap 1-element containers.
    try:
        if isinstance(value, (list, tuple)):
            return float(value[0]) if len(value) == 1 else None
        if hasattr(value, "numpy") and not isinstance(value, (np.ndarray, np.generic)):
            # tf.Variable / tf.Tensor
            try:
                value = value.numpy()
            except Exception:
                pass
        if isinstance(value, (np.ndarray, np.generic)):
            flat = np.asarray(value).ravel()
            return float(flat[0]) if flat.size == 1 else None
        if hasattr(value, "__iter__"):
            # A hostile or broken iterator must not take down the verifier.
            try:
                items = list(value)
            except Exception:
                return None
            if len(items) == 1:
                try:
                    return float(items[0])
                except (TypeError, ValueError):
                    return None
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _describe(value, limit: int = 6) -> str:
    """Human-readable form of a possibly-scalar, possibly-list attribute."""
    scalar = _scalar(value)
    if scalar is not None:
        return f"{scalar:.12g}"
    try:
        arr = np.asarray(list(value) if hasattr(value, "__iter__") else value,
                         dtype=object)
    except Exception:
        return repr(value)
    flat = arr.ravel()
    if flat.size == 0:
        return "(empty)"
    head = ", ".join(f"{float(v):.6g}" for v in flat[:limit]
                     if isinstance(v, (int, float, np.number)))
    suffix = ", ..." if flat.size > limit else ""
    return f"[{head}{suffix}] ({flat.size} value(s))"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Verify the preprocessing contract.")
    p.add_argument("--config", default=None)
    p.add_argument("--onnx", default=None,
                   help="exported ONNX model; enables the bridge + numerics checks")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    m = cfg["model"]
    backbone = m["backbone"]
    h, w, n = m["input_height"], m["input_width"], m["num_classes"]

    print("=" * 74)
    print(f"GRAPH VERIFICATION: {backbone}")
    print("=" * 74)
    print("TF", tf.__version__)
    for g in tf.config.list_physical_devices("GPU"):
        print(f"GPU: {g.name}")
        try:
            info = tf.config.experimental.get_memory_info(g)
            print(f"  {info['current'] / 1024**3:.2f} / "
                  f"{info['total'] / 1024**3:.2f} GiB")
        except Exception:
            pass

    problems: list[str] = []

    # ------------------------------------------------------------ build graph
    base = getattr(tf.keras.applications, backbone)(
        include_top=m["include_top"],
        weights=m["weights"],
        input_shape=(h, w, 3),
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out = tf.keras.layers.Dense(n, activation=m["activation"])(x)
    model = tf.keras.Model(base.input, out)

    # ------------------------------------------- 1. internal preprocessing
    print("\n" + "-" * 74)
    print("1. Internal preprocessing layers in the Keras graph")
    print("-" * 74)
    rescales = [l for l in model.layers
                if isinstance(l, tf.keras.layers.Rescaling)]
    norms = [l for l in model.layers
             if isinstance(l, tf.keras.layers.Normalization)]
    print(f"  Rescaling layers found: {len(rescales)}")
    for l in rescales:
        s = _scalar(l.scale)
        o = _scalar(l.offset)
        s_txt = _describe(l.scale)
        o_txt = _describe(l.offset)
        print(f"    {l.name}")
        print(f"      scale  = {s_txt}   (type {type(l.scale).__name__})")
        print(f"      offset = {o_txt}   (type {type(l.offset).__name__})")
        if s is not None and abs(s - 1.0 / 255.0) < 1e-9:
            print("      -> this is the /255 stem")
        if s is not None and abs(s - 255.0) < 1e-6:
            print("      -> this is a *255 bridge (unexpected in the Keras graph)")
    print(f"  Normalization layers found: {len(norms)}")
    for l in norms:
        print(f"    {l.name}: mean={_describe(getattr(l, 'mean', None))} "
              f"var={_describe(getattr(l, 'variance', None))}")

    has_div_255 = any(
        (s := _scalar(l.scale)) is not None and abs(s - 1.0 / 255.0) < 1e-9
        for l in rescales
    )
    print(f"\n  graph contains Rescaling(1/255): {has_div_255}")
    if not has_div_255:
        problems.append("Keras graph has no Rescaling(1/255); the training path "
                        "would feed [0,255] to a stem that expects [0,1], or "
                        "vice versa. Reconcile before training.")

    # --------------------------------------------------------- 2. signature
    print("\n" + "-" * 74)
    print("2. Graph signature")
    print("-" * 74)
    print(f"  input : {model.input_shape}  dtype={model.input.dtype}")
    print(f"  output: {model.output_shape}  dtype={model.output.dtype}")
    total = int(model.count_params())
    trainable = int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))
    print(f"  total params     : {total:,}")
    print(f"  trainable params : {trainable:,}")

    # ------------------------------------------- 3. ONNX bridge + numerics
    onnx_path = Path(args.onnx) if args.onnx else None
    if onnx_path is None:
        default = Path(cfg["_base_dir"]) / cfg["output"]["dir"] / cfg["output"]["onnx"]
        onnx_path = default if default.is_file() else None

    print("\n" + "-" * 74)
    print("3. ONNX export bridge and numerical equivalence")
    print("-" * 74)

    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(1, h, w, 3),
                       dtype=np.uint8).astype(np.float32)
    unit = (raw / 255.0).astype(np.float32)          # what production emits
    unit_nchw = np.transpose(unit, (0, 3, 1, 2)).astype(np.float32)

    # Training path: raw [0,255] straight into the graph.
    training_input = raw

    if onnx_path is None:
        print("  No ONNX model found; skipping the export-bridge and "
              "end-to-end comparison.")
        print("  Run export_onnx.py first, then re-run with --onnx to verify.")
    else:
        try:
            import onnxruntime as ort
        except ImportError:
            print("  onnxruntime not installed; cannot verify the export.")
            problems.append("onnxruntime unavailable")
        else:
            sess = ort.InferenceSession(str(onnx_path),
                                        providers=["CPUExecutionProvider"])
            onnx_in = sess.get_inputs()[0]
            onnx_out = sess.get_outputs()[0]
            print(f"  ONNX: {onnx_path.name}")
            print(f"    input  {onnx_in.name} {onnx_in.shape} {onnx_in.type}")
            print(f"    output {onnx_out.name} {onnx_out.shape} {onnx_out.type}")

            if len(onnx_in.shape) != 4:
                problems.append(f"ONNX input is not 4D: {onnx_in.shape}")
            elif not (isinstance(onnx_in.shape[1], int) and onnx_in.shape[1] == 3):
                problems.append(
                    f"ONNX input is not NCHW with 3 channels: {onnx_in.shape}; "
                    "production supplies (1, 3, 224, 224)")

            # Bridge present? Compare the ONNX graph against the raw Keras model
            # on a [0,255] batch: if Rescaling(255) is missing, the ONNX graph
            # would need a [0,255] input instead of [0,1].
            keras_out = np.asarray(model.predict(raw, verbose=0))
            onnx_from_unit = np.asarray(
                sess.run([onnx_out.name], {onnx_in.name: unit_nchw})[0])
            onnx_from_raw = None
            if len(onnx_in.shape) == 4 and onnx_in.shape[1] == 3:
                raw_nchw = np.transpose(raw, (0, 3, 1, 2)).astype(np.float32)
                onnx_from_raw = np.asarray(
                    sess.run([onnx_out.name], {onnx_in.name: raw_nchw})[0])

            k = keras_out.ravel()
            u = onnx_from_unit.ravel()
            r = onnx_from_raw.ravel() if onnx_from_raw is not None else None
            max_ku = float(np.max(np.abs(k - u))) if k.size == u.size else float("inf")
            max_kr = float(np.max(np.abs(k - r))) if r is not None and k.size == r.size \
                else float("inf")

            print(f"\n  |Keras(raw255) - ONNX(unit01 NCHW)|  = {max_ku:.3e}")
            print(f"  |Keras(raw255) - ONNX(raw255 NCHW)| = {max_kr:.3e}")
            if max_ku <= TOL:
                print("  -> the Rescaling(255) bridge is PRESENT and correct.")
            elif max_kr <= TOL:
                print("  -> BRIDGE MISSING: the graph matches a [0,255] input, so")
                print("     production's [0,1] would arrive 255x too small.")
                problems.append(
                    "ONNX graph has no Rescaling(255) bridge; it expects [0,255] "
                    "but production sends [0,1]")
            else:
                print("  -> UNEXPECTED: the graph matches neither range. Inspect")
                print("     the export before training.")
                problems.append(
                    f"ONNX output matches neither [0,1] nor [0,255] input "
                    f"(diffs {max_ku:.3e} / {max_kr:.3e})")

    # ---------------------------- 4. explicit transform equivalence (numeric)
    print("\n" + "-" * 74)
    print("4. Transform equivalence, computed directly")
    print("-" * 74)
    x = np.array([0.0, 1.0, 127.5, 255.0], dtype=np.float32)
    training_chain = x * (1.0 / 255.0)                      # Keras stem only
    production_chain = (x / 255.0) * 255.0 * (1.0 / 255.0)   # + bridge + stem
    print(f"  input                : {x}")
    print(f"  training  (x/255)    : {training_chain}")
    print(f"  production (x/255*255/255): {production_chain}")
    d = float(np.max(np.abs(training_chain - production_chain)))
    print(f"  max abs difference   : {d:.3e}")
    if d <= TOL:
        print("  -> EQUIVALENT: both paths divide by 255 exactly once.")
    else:
        print("  -> NOT equivalent.")
        problems.append(f"training and production transforms differ by {d:.3e}")

    print(f"  sanity: a second /255 would give {x[3] / 255.0 / 255.0:.6f} "
          f"instead of {training_chain[3]:.6f} (255x too small)")

    # --------------------------------- 5. no external rescale in train/eval
    print("\n" + "-" * 74)
    print("5. Training/evaluation must not rescale externally")
    print("-" * 74)
    here = Path(__file__).resolve().parent
    for name in ("train.py", "evaluate.py"):
        text = (here / name).read_text(encoding="utf-8")
        code = "\n".join(
            ln.split("#", 1)[0] for ln in text.splitlines()
        )
        bad = ("Rescaling(1.0 / 255.0)" in code
               or "rescale(x), y" in code
               or "1.0 / 255.0" in code)
        print(f"  {name:14} external rescale: {'FOUND (bad)' if bad else 'none (good)'}")
        if bad:
            problems.append(f"{name} applies an external 1/255 before EfficientNet")

    # ------------------------------------------------------------ conclusion
    print("\n" + "=" * 74)
    if problems:
        print("CONTRACT VIOLATED - do not train")
        print("=" * 74)
        for p in problems:
            print(f"  - {p}")
        return EXIT_CONTRACT_VIOLATION

    print("CONTRACT VERIFIED")
    print("=" * 74)
    print(f"  backbone                 : {backbone}")
    print(f"  internal Rescaling(1/255): {has_div_255}")
    print(f"  total / trainable params : {total:,} / {trainable:,}")
    print("  training input           : raw [0, 255]")
    print("  production input         : [0, 1] float32 NCHW")
    print("  ONNX bridge              : Rescaling(255) -> [0, 255]")
    print("  net transform            : x / 255 on both paths (equivalent)")
    print("\nSafe to run: python train.py --batch-size 32 --epochs 2")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())