"""Single-image inference.

Usage:
  python scripts/image_inference.py --image leaf.jpg
  python scripts/image_inference.py --image leaf.jpg --confidence 0.80 --json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/image_inference.py` without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_disease.config import load_config  # noqa: E402
from ai_disease.inference import InferenceError  # noqa: E402
from ai_disease.pipeline import build_engine, run_inference  # noqa: E402
from ai_disease.preprocessing import PreprocessingError, load_image  # noqa: E402
from ai_disease.results import save_result  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Classify a crop disease from one image.")
    p.add_argument("--image", required=True, help="path to the input image")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--model", default=None, help="override the model path")
    p.add_argument("--labels", default=None, help="override the labels path")
    p.add_argument("--backend", default=None, choices=["onnx", "mock", "tflite"],
                   help="inference backend (default from config)")
    p.add_argument("--confidence", type=float, default=None,
                   help="override the confidence threshold, e.g. 0.70")
    p.add_argument("--json", action="store_true", help="print JSON only")
    p.add_argument("--log", action="store_true", help="save the result as JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.model:
        cfg.model_path = Path(args.model)
    if args.labels:
        cfg.labels_path = Path(args.labels)
    if args.confidence is not None:
        cfg.inference.confidence_threshold = args.confidence

    try:
        image = load_image(args.image)
    except PreprocessingError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    try:
        engine = build_engine(cfg, args.backend)
    except InferenceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    result = run_inference(image, engine, cfg,
                           source="image", image_path=str(args.image))

    if args.json:
        print(result.to_json(indent=2))
    else:
        print(result.display())
        print(f"Model: {cfg.model_file.name}  ({engine.name})")

    if args.log or cfg.logging.enabled:
        results_dir = cfg.resolve(cfg.logging.results_dir)
        images_dir = cfg.resolve(cfg.logging.images_dir)
        out = save_result(result, results_dir,
                          images_dir if cfg.logging.save_annotated_images else None)
        if not args.json:
            print(f"Saved: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())