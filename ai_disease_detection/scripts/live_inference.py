"""Live camera inference with FPS and latency reporting.

Usage:
  python scripts/live_inference.py --camera 0
  python scripts/live_inference.py --backend file --path clip.mp4
  python scripts/live_inference.py --camera 0 --confidence 0.80 --no-display
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_disease.camera import Camera, CameraUnavailableError  # noqa: E402
from ai_disease.config import load_config  # noqa: E402
from ai_disease.inference import InferenceError  # noqa: E402
from ai_disease.pipeline import build_engine, run_inference  # noqa: E402
from ai_disease.results import annotate, save_result  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live crop disease classification.")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--backend", default=None, choices=["onnx", "mock", "tflite"],
                   help="inference backend (default from config)")
    p.add_argument("--camera", type=int, default=None, help="USB camera device index")
    p.add_argument("--backend-cam", default=None,
                   choices=["usb", "file", "picamera2"],
                   help="camera backend: usb (default), file, or picamera2")
    p.add_argument("--path", default=None, help="video file, with --backend-cam file")
    p.add_argument("--confidence", type=float, default=None,
                   help="override the confidence threshold, e.g. 0.70")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = run until 'q')")
    p.add_argument("--no-display", action="store_true",
                   help="do not open a window; prints one line per result")
    p.add_argument("--log", action="store_true", help="save results as JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.confidence is not None:
        cfg.inference.confidence_threshold = args.confidence
    if args.camera is not None:
        cfg.camera.device = args.camera

    try:
        engine = build_engine(cfg, args.backend)
    except InferenceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    cam_backend = args.backend_cam or "usb"
    cam = Camera.from_config(cfg.camera, backend=cam_backend, path=args.path)
    try:
        cam.open()
    except CameraUnavailableError as e:
        print(f"error: {e}", file=sys.stderr)
        return 4

    print(f"Camera open ({cam_backend}). Model: {cfg.model_file.name} ({engine.name})")
    print("Press 'q' to quit.", flush=True)

    # Rolling window keeps the FPS readout stable rather than jittery.
    frame_times = deque(maxlen=30)
    saved = 0
    processed = 0
    display = not args.no_display

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("end of stream")
                break

            result = run_inference(frame, engine, cfg, source=cam_backend)

            now = frame_times[-1][0] if frame_times else None
            import time as _time

            t = _time.perf_counter()
            frame_times.append(t)
            fps = 0.0
            if len(frame_times) > 1:
                span = frame_times[-1][0] - frame_times[0][0]
                if span > 0:
                    fps = (len(frame_times) - 1) / span

            if display:
                import cv2

                canvas = annotate(frame, result)
                lines = [
                    f"{result.class_name}",
                    f"{result.confidence * 100:.1f}%   {result.inference_ms:.0f} ms",
                    f"{fps:.1f} FPS",
                ]
                if result.reason:
                    lines.append(result.reason)
                y = canvas.shape[0] - 20 - 26 * (len(lines) - 2)
                for i, line in enumerate(lines):
                    colour = (0, 200, 255) if result.reason else (60, 220, 60)
                    cv2.putText(canvas, line, (12, y + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
                cv2.imshow("Crop Disease AI", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print(
                    f"{result.class_name}\t{result.confidence * 100:.1f}%\t"
                    f"{result.inference_ms:.1f}ms\t{fps:.1f}fps"
                    + (f"\t({result.reason})" if result.reason else ""),
                    flush=True,
                )

            processed += 1
            if (args.log or cfg.logging.enabled) and result.reliable:
                from ai_disease.results import annotate as _annotate

                out = save_result(
                    result,
                    cfg.resolve(cfg.logging.results_dir),
                    cfg.resolve(cfg.logging.images_dir)
                    if cfg.logging.save_annotated_images else None,
                    annotated_image=annotate(frame, result) if cfg.logging.save_annotated_images else None,
                )
                saved += 1

            if args.max_frames and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cam.release()
        if display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    print(f"Processed {processed} frame(s), saved {saved} result(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())