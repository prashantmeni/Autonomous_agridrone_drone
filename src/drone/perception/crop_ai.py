"""Adapter exposing the standalone AI module to the drone API.

The disease logic now lives in `ai_disease_detection.ai_disease`, which is
standalone and Pi-optimised. This module is a thin bridge so `server.py` keeps
its existing endpoint contract without importing the package awkwardly or
knowing its internals.

Deliberately contains no inference logic: it only translates between the
backend's config/dict shapes and the AI module's objects.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

log = logging.getLogger("drone.perception.crop_ai")

# The AI module is a sibling package, not part of the drone distribution, so it
# is not necessarily importable. Add the repo root to sys.path once, rather
# than relying on the caller, and resolve it by absolute path so this works
# regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ai = None
_ai_error: str | None = None
_lock = threading.Lock()


def _ai_module():
    """Import the AI package lazily, so a missing dependency cannot break boot."""
    global _ai, _ai_error
    with _lock:
        if _ai is not None or _ai_error is not None:
            return _ai
        try:
            from ai_disease_detection.ai_disease import (  # noqa: F401
                config,
                inference,
                image_quality,
                onnx_engine,
                pipeline,
                preprocessing,
            )
            _ai = {
                "config": config,
                "inference": inference,
                "image_quality": image_quality,
                "onnx_engine": onnx_engine,
                "pipeline": pipeline,
                "preprocessing": preprocessing,
            }
        except Exception as e:  # pragma: no cover - dependency problem
            _ai_error = str(e)
            log.error("crop_ai module unavailable: %s", e)
            _ai = None
    return _ai


def available() -> bool:
    return _ai_module() is not None


def unavailable_reason() -> str | None:
    _ai_module()
    return _ai_error


# --------------------------------------------------------------------- paths
def _ai_root() -> Path:
    return _REPO_ROOT / "ai_disease_detection"


def labels_path() -> Path:
    return _ai_root() / "models" / "labels.txt"


def model_dir() -> Path:
    return _ai_root() / "models"


# ------------------------------------------------------- labels / thresholds
def load_labels() -> list[str]:
    """The 38 production labels, read straight from the AI module."""
    mod = _ai_module()
    if mod is None:
        return []
    return mod["pipeline"].load_labels(str(labels_path()))


def load_production_labels() -> list[str]:
    return load_labels()


# ---------------------------------------------------------------------- engine
def build_engine(model_path, labels: list[str], expected_classes: int = 38):
    """Instantiate the AI module's configured backend (onnx by default)."""
    mod = _ai_module()
    if mod is None:
        raise mod_error("crop_ai module unavailable")
    return mod["inference"].create_engine(
        "onnx",
        str(model_path),
        labels,
        expected_classes=expected_classes,
        strict_label_count=True,
    )


def mod_error(message: str):
    mod = _ai_module()
    err = mod["inference"].InferenceError if mod else RuntimeError
    return err(f"{message}: {_ai_error or 'module unavailable'}")


# ------------------------------------------------------------------- metadata
def model_status(model_path, enabled: bool, confidence_threshold: float) -> dict:
    """Metadata for GET /api/disease/model, in the backend's existing shape."""
    mod = _ai_module()
    if mod is None:
        return {
            "status": "MODULE_UNAVAILABLE",
            "enabled": enabled,
            "model_path": str(model_path),
            "detail": _ai_error or "ai_disease_detection not importable",
            "labels": 0,
            "supported_formats": [".onnx", ".tflite"],
        }

    inference = mod["inference"]

    raw = str(model_path or "").strip()
    path = Path(raw) if raw else None
    info: dict = {
        "enabled": bool(enabled),
        "model_path": raw,
        "model_name": path.name if path and path.name else None,
        "supported_formats": sorted(inference.SUPPORTED.keys())
        if hasattr(inference, "SUPPORTED") else [".onnx", ".tflite"],
        "load_error": None,
    }
    if not enabled or not raw:
        info.update({"status": "MODEL_NOT_AVAILABLE", "format": None,
                     "runtime": None, "labels": 0,
                     "detail": "No model uploaded yet."})
        return info
    if not path.is_file():
        info.update({"status": "MODEL_NOT_FOUND", "format": path.suffix.lower(),
                     "labels": 0, "detail": f"{path} does not exist."})
        return info

    labels = load_labels()
    try:
        engine = build_engine(path, labels)
        info.update(engine.describe())
        info["status"] = "OK"
        return info
    except Exception as e:
        info.update({"status": "LOAD_FAILED", "labels": len(labels),
                     "detail": str(e), "load_error": str(e)})
        return info


def quality_thresholds():
    """Expose the AI module's image-quality gates to the API."""
    mod = _ai_module()
    if mod is None:
        return None
    from dataclasses import asdict

    return asdict(mod["image_quality"].QualityThresholds())


# -------------------------------------------------------------- inference
def analyze(frame, model_path, confidence_threshold: float,
            healthy_keyword: str = "healthy", source: str = "camera",
            image_path: str | None = None,
            high_threshold: float = 0.80,
            medium_threshold: float = 0.50,
            tiled: bool | None = None) -> dict:
    """Run the AI module's measured pipeline on one frame.

    Returns the AI module's DetectionResult as a dict, so the backend keeps
    publishing the same JSON shape the dashboard already renders.

    `tiled` selects the sliding-window scan for wide/aerial frames; default
    None defers to the module's `tiled.enabled` setting, keeping still-image
    uploads single-shot unless scanning is switched on.
    """
    mod = _ai_module()
    if mod is None:
        raise mod_error("cannot analyze frame")

    labels = load_labels()
    engine = build_engine(model_path, labels)

    # Use the module's deployed config (ai_disease_detection/config.yaml) so
    # preprocessing (type, mean/std) and model contract always match the model
    # that config points at — e.g. cropguard needs ImageNet normalization, not
    # the divide_255 default. Only the drone's own handling knobs are overridden.
    cfg_cls = mod["config"]
    cfg = cfg_cls.load_config()
    cfg.inference.confidence_threshold = float(confidence_threshold)
    cfg.inference.healthy_keyword = healthy_keyword
    cfg.skip_image_quality_check = False

    use_tiled = cfg.tiled.enabled if tiled is None else bool(tiled)
    if use_tiled:
        result = mod["pipeline"].run_inference_tiled(
            frame, engine, cfg, source=source, image_path=image_path
        )
    else:
        result = mod["pipeline"].run_inference(
            frame, engine, cfg, source=source, image_path=image_path
        )
    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)

    # The pipeline gates on frame quality and records it under metadata; surface
    # it as a first-class field so the dashboard can render a warning without
    # digging through metadata.
    quality = (data.get("metadata") or {}).get("quality")
    if quality is not None:
        data["image_quality"] = quality

    # A tiled scan reports the best tile plus the full grid; expose the scan
    # summary flat so the dashboard can render overlay boxes without parsing
    # metadata.
    meta = data.get("metadata") or {}
    if meta.get("scanned"):
        data["tile_count"] = int(meta.get("tile_count") or 0)
        data["tiles"] = meta.get("tiles") or []

    reliable = bool(data.get("reliable"))
    confidence = float(data.get("confidence") or 0.0)

    if not reliable and data.get("reason"):
        # A rejected frame must look rejected, not like a low-confidence guess.
        code = data["reason"]
        iq = mod["image_quality"]
        data.setdefault("message", iq.MESSAGES.get(
            code, "Image cannot be used for disease detection."))
        data.setdefault("suggestion", iq.SUGGESTIONS.get(
            code, "Capture a clearer, well-lit image."))

        # A broken install is a server fault (503), not a rejected photo.
        if code == "IMAGE_QUALITY_UNAVAILABLE":
            raise RuntimeError(data.get("message", "quality checks unavailable"))

    # Map the module's result onto the JSON the dashboard already renders.
    status = "OK" if reliable else str(data.get("reason") or "UNKNOWN")
    data["status"] = status
    data["success"] = reliable
    data["confidence"] = confidence
    data["confidence_level"] = _confidence_band(
        confidence, reliable, high_threshold, medium_threshold)
    data["reliable"] = reliable
    data["label"] = data.get("disease") or data.get("class_name")
    data["note"] = _note_for(data, healthy_keyword)
    # Present on every response so the UI's error branch never reads undefined.
    data.setdefault("message", None)
    data.setdefault("suggestion", None)
    # The UI branches on error_code, so always present: null on success, the
    # rejection code otherwise.
    data["error_code"] = None if reliable else status
    # Persistence is the server's job; default so the UI never sees undefined.
    data.setdefault("marked", False)
    data["timing"] = {
        "preprocess_ms": float(data.get("preprocess_ms") or 0.0),
        "inference_ms": float(data.get("inference_ms") or 0.0),
        "postprocess_ms": 0.0,
        "total_ms": float(data.get("total_ms") or 0.0),
    }
    data["model"] = {"name": Path(model_path).name or None, "version": None}
    return data


def _confidence_band(confidence: float, reliable: bool, high: float,
                     medium: float) -> str:
    """HIGH / MEDIUM / LOW / NONE, using the backend's configured cut-offs.

    These must match the thresholds published by /api/crop-ai/model, otherwise
    the UI advertises one set of bands and renders another.
    """
    if not reliable:
        return "NONE"
    if confidence >= high:
        return "HIGH"
    if confidence >= medium:
        return "MEDIUM"
    return "LOW"


def _note_for(data: dict, healthy_keyword: str) -> str | None:
    """Short human-readable summary for the UI, or None when unremarkable."""
    if not data.get("reliable"):
        return data.get("message")
    if data.get("healthy"):
        return "No disease detected."
    return None


def check_image(frame):
    """Run the AI module's frame-quality gate, if available."""
    mod = _ai_module()
    if mod is None:
        return {"valid": True, "metrics": {}}
    iq = mod["image_quality"]
    return iq.check_image(frame, iq.QualityThresholds())


MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _strip_label_suffix(name: str) -> str:
    """`model.onnx.txt` -> `model.onnx`, so labels attach to their model."""
    for suffix in (".txt", ".json", ".labels"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name
