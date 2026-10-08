"""Shared pipeline: labels -> engine -> preprocessing -> result.

One code path serves still images and camera frames, so both modes cannot drift.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import tiling
from .config import Config
from .image_quality import check_image
from .inference import InferenceEngine, create_engine
from .preprocessing import preprocess
from .results import DetectionResult, classify


def load_labels(labels_path) -> list[str]:
    """Read one label per line, ignoring blanks and # comments."""
    p = Path(labels_path)
    if not p.is_file():
        raise FileNotFoundError(f"labels file not found: {p}")
    out: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def build_engine(cfg: Config, backend: str | None = None) -> InferenceEngine:
    """Create and load the configured backend.

    The mock backend skips the label-count check by construction, since it has
    no output shape to compare against.
    """
    kind = (backend or cfg.inference.backend).strip().lower()
    labels = load_labels(cfg.labels_file) if cfg.labels_file.is_file() else []
    return create_engine(
        kind,
        cfg.model_file,
        labels,
        expected_classes=cfg.inference.expected_classes,
        strict_label_count=cfg.inference.strict_label_count,
    )


def _quality_gate(image, engine: InferenceEngine, cfg: Config) -> dict:
    """Frame quality check shared by the still and tiled paths."""
    from .image_quality import QualityThresholds

    skip_quality = bool(getattr(cfg, "skip_image_quality_check", False))
    if skip_quality:
        return {"valid": True, "metrics": {}, "skipped": True}
    return check_image(
        image,
        QualityThresholds.from_config(cfg),
        has_labels=bool(getattr(engine, "labels", [])),
    )


def _rejected(quality: dict, *, source, image_path, started) -> DetectionResult:
    """Build the uniform rejection result for an unusable frame."""
    return DetectionResult(
        class_name="Unknown",
        confidence=0.0,
        healthy=False,
        reason=quality.get("error_code", "IMAGE_INVALID"),
        inference_ms=0.0,
        preprocess_ms=0.0,
        total_ms=round((time.perf_counter() - started) * 1000.0, 3),
        source=source,
        image_path=image_path,
        metadata={"quality": quality},
    )


def run_inference(image, engine: InferenceEngine, cfg: Config,
                  source: str | None = None,
                  image_path: str | None = None) -> DetectionResult:
    """Quality-gate, then preprocess, run, and build a DetectionResult.

    A frame that fails the quality gate never reaches the model, so no disease
    can be reported from an unusable image.
    """
    started = time.perf_counter()

    # 1. Quality gate.
    quality = _quality_gate(image, engine, cfg)
    if not quality.get("valid"):
        # Reject before inference: no model call, no timing beyond the gate.
        return _rejected(quality, source=source, image_path=image_path, started=started)

    # 2. Preprocess.
    t0 = time.perf_counter()
    tensor = preprocess(image, cfg.preprocess)
    preprocess_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Inference.
    t1 = time.perf_counter()
    scores = engine.predict(tensor)
    inference_ms = (time.perf_counter() - t1) * 1000.0
    engine._last_inference_ms = inference_ms

    labels = list(engine.labels) if getattr(engine, "labels", None) else []
    if not labels:
        # No labels file: fall back to indexed names rather than inventing diseases.
        labels = [f"class_{i}" for i in range(len(scores))]

    # 4. Postprocess + confidence band.
    result = classify(
        scores,
        labels,
        confidence_threshold=cfg.inference.confidence_threshold,
        healthy_keyword=cfg.inference.healthy_keyword,
        inference_ms=inference_ms,
        preprocess_ms=preprocess_ms,
        source=source,
        image_path=image_path,
    )
    result.total_ms = round(preprocess_ms + inference_ms, 3)
    result.metadata["quality"] = quality
    return result


def run_inference_tiled(image, engine: InferenceEngine, cfg: Config,
                        source: str | None = None,
                        image_path: str | None = None) -> DetectionResult:
    """Scan a wide frame tile-by-tile and report the best tile.

    Frames no larger than one tile (or with tiling disabled) fall back to the
    single-inference path, so a close-up leaf and an aerial shot are handled
    by the same pipeline.

    The whole frame still passes the quality gate first: no disease is ever
    reported from a blurry, dark, or otherwise unusable shot.
    """
    if not tiling.should_tile(image, cfg):
        return run_inference(image, engine, cfg, source=source, image_path=image_path)

    started = time.perf_counter()

    # 1. Quality gate on the whole frame, same as the single-shot path.
    quality = _quality_gate(image, engine, cfg)
    if not quality.get("valid"):
        return _rejected(quality, source=source, image_path=image_path, started=started)

    # 2. Scan every tile.
    results, boxes = tiling.scan(image, engine, cfg)
    if results:
        best = max(results, key=lambda r: r.confidence)
        best.source = source
        best.image_path = image_path
        best.preprocess_ms = round(sum(r.preprocess_ms for r in results), 3)
        best.inference_ms = round(sum(r.inference_ms for r in results), 3)
        best.total_ms = round((time.perf_counter() - started) * 1000.0, 3)
        best.metadata["quality"] = quality
        best.metadata["scanned"] = True
        best.metadata["tile_count"] = len(results)
        best.metadata["tiles"] = [
            {
                "class_name": r.class_name,
                "confidence": round(float(r.confidence), 4),
                "reliable": r.reliable,
                "reason": r.reason,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
            for r, (x, y, w, h) in zip(results, boxes)
        ]
        return best

    # Scanning produced nothing (should not happen with a valid config); be
    # honest about it rather than inventing a class.
    return DetectionResult(
        class_name="Unknown",
        confidence=0.0,
        healthy=False,
        reason="no_tiles",
        inference_ms=0.0,
        preprocess_ms=0.0,
        total_ms=round((time.perf_counter() - started) * 1000.0, 3),
        source=source,
        image_path=image_path,
        metadata={"quality": quality, "scanned": True, "tile_count": 0},
    )