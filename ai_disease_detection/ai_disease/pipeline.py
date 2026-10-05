"""Shared pipeline: labels -> engine -> preprocessing -> result.

One code path serves still images and camera frames, so both modes cannot drift.
"""
from __future__ import annotations

import time
from pathlib import Path

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


def run_inference(image, engine: InferenceEngine, cfg: Config,
                  source: str | None = None,
                  image_path: str | None = None) -> DetectionResult:
    """Quality-gate, then preprocess, run, and build a DetectionResult.

    A frame that fails the quality gate never reaches the model, so no disease
    can be reported from an unusable image.
    """
    started = time.perf_counter()

    # 1. Quality gate.
    from .image_quality import QualityThresholds

    skip_quality = bool(getattr(cfg, "skip_image_quality_check", False))
    if skip_quality:
        quality = {"valid": True, "metrics": {}, "skipped": True}
    else:
        quality = check_image(
            image,
            QualityThresholds.from_config(cfg),
            has_labels=bool(getattr(engine, "labels", [])),
        )

    if not quality.get("valid"):
        # Reject before inference: no model call, no timing beyond the gate.
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