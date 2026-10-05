"""Result construction for disease detection.

Turns raw engine output into a stable, human-readable record. Never invents a
disease: low-confidence output is reported as Unknown with a reason.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

UNKNOWN_CLASS = "Unknown"
REASON_LOW_CONFIDENCE = "below_confidence_threshold"


@dataclass
class DetectionResult:
    """One classification outcome."""

    class_id: int | None = None
    class_name: str = UNKNOWN_CLASS
    confidence: float = 0.0
    healthy: bool = False
    reason: str | None = None
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    total_ms: float = 0.0
    timestamp: str = ""
    # Populated only when logging is enabled; reserved for the future
    # Pixhawk telemetry bridge (lat/lon/alt are never filled in by this module).
    source: str | None = None
    image_path: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ views
    @property
    def reliable(self) -> bool:
        """True when a class was actually assigned (not Unknown)."""
        return self.reason is None and self.class_name != UNKNOWN_CLASS

    @property
    def disease(self) -> str | None:
        """Disease component of a PlantVillage-style label, or None if healthy."""
        if not self.reliable or self.healthy:
            return None
        label = self.class_name
        if "___" in label:
            return label.split("___", 1)[1].replace("_", " ").strip() or None
        return label

    @property
    def crop(self) -> str | None:
        if not self.reliable:
            return None
        return self.class_name.split("___", 1)[0].strip() or None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reliable"] = self.reliable
        d["disease"] = self.disease
        d["crop"] = self.crop
        return d

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def display(self) -> str:
        """Human-readable one-line-per-field summary."""
        lines = [f"Disease: {self.class_name}"]
        lines.append(f"Confidence: {self.confidence * 100:.1f}%")
        if self.reason:
            lines.append(f"Reason: {self.reason}")
        lines.append(f"Inference: {self.inference_ms:.1f} ms")
        lines.append(f"Healthy: {'yes' if self.healthy else 'no'}")
        return "\n".join(lines)


def is_healthy_label(label: str, keyword: str = "healthy") -> bool:
    """True for PlantVillage healthy classes, matched case-insensitively."""
    return keyword.strip().lower() in str(label).strip().lower()


def classify(
    probabilities,
    labels: list[str],
    *,
    confidence_threshold: float = 0.70,
    healthy_keyword: str = "healthy",
    inference_ms: float = 0.0,
    preprocess_ms: float = 0.0,
    source: str | None = None,
    image_path: str | None = None,
) -> DetectionResult:
    """Build a DetectionResult from a class-score vector.

    `probabilities` must be index-aligned with `labels`. Output index is taken
    from argmax, never hard-coded.
    """
    started = time.perf_counter()
    scores = [float(p) for p in probabilities]
    total = len(scores)

    if total == 0:
        return DetectionResult(
            class_name=UNKNOWN_CLASS,
            reason="empty_output",
            inference_ms=inference_ms,
            preprocess_ms=preprocess_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            source=source,
            image_path=image_path,
        )

    best_idx = max(range(total), key=lambda i: scores[i])
    confidence = scores[best_idx]
    label = labels[best_idx] if best_idx < len(labels) else f"class_{best_idx}"

    # A threshold miss is reported honestly rather than dressed up as a class.
    if confidence < confidence_threshold:
        return DetectionResult(
            class_id=best_idx,
            class_name=UNKNOWN_CLASS,
            confidence=confidence,
            healthy=False,
            reason=REASON_LOW_CONFIDENCE,
            inference_ms=inference_ms,
            preprocess_ms=preprocess_ms,
            total_ms=(time.perf_counter() - started) * 1000.0,
            source=source,
            image_path=image_path,
            metadata={"top_label": label},
        )

    return DetectionResult(
        class_id=best_idx,
        class_name=label,
        confidence=confidence,
        healthy=is_healthy_label(label, healthy_keyword),
        inference_ms=inference_ms,
        preprocess_ms=preprocess_ms,
        total_ms=(time.perf_counter() - started) * 1000.0,
        source=source,
        image_path=image_path,
    )


def save_result(result: DetectionResult, results_dir, images_dir=None,
                annotated_image: object | None = None) -> Path | None:
    """Persist one result as timestamped JSON. Optional; never mandatory."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    path = out_dir / f"result_{stamp}.json"
    path.write_text(result.to_json(indent=2), encoding="utf-8")

    if annotated_image is not None and images_dir is not None:
        try:
            import cv2

            img_dir = Path(images_dir)
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"frame_{stamp}.jpg"
            cv2.imwrite(str(img_path), annotated_image)
            result.image_path = str(img_path)
            path.write_text(result.to_json(indent=2), encoding="utf-8")
        except Exception:
            # Annotation saving is best-effort; never fail the run over it.
            pass
    return path


def annotate(frame, result: DetectionResult):
    """Draw the result onto a copy of the frame for on-screen/overlay use."""
    import cv2
    import numpy as np

    canvas = frame.copy()
    h, w = canvas.shape[:2]
    top = [result.class_name, f"{result.confidence * 100:.1f}%  {result.inference_ms:.0f} ms"]
    if result.reason:
        top.append(f"({result.reason})")
    elif result.healthy:
        top.append("healthy")

    y = 34
    for i, line in enumerate(top):
        colour = (60, 220, 60) if result.healthy else (0, 165, 255)
        if i == 0 and result.reason:
            colour = (0, 200, 255)
        cv2.putText(canvas, line, (12, y + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (w - 1, 8 + 30 * len(top)), (30, 30, 30), -1)
    return canvas