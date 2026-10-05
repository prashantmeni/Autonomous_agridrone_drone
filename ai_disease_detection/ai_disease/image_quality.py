"""Frame quality gating for disease inference.

Runs before the model. A frame that is too dark, blown out, blurry, corrupt,
too small or featureless is rejected with a specific error code, so no disease
is ever reported from an unusable image.

Thresholds are data, not literals, so they can be tuned per deployment. The
pipeline calls `check_image()` via `run_inference()`; the drone backend exposes
the same gate through `perception.crop_ai`.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

try:  # cv2 is a hard dependency of preprocessing too; detect it once here.
    import cv2
except ImportError:  # pragma: no cover - dependency problem
    cv2 = None

from .preprocessing import ensure_bgr

log = logging.getLogger("ai_disease.image_quality")


@dataclass(frozen=True)
class QualityThresholds:
    """Acceptable bounds for a frame, on a 0-255 luminance scale."""

    # Mean luminance of the greyscale frame.
    min_brightness: float = 18.0
    max_brightness: float = 245.0
    # Variance of the Laplacian: the standard focus measure. A blurred frame
    # scores near zero; a sharp leaf texture scores comfortably above 40.
    min_blur_score: float = 12.0
    # Minimum accepted frame geometry before any resize happens.
    min_width: int = 32
    min_height: int = 32
    # Fraction of the frame that must not be identical: guards against a blank
    # frame of uniform colour, which can have healthy brightness and blur.
    min_contrast_std: float = 4.0
    # Refuse frames when the active model has no labels file attached.
    require_labels: bool = False

    @classmethod
    def from_config(cls, cfg) -> "QualityThresholds":
        """Build from a config object, falling back to defaults when unset."""
        if cfg is None:
            return cls()
        kwargs: dict = {}
        for field, attr in (
            ("min_brightness", "image_min_brightness"),
            ("max_brightness", "image_max_brightness"),
            ("min_blur_score", "image_min_blur_score"),
            ("min_width", "image_min_width"),
            ("min_height", "image_min_height"),
            ("min_contrast_std", "image_min_contrast_std"),
            ("require_labels", "image_require_labels"),
        ):
            value = getattr(cfg, attr, None)
            if value is not None:
                kwargs[field] = value
        return cls(**kwargs)


# Human-readable text keyed by error code, so callers get actionable advice.
MESSAGES = {
    "IMAGE_TOO_DARK": "Image is too dark for reliable disease detection.",
    "IMAGE_TOO_BRIGHT": "Image is overexposed for reliable disease detection.",
    "IMAGE_TOO_BLURRY": "Image is too blurry for reliable disease detection.",
    "IMAGE_TOO_SMALL": "Image resolution is too low for reliable disease detection.",
    "IMAGE_NO_DETAIL": "Image lacks usable visual detail for disease detection.",
    "IMAGE_INVALID": "Image is invalid or corrupt.",
    "IMAGE_NO_LABELS": "No labels file is attached to the active model.",
    # A broken install, not a bad photo. Kept distinct so callers do not tell a
    # user their image is corrupt when the real fault is a missing dependency.
    "IMAGE_QUALITY_UNAVAILABLE": "Image quality checks are unavailable on this system.",
}

SUGGESTIONS = {
    "IMAGE_TOO_DARK": "Improve lighting and capture the leaf again.",
    "IMAGE_TOO_BRIGHT": "Reduce direct glare or move away from strong backlight.",
    "IMAGE_TOO_BLURRY": "Hold the camera steady or increase focus before capturing.",
    "IMAGE_TOO_SMALL": "Move closer so the leaf fills more of the frame.",
    "IMAGE_NO_DETAIL": "Frame a single leaf against a plain background.",
    "IMAGE_INVALID": "The image could not be decoded. Capture a fresh JPEG or PNG.",
    "IMAGE_NO_LABELS": "Attach a labels file so results name diseases instead of class_N.",
    "IMAGE_QUALITY_UNAVAILABLE":
        "Install the AI module's dependencies (opencv-python, numpy) and restart.",
}


def error_payload(code: str, detail: str | None = None) -> dict:
    """Quality-gate rejection.

    Deliberately lean: this nests under `image_quality` in the API response, so
    it carries only what describes the image, not the top-level result fields.
    """
    out = {
        "valid": False,
        "error_code": code,
        "message": MESSAGES.get(code, "Image cannot be used for disease detection."),
        "suggestion": SUGGESTIONS.get(code, "Capture a clearer, well-lit image."),
    }
    if detail:
        out["detail"] = detail
    return out


def _measure(image, thresholds: QualityThresholds) -> dict:
    """Compute the raw metrics. Expects a validated BGR ndarray."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return {
        "brightness": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        # Variance of the Laplacian, the familiar OpenCV focus measure.
        "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def check_image(image, thresholds: QualityThresholds | None = None,
                has_labels: bool = True) -> dict:
    """Validate a frame.

    Returns a dict with ``valid`` plus the measured metrics, and on failure an
    ``error_code`` the caller can surface directly. Never returns a disease.
    """
    th = thresholds or QualityThresholds()

    # A missing OpenCV install must not masquerade as a bad photo.
    if cv2 is None:
        out = error_payload("IMAGE_QUALITY_UNAVAILABLE",
                            "opencv-python is not installed")
        out.update({"valid": False, "metrics": {}})
        return out

    if image is None:
        out = error_payload("IMAGE_INVALID", "no image supplied")
        out.update({"valid": False, "metrics": {}})
        return out

    # Structural validation is preprocessing's job and raises PreprocessingError,
    # so reuse it rather than duplicating the rules and diverging.
    ensure_bgr(image)

    h, w = image.shape[:2]
    if h < th.min_height or w < th.min_width:
        out = error_payload(
            "IMAGE_TOO_SMALL",
            f"{w}x{h} is below the {th.min_width}x{th.min_height} minimum")
        out.update({"valid": False, "metrics": {"width": int(w),
                                                "height": int(h)}})
        return out

    metrics = _measure(image, th)

    # Ordered dark -> bright -> blur -> detail -> labels, so the first failing
    # check is the one reported.
    code: str | None = None
    detail: str | None = None
    if metrics["brightness"] < th.min_brightness:
        code = "IMAGE_TOO_DARK"
        detail = f"brightness {metrics['brightness']} < {th.min_brightness}"
    elif metrics["brightness"] > th.max_brightness:
        code = "IMAGE_TOO_BRIGHT"
        detail = f"brightness {metrics['brightness']} > {th.max_brightness}"
    elif metrics["blur_score"] < th.min_blur_score:
        code = "IMAGE_TOO_BLURRY"
        detail = f"blur score {metrics['blur_score']} < {th.min_blur_score}"
    elif metrics["contrast_std"] < th.min_contrast_std:
        code = "IMAGE_NO_DETAIL"
        detail = f"contrast std {metrics['contrast_std']} < {th.min_contrast_std}"
    elif th.require_labels and not has_labels:
        code = "IMAGE_NO_LABELS"
        detail = "model has no labels file"

    if code:
        out = error_payload(code, detail)
        out.update({"valid": False, "metrics": metrics})
        return out

    return {
        "valid": True,
        "metrics": metrics,
        "thresholds": asdict(th),
    }
