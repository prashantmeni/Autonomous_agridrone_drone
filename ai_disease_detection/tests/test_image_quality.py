"""Frame quality gating: a rejected photo must never yield a disease."""
from __future__ import annotations

import numpy as np
import pytest

from ai_disease.image_quality import (
    MESSAGES,
    SUGGESTIONS,
    QualityThresholds,
    check_image,
    error_payload,
)


def textured(seed: int = 0, size: int = 224, mean: float = 120,
             std: float = 45) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, std, (size, size, 3)), 0, 255).astype(np.uint8)


def blurred(img: np.ndarray, k: int = 21) -> np.ndarray:
    import cv2

    return cv2.blur(img, (k, k))


# ------------------------------------------------------------------ thresholds
def test_defaults_are_ordered_sensibly():
    t = QualityThresholds()
    assert t.min_brightness < t.max_brightness
    assert t.min_width > 0 and t.min_height > 0


def test_thresholds_from_config_overrides():
    class Cfg:
        image_min_brightness = 40.0
        image_max_brightness = 200.0
        image_min_blur_score = 30.0
        image_min_width = 64
        image_min_height = 64
        image_min_contrast_std = 9.0
        image_require_labels = True

    t = QualityThresholds.from_config(Cfg())
    assert t.min_brightness == 40.0
    assert t.min_width == 64
    assert t.require_labels is True


def test_thresholds_from_config_tolerates_missing_attrs():
    t = QualityThresholds.from_config(object())
    assert t == QualityThresholds()


def test_thresholds_from_config_none_uses_defaults():
    assert QualityThresholds.from_config(None) == QualityThresholds()


# ------------------------------------------------------------------ acceptance
def test_textured_frame_passes():
    r = check_image(textured(1))
    assert r["valid"] is True
    assert set(r["metrics"]) >= {"brightness", "blur_score", "contrast_std",
                                  "width", "height"}


def test_accepted_result_reports_thresholds_used():
    r = check_image(textured(2))
    assert r["thresholds"]["min_brightness"] == 18.0


def test_grayscale_frame_is_accepted():
    g = textured(3, mean=120)[:, :, 0].copy()
    assert check_image(g)["valid"] is True


# ------------------------------------------------------------------ rejection
def test_dark_frame_rejected():
    r = check_image(textured(4, mean=2, std=1))
    assert r["valid"] is False
    assert r["error_code"] == "IMAGE_TOO_DARK"
    assert r["message"] == MESSAGES["IMAGE_TOO_DARK"]


def test_overexposed_frame_rejected():
    r = check_image(textured(5, mean=253, std=1))
    assert r["error_code"] == "IMAGE_TOO_BRIGHT"


def test_blurred_frame_rejected():
    r = check_image(blurred(textured(6)))
    assert r["valid"] is False
    assert r["error_code"] == "IMAGE_TOO_BLURRY"


def test_flat_frame_rejected_for_lack_of_detail():
    r = check_image(np.full((224, 224, 3), 120, dtype=np.uint8))
    assert r["valid"] is False
    assert r["error_code"] in ("IMAGE_TOO_BLURRY", "IMAGE_NO_DETAIL")


def test_small_frame_rejected():
    r = check_image(textured(7, size=16))
    assert r["error_code"] == "IMAGE_TOO_SMALL"
    assert r["metrics"]["width"] == 16


def test_none_image_rejected():
    r = check_image(None)
    assert r["valid"] is False
    assert r["error_code"] == "IMAGE_INVALID"


def test_structurally_invalid_raises_preprocessing_error():
    """Empty arrays are a caller bug, not a merely poor photo."""
    from ai_disease.preprocessing import PreprocessingError

    with pytest.raises(PreprocessingError):
        check_image(np.array([], dtype=np.uint8))


def test_missing_labels_rejected_when_required():
    t = QualityThresholds(require_labels=True)
    r = check_image(textured(8), t, has_labels=False)
    assert r["error_code"] == "IMAGE_NO_LABELS"


def test_missing_labels_ignored_by_default():
    r = check_image(textured(9), QualityThresholds(), has_labels=False)
    assert r["valid"] is True


def test_every_rejection_carries_actionable_text():
    for code in MESSAGES:
        p = error_payload(code)
        assert p["message"] and p["suggestion"]
        assert p["valid"] is False


def test_first_failing_check_is_reported():
    """A frame that is both dark and blurry reports darkness."""
    r = check_image(blurred(textured(10, mean=2, std=1)))
    assert r["error_code"] == "IMAGE_TOO_DARK"


def test_rejection_includes_metrics_for_diagnosis():
    r = check_image(textured(11, mean=2, std=1))
    assert "brightness" in r["metrics"]
    assert "blur_score" in r["metrics"]
