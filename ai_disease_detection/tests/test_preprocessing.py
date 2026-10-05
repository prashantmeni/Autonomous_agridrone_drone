"""Test preprocessing: RGB conversion, resize, normalisation, tensor shape."""
import numpy as np
import pytest

from ai_disease.config import PreprocessConfig
from ai_disease.preprocessing import (
    NORMALIZATIONS,
    PreprocessingError,
    ensure_bgr,
    normalize,
    preprocess,
    resize,
    to_rgb,
    validate_normalization,
)

cv2 = pytest.importorskip("cv2")


def test_ensure_bgr_accepts_3channel():
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    assert ensure_bgr(img).shape == (10, 12, 3)


def test_ensure_bgr_accepts_grayscale():
    assert ensure_bgr(np.zeros((10, 12), dtype=np.uint8)).ndim == 2


def test_ensure_bgr_rejects_none():
    with pytest.raises(PreprocessingError, match="None"):
        ensure_bgr(None)


def test_ensure_bgr_rejects_empty():
    with pytest.raises(PreprocessingError, match="empty"):
        ensure_bgr(np.array([], dtype=np.uint8))


def test_ensure_bgr_rejects_bad_shape():
    with pytest.raises(PreprocessingError, match="unexpected image shape"):
        ensure_bgr(np.zeros((4, 4, 5), dtype=np.uint8))


def test_to_rgb_swaps_channels():
    # BGR ordering means a blue pixel has 255 in slot 0. After conversion to
    # RGB that same blue must sit in slot 2.
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255          # blue in BGR
    rgb = to_rgb(bgr)
    assert rgb.shape == (4, 4, 3)
    assert rgb[0, 0, 2] == 255   # blue now in the RGB blue slot
    assert rgb[0, 0, 0] == 0
    assert rgb[0, 0, 1] == 0


def test_to_rgb_red_moves_to_first_slot():
    bgr = np.zeros((4, 4, 3), dtype=np.uint8)
    bgr[:, :, 2] = 255          # red in BGR
    rgb = to_rgb(bgr)
    assert rgb[0, 0, 0] == 255   # red now first in RGB


def test_to_rgb_from_grayscale_expands():
    gray = np.full((4, 4), 128, dtype=np.uint8)
    rgb = to_rgb(gray)
    assert rgb.shape == (4, 4, 3)
    assert (rgb == 128).all()


def test_to_rgb_drops_alpha():
    bgra = np.zeros((4, 4, 4), dtype=np.uint8)
    bgra[:, :, 3] = 255
    assert to_rgb(bgra).shape == (4, 4, 3)


def test_resize_to_target():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = resize(img, 224, 224)
    assert out.shape[:2] == (224, 224)


def test_resize_noop_when_already_correct():
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    assert resize(img, 224, 224) is img


def test_resize_rejects_zero_size():
    with pytest.raises(PreprocessingError, match="invalid target size"):
        resize(np.zeros((10, 10, 3), dtype=np.uint8), 0, 224)


def test_validate_normalization_known_values():
    for kind in NORMALIZATIONS:
        assert validate_normalization(kind) == kind


def test_validate_normalization_rejects_unknown():
    with pytest.raises(PreprocessingError, match="unknown normalization"):
        validate_normalization("scale_to_unit_ball")


def test_normalize_divide_255_maps_to_unit_range():
    cfg = PreprocessConfig(type="divide_255")
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    out = normalize(img, cfg)
    assert out.dtype == np.float32
    assert out[0, 0, 0] == pytest.approx(0.0)
    assert out[0, 0, 1] == pytest.approx(128 / 255.0, abs=1e-6)
    assert out[0, 0, 2] == pytest.approx(1.0)


def test_normalize_none_keeps_raw_range():
    cfg = PreprocessConfig(type="none")
    img = np.array([[[255]]], dtype=np.uint8)
    assert normalize(img, cfg)[0, 0, 0] == pytest.approx(255.0)


def test_normalize_imagenet_centres_around_zero():
    cfg = PreprocessConfig(type="imagenet")
    grey = np.full((2, 2, 3), 128, dtype=np.uint8)
    out = normalize(grey, cfg)
    # A mid-grey image should land near zero after mean subtraction.
    assert abs(float(out.mean())) < 0.6


def test_preprocess_produces_nchw_batch():
    cfg = PreprocessConfig(input_size=(224, 224), type="divide_255")
    img = np.full((480, 640, 3), 120, dtype=np.uint8)
    tensor = preprocess(img, cfg)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    # divide_255 on a uniform 120 image -> ~0.47 everywhere.
    assert 0.4 < float(tensor.mean()) < 0.55


def test_preprocess_from_grayscale_input():
    cfg = PreprocessConfig(input_size=(224, 224))
    tensor = preprocess(np.full((100, 100), 200, dtype=np.uint8), cfg)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_matches_training_pipeline_not_live_script():
    """Confirms we normalise, unlike the reference Live_Inference script."""
    cfg = PreprocessConfig(type="divide_255")
    img = np.full((8, 8, 3), 255, dtype=np.uint8)
    tensor = preprocess(img, cfg)
    # A white image must map to 1.0, not 255.0.
    assert float(tensor.max()) == pytest.approx(1.0)
    assert float(tensor.max()) < 2.0


def test_preprocess_respects_custom_input_size():
    cfg = PreprocessConfig(input_size=(64, 96), type="divide_255")
    tensor = preprocess(np.zeros((200, 200, 3), dtype=np.uint8), cfg)
    assert tensor.shape == (1, 3, 64, 96)
    assert cfg.width == 96 and cfg.height == 64