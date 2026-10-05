"""Image preprocessing for the disease classifier.

Follows the reference repository's *training* pipeline (project.py used
Keras `image_size=(224,224)`, which normalises to [0,1]), deliberately
ignoring the raw 0-255 handling in its Live_Inference script.
"""
from __future__ import annotations

import numpy as np

from .config import PreprocessConfig

# Valid normalisation modes.
NORMALIZATIONS = ("divide_255", "imagenet", "none")


class PreprocessingError(ValueError):
    """Raised when an image cannot be preprocessed."""


def validate_normalization(kind: str) -> str:
    kind = (kind or "divide_255").strip().lower()
    if kind not in NORMALIZATIONS:
        raise PreprocessingError(
            f"unknown normalization type {kind!r}; expected one of {NORMALIZATIONS}"
        )
    return kind


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Validate an input frame is a usable BGR or grayscale array.

    Does not convert: the BGR->RGB step happens in preprocess() so this only
    guards shape/type problems.
    """
    if image is None:
        raise PreprocessingError("image is None")
    if not isinstance(image, np.ndarray):
        raise PreprocessingError(f"expected numpy.ndarray, got {type(image).__name__}")
    if image.size == 0:
        raise PreprocessingError("image is empty")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return image
    raise PreprocessingError(f"unexpected image shape {image.shape}; expected HxW, HxWx3 or HxWx4")


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV convention) to RGB. Grayscale is expanded."""
    import cv2

    image = ensure_bgr(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to the model's expected (width, height)."""
    import cv2

    image = ensure_bgr(image)
    if width <= 0 or height <= 0:
        raise PreprocessingError(f"invalid target size {height}x{width}")
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def normalize(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Scale to float32 using the configured normalisation."""
    kind = validate_normalization(cfg.type)
    x = image.astype(np.float32)
    if kind == "divide_255":
        return x / 255.0
    if kind == "imagenet":
        mean = np.asarray(cfg.imagenet_mean, dtype=np.float32)
        std = np.asarray(cfg.imagenet_std, dtype=np.float32)
        return (x / 255.0 - mean) / std
    return x  # "none"


def preprocess(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Full path: BGR -> RGB -> resize -> float32 -> normalise -> NCHW batch.

    Returns a float32 array shaped (1, C, H, W), which is what an ONNX model
    exported from a Keras NHWC input expects.
    """
    rgb = to_rgb(image)
    resized = resize(rgb, cfg.width, cfg.height)
    scaled = normalize(resized, cfg)
    # HWC -> CHW, then add the batch dimension.
    return np.transpose(scaled, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def load_image(path) -> np.ndarray:
    """Read an image file as BGR. Raises PreprocessingError when undecodable."""
    import cv2

    p = str(path)
    image = cv2.imread(p, cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessingError(f"could not decode image: {p}")
    return image


def decode_image(data: bytes) -> np.ndarray:
    """Decode in-memory image bytes as BGR.

    For uploaded files, where there is no path on disk. Returns None when the
    bytes are not a decodable image, so callers can report a bad upload
    without treating it as an internal error.
    """
    import cv2

    if not data:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return image