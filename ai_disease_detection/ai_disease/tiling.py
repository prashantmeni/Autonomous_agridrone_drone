"""Sliding-window scanning for wide aerial frames.

The classifier expects a single close-up leaf roughly filling the 224x224
input. Drone frames show whole plants or many leaves, so one downscale would
destroy the signal. This module scans the frame with tiles sized to the model
input and classifies each one; the best tile backs the reported result while
the rest are kept for display and geo-tagging.

Scanning runs tile-by-tile on the engine's single-input contract: onnxruntime
gains nothing from batching on a Raspberry Pi, and per-tile predicts keep
this compatible with models exported with a fixed batch dimension.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .preprocessing import normalize, to_rgb
from .results import classify


def tile_boxes(width: int, height: int, tile_size: int,
               stride: int) -> list[tuple[int, int, int, int]]:
    """Gap-free coverage: step by `stride`, clamp each last tile to the edge.

    Returns (x, y, w, h) boxes in row-major order (left-to-right, top-to-bottom).
    A frame smaller than the tile in a dimension still yields one tile along
    that axis, so windows stay complete even for odd frame sizes.
    """
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    boxes: list[tuple[int, int, int, int]] = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            boxes.append((x, y, w, h))
            x += stride
        y += stride
    return boxes


def should_tile(image, cfg: Config) -> bool:
    """True when scanning is worthwhile: the frame exceeds one tile.

    Geometric test only — whether to scan is the caller's decision (the drone
    monitor always scans camera frames; uploads scan when `tiled.enabled`).
    """
    h, w = image.shape[:2]
    return w > cfg.tiled.tile_size and h > cfg.tiled.tile_size


def build_batch(image, cfg: Config) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Extract, resize and normalise every tile into one CHW stack.

    Returns (float32 (N, C, H, W) tensor, boxes). The whole frame is converted
    to RGB once; tiles are then resized to the model input, so edge tiles
    shorter than a full tile are handled by the same resize path as a still.
    """
    rgb = to_rgb(image)
    boxes = tile_boxes(rgb.shape[1], rgb.shape[0], cfg.tiled.tile_size,
                       cfg.tiled.stride)
    if not boxes:
        raise ValueError("no tiles produced by current tiling config")
    size = (cfg.preprocess.width, cfg.preprocess.height)
    tiles = np.stack([
        cv2.resize(rgb[y:y + h, x:x + w], size, interpolation=cv2.INTER_LINEAR)
        for (x, y, w, h) in boxes
    ]).astype(np.float32)
    scaled = normalize(tiles, cfg.preprocess)
    return np.transpose(scaled, (0, 3, 1, 2)), boxes


def scan(image, engine, cfg: Config) -> tuple[list, list[tuple[int, int, int, int]]]:
    """Classify every tile of a frame; return (results, boxes).

    `results[i]` is the DetectionResult for `boxes[i]`, index-aligned.
    """
    tensor, boxes = build_batch(image, cfg)
    labels = list(engine.labels) if getattr(engine, "labels", None) else []
    results = []
    for i in range(tensor.shape[0]):
        scores = engine.predict(tensor[i:i + 1])
        results.append(classify(
            scores, labels,
            confidence_threshold=cfg.inference.confidence_threshold,
            healthy_keyword=cfg.inference.healthy_keyword,
        ))
    return results, boxes