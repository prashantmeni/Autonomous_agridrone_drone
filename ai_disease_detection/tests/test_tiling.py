"""Sliding-window scanning for wide/aerial frames.

Covers tile geometry, batched tile extraction, per-tile scanning, the
best-tile result aggregation and the fallback to single-shot inference for
frames no larger than one tile.
"""
from typing import ClassVar

import numpy as np
import pytest
from ai_disease.config import Config, PreprocessConfig, TiledConfig
from ai_disease.inference import InferenceEngine
from ai_disease.tiling import build_batch, scan, should_tile, tile_boxes

cv2 = pytest.importorskip("cv2")


class BrightnessEngine(InferenceEngine):
    """Two-class engine whose confidence tracks tile brightness."""

    name = "brightness"
    labels: ClassVar[list[str]] = ["Leaf___disease", "Leaf___healthy"]

    def load(self):
        self._loaded = True

    @property
    def num_classes(self):
        return 2

    def predict(self, tensor):
        mean = float(np.mean(tensor))
        conf = min(0.99, max(0.01, mean * 1.5))
        score = [conf, 1.0 - conf]
        self._last_inference_ms = 1.0
        return score


def tiled_cfg(enabled: bool = True, tile_size: int = 64, stride: int = 64,
              threshold: float = 0.60) -> Config:
    cfg = Config(
        preprocess=PreprocessConfig(type="divide_255"),
        tiled=TiledConfig(enabled=enabled, tile_size=tile_size, stride=stride),
    )
    cfg.inference.confidence_threshold = threshold
    return cfg


def textured_frame(height: int = 90, width: int = 130) -> np.ndarray:
    """Non-flat frame that passes the quality gate; a bright 64x64 block sits
    at the top-left so the first tile is clearly the brightest."""
    rng = np.random.default_rng(3)
    frame = rng.normal(90, 30, (height, width, 3))
    frame[:64, :64] = 255
    return np.clip(frame, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ geometry
def test_tile_boxes_complete_gapless_coverage():
    boxes = tile_boxes(640, 480, 224, 224)
    assert len(boxes) == 9
    assert boxes[0] == (0, 0, 224, 224)
    assert (640, 480) == (boxes[-1][0] + boxes[-1][2], boxes[-1][1] + boxes[-1][3])


def test_tile_boxes_clamps_last_tile_to_edge():
    boxes = tile_boxes(192, 96, 64, 64)
    assert boxes[-1] == (128, 64, 64, 32)
    covered = {(x, y) for (x, y, w, h) in boxes}
    assert (0, 0) in covered and (128, 64) in covered


def test_tile_boxes_overlaps_when_stride_below_tile():
    boxes = tile_boxes(120, 120, 64, 32)
    assert len(boxes) == 16
    assert boxes[0] == (0, 0, 64, 64)
    assert boxes[1] == (32, 0, 64, 64)  # shifted, still full-width tile


def test_tile_boxes_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        tile_boxes(0, 100, 64, 64)
    with pytest.raises(ValueError):
        tile_boxes(100, 100, 0, 64)
    with pytest.raises(ValueError):
        tile_boxes(100, 100, 64, 0)


# -------------------------------------------------------------------- policy
def test_should_tile_true_only_for_frames_larger_than_a_tile():
    cfg = tiled_cfg()
    assert should_tile(np.zeros((130, 90, 3), np.uint8), cfg) is True
    assert should_tile(np.zeros((64, 64, 3), np.uint8), cfg) is False
    assert should_tile(np.zeros((50, 90, 3), np.uint8), cfg) is False  # one axis only
    assert should_tile(np.zeros((90, 50, 3), np.uint8), cfg) is False


# ------------------------------------------------------------------- batches
def test_build_batch_aligns_tiles_and_boxes():
    frame = textured_frame()
    cfg = tiled_cfg()
    tensor, boxes = build_batch(frame, cfg)
    assert tensor.shape == (6, 3, 224, 224)  # 3 cols x 2 rows tiles
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0  # divide_255
    assert len(boxes) == tensor.shape[0]
    assert boxes[0] == (0, 0, 64, 64)


def test_scan_returns_one_result_per_tile_aligned_with_boxes():
    engine = BrightnessEngine()
    engine.load()
    results, boxes = scan(textured_frame(), engine, tiled_cfg())
    assert len(results) == len(boxes) == 6
    for r in results:
        assert 0.0 <= r.confidence <= 1.0
        assert r.class_id == 0
    assert results[0].confidence > results[1].confidence


# ----------------------------------------------------------- tiled pipeline
def test_run_inference_tiled_reports_best_tile_and_grid():
    from ai_disease.pipeline import run_inference_tiled

    engine = BrightnessEngine()
    engine.load()
    result = run_inference_tiled(textured_frame(), engine, tiled_cfg(),
                                 source="aerial")
    assert result.class_name == "Leaf___disease"
    assert result.reliable is True
    assert result.metadata["scanned"] is True
    assert result.metadata["tile_count"] == 6
    tiles = result.metadata["tiles"]
    assert len(tiles) == 6
    best = tiles[0]  # the bright top-left tile wins
    assert best["confidence"] > tiles[1]["confidence"]
    assert best["x"] == 0 and best["y"] == 0
    assert result.total_ms >= result.inference_ms >= 0.0


def test_run_inference_tiled_falls_back_to_single_shot_for_leaf_frames():
    from ai_disease.pipeline import run_inference_tiled

    engine = BrightnessEngine()
    engine.load()
    small = np.clip(np.random.default_rng(5).normal(120, 45, (64, 64, 3)),
                    0, 255).astype(np.uint8)
    result = run_inference_tiled(small, engine, tiled_cfg(), source="camera")
    assert result.metadata.get("scanned") is not True
    assert result.class_name == "Leaf___disease"


def test_run_inference_tiled_still_quality_gates_wide_frames():
    from ai_disease.pipeline import run_inference_tiled

    engine = BrightnessEngine()
    engine.load()
    flat = np.full((130, 90, 3), 127, np.uint8)  # blurry: the gate rejects it
    result = run_inference_tiled(flat, engine, tiled_cfg())
    assert result.class_name == "Unknown"
    assert result.reason and result.reason.startswith("IMAGE_")
    assert result.metadata.get("scanned") is not True


# --------------------------------------------------------------- config load
def test_config_defaults_tiling_off():
    from ai_disease.config import load_config

    cfg = load_config()
    assert cfg.tiled.enabled is False
    assert cfg.tiled.tile_size == 224
    assert cfg.tiled.stride == 224


def test_config_parses_tiled_settings():
    from ai_disease.config import load_config

    cfg = load_config()
    cfg.tiled.enabled = True
    cfg.tiled.stride = 112
    assert cfg.tiled.enabled is True
    assert cfg.tiled.tile_size == cfg.preprocess.width