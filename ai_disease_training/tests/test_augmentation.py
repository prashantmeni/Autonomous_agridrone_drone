"""Augmentation math in train.py.

TF is absent on Windows, so these skip here and run in Colab, which is where
training actually happens. The rotation test exists because an earlier version
wrote the sampled angle directly into the affine matrix, making rotation and
shear silent no-ops that still passed every other test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

tf = pytest.importorskip("tensorflow", reason="TensorFlow runs in Colab only")

import train  # noqa: E402

SIZE = 32


def cfg(**aug) -> dict:
    base = {"rotation_range": 0.0, "width_shift_range": 0.0,
            "height_shift_range": 0.0, "shear_range": 0.0, "zoom_range": 0.0,
            "horizontal_flip": False}
    base.update(aug)
    return {"training": {"augmentation": base}}


def frames(n=2, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8) for _ in range(n)]


def dataset_for(images):
    return tf.data.Dataset.from_tensor_slices(
        (np.stack(images).astype(np.uint8),
         np.zeros(len(images), np.float32))
    ).batch(1)


def run(images, **aug):
    out = list(train.augment_batches(dataset_for(images), cfg(**aug),
                                     SIZE, SIZE))
    return [o[0].numpy() for o in out]


def test_augmentation_runs_without_name_error():
    """Regression: np.pi was referenced with no numpy import."""
    out = run(frames(4), rotation_range=15.0, width_shift_range=0.1,
              height_shift_range=0.1, shear_range=10.0, zoom_range=0.2,
              horizontal_flip=True)
    assert len(out) == 4


def test_rotation_actually_moves_pixels():
    base = frames(2)
    out = run(base, rotation_range=25.0)
    assert sum(1 for b, a in zip(base, out) if not (b == a).all()) == 2, \
        "rotation left the image untouched"


def test_zero_params_is_an_identity_transform():
    base = frames(2)
    out = run(base)
    for b, a in zip(base, out):
        assert (b == a).all(), "zero params altered pixels"


def test_translation_alone_moves_content():
    base = frames(2)
    out = run(base, width_shift_range=0.2, height_shift_range=0.2)
    assert sum(1 for b, a in zip(base, out) if not (b == a).all()) == 2


def test_shear_alone_moves_content():
    base = frames(2)
    out = run(base, shear_range=20.0)
    assert sum(1 for b, a in zip(base, out) if not (b == a).all()) == 2


def test_output_stays_uint8():
    out = run(frames(2), rotation_range=30.0, width_shift_range=0.2,
              height_shift_range=0.2, shear_range=20.0, zoom_range=0.3,
              horizontal_flip=True)
    for arr in out:
        assert arr.dtype == np.uint8


def test_shape_is_preserved():
    out = run(frames(2), rotation_range=10.0, shear_range=5.0, zoom_range=0.1)
    for arr in out:
        assert arr.shape == (SIZE, SIZE, 3)


def test_rotation_matrix_layout_is_batch_by_2_by_3():
    """Guards the matrix layout the rotation fix depends on."""
    theta = tf.constant([0.1, 0.2], dtype=tf.float32)
    cos_t, sin_t = tf.math.cos(theta), tf.math.sin(theta)
    zeros = tf.zeros_like(theta)
    m = tf.stack([
        tf.stack([cos_t, -sin_t, zeros], axis=-1),
        tf.stack([sin_t, cos_t, zeros], axis=-1),
    ], axis=1)
    assert tuple(m.shape) == (2, 2, 3)
    # A zero angle must build the identity, so compare against tf.repeat of
    # tf.eye. tf.repeat, not Tensor.repeat: the latter is numpy-only and does
    # not exist on an EagerTensor.
    ident = tf.repeat(tf.eye(2)[None], 2, axis=0)
    zero = tf.zeros([2], dtype=tf.float32)
    built = tf.stack([
        tf.stack([tf.math.cos(zero), -tf.math.sin(zero), zero], axis=-1),
        tf.stack([tf.math.sin(zero), tf.math.cos(zero), zero], axis=-1),
    ], axis=1)
    assert np.allclose(built.numpy(), ident.numpy(), atol=1e-6)
