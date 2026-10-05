"""Affine matrix algebra used by train.py's augmentation.

Pure NumPy, so it runs without TensorFlow. TF is absent on Windows and present
in Colab, but the composition R @ S is plain linear algebra: verifying it here
means the Colab smoke test is not the first time this math runs.

Mirrors the tf.stack composition in train.py._augment exactly.
"""
from __future__ import annotations

import numpy as np
import pytest


def compose(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotation by theta composed with shear by phi -> (batch, 2, 3)."""
    cos_t, sin_t, tan_p = np.cos(theta), np.sin(theta), np.tan(phi)
    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)
    rotation = np.stack([
        np.stack([cos_t, -sin_t, zeros], axis=-1),
        np.stack([sin_t, cos_t, zeros], axis=-1),
    ], axis=1)
    shear = np.stack([
        np.stack([ones, tan_p, zeros], axis=-1),
        np.stack([zeros, ones, zeros], axis=-1),
    ], axis=1)
    a = np.stack([
        np.stack([
            rotation[:, 0, 0] * shear[:, 0, 0] + rotation[:, 0, 1] * shear[:, 1, 0],
            rotation[:, 0, 0] * shear[:, 0, 1] + rotation[:, 0, 1] * shear[:, 1, 1],
            zeros,
        ], axis=-1),
        np.stack([
            rotation[:, 1, 0] * shear[:, 0, 0] + rotation[:, 1, 1] * shear[:, 1, 0],
            rotation[:, 1, 0] * shear[:, 0, 1] + rotation[:, 1, 1] * shear[:, 1, 1],
            zeros,
        ], axis=-1),
    ], axis=1)
    return a


def test_shape_is_batch_2_3():
    theta = np.array([0.1, 0.2, -0.3])
    phi = np.array([0.0, 0.1, -0.1])
    assert compose(theta, phi).shape == (3, 2, 3)


def test_zero_angles_give_identity():
    m = compose(np.zeros(2), np.zeros(2))
    assert np.allclose(m[:, :, :2], np.eye(2)[None], atol=1e-12)
    assert np.allclose(m[:, :, 2], 0.0)


def test_no_shear_is_a_pure_rotation():
    """The regression: raw angles must not land in the matrix."""
    theta = np.array([0.0])
    m = compose(theta, np.zeros(1))
    assert np.allclose(m[0, :, :2],
                       [[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]],
                       atol=1e-12)


def test_ninety_degree_rotation():
    m = compose(np.array([np.pi / 2]), np.zeros(1))
    expected = np.array([[0.0, -1.0], [1.0, 0.0]])
    assert np.allclose(m[0, :, :2], expected, atol=1e-9)


def test_rotation_is_not_a_no_op():
    """A 25-degree rotation must differ from identity."""
    m = compose(np.array([np.deg2rad(25.0)]), np.zeros(1))
    assert not np.allclose(m[0, :, :2], np.eye(2), atol=1e-3)


def test_no_rotation_is_a_pure_shear():
    m = compose(np.zeros(1), np.array([np.deg2rad(20.0)]))
    assert np.allclose(m[0, :, :2], [[1.0, np.tan(np.deg2rad(20.0))], [0.0, 1.0]],
                       atol=1e-12)


def test_composition_equals_explicit_matrix_product():
    """Cross-check against a hand-built R @ S."""
    theta = np.array([0.3])
    phi = np.array([-0.15])
    cos_t, sin_t, tan_p = np.cos(theta[0]), np.sin(theta[0]), np.tan(phi[0])
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    S = np.array([[1.0, tan_p], [0.0, 1.0]])
    assert np.allclose(compose(theta, phi)[0, :, :2], R @ S, atol=1e-12)


def test_rotation_preserves_length():
    """A rotation must not stretch the image."""
    theta = np.array([np.deg2rad(37.0)])
    m = compose(theta, np.zeros(1))[0, :, :2]
    assert np.allclose(m @ m.T, np.eye(2), atol=1e-9)


def test_batch_elements_are_independent():
    theta = np.array([0.0, 0.5, -0.5])
    phi = np.zeros(3)
    m = compose(theta, phi)
    assert np.allclose(m[0, :, :2], np.eye(2), atol=1e-12)
    assert not np.allclose(m[1, :, :2], m[2, :, :2], atol=1e-6)


@pytest.mark.parametrize("deg", [1.0, 15.0, 90.0, 179.0])
def test_no_angle_produces_nan(deg):
    m = compose(np.array([np.deg2rad(deg)]), np.array([0.0]))
    assert np.isfinite(m).all()
