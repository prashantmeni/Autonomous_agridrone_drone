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


# --------------------------------------- forward matrix -> op's flat transform
# train.py folds zoom into the forward matrix, inverts the 3x3 homogeneous
# matrix, then flattens it to the (batch, 8) the projective op accepts. That
# step is pure linear algebra, so it is verified here rather than first running
# in Colab.
def forward(theta, phi, scale, tx, ty) -> np.ndarray:
    """train.py's `fwd`: forward 3x3 homogeneous matrix, (batch, 3, 3)."""
    lin = compose(theta, phi)[:, :, :2] / scale[:, None, None]
    n = theta.shape[0]
    fwd = np.zeros((n, 3, 3), dtype=lin.dtype)
    fwd[:, :2, :2] = lin
    fwd[:, 0, 2] = tx
    fwd[:, 1, 2] = ty
    fwd[:, 2, 2] = 1.0
    return fwd


def op_transform(theta, phi, scale, tx, ty) -> np.ndarray:
    """train.py's `transforms`: inverted and flattened, (batch, 8)."""
    return np.linalg.inv(forward(theta, phi, scale, tx, ty)).reshape(-1, 9)[:, :8]


def to_homogeneous(flat: np.ndarray) -> np.ndarray:
    """(n, 8) flat affine form back to (n, 3, 3): append the [0, 0, 1] row."""
    return np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1) \
        .reshape(-1, 3, 3)


def test_op_transform_shape_is_batch_by_8():
    n = 4
    out = op_transform(np.zeros(n), np.zeros(n), np.ones(n),
                       np.zeros(n), np.zeros(n))
    assert out.shape == (n, 8)


def test_op_transform_is_identity_when_every_param_is_zero():
    m = op_transform(np.array([0.0]), np.array([0.0]), np.array([1.0]),
                     np.array([0.0]), np.array([0.0]))
    assert np.allclose(m[0], [1, 0, 0, 0, 1, 0, 0, 0], atol=1e-12)


def test_op_transform_is_the_exact_inverse_of_forward():
    """The op maps output coords -> input coords, the opposite of `fwd`."""
    theta = np.array([0.3, -0.7])
    phi = np.array([0.1, -0.2])
    scale = np.array([0.9, 1.0])
    tx = np.array([10.0, -5.0])
    ty = np.array([-3.0, 7.0])
    homog = to_homogeneous(op_transform(theta, phi, scale, tx, ty))
    assert np.allclose(homog @ forward(theta, phi, scale, tx, ty),
                       np.eye(3)[None], atol=1e-9)


def test_zoom_less_than_one_zooms_in():
    """config.yaml documents scale < 1 as zoom IN: the op must sample closer
    to the origin, so its linear part has magnitude `scale`, not 1/`scale`."""
    m = op_transform(np.array([0.0]), np.array([0.0]), np.array([0.5]),
                     np.array([0.0]), np.array([0.0]))
    assert np.allclose(to_homogeneous(m)[0, :2, :2], 0.5 * np.eye(2),
                       atol=1e-12)


def test_translation_is_negated_by_the_inversion():
    tx, ty = 12.0, -4.0
    m = op_transform(np.array([0.0]), np.array([0.0]), np.array([1.0]),
                     np.array([tx]), np.array([ty]))
    assert np.allclose(m[0, [2, 5]], [-tx, -ty], atol=1e-12)


def test_inverted_rotation_is_still_a_pure_rotation():
    """Inversion must not introduce scaling creep into a rotation."""
    theta = np.deg2rad(37.0)
    m = op_transform(np.array([theta]), np.zeros(1), np.ones(1),
                     np.zeros(1), np.zeros(1))
    lin = to_homogeneous(m)[0, :2, :2]
    expected = np.linalg.inv(
        np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]))
    assert np.allclose(lin, expected, atol=1e-9)
    assert np.allclose(lin @ lin.T, np.eye(2), atol=1e-9)


@pytest.mark.parametrize("deg", [1.0, 40.0, 90.0, 179.0])
def test_op_transform_never_produces_nan(deg):
    m = op_transform(np.array([np.deg2rad(deg)]), np.array([np.deg2rad(20.0)]),
                     np.array([0.8]), np.array([15.0]), np.array([-15.0]))
    assert np.isfinite(m).all()
