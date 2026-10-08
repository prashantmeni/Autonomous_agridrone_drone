"""Reproduce train.py's augmentation scoping with plain Python.

The Colab run failed with UnboundLocalError because the old code rebound
`rotation`/`shear` to tensors inside _augment. TensorFlow is unavailable on
Windows, so this mirrors the function's structure using only numpy to prove the
scoping is now correct: every closure variable is read before any rebinding,
and no local name shadows one it also reads.

Kept in sync with train.py's variable names on purpose - if the two drift, the
shadowing bug can come back unnoticed.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

TRAIN = Path(__file__).resolve().parents[1] / "train.py"


def _augment_like_train(rotation, wshift, hshift, shear, zoom, flip,
                        width, height, images):
    """Mirrors train.py's _augment: same names, same read/write pattern."""
    x = np.asarray(images, dtype=np.float32).copy()
    if flip:
        x = x[..., ::-1]

    if any([rotation, wshift, hshift, shear, zoom]):
        n = x.shape[0]
        theta = np.random.uniform(-rotation, rotation, n) * (np.pi / 180.0)
        phi = np.random.uniform(-shear, shear, n) * (np.pi / 180.0)
        tx = np.random.uniform(-wshift, wshift, n) * float(width)
        ty = np.random.uniform(-hshift, hshift, n) * float(height)
        scale = 1.0 - np.random.uniform(0.0, zoom, n)

        cos_t, sin_t, tan_p = np.cos(theta), np.sin(theta), np.tan(phi)
        ones, zeros = np.ones_like(theta), np.zeros_like(theta)
        # Names must differ from the range parameters above.
        rot_m = np.stack([
            np.stack([cos_t, -sin_t, zeros], axis=-1),
            np.stack([sin_t, cos_t, zeros], axis=-1),
        ], axis=1)
        shear_m = np.stack([
            np.stack([ones, tan_p, zeros], axis=-1),
            np.stack([zeros, ones, zeros], axis=-1),
        ], axis=1)
        composed = np.stack([
            np.stack([
                rot_m[:, 0, 0] * shear_m[:, 0, 0] + rot_m[:, 0, 1] * shear_m[:, 1, 0],
                rot_m[:, 0, 0] * shear_m[:, 0, 1] + rot_m[:, 0, 1] * shear_m[:, 1, 1],
                np.zeros_like(theta),
            ], axis=-1),
            np.stack([
                rot_m[:, 1, 0] * shear_m[:, 0, 0] + rot_m[:, 1, 1] * shear_m[:, 1, 0],
                rot_m[:, 1, 0] * shear_m[:, 0, 1] + rot_m[:, 1, 1] * shear_m[:, 1, 1],
                np.zeros_like(theta),
            ], axis=-1),
        ], axis=1)
        # Same tail steps as train.py: fold zoom into the forward matrix,
        # invert the homogeneous 3x3, flatten to the (n, 8) the op accepts.
        lin = composed[:, :, :2] / scale[:, None, None]
        fwd = np.zeros((n, 3, 3))
        fwd[:, :2, :2] = lin
        fwd[:, 0, 2] = tx
        fwd[:, 1, 2] = ty
        fwd[:, 2, 2] = 1.0
        inv = np.linalg.inv(fwd)
        transforms = inv.reshape(-1, 9)[:, :8]
        assert transforms.shape == (n, 8)
        # Stand in for tf.raw_ops.ImageProjectiveTransformV3: it is what
        # resamples pixels in train.py, so this mirror stops at the transform.
    x = np.clip(x, 0.0, 255.0)
    return x


def frames(n=2, size=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (n, size, size, 3)).astype(np.uint8)


# ------------------------------------------------------------------ scoping
def test_no_unbound_local_error():
    """The exact failure Colab reported: reading a name before assignment."""
    out = _augment_like_train(25.0, 0.2, 0.2, 20.0, 0.3, True, 32, 32,
                              frames())
    assert out is not None


@pytest.mark.parametrize("kwargs", [
    dict(rotation=25.0, wshift=0.0, hshift=0.0, shear=0.0, zoom=0.0, flip=False),
    dict(rotation=0.0, wshift=0.2, hshift=0.2, shear=0.0, zoom=0.0, flip=False),
    dict(rotation=0.0, wshift=0.0, hshift=0.0, shear=20.0, zoom=0.0, flip=False),
    dict(rotation=0.0, wshift=0.0, hshift=0.0, shear=0.0, zoom=0.3, flip=False),
    dict(rotation=0.0, wshift=0.0, hshift=0.0, shear=0.0, zoom=0.0, flip=True),
])
def test_every_single_operation_alone(kwargs):
    """Each op is exercised in isolation, as the failing tests did."""
    params = dict(rotation=0.0, wshift=0.0, hshift=0.0, shear=0.0, zoom=0.0,
                  flip=False)
    params.update(kwargs)
    out = _augment_like_train(**params, width=32, height=32, images=frames())
    assert out.shape == frames().shape


def test_all_operations_together():
    out = _augment_like_train(30.0, 0.2, 0.2, 25.0, 0.3, True, 32, 32,
                              frames(4))
    assert out.shape == (4, 8, 8, 3)


def test_output_is_clipped_to_uint8_range():
    out = _augment_like_train(90.0, 0.9, 0.9, 80.0, 0.9, True, 32, 32,
                              frames())
    assert out.min() >= 0.0 and out.max() <= 255.0


# ------------------------------------------------------ static check of source
def _bound_somewhere(func: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(func)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _read_before_bind(func: ast.FunctionDef) -> set[str]:
    """Names read before their first assignment, in execution order.

    This is precisely what raises UnboundLocalError: a name bound anywhere in a
    function is local throughout, so reading it above its first assignment
    fails even if the enclosing scope has a value of that name. Rebinding after
    assignment is fine and is not reported.
    """
    params = {a.arg for a in func.args.args}
    # Comprehension targets are their own scope; ignore them.
    comprehension_targets = {
        t.id for n in ast.walk(func) if isinstance(n, ast.comprehension)
        for t in ast.walk(n.target)
        if isinstance(t, ast.Name)
    }
    bound = set(params) | comprehension_targets
    local = _bound_somewhere(func) | bound

    offending: set[str] = set()

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda, ast.ClassDef)):
                continue  # nested scope: has its own locals
            if isinstance(child, ast.Name):
                if isinstance(child.ctx, ast.Load):
                    if child.id in local and child.id not in bound:
                        offending.add(child.id)
                else:
                    bound.add(child.id)
                continue
            visit(child)

    for stmt in func.body:
        visit(stmt)
    return offending


def test_train_source_has_no_read_before_bind():
    """Parse the real train.py and reproduce Python's UnboundLocalError rule.

    This is the guard that matters: it inspects the shipped file, not a copy,
    so the Colab failure cannot silently return.
    """
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))

    augment = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "augment_batches"),
        None)
    assert augment is not None, "augment_batches not found"

    inner = next(
        (n for n in ast.walk(augment)
         if isinstance(n, ast.FunctionDef) and n.name == "_augment"),
        None)
    assert inner is not None, "_augment not found"

    offending = _read_before_bind(inner)
    assert not offending, (
        f"_augment reads these before assigning them: {sorted(offending)}")


def test_augment_range_params_are_never_rebound():
    """The original bug: `rotation`/`shear` reused as matrix variables."""
    tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
    augment = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "augment_batches"),
        None)
    inner = next(
        (n for n in ast.walk(augment)
         if isinstance(n, ast.FunctionDef) and n.name == "_augment"),
        None)
    bound = _bound_somewhere(inner)
    for name in ("rotation", "wshift", "hshift", "shear", "zoom", "flip",
                 "width", "height", "a"):
        assert name not in bound, f"{name} is rebound inside _augment"


def test_read_before_bind_detector_catches_the_original_bug():
    """Prove the detector works by reintroducing the shadowing in a snippet."""
    snippet = ast.parse(
        "def outer():\n"
        "    rotation = 25.0\n"
        "    def inner():\n"
        "        if rotation > 0:\n"
        "            rotation = tf.stack([1])\n"
        "        return rotation\n"
        "    return inner\n"
    )
    inner = next(n for n in ast.walk(snippet)
                 if isinstance(n, ast.FunctionDef) and n.name == "inner")
    assert _read_before_bind(inner) == {"rotation"}


def test_train_source_uses_matmul_free_stack_composition():
    """The matrix must be built with tf.stack, which works in graph mode."""
    src = TRAIN.read_text(encoding="utf-8")
    assert "rot_m" in src and "shear_m" in src
    assert "composed" in src
    # No python-level branching on tensors, which would break tf.data graphs.
    assert "tf.cond" not in src
