"""Tests for verify_graph.py's value coercion and contract logic.

The verifier must survive Keras's inconsistent representations of layer
attributes: TF 2.20 returns a TrackedList for Rescaling.scale when ImageNet
normalisation uses per-channel values, while older releases return a float.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

VERIFY = ROOT / "verify_graph.py"


def load_verifier():
    """Import verify_graph without executing TF-dependent code.

    The module imports tensorflow at top level, so extract only the helpers
    and exec them in a namespace with numpy available.
    """
    src = VERIFY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"_scalar", "_describe"}
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            keep.append(node)
    assert keep, "helpers not found in verify_graph.py"
    module = ast.Module(body=keep, type_ignores=[])
    ns = {"np": np, "tf": None}
    exec(compile(ast.fix_missing_locations(module), "verify_graph", "exec"), ns)
    return ns["_scalar"], ns["_describe"]


scalar, describe = load_verifier()


# ------------------------------------------------------------- _scalar
def test_scalar_passes_floats_through():
    assert scalar(1.0) == pytest.approx(1.0)
    assert scalar(255.0) == pytest.approx(255.0)
    assert scalar(0) == 0.0


def test_scalar_handles_one_element_list():
    """TF 2.20's TrackedList case: a length-1 list of one scale."""
    assert scalar([1.0 / 255.0]) == pytest.approx(1.0 / 255.0)
    assert scalar((0.5,)) == pytest.approx(0.5)


def test_scalar_returns_none_for_multi_value_containers():
    """A per-channel list has no single scale, so report None, not a guess."""
    assert scalar([0.485, 0.456, 0.406]) is None
    assert scalar((1.0, 2.0)) is None
    assert scalar([]) is None


def test_scalar_handles_tracked_list_subclass():
    """Emulate TrackedList: a list subclass. Must not raise."""
    class TrackedList(list):
        pass

    tl = TrackedList([1.0 / 255.0])
    assert scalar(tl) == pytest.approx(1.0 / 255.0)

    wide = TrackedList([0.485, 0.456, 0.406])
    assert scalar(wide) is None


def test_scalar_handles_numpy_values():
    assert scalar(np.float32(0.5)) == pytest.approx(0.5)
    assert scalar(np.array([0.25])) == pytest.approx(0.25)
    assert scalar(np.array(0.75)) == pytest.approx(0.75)
    assert scalar(np.array([1.0, 2.0])) is None


def test_scalar_handles_nested_numpy():
    assert scalar(np.array([[1.0 / 255.0]])) == pytest.approx(1.0 / 255.0)


def test_scalar_handles_objects_exposing_numpy():
    class FakeVariable:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def numpy(self):
            return self._arr

    assert scalar(FakeVariable(0.5)) == pytest.approx(0.5)
    assert scalar(FakeVariable([0.5])) == pytest.approx(0.5)
    assert scalar(FakeVariable([1.0, 2.0])) is None


def test_scalar_returns_none_for_none_and_garbage():
    assert scalar(None) is None
    assert scalar("not a number") is None
    assert scalar(object()) is None


def test_scalar_never_raises():
    class Hostile:
        def __iter__(self):
            raise RuntimeError("boom")

    assert scalar(Hostile()) is None


# ----------------------------------------------------------- _describe
def test_describe_scalar_value():
    """A scalar prints as a plain number, with 1/255 recognisable."""
    out = describe(1.0 / 255.0)
    assert out.startswith("0.0039")
    assert not out.startswith("[")


def test_describe_multi_value_container():
    out = describe([0.485, 0.456, 0.406])
    assert "0.485" in out
    assert "3 value" in out


def test_describe_tracked_list_does_not_raise():
    class TrackedList(list):
        pass

    assert describe(TrackedList([1.0])) != ""
    assert describe(TrackedList([0.485, 0.456, 0.406])) != ""


def test_describe_none():
    assert describe(None) != ""


# ------------------------------------------------ verifier contract checks
def test_verifier_requires_internal_rescale():
    src = VERIFY.read_text(encoding="utf-8")
    assert "has_div_255" in src
    assert "no Rescaling(1/255)" in src
    assert "EXIT_CONTRACT_VIOLATION" in src


def test_verifier_checks_onnx_bridge():
    src = VERIFY.read_text(encoding="utf-8")
    assert "BRIDGE MISSING" in src
    assert "Rescaling(255)" in src
    assert "--onnx" in src


def test_verifier_checks_numeric_equivalence():
    src = VERIFY.read_text(encoding="utf-8")
    assert "Transform equivalence" in src
    assert "EQUIVALENT" in src
    assert "NOT equivalent" in src


def test_verifier_detects_external_rescale():
    src = VERIFY.read_text(encoding="utf-8")
    assert "external rescale" in src
    assert "applies an external 1/255" in src


def test_verifier_reports_both_rescale_layers():
    src = VERIFY.read_text(encoding="utf-8")
    assert "Rescaling layers found" in src
    assert "for l in rescales" in src
    # Must print every layer, not just the first.
    assert "type {type(l.scale).__name__}" in src


def test_verifier_uses_coercion_everywhere():
    """No bare float(l.scale) anywhere; that is what crashed on TF 2.20."""
    src = VERIFY.read_text(encoding="utf-8")
    assert "float(l.scale)" not in src
    assert "float(l.offset)" not in src
    assert "float(l.mean)" not in src
    assert "_scalar(l.scale)" in src


def test_verifier_checks_nchw_input():
    src = VERIFY.read_text(encoding="utf-8")
    assert "is not NCHW" in src


def test_verifier_does_not_depend_on_training_config_for_input_shape():
    """The ONNX input must match production's fixed (1,3,224,224) contract."""
    src = VERIFY.read_text(encoding="utf-8")
    assert "unit_nchw" in src
    assert "np.transpose(unit, (0, 3, 1, 2))" in src
