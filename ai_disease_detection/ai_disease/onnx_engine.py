"""ONNX Runtime backend - the primary engine for Raspberry Pi (Python 3.13).

onnxruntime has a cp313 manylinux aarch64 wheel; tflite-runtime does not.
See README "Runtime decision".
"""
from __future__ import annotations

import os
from pathlib import Path

from .inference import (
    InferenceEngine,
    InferenceError,
    LabelMismatchError,
    ModelNotFoundError,
    timed_run,
)


class ONNXInferenceEngine(InferenceEngine):
    """Loads a .onnx classifier and returns per-class probabilities.

    The model is inspected rather than assumed: input rank and channel layout
    are read from the graph, and the output width is cross-checked against the
    label count.
    """

    name = "onnx"

    def __init__(self, model_path, labels: list[str] | None = None,
                 expected_classes: int | None = None, strict_label_count: bool = True):
        self.model_path = Path(model_path)
        self.labels = list(labels or [])
        self.expected_classes = expected_classes
        self.strict_label_count = strict_label_count

        self.session = None
        self._input = None
        self._output = None
        self._input_rank = None
        self._layout = "NCHW"
        self._num_classes = None
        self._loaded = False
        self._last_inference_ms = 0.0

    # ------------------------------------------------------------------- load
    def load(self) -> None:
        if not self.model_path.is_file():
            raise ModelNotFoundError(
                f"Model file not found: {self.model_path}\n\n"
                "The reference repository does not contain trained model weights.\n"
                "Please provide a converted ONNX model before running real inference."
            )
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise InferenceError(
                "onnxruntime is not installed. Install it with: "
                "pip install onnxruntime"
            ) from e

        opts = ort.SessionOptions()
        # 3 = errors only; keeps the Raspberry Pi console readable.
        opts.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), opts, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            raise InferenceError(
                f"onnxruntime could not load {self.model_path.name}: {e}"
            ) from e

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs:
            raise InferenceError("model exposes no input tensor")
        if not outputs:
            raise InferenceError("model exposes no output tensor")

        self._input = inputs[0]
        self._output = outputs[0]
        self._inspect_input()
        self._inspect_output()
        self._check_labels()
        self._loaded = True

    def _inspect_input(self) -> None:
        shape = list(self._input.shape or [])
        self._input_rank = len(shape)
        self._input_shape = tuple(shape)

        if self._input_rank == 4:
            # NCHW -> channels at axis 1; NHWC -> channels at axis 3.
            if isinstance(shape[1], int) and isinstance(shape[3], int):
                self._layout = "NCHW"
            elif isinstance(shape[3], int):
                self._layout = "NHWC"
        elif self._input_rank != 3:
            raise InferenceError(
                f"expected a 3D or 4D input tensor, got rank {self._input_rank} "
                f"with shape {shape}"
            )

    def _inspect_output(self) -> None:
        shape = list(self._output.shape or [])
        for dim in shape:
            # Skip the batch axis; the first other concrete dim is the class count.
            if isinstance(dim, int) and dim > 1:
                self._num_classes = int(dim)
                break

    def _check_labels(self) -> None:
        if not self.labels:
            return
        if self._num_classes is None:
            # Dynamic output shape: cannot verify, so do not pretend to.
            return
        if len(self.labels) != self._num_classes:
            message = (
                f"label count mismatch: {len(self.labels)} labels vs "
                f"{self._num_classes} model output classes"
            )
            if self.strict_label_count:
                raise LabelMismatchError(message)
        if self.expected_classes and self._num_classes != self.expected_classes:
            raise LabelMismatchError(
                f"model outputs {self._num_classes} classes but the reference "
                f"architecture expects {self.expected_classes}"
            )

    # -------------------------------------------------------------- inference
    def predict(self, tensor) -> list[float]:
        if not self._loaded or self.session is None:
            raise InferenceError("engine not loaded; call load() first")
        try:
            feed = self._prepare(tensor)
            raw = self.session.run([self._output.name], {self._input.name: feed})[0]
        except InferenceError:
            raise
        except Exception as e:
            raise InferenceError(f"inference failed: {e}") from e
        return self._scores(raw)

    def _prepare(self, tensor):
        import numpy as np

        arr = np.asarray(tensor, dtype=np.float32)
        if self._input_rank == 4 and arr.ndim == 3:
            arr = arr[np.newaxis, ...]  # add batch
        if self._input_rank == 4 and self._layout == "NHWC" and arr.ndim == 4:
            # Preprocessing yields NCHW; transpose only if the graph wants NHWC.
            arr = np.transpose(arr, (0, 2, 3, 1))
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _scores(self, raw) -> list[float]:
        import numpy as np

        arr = np.asarray(raw)
        arr = arr.reshape(-1)
        if arr.size == 0:
            raise InferenceError("model returned an empty output")
        if not np.all(np.isfinite(arr)):
            raise InferenceError("model output contains non-finite values")
        if arr.min() < 0.0 or arr.max() > 1.0 + 1e-6:
            # Raw logits rather than probabilities; softmax so downstream
            # confidence filtering compares against a 0-1 scale.
            shifted = arr.astype(np.float64) - float(arr.max())
            exp = np.exp(shifted)
            arr = exp / float(exp.sum())
        return [float(v) for v in arr]

    # --------------------------------------------------------------- metadata
    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    @property
    def input_shape(self) -> tuple | None:
        return getattr(self, "_input_shape", None)

    @property
    def layout(self) -> str:
        return self._layout

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "model": self.model_path.name,
            "layout": self._layout,
            "output": self._output.name if self._output else None,
            "labels": len(self.labels),
            "runtime_version": self._runtime_version(),
        })
        return d

    def _runtime_version(self) -> str:
        try:
            import onnxruntime as ort

            return getattr(ort, "__version__", "unknown")
        except Exception:
            return "unavailable"

    def run(self, tensor) -> tuple[list[float], float]:
        """Convenience: predict() with latency measurement."""
        return timed_run(self, tensor)