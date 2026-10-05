"""Optional TFLite backend - NOT enabled by default.

Why this is not implemented yet
-------------------------------
`tflite-runtime` publishes no cp313 aarch64 wheel, and the fallback
(`tensorflow`) pulls in ~600 MB. The module therefore targets ONNX Runtime,
which has a working cp313 manylinux aarch64 wheel.

This file exists so the backend slot is real rather than hypothetical, and so
adding it later is a config change rather than a rewrite. It is not imported by
the default code path and is not covered by the test suite.

If a TFLite model is genuinely needed later: either move this module to a
Python 3.11 virtualenv (where tflite-runtime does publish aarch64 wheels), or
convert the TFLite model to ONNX and use ONNXInferenceEngine.
"""
from __future__ import annotations

from pathlib import Path

from .inference import (
    InferenceEngine,
    InferenceError,
    LabelMismatchError,
    ModelNotFoundError,
)


class TFLiteInferenceEngine(InferenceEngine):
    """Loads a .tflite classifier via tflite_runtime or tensorflow.

    Not exercised by tests; see module docstring.
    """

    name = "tflite"

    def __init__(self, model_path, labels: list[str] | None = None,
                 expected_classes: int | None = None, strict_label_count: bool = True):
        self.model_path = Path(model_path)
        self.labels = list(labels or [])
        self.expected_classes = expected_classes
        self.strict_label_count = strict_label_count
        self.interpreter = None
        self._input = None
        self._output = None
        self._num_classes = None
        self._loaded = False
        self._last_inference_ms = 0.0
        self._runtime = None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise ModelNotFoundError(f"Model file not found: {self.model_path}")

        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore

            self._runtime = "tflite_runtime"
        except ImportError:
            try:
                from tensorflow.lite import Interpreter  # type: ignore

                self._runtime = "tensorflow"
            except ImportError as e:
                raise InferenceError(
                    "TFLite backend unavailable: no tflite-runtime or tensorflow.\n"
                    "On Python 3.13/aarch64 neither is installable. Use the ONNX "
                    "backend, or move this module to Python 3.11."
                ) from e

        try:
            self.interpreter = Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise InferenceError(f"could not load TFLite model: {e}") from e

        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]

        shape = list(self._input.get("shape", []))
        if len(shape) != 4:
            raise InferenceError(f"expected a 4D input tensor, got shape {shape}")
        for dim in shape:
            if isinstance(dim, int) and dim > 1 and dim != shape[0]:
                # First non-batch, non-channel dim that looks like classes.
                pass

        out_shape = list(self._output.get("shape", []))
        for dim in out_shape:
            if isinstance(dim, int) and dim > 1:
                self._num_classes = int(dim)
                break

        if self.labels and self._num_classes:
            if len(self.labels) != self._num_classes and self.strict_label_count:
                raise LabelMismatchError(
                    f"label count mismatch: {len(self.labels)} labels vs "
                    f"{self._num_classes} model output classes"
                )

        self._loaded = True

    def predict(self, tensor) -> list[float]:
        if not self._loaded:
            raise InferenceError("engine not loaded; call load() first")
        import numpy as np

        try:
            # TFLite inputs here are NHWC float32.
            arr = np.asarray(tensor, dtype=np.float32)
            if arr.ndim == 4 and arr.shape[1] == 3:
                arr = np.transpose(arr, (0, 2, 3, 1))
            self.interpreter.set_tensor(self._input["index"], arr)
            self.interpreter.invoke()
            raw = self.interpreter.get_tensor(self._output["index"])[0]
        except Exception as e:
            raise InferenceError(f"TFLite inference failed: {e}") from e

        scores = np.asarray(raw, dtype=np.float64)
        if scores.min() < 0.0 or scores.max() > 1.0 + 1e-6:
            shifted = scores - scores.max()
            exp = np.exp(shifted)
            scores = exp / float(exp.sum())
        return [float(v) for v in scores]

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    @property
    def input_shape(self) -> tuple | None:
        if self._input is None:
            return None
        return tuple(self._input.get("shape", []))

    def describe(self) -> dict:
        d = super().describe()
        d.update({"model": self.model_path.name, "runtime": self._runtime})
        return d