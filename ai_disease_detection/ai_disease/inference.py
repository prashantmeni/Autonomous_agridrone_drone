"""Model-independent inference interface.

The engine takes a preprocessed float32 tensor and returns class scores. It
knows nothing about OpenCV, cameras, or flight control.

Backends:
  MockInferenceEngine  - deterministic, for tests with no model file
  ONNXInferenceEngine  - onnxruntime (primary, works on Python 3.13/aarch64)
  TFLiteInferenceEngine- declared only; see notes in the class docstring
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod


class InferenceError(RuntimeError):
    """Raised when a model cannot be loaded or an inference run fails."""


class ModelNotFoundError(InferenceError):
    """Raised when the configured model file is absent."""


class LabelMismatchError(InferenceError):
    """Raised when label count does not match the model's output classes."""


class InferenceEngine(ABC):
    """Contract every backend implements."""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Load weights and validate them. Raise on failure."""

    @abstractmethod
    def predict(self, tensor) -> list[float]:
        """Run inference on a preprocessed tensor, returning class scores."""

    # ---------------------------------------------------------------- metadata
    @property
    @abstractmethod
    def num_classes(self) -> int | None:
        """Number of output classes, or None if not determinable."""

    @property
    def input_shape(self) -> tuple | None:
        return None

    @property
    def last_inference_ms(self) -> float:
        """Latency of the most recent predict() call, in milliseconds."""
        return float(getattr(self, "_last_inference_ms", 0.0))

    @property
    def is_loaded(self) -> bool:
        return bool(getattr(self, "_loaded", False))

    def describe(self) -> dict:
        return {
            "engine": self.name,
            "loaded": self.is_loaded,
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
        }


def timed_run(engine: InferenceEngine, tensor) -> tuple[list[float], float]:
    """Run predict() and record wall-clock latency on the engine."""
    t0 = time.perf_counter()
    scores = engine.predict(tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    engine._last_inference_ms = elapsed_ms
    return scores, elapsed_ms


def create_engine(backend: str, model_path, labels: list[str],
                  expected_classes: int = 38, strict_label_count: bool = True):
    """Build and load an engine. Kept here so callers need no backend imports."""
    kind = (backend or "onnx").strip().lower()

    if kind == "mock":
        from .mock_engine import MockInferenceEngine

        engine = MockInferenceEngine(num_classes=len(labels) or expected_classes)
        engine.load()
        return engine

    if kind == "onnx":
        from .onnx_engine import ONNXInferenceEngine

        engine = ONNXInferenceEngine(
            model_path, labels, expected_classes=expected_classes,
            strict_label_count=strict_label_count,
        )
        engine.load()
        return engine

    if kind == "tflite":
        # Declared for future use. Not implemented in this module: on Python
        # 3.13/aarch64 there is no tflite-runtime wheel, so shipping it would
        # mean pulling in full TensorFlow, which the module avoids deliberately.
        from .tflite_engine import TFLiteInferenceEngine

        engine = TFLiteInferenceEngine(
            model_path, labels, expected_classes=expected_classes,
            strict_label_count=strict_label_count,
        )
        engine.load()
        return engine

    raise InferenceError(
        f"unknown backend {backend!r}; expected onnx, mock or tflite"
    )