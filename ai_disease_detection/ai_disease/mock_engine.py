"""Deterministic mock engine for tests.

Lets the whole pipeline be exercised with no model file present. It produces
*plausible scores*, not real disease predictions - results from this engine say
nothing about recognition accuracy.
"""
from __future__ import annotations

from .inference import InferenceEngine, timed_run

# Recognisable PlantVillage-style names so tests can assert on healthy/disease
# handling without a real model.
MOCK_LABELS = [
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Apple___healthy",
    "Corn_(maize)___Common_rust",
]


class MockInferenceEngine(InferenceEngine):
    """Returns a fixed or caller-supplied probability vector."""

    name = "mock"

    def __init__(self, num_classes: int = 4, scores: list[float] | None = None,
                 fail_on_predict: bool = False, labels: list[str] | None = None):
        self._n = num_classes
        self._scores = list(scores) if scores else self._default_scores(num_classes)
        self.fail_on_predict = fail_on_predict
        # Carries labels so the shared pipeline is exercised end to end without
        # a real model, exactly as ONNXInferenceEngine does. When more classes
        # are requested than MOCK_LABELS defines, pad with indexed names so
        # every score index still resolves to a label.
        if labels:
            self.labels = list(labels)
        elif num_classes <= len(MOCK_LABELS):
            self.labels = list(MOCK_LABELS[:num_classes])
        else:
            self.labels = list(MOCK_LABELS) + [
                f"class_{i}" for i in range(len(MOCK_LABELS), num_classes)
            ]
        self._loaded = False
        self._last_inference_ms = 0.0
        self.call_count = 0

    @staticmethod
    def _default_scores(n: int) -> list[float]:
        # Decaying distribution that sums to ~1 and puts the top at ~0.6.
        raw = [0.6 ** i for i in range(n)]
        total = sum(raw)
        return [v / total for v in raw]

    def load(self) -> None:
        self._loaded = True

    def predict(self, tensor) -> list[float]:
        if not self._loaded:
            from .inference import InferenceError

            raise InferenceError("mock engine not loaded; call load() first")
        if self.fail_on_predict:
            from .inference import InferenceError

            raise InferenceError("mock engine configured to fail")
        # Touch the tensor so callers see that it is genuinely consumed.
        if tensor is not None and getattr(tensor, "size", 0) == 0:
            from .inference import InferenceError

            raise InferenceError("empty tensor passed to mock engine")
        self.call_count += 1
        return list(self._scores)

    @property
    def num_classes(self) -> int | None:
        return self._n

    @property
    def input_shape(self) -> tuple | None:
        return (1, 3, 224, 224)

    def run(self, tensor) -> tuple[list[float], float]:
        return timed_run(self, tensor)