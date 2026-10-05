"""Test engines, config, camera fallback, and the missing-model path.

These tests use MockInferenceEngine. They verify plumbing only - they do NOT
validate disease recognition, because no trained model is present.
"""
import numpy as np
import pytest

from ai_disease.config import load_config


def realistic_frame(seed: int = 0, size: int = 224) -> np.ndarray:
    """A textured, well-exposed frame that passes the quality gate.

    run_inference() now gates on frame quality before the model, so tests
    exercising classification need a frame a real camera could produce. A flat
    grey fill is rejected as IMAGE_TOO_BLURRY (no detail to measure).
    """
    rng = np.random.default_rng(seed)
    frame = rng.normal(120, 45, (size, size, 3))
    return np.clip(frame, 0, 255).astype(np.uint8)

from ai_disease.inference import (
    InferenceEngine,
    InferenceError,
    LabelMismatchError,
    ModelNotFoundError,
    create_engine,
    timed_run,
)
from ai_disease.mock_engine import MOCK_LABELS, MockInferenceEngine

cv2 = pytest.importorskip("cv2")


# ------------------------------------------------------------------ mock engine
def test_mock_engine_loads():
    e = MockInferenceEngine()
    e.load()
    assert e.is_loaded is True


def test_mock_engine_returns_scores():
    e = MockInferenceEngine(num_classes=4)
    e.load()
    scores = e.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert len(scores) == 4
    assert pytest.approx(sum(scores), abs=1e-6) == 1.0


def test_mock_engine_is_deterministic():
    e = MockInferenceEngine(num_classes=5)
    e.load()
    t = np.zeros((1, 3, 224, 224), dtype=np.float32)
    assert e.predict(t) == e.predict(t)


def test_mock_engine_custom_scores():
    e = MockInferenceEngine(num_classes=3, scores=[0.1, 0.2, 0.7])
    e.load()
    scores = e.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert scores[2] == pytest.approx(0.7)


def test_mock_engine_predict_before_load_raises():
    e = MockInferenceEngine()
    with pytest.raises(InferenceError, match="not loaded"):
        e.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))


def test_mock_engine_failure_path():
    e = MockInferenceEngine(fail_on_predict=True)
    e.load()
    with pytest.raises(InferenceError, match="configured to fail"):
        e.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))


def test_mock_engine_empty_tensor_rejected():
    e = MockInferenceEngine()
    e.load()
    with pytest.raises(InferenceError, match="empty tensor"):
        e.predict(np.zeros((0,), dtype=np.float32))


def test_timed_run_records_latency():
    e = MockInferenceEngine()
    e.load()
    scores, ms = timed_run(e, np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert len(scores) == 4
    assert ms >= 0.0
    assert e.last_inference_ms == ms


def test_mock_engine_describe():
    e = MockInferenceEngine(num_classes=7)
    e.load()
    d = e.describe()
    assert d["engine"] == "mock"
    assert d["num_classes"] == 7


def test_mock_labels_include_healthy_and_disease():
    assert any("healthy" in l for l in MOCK_LABELS)
    assert any("blight" in l or "rust" in l for l in MOCK_LABELS)


# ------------------------------------------------------------ engine factory
def test_create_mock_engine():
    e = create_engine("mock", None, MOCK_LABELS)
    assert isinstance(e, MockInferenceEngine)
    assert e.is_loaded


def test_create_unknown_backend_raises():
    with pytest.raises(InferenceError, match="unknown backend"):
        create_engine("caffe", None, MOCK_LABELS)


# --------------------------------------------------------- missing real model
def test_missing_onnx_model_raises_with_guidance(tmp_path):
    """The real ONNX path must fail clearly, never fall back to a fake model."""
    pytest.importorskip("onnxruntime")
    from ai_disease.onnx_engine import ONNXInferenceEngine

    engine = ONNXInferenceEngine(tmp_path / "absent.onnx", [])
    with pytest.raises(ModelNotFoundError) as exc:
        engine.load()
    message = str(exc.value)
    assert "Model file not found" in message
    assert "does not contain trained model weights" in message


def test_invalid_onnx_file_rejected(tmp_path):
    pytest.importorskip("onnxruntime")
    from ai_disease.onnx_engine import ONNXInferenceEngine

    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx model")
    engine = ONNXInferenceEngine(bad, [])
    with pytest.raises(InferenceError):
        engine.load()


def test_onnx_engine_predict_before_load_raises(tmp_path):
    pytest.importorskip("onnxruntime")
    from ai_disease.onnx_engine import ONNXInferenceEngine

    engine = ONNXInferenceEngine(tmp_path / "x.onnx", [])
    with pytest.raises(InferenceError, match="not loaded"):
        engine.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))


# --------------------------------------------------------------------- config
def test_default_config_targets_224_and_38_classes():
    cfg = load_config()
    assert cfg.preprocess.input_size == (224, 224)
    assert cfg.inference.expected_classes == 38
    assert cfg.inference.backend == "onnx"


def test_default_normalization_is_divide_255():
    assert load_config().preprocess.type == "divide_255"


def test_model_path_points_at_expected_location():
    cfg = load_config()
    assert cfg.model_file.name == "disease_model.onnx"
    assert cfg.model_file.parent.name == "models"


def test_labels_file_resolves():
    cfg = load_config()
    assert cfg.labels_file.is_file()


# ---------------------------------------------------------------- end-to-end
def test_pipeline_runs_with_mock_engine():
    from ai_disease.pipeline import load_labels, run_inference

    cfg = load_config()
    labels = load_labels(cfg.labels_file)
    engine = MockInferenceEngine(
        num_classes=38, scores=[0.05] * 37 + [0.95], labels=labels
    )
    engine.load()
    frame = np.clip(np.random.default_rng(0).normal(120, 45, (480, 640, 3)),
                    0, 255).astype(np.uint8)
    result = run_inference(frame, engine, cfg, source="image")
    assert result.class_name.endswith("healthy")
    assert result.healthy is True
    assert result.inference_ms >= 0.0
    assert result.total_ms >= result.inference_ms


def test_mock_engine_pads_labels_beyond_mock_set():
    e = MockInferenceEngine(num_classes=38)
    assert len(e.labels) == 38


def test_pipeline_labels_healthy_class_correctly():
    from ai_disease.pipeline import load_labels, run_inference

    labels = load_labels(load_config().labels_file)
    assert labels[37] == "Tomato___healthy"
    scores = [0.001] * 38
    scores[37] = 0.962

    cfg = load_config()
    cfg.inference.confidence_threshold = 0.70
    engine = MockInferenceEngine(num_classes=38, scores=scores, labels=labels)
    engine.load()
    result = run_inference(realistic_frame(1), engine, cfg)
    assert result.class_name == "Tomato___healthy"
    assert result.healthy is True


def test_pipeline_uses_real_labels_for_disease_class():
    """argmax index must resolve to the matching label from labels.txt."""
    from ai_disease.pipeline import load_labels, run_inference

    labels = load_labels(load_config().labels_file)
    scores = [0.001] * 38
    scores[20] = 0.95   # Potato___Early_blight
    assert labels[20] == "Potato___Early_blight"

    cfg = load_config()
    engine = MockInferenceEngine(num_classes=38, scores=scores, labels=labels)
    engine.load()
    result = run_inference(realistic_frame(1), engine, cfg)
    assert result.class_id == 20
    assert result.class_name == "Potato___Early_blight"
    assert result.crop == "Potato"
    assert result.disease == "Early blight"


def test_mock_engine_carries_labels():
    e = MockInferenceEngine(num_classes=4)
    assert len(e.labels) == 4
    assert any("healthy" in l for l in e.labels)


def test_pipeline_reports_unknown_below_threshold():
    from ai_disease.pipeline import run_inference

    cfg = load_config()
    cfg.inference.confidence_threshold = 0.95
    engine = MockInferenceEngine(num_classes=38, scores=[0.4] + [0.6 / 37] * 37)
    engine.load()
    result = run_inference(realistic_frame(1), engine, cfg)
    assert result.class_name == "Unknown"
    assert result.reason == "below_confidence_threshold"


def test_pipeline_falls_back_to_class_n_without_labels():
    """With no labels anywhere, results must be class_N - never a guessed disease."""
    from ai_disease.pipeline import run_inference

    cfg = load_config()
    cfg.inference.confidence_threshold = 0.10
    scores = [0.001] * 38
    scores[5] = 0.97
    engine = MockInferenceEngine(num_classes=38, scores=scores, labels=[])
    engine.load()
    result = run_inference(realistic_frame(1), engine, cfg)
    assert result.class_name == "class_5"
    assert result.crop == "class_5"


def test_pipeline_invalid_image_raises():
    from ai_disease.pipeline import run_inference
    from ai_disease.preprocessing import PreprocessingError

    cfg = load_config()
    engine = MockInferenceEngine(num_classes=38)
    engine.load()
    with pytest.raises(PreprocessingError):
        run_inference(np.array([], dtype=np.uint8), engine, cfg)


# ------------------------------------------------------------------- camera
def test_camera_unavailable_error_is_typed():
    from ai_disease.camera import Camera, CameraUnavailableError

    assert issubclass(CameraUnavailableError, RuntimeError)
    c = Camera(backend="usb", device=999)
    assert c.is_open is False


def test_camera_missing_file_raises():
    from ai_disease.camera import Camera, CameraUnavailableError

    cam = Camera(backend="file", path="definitely-not-here.mp4")
    with pytest.raises(CameraUnavailableError, match="not found"):
        cam.open()


def test_camera_picamera2_without_package_raises_clearly():
    from ai_disease.camera import Camera, CameraUnavailableError

    cam = Camera(backend="picamera2")
    try:
        cam.open()
    except CameraUnavailableError as e:
        assert "picamera2" in str(e)
    else:
        pytest.skip("picamera2 installed on this host")


def test_camera_release_is_safe_when_not_open():
    from ai_disease.camera import Camera

    Camera(backend="usb").release()  # must not raise


# ------------------------------------------------------------ flight isolation
def test_module_has_no_flight_control_imports():
    """Hard guarantee: no MAVLink / PX4 / autopilot anywhere in src/ or scripts/."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    banned = ("pymavlink", "mavutil", "ardupilot", "mavproxy", "rclpy", "rospy")
    offenders = []
    for path in list(root.glob("src/*.py")) + list(root.glob("scripts/*.py")):
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, f"flight-control references found: {offenders}"


def test_requirements_exclude_heavy_and_flight_packages():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(
        encoding="utf-8"
    ).lower()
    for banned in ("tensorflow", "torch", "yolov5", "pymavlink", "ardupilot", "mavproxy"):
        # Mentions in explanatory comments are fine; install directives are not.
        assert f"\n{banned}" not in text, f"{banned} appears as a dependency"


def test_tflite_backend_not_imported_by_default_path():
    """TFLite is a stub; importing the package must not pull it in."""
    import sys

    import ai_disease  # noqa: F401

    assert "ai_disease.tflite_engine" not in sys.modules


def test_abstract_engine_cannot_be_instantiated():
    with pytest.raises(TypeError):
        InferenceEngine()  # type: ignore[abstract]


def test_label_mismatch_is_inference_error_subclass():
    assert issubclass(LabelMismatchError, InferenceError)