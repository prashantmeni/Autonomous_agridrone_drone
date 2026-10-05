"""Configuration loading for the crop disease AI module.

Kept dependency-light: yaml plus plain dataclasses. No framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

MODEL_NOT_PRESENT = (
    "Model file not found: {path}\n\n"
    "The reference repository does not contain trained model weights.\n"
    "Please provide a converted ONNX model before running real inference."
)


@dataclass
class PreprocessConfig:
    input_size: tuple[int, int] = (224, 224)
    type: str = "divide_255"
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def width(self) -> int:
        return self.input_size[1]

    @property
    def height(self) -> int:
        return self.input_size[0]


@dataclass
class InferenceConfig:
    backend: str = "onnx"
    confidence_threshold: float = 0.70
    healthy_keyword: str = "healthy"
    strict_label_count: bool = True
    expected_classes: int = 38


@dataclass
class CameraConfig:
    device: int = 0
    width: int = 640
    height: int = 480
    fps: int = 15


@dataclass
class LoggingConfig:
    enabled: bool = False
    results_dir: Path = Path("logs/results")
    images_dir: Path = Path("logs/images")
    save_annotated_images: bool = False


@dataclass
class Config:
    model_path: Path = Path("models/disease_model.onnx")
    labels_path: Path = Path("models/labels.txt")
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, p: Path | str) -> Path:
        """Resolve a config path relative to the module root when relative."""
        p = Path(p)
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def model_file(self) -> Path:
        return self.resolve(self.model_path)

    @property
    def labels_file(self) -> Path:
        return self.resolve(self.labels_path)


def _resolve_base(config_path: Path) -> Path:
    # config.yaml sits at the module root; relative paths in it are relative there.
    return config_path.parent


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration, applying defaults for anything absent."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    model = raw.get("model", {}) or {}
    pre = raw.get("preprocessing", {}) or {}
    inf = raw.get("inference", {}) or {}
    cam = raw.get("camera", {}) or {}
    log = raw.get("logging", {}) or {}

    size = pre.get("input_size", model.get("input_size", [224, 224]))
    base = _resolve_base(path)

    return Config(
        model_path=Path(model.get("path", "models/disease_model.onnx")),
        labels_path=Path(model.get("labels", "models/labels.txt")),
        preprocess=PreprocessConfig(
            input_size=(int(size[0]), int(size[1])),
            type=str(pre.get("type", "divide_255")),
            imagenet_mean=tuple(pre.get("imagenet_mean", [0.485, 0.456, 0.406])),
            imagenet_std=tuple(pre.get("imagenet_std", [0.229, 0.224, 0.225])),
        ),
        inference=InferenceConfig(
            backend=str(inf.get("backend", "onnx")),
            confidence_threshold=float(inf.get("confidence_threshold", 0.70)),
            healthy_keyword=str(inf.get("healthy_keyword", "healthy")),
            strict_label_count=bool(inf.get("strict_label_count", True)),
            expected_classes=int(model.get("expected_classes", 38)),
        ),
        camera=CameraConfig(
            device=int(cam.get("device", 0)),
            width=int(cam.get("width", 640)),
            height=int(cam.get("height", 480)),
            fps=int(cam.get("fps", 15)),
        ),
        logging=LoggingConfig(
            enabled=bool(log.get("enabled", False)),
            results_dir=Path(log.get("results_dir", "logs/results")),
            images_dir=Path(log.get("images_dir", "logs/images")),
            save_annotated_images=bool(log.get("save_annotated_images", False)),
        ),
        base_dir=base,
    )