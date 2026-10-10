"""Typed configuration loader. Env vars override YAML. No hard-coded secrets."""
from __future__ import annotations
import os, re, yaml
from pathlib import Path
from pydantic import BaseModel, Field

class MavlinkConfig(BaseModel):
    connection: str = "/dev/ttyACM0"
    baud: int = 115200
    heartbeat_timeout_s: float = 5.0
    reconnect_delay_s: float = 2.0

class TelemetryConfig(BaseModel):
    rate_hz: float = 5.0
    log_dir: str = "data/logs"
    max_log_mb: float = 50.0

class MissionConfig(BaseModel):
    max_altitude_m: float = 30.0
    max_speed_mps: float = 8.0
    default_takeoff_alt_m: float = 15.0

class SafetyConfig(BaseModel):
    minimum_battery_percent: int = 25
    critical_battery_percent: int = 15
    gps_required: bool = True
    min_satellites: int = 8
    max_hdop: float = 1.5
    geofence_margin_m: float = 5.0
    # When true, arming requires the aircraft to be inside a stored boundary.
    # Off by default: with no boundary configured it would block every arm.
    geofence_required: bool = False
    # Low-battery action thresholds used by the failsafe supervisor.
    rtl_battery_percent: int = 25
    land_battery_percent: int = 15

class CameraConfig(BaseModel):
    enabled: bool = True
    type: str = "auto"
    width: int = 1280
    height: int = 720
    fps: int = 15
    # Explicit V4L2 node; empty means auto-probe video0. Only sensor capture
    # nodes work — the isp/pipeline nodes cannot be opened directly.
    device: str = ""

class ObstacleConfig(BaseModel):
    enabled: bool = False
    warning_distance_m: float = 5.0
    critical_distance_m: float = 2.0
    emergency_distance_m: float = 1.0
    # Rangefinder wiring. ``sensors`` maps a direction to its device, e.g.
    #   sensors: {front: {type: ultrasonic, trig_pin: 23, echo_pin: 24}}
    # Serial devices use {type: serial, port: /dev/ttyAMA0}. An empty map means
    # no rangefinder is fitted, which is reported honestly rather than guessed.
    sensors: dict[str, dict] = Field(default_factory=dict)
    # Forward-camera hazard detection (people/vehicles ahead). Reports and alerts
    # only; it never commands the aircraft.
    camera_hazard_detection: bool = False
    camera_hazard_interval_s: float = 1.0
    # Companion-side braking: hold when clearance drops below this distance.
    brake_enabled: bool = True

class LandingConfig(BaseModel):
    enabled: bool = False
    marker_type: str = "aruco"
    max_descent_speed_mps: float = 0.8
    marker_lost_timeout_s: float = 2.0

class DiseaseConfig(BaseModel):
    enabled: bool = False
    model_path: str = ""
    confidence_threshold: float = 0.6
    inference_interval_s: float = 2.0

    # Confidence banding. These describe how much to trust the model's own
    # score for this frame; they are NOT statements about model accuracy.
    confidence_high: float = 0.80
    confidence_medium: float = 0.50
    # A result below this is not stored as a detection.
    reliable_min_confidence: float = 0.50

    # Frame-quality gates. A frame failing any of these is rejected before
    # inference, so no disease is ever reported from an unusable image.
    image_min_brightness: float = 18.0
    image_max_brightness: float = 245.0
    image_min_blur_score: float = 12.0
    image_min_width: int = 32
    image_min_height: int = 32
    image_min_contrast_std: float = 4.0
    image_require_labels: bool = False
    # Skip the quality gate (diagnostics only; leaves are usually fine).
    skip_image_quality_check: bool = False

class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000

class DbConfig(BaseModel):
    path: str = "data/drone.db"

class AppConfig(BaseModel):
    drone_name: str = "smart_farming_drone"
    environment: str = "default"
    mavlink: MavlinkConfig = Field(default_factory=MavlinkConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    mission: MissionConfig = Field(default_factory=MissionConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    obstacle_avoidance: ObstacleConfig = Field(default_factory=ObstacleConfig)
    precision_landing: LandingConfig = Field(default_factory=LandingConfig)
    disease_detection: DiseaseConfig = Field(default_factory=DiseaseConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    database: DbConfig = Field(default_factory=DbConfig)

_ENV_RE = re.compile(r"\$\{([^:}]+)(?::([^}]*))?\}")

def _expand_env(v: str) -> str:
    def repl(m):
        return os.environ.get(m.group(1), m.group(2) or "")
    return _ENV_RE.sub(repl, v)

def _deep_expand(o):
    if isinstance(o, dict): return {k: _deep_expand(v) for k, v in o.items()}
    if isinstance(o, list): return [_deep_expand(v) for v in o]
    if isinstance(o, str): return _expand_env(o)
    return o

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    base = {}
    default_p = path.parent / "default.yaml"
    if default_p.exists():
        base = yaml.safe_load(default_p.read_text()) or {}
    cur = yaml.safe_load(path.read_text()) if path.exists() else {}
    merged = _deep_merge(base, cur)
    merged = _deep_expand(merged)
    # top-level `drone.name` mapping
    if "drone" in merged and isinstance(merged["drone"], dict):
        merged["drone_name"] = merged["drone"].get("name", merged.get("drone_name", "smart_farming_drone"))
    # env overrides for mavlink (never hard-code /dev/ttyACM0 in code)
    if os.environ.get("MAVLINK_CONNECTION"):
        merged.setdefault("mavlink", {})["connection"] = os.environ["MAVLINK_CONNECTION"]
    if os.environ.get("MAVLINK_BAUD"):
        merged.setdefault("mavlink", {})["baud"] = int(os.environ["MAVLINK_BAUD"])
    return AppConfig(**{k: v for k, v in merged.items() if k in AppConfig.model_fields})
