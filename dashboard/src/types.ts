export interface TelemetryData {
  connected: boolean;
  armed: boolean;
  mode: string;
  lat: number | null;
  lon: number | null;
  alt_m: number;
  relative_alt_m: number;
  ground_speed_mps: number;
  heading_deg: number;
  battery_voltage_v: number;
  battery_current_a?: number;
  battery_remaining_pct: number;
  gps_fix: number;
  satellites: number;
  hdop: number;
  roll_deg: number;
  pitch_deg: number;
  yaw_deg?: number;
  accel_x?: number;
  accel_y?: number;
  accel_z?: number;
  alt_baro_m?: number;
  fsm_state?: string;
  rc_connected?: boolean;
  rc_age_s?: number;
  rc_rssi?: number;
  rc_channels?: number;
  updated_at?: number;
}

export interface PreflightReport {
  ready: boolean;
  checks?: Record<string, string>;
  issues: string[];
  [key: string]: any;
}

export interface HealthReport {
  environment: string;
  fsm: string;
  status: string;
  checks: Record<string, string>;
}

export interface CameraStatus {
  status: string;
  available: boolean;
  width: number;
  height: number;
  fps: number;
  frame_age_s: number | null;
  recording: boolean;
  recording_path: string | null;
  recording_s: number;
  frames_written: number;
}

export interface GimbalStatus {
  pan_deg: number;
  tilt_deg: number;
  pan_min: number;
  pan_max: number;
  tilt_min: number;
  tilt_max: number;
  control: string;
  available: boolean;
  last_error: string | null;
}

export interface ChargingStatus {
  docked: boolean;
  charging: boolean;
  auto_charge_on_landing: boolean;
  source: string;
  voltage_v: number | null;
  battery_pct: number | null;
  voltage_trend_mv: number | null;
  samples: number;
  pad_id: string;
}

export interface ObstacleStatus {
  enabled: boolean;
  available: boolean;
  sensor_status: string;
  state: string;
  distances_m: Record<string, number | null>;
  min_distance_m: number | null;
  thresholds_m: { warning: number; critical: number; emergency: number };
}

export interface PrecisionLandingStatus {
  active: boolean;
  phase: string;
  marker_found: boolean;
  offset_x: number | null;
  offset_y: number | null;
  confidence: number;
  decision: string | null;
  altitude_m: number | null;
  fallback: string | null;
  message: string;
}

export interface BehaviourRules {
  survey_altitude_m: number;
  cruise_speed_mps: number;
  overlap_pct: number;
  rth_altitude_m: number;
  rth_battery_reserve_pct: number;
  obstacle_brake_distance_m: number;
  auto_charge_on_landing: boolean;
  auto_nadir_lock: boolean;
  geofence_enabled: boolean;
  landing_pad_id: string;
}

export interface MissionWaypoint {
  lat: number;
  lon: number;
  alt: number;
  command?: number;
  hold_s?: number;
}

export interface Mission {
  id: string;
  name: string;
  waypoints?: MissionWaypoint[];
  takeoff?: { altitude_m: number };
}

/** Live mission supervision state reported by the companion computer. */
export interface MissionStatus {
  phase:
    | "IDLE"
    | "VALIDATING"
    | "UPLOADING"
    | "EXECUTING"
    | "PAUSED"
    | "ABORTING"
    | "COMPLETE"
    | "FAILED";
  mission_id: string | null;
  name: string | null;
  total: number;
  done: number;
  current_seq: number;
  error: string | null;
  started_at: number | null;
  finished_at: number | null;
  active: boolean;
  elapsed_s: number | null;
}

export interface Boundary {
  id: string;
  name: string;
  area_m2: number;
  perimeter_m?: number;
  geojson: any;
}

export interface Detection {
  crop: string;
  disease: string;
  confidence: number;
  latitude: number;
  longitude: number;
  timestamp: number;
  image?: string;
}

export interface LogEntry {
  ts: number;
  level: "INFO" | "WARNING" | "ERROR" | "CRITICAL";
  component: string;
  event: string;
  data?: any;
}

export interface DiseaseAnalysis {
  status: string;
  crop?: string | null;
  disease?: string | null;
  confidence?: number;
  marked?: boolean;
  note?: string;
  detail?: string;
  // Production pipeline additions
  success?: boolean;
  error_code?: string;
  message?: string;
  suggestion?: string;
  confidence_level?: "HIGH" | "MEDIUM" | "LOW" | "NONE";
  reliable?: boolean;
  label?: string;
  source?: "camera" | "upload";
  mark_skipped_reason?: string;
  image_quality?: {
    valid: boolean;
    skipped?: boolean;
    error_code?: string;
    metrics?: {
      brightness?: number;
      contrast_std?: number;
      blur_score?: number;
      width?: number;
      height?: number;
    };
  };
  model?: { name?: string | null; version?: string | null };
  timing?: {
    preprocess_ms?: number;
    inference_ms?: number;
    postprocess_ms?: number;
    total_ms?: number;
  };
}

export interface DiseaseModelStatus {
  status: string;
  enabled?: boolean;
  model_path?: string | null;
  model_name?: string | null;
  format?: string | null;
  runtime?: string | null;
  name?: string | null;
  path?: string;
  input?: string;
  input_size?: number[];
  input_width?: number;
  input_height?: number;
  layout?: string;
  output?: string;
  labels?: number;
  label_list?: string[];
  num_classes?: number | null;
  model_version?: string | null;
  size_bytes?: number;
  modified_at?: number;
  producer?: string;
  detail?: string;
  load_error?: string | null;
  supported_formats?: string[];
  uploaded?: { name: string; bytes: number; kind?: string };
  confidence_thresholds?: { high: number; medium: number; reliable_min: number; store_min: number };
  image_quality_thresholds?: {
    min_brightness: number;
    max_brightness: number;
    min_blur_score: number;
    min_width: number;
    min_height: number;
    min_contrast_std: number;
    require_labels: boolean;
    skip_check: boolean;
  };
  available_models?: Array<{
    name: string;
    size_bytes: number;
    modified_at: number;
    format: string;
    has_labels: boolean;
  }>;
  validation?: {
    valid?: boolean;
    code?: string;
    num_classes?: number;
    labels_match?: boolean | null;
    errors?: string[];
    warnings?: string[];
    test_inference?: string;
    replaced?: boolean;
  };
}
