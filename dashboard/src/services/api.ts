import {
  TelemetryData,
  PreflightReport,
  HealthReport,
  Mission,
  Boundary,
  Detection,
  LogEntry,
  CameraStatus,
  GimbalStatus,
  ChargingStatus,
  ObstacleStatus,
  PrecisionLandingStatus,
  BehaviourRules,
  DiseaseAnalysis,
  DiseaseModelStatus,
} from "../types";

const getHeaders = (multipart = false) => {
  const token = localStorage.getItem("agridrone_token") || "";
  // FormData uploads must let the browser set the multipart boundary itself.
  const headers: Record<string, string> = multipart ? {} : { "Content-Type": "application/json" };
  if (token) headers["X-Token"] = token;
  return headers;
};

const extractDetail = (body: string): string => {
  try {
    const parsed = JSON.parse(body);
    const detail = parsed?.detail;
    if (typeof detail === "string") return detail;
    if (detail != null) return JSON.stringify(detail);
    if (typeof parsed?.message === "string") return parsed.message;
  } catch {
    /* not JSON */
  }
  return body;
};

export const api = {
  async get<T>(path: string): Promise<T> {
    const res = await fetch(path, { headers: getHeaders() });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(extractDetail(err) || `GET ${path} failed with status ${res.status}`);
    }
    return res.json();
  },

  async post<T>(path: string, body?: any): Promise<T> {
    const res = await fetch(path, {
      method: "POST",
      headers: getHeaders(),
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(extractDetail(err) || `POST ${path} failed with status ${res.status}`);
    }
    return res.json();
  },

  // Flight commands
  arm: () => api.post<{ armed: boolean }>("/api/drone/arm"),
  disarm: () => api.post<{ armed: boolean }>("/api/drone/disarm"),
  takeoff: (altitude_m: number) =>
    api.post<{ status: string; alt_m: number }>("/api/drone/takeoff", { altitude_m }),
  rtl: () => api.post<{ ok?: boolean; skipped?: string }>("/api/drone/rtl"),
  land: () => api.post<{ ok?: boolean; skipped?: string }>("/api/drone/land"),
  setMode: (mode: string) => api.post<{ mode: string }>("/api/drone/mode", { mode }),

  // Precision landing (real vision-servo loop)
  precisionLand: () => api.post<PrecisionLandingStatus>("/api/drone/precision-landing"),
  precisionLandingStatus: () => api.get<PrecisionLandingStatus>("/api/drone/precision-landing"),

  // Pan-tilt servo gimbal
  getGimbal: () => api.get<GimbalStatus>("/api/drone/gimbal"),
  setGimbal: (pan_deg: number, tilt_deg: number) =>
    api.post<GimbalStatus>("/api/drone/gimbal", { pan_deg, tilt_deg }),

  // Charging dock
  getChargingStatus: () => api.get<ChargingStatus>("/api/drone/charging"),

  // Obstacle avoidance
  getObstacleStatus: () => api.get<ObstacleStatus>("/api/drone/obstacles"),

  // Camera
  getCameraStatus: () => api.get<CameraStatus>("/api/camera/status"),
  toggleRecord: (action: "start" | "stop" | "toggle" = "toggle") =>
    api.post<{ recording: boolean; path?: string; error?: string }>("/api/camera/record", { action }),
  snapshotUrl: () => "/api/camera/snapshot",
  streamUrl: () => "/api/camera/stream",

  // Disease detection
  analyzeDisease: (mark = false) =>
    api.post<DiseaseAnalysis>("/api/disease/analyze", { mark }),
  // Same pipeline, but fed a still image instead of the live camera frame.
  analyzeDiseaseImage: (file: File, mark = false) => {
    const form = new FormData();
    form.append("file", file);
    form.append("mark", String(mark));
    return fetch("/api/disease/analyze", {
      method: "POST",
      headers: getHeaders(true),
      body: form,
    }).then(async (res) => {
      const text = await res.text();
      if (!res.ok) throw new Error(extractDetail(text) || `analyze failed with status ${res.status}`);
      return JSON.parse(text) as DiseaseAnalysis;
    });
  },
  getDiseaseModel: () => api.get<DiseaseModelStatus>("/api/disease/model"),
  uploadDiseaseModel: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return fetch("/api/disease/model", {
      method: "POST",
      headers: getHeaders(true),
      body: form,
    }).then(async (res) => {
      const text = await res.text();
      if (!res.ok) throw new Error(extractDetail(text) || `upload failed with status ${res.status}`);
      return JSON.parse(text) as DiseaseModelStatus;
    });
  },
  // Label companion for an already-uploaded model: `<model>.onnx.txt`.
  uploadDiseaseLabels: (modelName: string, file: File) => {
    const form = new FormData();
    form.append("file", file, `${modelName}.txt`);
    return fetch("/api/disease/model", {
      method: "POST",
      headers: getHeaders(true),
      body: form,
    }).then(async (res) => {
      const text = await res.text();
      if (!res.ok) throw new Error(extractDetail(text) || `label upload failed with status ${res.status}`);
      return JSON.parse(text) as DiseaseModelStatus;
    });
  },
  markDetection: (
    crop: string,
    disease: string,
    confidence: number,
    latitude: number,
    longitude: number
  ) => api.post<{ status: string }>("/api/detections/mark", { crop, disease, confidence, latitude, longitude }),

  // Behaviour rules
  getBehaviour: () => api.get<BehaviourRules>("/api/behaviour"),
  setBehaviour: (rules: Partial<BehaviourRules>) => api.post<BehaviourRules>("/api/behaviour", rules),

  // Status & health
  getStatus: () => api.get<TelemetryData>("/api/drone/status"),
  getPreflight: () => api.get<PreflightReport>("/api/drone/preflight"),
  getHealth: () => api.get<HealthReport>("/api/health"),

  // Missions & boundaries
  listMissions: () => api.get<Mission[]>("/api/missions"),
  createMission: (mission: { name: string; waypoints: any[]; takeoff?: { altitude_m: number } }) =>
    api.post<{ id: string }>("/api/missions", mission),
  getMission: (id: string) => api.get<any>(`/api/missions/${id}`),
  startMission: (id: string) =>
    api.post<{ mission_id: string; status: string }>(`/api/missions/${id}/start`),
  abortMission: (id: string) =>
    api.post<{ status: string; reason: string }>(`/api/missions/${id}/abort`),

  createBoundary: (name: string, geojson: any) =>
    api.post<Boundary>("/api/boundaries", { name, geojson }),
  importKmlBoundary: (name: string, kml: string) =>
    api.post<Boundary>("/api/boundaries/import-kml", { name, kml }),
  generateSurvey: (boundary_geojson: any, altitude_m = 20, speed_mps = 5, overlap = 0.7) =>
    api.post<{
      waypoints: { lat: number; lon: number; alt: number }[];
      lines: [number, number][][];
      distance_m: number;
      eta_s: number;
      footprint_m: number;
    }>("/api/survey/generate", {
      boundary_geojson,
      altitude_m,
      speed_mps,
      overlap,
    }),

  // Detections & logs
  getDetections: () => api.get<Detection[]>("/api/detections"),
  getLogs: (limit = 100) => api.get<LogEntry[]>(`/api/logs?limit=${limit}`),
};
