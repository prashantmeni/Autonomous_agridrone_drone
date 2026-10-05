import React, { useState, useEffect } from "react";
import { MapView } from "../map/MapView";
import { api } from "../services/api";
import { TelemetryData } from "../types";
import { Upload, Play, CheckCircle, AlertCircle, RefreshCw, Compass, MapPin, Sliders } from "lucide-react";

interface MissionPlannerProps {
  telemetry: TelemetryData;
}

const SAMPLE_FIELD_GEOJSON = {
  type: "Polygon",
  coordinates: [
    [
      [77.5940, 12.9710],
      [77.5960, 12.9710],
      [77.5962, 12.9725],
      [77.5938, 12.9725],
      [77.5940, 12.9710],
    ],
  ],
};

export const MissionPlanner: React.FC<MissionPlannerProps> = ({ telemetry }) => {
  const [boundaryName, setBoundaryName] = useState<string>("Untitled field");
  const [altitude, setAltitude] = useState<number>(20);
  const [speed, setSpeed] = useState<number>(5);
  const [overlap, setOverlap] = useState<number>(0.7);

  const [activeBoundary, setActiveBoundary] = useState<any>(SAMPLE_FIELD_GEOJSON);
  const [surveyPlan, setSurveyPlan] = useState<{
    waypoints: { lat: number; lon: number; alt: number }[];
    lines: [number, number][][];
    distance_m: number;
    eta_s: number;
    footprint_m: number;
  } | null>(null);

  const [savedMissions, setSavedMissions] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: "success" | "error" } | null>(null);

  // Generate survey path automatically on parameter change
  const generateSurvey = async (boundary = activeBoundary) => {
    if (!boundary) return;
    setLoading(true);
    try {
      const plan = await api.generateSurvey(boundary, altitude, speed, overlap);
      setSurveyPlan(plan);
      setStatusMsg({
        text: `Generated ${plan.waypoints.length} survey waypoints (${Math.round(plan.distance_m)}m path, ETA ${(plan.eta_s / 60).toFixed(1)} mins)`,
        type: "success",
      });
    } catch (e: any) {
      setSurveyPlan(null);
      setStatusMsg({
        text: `Survey generation failed: ${e.message}`,
        type: "error",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    generateSurvey();
    api.listMissions().then(setSavedMissions).catch(() => {});
  }, []);

  const handleUploadAndSave = async () => {
    if (!surveyPlan || surveyPlan.waypoints.length === 0) return;
    setLoading(true);
    try {
      const res = await api.createMission({
        name: boundaryName,
        waypoints: surveyPlan.waypoints,
        takeoff: { altitude_m: 15 },
      });
      setStatusMsg({ text: `Mission saved successfully (ID: ${res.id.slice(0, 8)})`, type: "success" });
      api.listMissions().then(setSavedMissions).catch(() => {});
    } catch (e: any) {
      setStatusMsg({ text: `Error saving mission: ${e.message}`, type: "error" });
    } finally {
      setLoading(false);
    }
  };

  const handleExecuteMission = async (mid: string) => {
    try {
      await api.startMission(mid);
      setStatusMsg({ text: `Autonomous mission #${mid.slice(0, 8)} started! PX4 executing flight.`, type: "success" });
    } catch (e: any) {
      setStatusMsg({ text: `Could not start mission: ${e.message}`, type: "error" });
    }
  };

  return (
    <div style={{ display: "flex", height: "100%", width: "100%", overflow: "hidden" }}>
      {/* Map View showing planned path */}
      <div style={{ flex: 1, position: "relative" }}>
        <MapView
          telemetry={telemetry}
          boundaryGeojson={activeBoundary}
          surveyWaypoints={surveyPlan?.waypoints || []}
          surveyLines={surveyPlan?.lines || []}
        />
      </div>

      {/* Control Configuration Sidebar */}
      <div style={{
        width: 420,
        background: "#0c101c",
        borderLeft: "1px solid var(--border-color)",
        display: "flex",
        flexDirection: "column",
        overflowY: "auto",
        padding: 20,
        gap: 16
      }}>
        <div>
          <h2 style={{ fontSize: "1.15rem", fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
            <Sliders size={20} color="#10b981" />
            Field Survey Planner
          </h2>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 2 }}>
            Boustrophedon autonomous flight path generation
          </p>
        </div>

        {statusMsg && (
          <div style={{
            padding: "10px 14px",
            borderRadius: "var(--radius-md)",
            fontSize: "0.8rem",
            background: statusMsg.type === "success" ? "rgba(16, 185, 129, 0.15)" : "rgba(239, 68, 68, 0.15)",
            color: statusMsg.type === "success" ? "#34d399" : "#f87171",
            border: `1px solid ${statusMsg.type === "success" ? "rgba(16, 185, 129, 0.3)" : "rgba(239, 68, 68, 0.3)"}`
          }}>
            {statusMsg.text}
          </div>
        )}

        {/* Boundary Parameters */}
        <div className="card" style={{ padding: 16 }}>
          <div className="card-header" style={{ marginBottom: 12 }}>
            <span className="card-title" style={{ fontSize: "0.88rem" }}>
              <MapPin size={16} color="#06b6d4" />
              Field Boundary
            </span>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => generateSurvey()}
            >
              <RefreshCw size={12} /> Replan
            </button>
          </div>

          <div className="form-group">
            <label className="form-label">Field Name</label>
            <input
              type="text"
              className="form-control"
              value={boundaryName}
              onChange={(e) => setBoundaryName(e.target.value)}
            />
            <p style={{ fontSize: "0.68rem", color: "var(--text-dim)", marginTop: 4 }}>
              Default boundary is a sample polygon — replace with your field before flying.
            </p>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div className="form-group">
              <label className="form-label">Flight Alt (m)</label>
              <input
                type="number"
                min={10}
                max={50}
                className="form-control"
                value={altitude}
                onChange={(e) => {
                  setAltitude(Number(e.target.value));
                  generateSurvey();
                }}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Speed (m/s)</label>
              <input
                type="number"
                min={2}
                max={12}
                className="form-control"
                value={speed}
                onChange={(e) => {
                  setSpeed(Number(e.target.value));
                  generateSurvey();
                }}
              />
            </div>
          </div>

          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <label className="form-label">Side Overlap ({Math.round(overlap * 100)}%)</label>
              <span style={{ fontSize: "0.75rem", color: "var(--text-dim)" }}>Footprint: {surveyPlan?.footprint_m.toFixed(1) ?? "—"}m</span>
            </div>
            <input
              type="range"
              min={0.4}
              max={0.85}
              step={0.05}
              value={overlap}
              onChange={(e) => {
                setOverlap(Number(e.target.value));
                generateSurvey();
              }}
              style={{ accentColor: "#10b981", width: "100%", marginTop: 6 }}
            />
          </div>
        </div>

        {/* Survey Calculations Overview */}
        {surveyPlan && (
          <div className="card" style={{ padding: 16, background: "rgba(16, 185, 129, 0.05)" }}>
            <div className="card-header" style={{ marginBottom: 12 }}>
              <span className="card-title" style={{ fontSize: "0.88rem" }}>
                <CheckCircle size={16} color="#10b981" />
                Flight Metrics
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", textTransform: "uppercase" }}>Total Distance</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700 }}>
                  {(surveyPlan.distance_m / 1000).toFixed(2)} km
                </div>
              </div>
              <div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", textTransform: "uppercase" }}>Estimated Time</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700 }}>
                  {(surveyPlan.eta_s / 60).toFixed(1)} mins
                </div>
              </div>
              <div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)", textTransform: "uppercase" }}>Waypoints</div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", fontWeight: 700 }}>
                  {surveyPlan.waypoints.length} points
                </div>
              </div>
            </div>

            <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
              <button
                className="btn btn-primary"
                style={{ flex: 1 }}
                onClick={handleUploadAndSave}
                disabled={loading}
              >
                <Upload size={14} />
                Save Mission
              </button>
            </div>
          </div>
        )}

        {/* Saved Missions list */}
        <div className="card" style={{ padding: 16, flex: 1 }}>
          <div className="card-header" style={{ marginBottom: 8 }}>
            <span className="card-title" style={{ fontSize: "0.88rem" }}>
              Saved Field Missions
            </span>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 180, overflowY: "auto" }}>
            {savedMissions.length === 0 ? (
              <p style={{ fontSize: "0.8rem", color: "var(--text-dim)" }}>No saved missions in database yet.</p>
            ) : (
              savedMissions.map((m) => (
                <div
                  key={m.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "8px 12px",
                    background: "rgba(255, 255, 255, 0.03)",
                    border: "1px solid var(--border-color)",
                    borderRadius: "var(--radius-sm)"
                  }}
                >
                  <div>
                    <div style={{ fontSize: "0.82rem", fontWeight: 600 }}>{m.name}</div>
                    <div style={{ fontSize: "0.68rem", color: "var(--text-dim)", fontFamily: "var(--font-mono)" }}>
                      ID: {m.id.slice(0, 8)}...
                    </div>
                  </div>
                  <button
                    className="btn btn-sm btn-primary"
                    onClick={() => handleExecuteMission(m.id)}
                    title="Upload to PX4 and Fly"
                  >
                    <Play size={12} /> Fly
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
