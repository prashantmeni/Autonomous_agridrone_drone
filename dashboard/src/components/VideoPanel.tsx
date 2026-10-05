import React, { useEffect, useRef, useState } from "react";
import {
  Video,
  VideoOff,
  Circle,
  Camera,
  Crosshair,
  Zap,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  Scan,
  Locate,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { TelemetryData, CameraStatus, GimbalStatus, PrecisionLandingStatus, DiseaseAnalysis } from "../types";
import { api } from "../services/api";

interface VideoPanelProps {
  telemetry: TelemetryData;
  precision: PrecisionLandingStatus | null;
  charging: boolean;
  docked: boolean;
  onMarkDisease: (disease: string, confidence: number) => void;
  notify: (text: string, type?: "success" | "warning" | "error") => void;
  embedded?: boolean;
}

const fmtTimer = (secs: number) => {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};

export const VideoPanel: React.FC<VideoPanelProps> = ({
  telemetry,
  precision,
  charging,
  docked,
  onMarkDisease,
  notify,
  embedded = false,
}) => {
  const [cam, setCam] = useState<CameraStatus | null>(null);
  const [gimbal, setGimbal] = useState<GimbalStatus | null>(null);
  const [analysis, setAnalysis] = useState<DiseaseAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [isFs, setIsFs] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const shellRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onChange = () => setIsFs(document.fullscreenElement === shellRef.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const toggleFs = () => {
    if (document.fullscreenElement === shellRef.current) {
      document.exitFullscreen().catch(() => {});
    } else {
      shellRef.current?.requestFullscreen().catch(() => {});
    }
  };

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const c = await api.getCameraStatus();
        if (alive) setCam(c);
      } catch {
        if (alive) setCam(null);
      }
      try {
        const g = await api.getGimbal();
        if (alive) setGimbal(g);
      } catch {
        /* keep last */
      }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const toggleRecord = async () => {
    try {
      const res = await api.toggleRecord("toggle");
      if (res.recording) notify(`Recording started${res.path ? ` → ${res.path}` : ""}`, "success");
      else notify("Recording stopped", "warning");
      setCam(await api.getCameraStatus().catch(() => cam));
    } catch (e: any) {
      notify(`Recorder: ${e.message}`, "error");
    }
  };

  const snapshot = async () => {
    try {
      const res = await fetch(api.snapshotUrl());
      if (!res.ok) throw new Error("snapshot unavailable");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `snapshot_${Date.now()}.jpg`;
      a.click();
      URL.revokeObjectURL(url);
      notify("Snapshot saved", "success");
    } catch (e: any) {
      notify(`Snapshot failed: ${e.message}`, "error");
    }
  };

  const nudgeGimbal = async (dPan: number, dTilt: number) => {
    const pan = (gimbal?.pan_deg ?? 0) + dPan;
    const tilt = Math.max(-90, Math.min(0, (gimbal?.tilt_deg ?? -90) + dTilt));
    try {
      const next = await api.setGimbal(pan, tilt);
      setGimbal(next);
      if (!next.available) {
        notify("Servo not reachable — check Pixhawk gimbal/mount output", "warning");
      }
    } catch (e: any) {
      notify(`Gimbal: ${e.message}`, "error");
    }
  };

  const runAnalysis = async () => {
    setAnalyzing(true);
    try {
      const res = await api.analyzeDisease(false);
      setAnalysis(res);
      if (res.status === "OK") notify(`Detected: ${res.disease} (${((res.confidence || 0) * 100).toFixed(0)}%)`, "success");
      else notify(res.note || res.status, "warning");
    } catch (e: any) {
      setAnalysis({ status: "ERROR", note: e.message });
      notify(`Analysis failed: ${e.message}`, "error");
    } finally {
      setAnalyzing(false);
    }
  };

  const live = cam?.available ?? false;
  const recording = cam?.recording ?? false;
  const precisionActive = precision?.active ?? false;

  return (
    <div ref={shellRef} className={`video-shell ${embedded ? "embedded" : ""}`}>
      {/* Stream */}
      <img
        ref={imgRef}
        key={streamKey}
        className="video-feed"
        src={`${api.streamUrl()}?t=${streamKey}`}
        alt="Live camera feed"
        onError={() => setStreamKey((k) => k + 1)}
      />

      {/* Top overlay strip */}
      <div className="video-top">
        <div className="chip-row">
          <span className={`chip ${live ? "chip-live" : "chip-off"}`}>
            <span className="pulse-dot" />
            {live ? `LIVE ${cam?.width}×${cam?.height}` : cam?.status === "NOT_STARTED" ? "CAMERA IDLE" : "NO SIGNAL"}
          </span>
          {recording && (
            <span className="chip chip-rec">
              <Circle size={8} fill="#ef4444" /> REC {fmtTimer(cam?.recording_s || 0)}
            </span>
          )}
          <span className="chip">
            GIMBAL {gimbal ? `${gimbal.pan_deg}° / ${gimbal.tilt_deg}°` : "—"}
          </span>
          {!gimbal?.available && gimbal && <span className="chip chip-warn">SERVO OFFLINE</span>}
        </div>

        <div className="chip-row">
          <button className="chip-btn" onClick={runAnalysis} disabled={analyzing} title="Run crop disease analysis on current frame">
            <Scan size={12} />
            {analyzing ? "Analyzing…" : "Analyze"}
          </button>
          <button className="chip-btn" onClick={toggleRecord} title="Start / stop video recording">
            <Circle size={9} fill={recording ? "#ef4444" : "#94a3b8"} />
            {recording ? "Stop" : "Record"}
          </button>
          <button className="chip-btn" onClick={snapshot} title="Save a snapshot">
            <Camera size={12} />
          </button>
          <button className="chip-btn" onClick={toggleFs} title={isFs ? "Exit fullscreen" : "Fullscreen"}>
            {isFs ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
          </button>
        </div>
      </div>

      {/* Telemetry footer */}
      {!embedded && (
        <div className="video-bottom">
          <div className="video-telemetry">
            <span><label>ALT</label>{telemetry.relative_alt_m.toFixed(1)} m</span>
            <span><label>SPD</label>{telemetry.ground_speed_mps.toFixed(1)} m/s</span>
            <span><label>HDG</label>{Math.round(telemetry.heading_deg)}°</span>
            <span><label>POS</label>
              {telemetry.lat != null ? `${telemetry.lat.toFixed(5)}, ${telemetry.lon?.toFixed(5)}` : "—"}
            </span>
          </div>
        </div>
      )}

      {/* Pan-tilt pad */}
      {!embedded && (
        <div className="pan-tilt">
          <div className="pan-tilt-title">PAN-TILT</div>
          <div className="pan-tilt-grid">
            <button onClick={() => nudgeGimbal(-15, 0)} title="Pan left"><ChevronLeft size={16} /></button>
            <button onClick={() => nudgeGimbal(0, 15)} title="Tilt up"><ChevronUp size={16} /></button>
            <button onClick={() => nudgeGimbal(0, -15)} title="Tilt down"><ChevronDown size={16} /></button>
            <button onClick={() => nudgeGimbal(15, 0)} title="Pan right"><ChevronRight size={16} /></button>
          </div>
          <button
            className="pan-tilt-center"
            onClick={() => {
              api
                .setGimbal(0, -90)
                .then(setGimbal)
                .catch(() => {});
            }}
            title="Nadir (straight down)"
          >
            <Locate size={11} /> NADIR
          </button>
        </div>
      )}

      {/* Precision landing overlay */}
      {precisionActive && (
        <div className="precision-overlay">
          <div className={`reticle ${precision?.marker_found ? "locked" : ""}`}>
            <Crosshair size={42} />
          </div>
          <div className="precision-label">
            {precision?.phase} · {precision?.marker_found ? `LOCK ${((precision?.offset_x || 0) * 100).toFixed(0)}%` : "SEARCHING PAD"}
            {precision?.fallback ? ` · fallback ${precision.fallback}` : ""}
          </div>
        </div>
      )}

      {/* Charging overlay */}
      {charging && (
        <div className="charging-overlay">
          <Zap size={16} fill="currentColor" />
          WIRELESS CHARGING — docked on landing pad
        </div>
      )}

      {/* Disease analysis result */}
      {analysis && (
        <div className={`analysis-card ${analysis.status === "OK" ? "ok" : "muted"}`}>
          <div className="analysis-head">
            <Scan size={12} />
            <span>CROP AI</span>
            <button onClick={() => setAnalysis(null)} title="Dismiss">×</button>
          </div>
          {analysis.status === "OK" ? (
            <>
              <div className="analysis-disease">{analysis.disease}</div>
              <div className="analysis-meta">
                {analysis.crop} · {((analysis.confidence || 0) * 100).toFixed(0)}% confidence
              </div>
              <button
                className="btn btn-sm btn-primary"
                onClick={() => onMarkDisease(analysis.disease || "Anomaly", analysis.confidence || 0.5)}
              >
                Mark zone on map
              </button>
            </>
          ) : (
            <div className="analysis-meta">{analysis.note || analysis.status}</div>
          )}
        </div>
      )}

      {/* Offline slate */}
      {!live && cam !== null && (
        <div className="video-slate">
          <VideoOff size={30} />
          <b>{cam?.status === "NOT_STARTED" ? "Camera idle" : "No camera signal"}</b>
          <span>Connect a USB/Pi camera to enable live video and recording</span>
        </div>
      )}
    </div>
  );
};
