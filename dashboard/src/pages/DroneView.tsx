import React, { useCallback, useRef } from "react";
import { Drone3D } from "../components/Drone3D";
import { CommandBar } from "../components/CommandBar";
import { ObstaclePanel } from "../components/ObstaclePanel";
import { MapView } from "../map/MapView";
import { TelemetryData, ChargingStatus, ObstacleStatus, Detection } from "../types";
import { api } from "../services/api";
import { Crosshair, BatteryCharging, Gauge, Navigation } from "lucide-react";

interface DroneViewProps {
  telemetry: TelemetryData;
  charging: ChargingStatus | null;
  obstacles: ObstacleStatus | null;
  obstaclesUnreachable: boolean;
  detections: Detection[];
  onTakeoff: (alt: number) => void;
  onLand: () => void;
  onRTL: () => void;
  onPrecisionLand: () => void;
  notify: (text: string, type?: "success" | "warning" | "error") => void;
}

const CARDINALS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];

interface Phase {
  label: string;
  cls: string;
}

const phaseOf = (t: TelemetryData, charging: ChargingStatus | null): Phase => {
  if (charging?.charging && !t.armed) return { label: "CHARGING", cls: "phase-charge" };
  const fsm = (t.fsm_state || "").toUpperCase();
  switch (fsm) {
    case "TAKEOFF":
      return { label: "TAKEOFF", cls: "phase-takeoff" };
    case "ARMING":
      return { label: "ARMING", cls: "phase-takeoff" };
    case "LANDING":
    case "PRECISION_LANDING":
      return { label: "LANDING", cls: "phase-land" };
    case "MISSION":
    case "SURVEY":
      return { label: "FLYING", cls: "phase-fly" };
    case "RETURN_HOME":
      return { label: "RETURN", cls: "phase-fly" };
    case "EMERGENCY":
    case "ABORT":
      return { label: "EMERGENCY", cls: "phase-emergency" };
    case "DISARMED":
    case "LANDED":
      return { label: "LANDED", cls: "phase-ground" };
    default:
      return t.armed ? { label: "FLYING", cls: "phase-fly" } : { label: "STANDBY", cls: "phase-ground" };
  }
};

const MODES = [
  { id: "TAKEOFF", label: "Takeoff", match: (m: string) => m === "AUTO/TAKEOFF" },
  { id: "AUTO", label: "Auto", match: (m: string) => m === "AUTO/MISSION" || m === "AUTO" },
  { id: "HOLD", label: "Hold", match: (m: string) => m === "AUTO/LOITER" || m === "HOLD" || m === "LOITER" },
  { id: "ALTCTL", label: "Altitude", match: (m: string) => m === "ALTCTL" },
  { id: "STABILIZED", label: "Stabilized", match: (m: string) => m === "STABILIZED" },
  { id: "POSCTL", label: "Position", match: (m: string) => m === "POSCTL" },
];

export const DroneView: React.FC<DroneViewProps> = ({
  telemetry,
  charging,
  obstacles,
  obstaclesUnreachable,
  detections,
  onTakeoff,
  onLand,
  onRTL,
  onPrecisionLand,
  notify,
}) => {
  const recenterRef = useRef<() => void>(() => {});
  const baroBase = useRef<number | null>(null);

  const registerRecenter = useCallback((fn: () => void) => {
    recenterRef.current = fn;
  }, []);

  const t = telemetry;
  const heading =
    t.gps_fix > 0 && t.heading_deg ? t.heading_deg : (t.yaw_deg ?? t.heading_deg) || 0;
  const cardinal = CARDINALS[Math.round((((heading % 360) + 360) % 360) / 22.5) % 16];
  const phase = phaseOf(t, charging);

  const baro = t.alt_baro_m;
  if (baroBase.current == null && baro != null && baro !== 0) baroBase.current = baro;
  const relAlt =
    baroBase.current != null && baro != null ? baro - baroBase.current : t.relative_alt_m || 0;

  const ax = t.accel_x ?? 0;
  const ay = t.accel_y ?? 0;
  const az = t.accel_z ?? 0;
  const bubbleX = Math.max(-1, Math.min(1, (t.roll_deg || 0) / 45)) * 24;
  const bubbleY = Math.max(-1, Math.min(1, (t.pitch_deg || 0) / 45)) * -24;

  const setMode = async (mode: string) => {
    try {
      await api.setMode(mode);
      notify(`Flight mode → ${mode}`, "success");
    } catch (e: any) {
      notify(`Mode change failed: ${e.message}`, "error");
    }
  };

  const stats = [
    { label: "Altitude", value: t.connected ? `${t.relative_alt_m.toFixed(1)}` : "—", unit: "m" },
    { label: "Ground speed", value: t.connected ? `${t.ground_speed_mps.toFixed(1)}` : "—", unit: "m/s" },
    { label: "Heading", value: t.connected ? `${Math.round(t.heading_deg)}` : "—", unit: "°" },
    { label: "Battery", value: t.battery_remaining_pct >= 0 ? `${t.battery_remaining_pct}` : "—", unit: "%" },
    { label: "Satellites", value: t.connected ? `${t.satellites}` : "—", unit: "sat" },
    { label: "HDOP", value: t.connected ? t.hdop.toFixed(1) : "—", unit: "" },
    {
      label: "Transmitter",
      value: t.rc_connected ? "LINKED" : t.rc_age_s != null && t.rc_age_s >= 0 ? "LOST" : "—",
      unit: t.rc_connected && t.rc_rssi != null && t.rc_rssi >= 0 ? `${t.rc_rssi}%` : "",
    },
  ];

  return (
    <div className="drone-page">
      <div className="drone-stage">
        <Drone3D telemetry={t} charging={charging} registerRecenter={registerRecenter} />

        {/* phase + flight card */}
        <div className="d3d-card d3d-tl">
          <div className={`d3d-phase ${phase.cls}`}>{phase.label}</div>
          <div className="d3d-row">
            <label>MODE</label>
            <b>{t.mode || "—"}</b>
          </div>
          <div className="d3d-row">
            <label>ALT</label>
            <b>{relAlt.toFixed(1)} m</b>
          </div>
          <div className="d3d-row">
            <label>SPEED</label>
            <b>{(t.ground_speed_mps || 0).toFixed(1)} m/s</b>
          </div>
          <div className="d3d-row">
            <label>BATT</label>
            <b>
              {t.battery_remaining_pct >= 0 ? `${t.battery_remaining_pct}%` : "—"}
              {t.battery_voltage_v ? ` · ${t.battery_voltage_v.toFixed(1)}V` : ""}
            </b>
          </div>
          <div className="d3d-row">
            <label>FSM</label>
            <b>{t.fsm_state || "—"}</b>
          </div>
        </div>

        {/* accelerometer */}
        <div className="d3d-card d3d-bl">
          <div className="d3d-card-title">
            <Gauge size={12} /> ACCELEROMETER <span className="d3d-unit">m/s²</span>
          </div>
          <div className="d3d-row">
            <label>X (fwd)</label>
            <b className={Math.abs(ax) > 1 ? "acc-hot" : ""}>{ax.toFixed(2)}</b>
          </div>
          <div className="d3d-row">
            <label>Y (right)</label>
            <b className={Math.abs(ay) > 1 ? "acc-hot" : ""}>{ay.toFixed(2)}</b>
          </div>
          <div className="d3d-row">
            <label>Z (down)</label>
            <b className={Math.abs(Math.abs(az) - 9.81) > 1.5 ? "acc-hot" : ""}>{az.toFixed(2)}</b>
          </div>
          <div className="d3d-bubble" title="Attitude bubble (from roll/pitch)">
            <i style={{ transform: `translate(calc(-50% + ${bubbleX}px), calc(-50% + ${bubbleY}px))` }} />
            <span className="bubble-label">HORIZON</span>
          </div>
        </div>

        {/* attitude + recenter */}
        <div className="d3d-card d3d-br">
          <div className="d3d-row">
            <label>ROLL</label>
            <b>{(t.roll_deg || 0).toFixed(1)}°</b>
          </div>
          <div className="d3d-row">
            <label>PITCH</label>
            <b>{(t.pitch_deg || 0).toFixed(1)}°</b>
          </div>
          <div className="d3d-row">
            <label>YAW</label>
            <b>{(t.yaw_deg ?? t.heading_deg ?? 0).toFixed(1)}°</b>
          </div>
          <button className="chip-btn d3d-recenter" onClick={() => recenterRef.current()} title="Reset drone position and altitude baseline">
            <Crosshair size={12} /> RECENTER
          </button>
        </div>

        {/* charging banner — only while charging */}
        {charging?.charging && (
          <div className="d3d-card d3d-charge">
            <BatteryCharging size={14} /> CHARGING
            {charging.voltage_v != null ? ` · ${charging.voltage_v.toFixed(2)} V` : ""}
            {charging.battery_pct != null ? ` · ${charging.battery_pct}%` : ""} · {charging.pad_id}
          </div>
        )}

        {/* flight command dock: arm/disarm, takeoff, land, RTL, precision land */}
        <CommandBar
          telemetry={t}
          chargingDocked={!!charging?.charging}
          onTakeoff={onTakeoff}
          onLand={onLand}
          onRTL={onRTL}
          onPrecisionLand={onPrecisionLand}
          notify={notify}
        />
      </div>

      <aside className="drone-side">
        {/* flight mode selector (moved from TopBar) */}
        <div className="d3d-card dside-card">
          <div className="d3d-card-title">FLIGHT MODE</div>
          <div className="drone-modes">
            {MODES.map((m) => (
              <button
                key={m.id}
                className={`dmode-btn ${m.match(t.mode || "") ? "active" : ""}`}
                onClick={() => setMode(m.id)}
                disabled={!t.connected}
                title={`Switch to ${m.label} mode`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* compass */}
        <div className="d3d-card dside-card">
          <div className="d3d-card-title">
            <Navigation size={12} /> COMPASS
          </div>
          <svg width="128" height="128" viewBox="0 0 128 128" className="d3d-compass">
            <circle cx="64" cy="64" r="58" fill="rgba(4,7,14,0.7)" stroke="var(--border-color)" />
            <g transform={`rotate(${-heading} 64 64)`}>
              {Array.from({ length: 16 }).map((_, i) => {
                const a = i * 22.5;
                const major = i % 4 === 0;
                const r1 = major ? 46 : 52;
                const x1 = 64 + Math.sin((a * Math.PI) / 180) * r1;
                const y1 = 64 - Math.cos((a * Math.PI) / 180) * r1;
                const x2 = 64 + Math.sin((a * Math.PI) / 180) * 57;
                const y2 = 64 - Math.cos((a * Math.PI) / 180) * 57;
                return (
                  <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={major ? "#94a3b8" : "#334155"} strokeWidth={major ? 2 : 1} />
                );
              })}
              {[["N", 0], ["E", 90], ["S", 180], ["W", 270]].map(([lbl, deg]) => {
                const rad = (deg as number) * (Math.PI / 180);
                const x = 64 + Math.sin(rad) * 36;
                const y = 64 - Math.cos(rad) * 36 + 4;
                return (
                  <text key={lbl as string} x={x} y={y} textAnchor="middle" fontSize="13" fontWeight="800" fill={lbl === "N" ? "#f87171" : "#cbd5e1"}>
                    {lbl as string}
                  </text>
                );
              })}
            </g>
            <polygon points="64,10 58,22 70,22" fill="#34d399" />
            <circle cx="64" cy="64" r="3" fill="#34d399" />
          </svg>
          <div className="d3d-heading">
            {heading.toFixed(1)}° <span>{cardinal}</span>
          </div>
        </div>

        {/* telemetry stats (moved from Live page) */}
        <div className="d3d-card dside-card">
          <div className="d3d-card-title">
            <Gauge size={12} /> TELEMETRY
          </div>
          <div className="stat-grid">
            {stats.map((s) => (
              <div className="stat-cell" key={s.label}>
                <span>{s.label}</span>
                <b>
                  {s.value}
                  <i>{s.unit}</i>
                </b>
              </div>
            ))}
          </div>
        </div>

        {/* map (moved from Live page) */}
        <div className="side-map">
          <MapView telemetry={t} detections={detections} />
        </div>

        {/* obstacles (moved from Live page) */}
        <ObstaclePanel obstacles={obstacles} unreachable={obstaclesUnreachable} />

        {/* charging dock — only while actually charging */}
        {charging?.charging && (
          <div className="card charge-card">
            <div className="card-header">
              <span className="card-title">Charging dock</span>
              <span className={`status-pill ${charging.charging ? "pill-live" : "pill-idle"}`}>
                {charging.charging ? "CHARGING" : "DOCKED"}
              </span>
            </div>
            <div className="charge-row">
              <b>{charging.voltage_v != null ? `${charging.voltage_v.toFixed(2)} V` : "—"}</b>
              <span>{charging.pad_id}</span>
            </div>
            <div className="charge-note">Detection: {charging.source.replace(/_/g, " ")}</div>
          </div>
        )}
      </aside>
    </div>
  );
};
