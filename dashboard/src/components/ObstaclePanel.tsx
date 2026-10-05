import React from "react";
import { Shield, ShieldAlert, ShieldX, ShieldQuestion } from "lucide-react";
import { ObstacleStatus } from "../types";

interface ObstaclePanelProps {
  obstacles: ObstacleStatus | null;
  unreachable: boolean;
}

const STATE_ICON: Record<string, React.ReactNode> = {
  CLEAR: <Shield size={16} />,
  WARNING: <ShieldAlert size={16} />,
  OBSTACLE_DETECTED: <ShieldAlert size={16} />,
  PATH_BLOCKED: <ShieldX size={16} />,
  EMERGENCY_STOP: <ShieldX size={16} />,
};

const STATE_COLOR: Record<string, string> = {
  CLEAR: "#34d399",
  WARNING: "#fbbf24",
  OBSTACLE_DETECTED: "#fbbf24",
  PATH_BLOCKED: "#f87171",
  EMERGENCY_STOP: "#ef4444",
};

export const ObstaclePanel: React.FC<ObstaclePanelProps> = ({ obstacles, unreachable }) => {
  const state = obstacles?.state || "UNKNOWN";
  const color = STATE_COLOR[state] || "#64748b";
  const dirs: { key: string; label: string; cls: string }[] = [
    { key: "front", label: "FRONT", cls: "dir-front" },
    { key: "left", label: "LEFT", cls: "dir-left" },
    { key: "right", label: "RIGHT", cls: "dir-right" },
    { key: "rear", label: "REAR", cls: "dir-rear" },
  ];

  const fmt = (v: number | null | undefined) => (v == null ? "—" : `${v.toFixed(1)}m`);

  return (
    <div className="card obstacle-card">
      <div className="card-header">
        <span className="card-title">
          {STATE_ICON[state] || <ShieldQuestion size={16} />}
          Obstacle avoidance
        </span>
        <span className="status-pill" style={{ color, background: `${color}18`, border: `1px solid ${color}40` }}>
          {unreachable ? "API OFFLINE" : !obstacles?.enabled ? "DISABLED" : state}
        </span>
      </div>

      <div className="radar">
        <div className="radar-ring r1" />
        <div className="radar-ring r2" />
        <div className="radar-cross" />
        <div className="radar-dot" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
        {dirs.map((d) => (
          <span key={d.key} className={`radar-dir ${d.cls}`}>
            <b>{fmt(obstacles?.distances_m?.[d.key])}</b>
            <i>{d.label}</i>
          </span>
        ))}
        <span className="radar-down">
          <b>{fmt(obstacles?.distances_m?.down)}</b>
          <i>AGL</i>
        </span>
      </div>

      <div className="obstacle-meta">
        {!obstacles || !obstacles.available
          ? "No rangefinders fitted — distances stay unknown until sensors are connected."
          : `Warning ${obstacles.thresholds_m.warning}m · Critical ${obstacles.thresholds_m.critical}m · Emergency ${obstacles.thresholds_m.emergency}m`}
      </div>
    </div>
  );
};
