import React, { useEffect, useState } from "react";
import { AlertTriangle, BatteryWarning, Eye, Satellite, ShieldAlert } from "lucide-react";
import { api } from "../services/api";
import { FailsafeStatus, HazardInfo } from "../types";

/**
 * Persistent safety strip shown on every view.
 *
 * It reports what the companion computer's failsafe supervisor and the
 * forward-camera detector have actually seen. The camera only raises alerts —
 * it never commands the aircraft — so nothing here implies a manoeuvre.
 */

const ADVICE: Record<string, { text: string; color: string }> = {
  BATTERY_CRITICAL: { text: "BATTERY CRITICAL", color: "#ef4444" },
  BATTERY_LOW: { text: "BATTERY LOW", color: "#fbbf24" },
  MAVLINK_LOST: { text: "MAVLINK LINK LOST", color: "#f87171" },
  GPS_DEGRADED: { text: "GPS DEGRADED", color: "#fbbf24" },
  EKF_ESTIMATE_ERROR: { text: "EKF ESTIMATE ERROR", color: "#f87171" },
};

export const SafetyBanner: React.FC = () => {
  const [failsafe, setFailsafe] = useState<FailsafeStatus | null>(null);
  const [hazard, setHazard] = useState<HazardInfo | null>(null);

  useEffect(() => {
    const poll = () => api.getFailsafe().then(setFailsafe).catch(() => {});
    poll();
    const id = setInterval(poll, 2000);
    return () => clearInterval(id);
  }, []);

  // Camera hazards arrive over the telemetry socket as {"type": "hazard"}.
  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      if (closed) return;
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      ws = new WebSocket(`${proto}//${window.location.host}/ws/telemetry`);
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg?.type === "hazard" && msg.data) setHazard(msg.data);
        } catch {
          /* ignore malformed frames */
        }
      };
      ws.onclose = () => {
        if (!closed) retry = setTimeout(connect, 2500);
      };
    };
    connect();

    return () => {
      closed = true;
      if (retry) clearTimeout(retry);
      ws?.close();
    };
  }, []);

  const advice = failsafe?.report?.advice ?? [];
  const chips = advice.filter((a) => ADVICE[a]).map((a) => ADVICE[a]);
  const hazardActive = Boolean(hazard?.hazard);

  if (chips.length === 0 && !hazardActive) return null;

  return (
    <div
      data-testid="safety-banner"
      style={{
        display: "flex",
        gap: 8,
        flexWrap: "wrap",
        padding: "6px 14px",
        borderBottom: "1px solid var(--border-color)",
        background: hazardActive ? "rgba(239, 68, 68, 0.10)" : "rgba(251, 191, 36, 0.08)",
      }}
    >
      {chips.map((c) => (
        <span
          key={c.text}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 5,
            fontSize: "0.7rem",
            fontWeight: 700,
            letterSpacing: "0.04em",
            color: c.color,
            background: `${c.color}18`,
            border: `1px solid ${c.color}40`,
            borderRadius: "var(--radius-sm)",
            padding: "2px 8px",
          }}
        >
          {c.text === "BATTERY LOW" || c.text === "BATTERY CRITICAL" ? (
            <BatteryWarning size={11} />
          ) : c.text === "GPS DEGRADED" ? (
            <Satellite size={11} />
          ) : (
            <ShieldAlert size={11} />
          )}
          {c.text}
        </span>
      ))}

      {hazardActive && (
        <span
          data-testid="hazard-chip"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 5,
            fontSize: "0.7rem",
            fontWeight: 700,
            color: "#ef4444",
            background: "rgba(239, 68, 68, 0.18)",
            border: "1px solid rgba(239, 68, 68, 0.4)",
            borderRadius: "var(--radius-sm)",
            padding: "2px 8px",
          }}
        >
          <Eye size={11} />
          {hazard?.kind === "person" ? "PERSON AHEAD" : "OBSTACLE AHEAD"}
          {hazard?.confidence ? ` ${Math.round((hazard.confidence ?? 0) * 100)}%` : ""}
        </span>
      )}

      {chips.length > 0 && (
        <span style={{ fontSize: "0.68rem", color: "var(--text-dim)", alignSelf: "center" }}>
          <AlertTriangle size={10} style={{ verticalAlign: "-1px" }} /> failsafe supervisor
        </span>
      )}
    </div>
  );
};