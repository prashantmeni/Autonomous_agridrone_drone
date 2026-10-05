import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { PreflightReport, TelemetryData } from "../types";
import { ShieldCheck, ShieldAlert, Cpu, HardDrive, Thermometer, Satellite, Radio, Battery, RefreshCw, Wifi, Zap } from "lucide-react";

interface FlightHealthProps {
  telemetry: TelemetryData;
}

interface HealthCard {
  label: string;
  value: string | number;
  unit?: string;
  sub?: string;
  status: "ok" | "warn" | "error";
  pct?: number;
  icon: React.ReactNode;
}

export const FlightHealth: React.FC<FlightHealthProps> = ({ telemetry }) => {
  const [report, setReport] = useState<PreflightReport | null>(null);
  const [healthData, setHealthData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const fetchChecks = async () => {
    setLoading(true);
    try {
      const [preflight, health] = await Promise.all([
        api.getPreflight().catch(() => null),
        api.getHealth().catch(() => null),
      ]);
      if (preflight) setReport(preflight);
      if (health) setHealthData(health);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChecks();
    const interval = setInterval(fetchChecks, 4000);
    return () => clearInterval(interval);
  }, []);

  const isGo = report?.ready ?? (telemetry.satellites >= 8 && telemetry.battery_remaining_pct >= 25);

  const battPct = telemetry.battery_remaining_pct;
  const battStatus: "ok" | "warn" | "error" = battPct <= 15 ? "error" : battPct <= 25 ? "warn" : "ok";

  const gpsOk = telemetry.satellites >= 8;
  const cpuPct = healthData?.pi?.cpu_pct;
  const tempC = healthData?.pi?.temp_c;
  const diskFree = healthData?.pi?.disk_free_pct;

  const statusColor = { ok: "#10b981", warn: "#f59e0b", error: "#ef4444" };
  const statusBg = { ok: "rgba(16, 185, 129, 0.1)", warn: "rgba(245, 158, 11, 0.1)", error: "rgba(239, 68, 68, 0.1)" };

  const flightCards: HealthCard[] = [
    {
      label: "Battery", value: battPct >= 0 ? `${battPct}%` : "—", unit: `${telemetry.battery_voltage_v.toFixed(1)}V`,
      sub: battStatus === "error" ? "CRITICAL — RTH NOW" : battStatus === "warn" ? "Low — consider landing" : "Nominal",
      status: battStatus, pct: battPct, icon: <Battery size={22} />,
    },
    {
      label: "GNSS / GPS", value: `${telemetry.satellites} sats`, unit: `HDOP ${telemetry.hdop.toFixed(1)}`,
      sub: gpsOk ? `3D Fix — ${telemetry.gps_fix >= 3 ? "RTK" : "Standard"}` : "Searching for satellites",
      status: gpsOk ? "ok" : "warn", pct: Math.min(100, (telemetry.satellites / 14) * 100), icon: <Satellite size={22} />,
    },
    {
      label: "MAVLink Link", value: telemetry.connected ? "CONNECTED" : "NO SIGNAL", unit: "",
      sub: telemetry.connected ? "Heartbeat within 5s threshold" : "Check USB/radio connection",
      status: telemetry.connected ? "ok" : "error", pct: telemetry.connected ? 100 : 0, icon: <Wifi size={22} />,
    },
    {
      label: "RC Transmitter",
      value: telemetry.rc_connected ? "CONNECTED" : telemetry.rc_age_s != null && telemetry.rc_age_s >= 0 ? "SIGNAL LOST" : "NO RC",
      unit: telemetry.rc_connected && telemetry.rc_rssi != null && telemetry.rc_rssi >= 0 ? `RSSI ${telemetry.rc_rssi}` : "",
      sub: telemetry.rc_connected
        ? `Handheld controller linked${telemetry.rc_channels ? ` · ${telemetry.rc_channels} channels` : ""}`
        : telemetry.rc_age_s != null && telemetry.rc_age_s >= 0
          ? `Last signal ${Math.round(telemetry.rc_age_s)}s ago — check transmitter`
          : "No RC stream — flying via MAVLink/autonomous control",
      status: telemetry.rc_connected ? "ok" : telemetry.rc_age_s != null && telemetry.rc_age_s >= 0 ? "error" : "warn",
      pct: telemetry.rc_connected ? 100 : 0, icon: <Radio size={22} />,
    },
  ];

  const hwCards: HealthCard[] = [
    {
      label: "CPU Load", value: cpuPct != null ? `${cpuPct}%` : "—", unit: cpuPct != null ? "Companion CPU" : "no data",
      sub: cpuPct == null ? "Companion stats unavailable" : cpuPct < 60 ? "Healthy" : cpuPct < 80 ? "Elevated" : "High — check processes",
      status: cpuPct == null ? "warn" : cpuPct >= 85 ? "warn" : "ok", pct: cpuPct, icon: <Cpu size={22} />,
    },
    {
      label: "Thermal", value: tempC != null ? `${tempC.toFixed(1)}°C` : "—", unit: tempC != null ? "SoC temperature" : "no data",
      sub: tempC == null ? "Companion stats unavailable" : tempC < 65 ? "Normal" : tempC < 75 ? "Warm" : "THROTTLING IMMINENT",
      status: tempC == null ? "warn" : tempC >= 75 ? "error" : tempC >= 65 ? "warn" : "ok", pct: tempC != null ? (tempC / 85) * 100 : undefined, icon: <Thermometer size={22} />,
    },
    {
      label: "Storage", value: diskFree != null ? `${diskFree}%` : "—", unit: diskFree != null ? "Free" : "no data",
      sub: diskFree == null ? "Companion stats unavailable" : diskFree > 20 ? "Adequate logging capacity" : "Low — clear old logs",
      status: diskFree == null ? "warn" : diskFree < 10 ? "error" : diskFree < 20 ? "warn" : "ok", pct: diskFree, icon: <HardDrive size={22} />,
    },
  ];

  const renderCard = (card: HealthCard, idx: number) => (
    <div key={idx} className="card" style={{ background: statusBg[card.status], borderColor: `${statusColor[card.status]}25` }}>
      <div style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
        <div style={{
          width: 44, height: 44, borderRadius: "var(--radius-md)", flexShrink: 0,
          background: `${statusColor[card.status]}18`,
          color: statusColor[card.status],
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          {card.icon}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: "0.68rem", color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 600, letterSpacing: "0.05em" }}>
            {card.label}
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 2 }}>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "1.3rem", fontWeight: 800, color: statusColor[card.status] }}>
              {card.value}
            </span>
            {card.unit && (
              <span style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>{card.unit}</span>
            )}
          </div>
          <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: 2 }}>{card.sub}</div>
          {card.pct !== undefined && (
            <div className="health-bar-track" style={{ marginTop: 8 }}>
              <div
                className="health-bar-fill"
                style={{ width: `${Math.max(0, Math.min(100, card.pct))}%`, background: statusColor[card.status] }}
              />
            </div>
          )}
        </div>
        <span className="badge" style={{
          background: statusBg[card.status],
          color: statusColor[card.status],
          border: `1px solid ${statusColor[card.status]}40`,
          fontSize: "0.6rem",
          flexShrink: 0,
        }}>
          {card.status === "ok" ? "OK" : card.status === "warn" ? "WARN" : "FAIL"}
        </span>
      </div>
    </div>
  );

  return (
    <div style={{ padding: "20px 24px", maxWidth: 1100, margin: "0 auto", overflowY: "auto", height: "100%" }}>
      {/* Page Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <div>
          <h2 style={{ fontSize: "1.2rem", fontWeight: 700, display: "flex", alignItems: "center", gap: 10 }}>
            {isGo ? <ShieldCheck size={24} color="#10b981" /> : <ShieldAlert size={24} color="#f59e0b" />}
            Drone Health & Diagnostics
          </h2>
          <p style={{ fontSize: "0.82rem", color: "var(--text-muted)", marginTop: 2 }}>
            Real-time autopilot sensors, companion computer, and safety interlocks
          </p>
        </div>
        <button className="btn btn-sm btn-secondary" onClick={fetchChecks} disabled={loading}>
          <RefreshCw size={13} style={{ animation: loading ? "spin 1s linear infinite" : "none" }} />
          {loading ? "Checking..." : "Refresh"}
        </button>
      </div>

      {/* Go/No-Go Banner */}
      <div className="card" style={{
        marginBottom: 20,
        background: isGo ? "rgba(16, 185, 129, 0.08)" : "rgba(245, 158, 11, 0.08)",
        borderColor: isGo ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.3)",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div style={{
              width: 48, height: 48, borderRadius: "var(--radius-md)", flexShrink: 0,
              background: isGo ? "rgba(16, 185, 129, 0.15)" : "rgba(245, 158, 11, 0.15)",
              color: isGo ? "#34d399" : "#fbbf24",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              {isGo ? <ShieldCheck size={28} /> : <ShieldAlert size={28} />}
            </div>
            <div>
              <div style={{ fontSize: "1.05rem", fontWeight: 800, color: isGo ? "#34d399" : "#fbbf24" }}>
                PREFLIGHT: {isGo ? "ALL SYSTEMS GO" : "HOLD — VERIFY BEFORE FLIGHT"}
              </div>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: 2 }}>
                {isGo
                  ? "All safety interlocks, geofence parameters, and battery thresholds are met."
                  : "One or more preflight checks have not cleared. Resolve issues before arming."}
              </p>
            </div>
          </div>
          <span className="badge" style={{
            fontSize: "0.82rem", padding: "8px 18px",
            background: isGo ? "rgba(16, 185, 129, 0.15)" : "rgba(245, 158, 11, 0.15)",
            color: isGo ? "#34d399" : "#fbbf24",
            border: `1px solid ${isGo ? "rgba(16, 185, 129, 0.4)" : "rgba(245, 158, 11, 0.4)"}`,
          }}>
            {isGo ? "✓ CLEARED" : "⚠ HOLD"}
          </span>
        </div>

        {report?.issues && report.issues.length > 0 && (
          <div style={{ marginTop: 14, borderTop: "1px solid var(--border-color)", paddingTop: 12 }}>
            <div style={{ fontSize: "0.7rem", color: "#f87171", fontWeight: 700, textTransform: "uppercase", marginBottom: 8, letterSpacing: "0.05em" }}>
              Issues Found:
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {report.issues.map((issue, idx) => (
                <span key={idx} className="badge" style={{ background: "rgba(239, 68, 68, 0.12)", color: "#f87171", border: "1px solid rgba(239, 68, 68, 0.25)", fontSize: "0.7rem" }}>
                  ⚠ {issue}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Flight Systems */}
      <div style={{ fontSize: "0.68rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-dim)", marginBottom: 10 }}>
        Autopilot & Telemetry
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12, marginBottom: 20 }}>
        {flightCards.map(renderCard)}
      </div>

      {/* Companion Computer */}
      <div style={{ fontSize: "0.68rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--text-dim)", marginBottom: 10 }}>
        Raspberry Pi Companion
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
        {hwCards.map(renderCard)}
      </div>
    </div>
  );
};
