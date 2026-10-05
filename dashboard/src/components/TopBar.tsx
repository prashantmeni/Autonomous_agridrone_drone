import React from "react";
import {
  Plane,
  BatteryCharging,
  BatteryLow,
  Battery,
  Satellite,
  Wifi,
  WifiOff,
  Zap,
  AlertOctagon,
  Radio,
} from "lucide-react";
import { TelemetryData, ChargingStatus } from "../types";

interface TopBarProps {
  telemetry: TelemetryData;
  isConnected: boolean;
  charging: ChargingStatus | null;
  onEmergency: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ telemetry, isConnected, charging, onEmergency }) => {
  const batt = telemetry.battery_remaining_pct;
  const BattIcon = !telemetry.connected ? WifiOff : batt <= 15 ? BatteryLow : batt <= 25 ? Battery : BatteryCharging;
  const battColor = batt < 0 ? "var(--text-dim)" : batt <= 15 ? "#ef4444" : batt <= 25 ? "#f59e0b" : "#34d399";

  return (
    <header className="top-bar">
      <div className="tb-left">
        <div className="tb-brand">
          <Plane size={16} />
          <span>AgriDrone</span>
          <em>GCS</em>
        </div>

        <span className={`badge ${isConnected ? "badge-connected" : "badge-disconnected"}`}>
          <span className="pulse-dot" />
          {isConnected ? "LINKED" : "NO LINK"}
        </span>

        <span className="badge badge-fsm">{telemetry.fsm_state || "DISARMED"}</span>

        {telemetry.rc_connected ? (
          <span
            className="badge badge-connected"
            title={`Transmitter link${telemetry.rc_rssi != null && telemetry.rc_rssi >= 0 ? ` · RSSI ${telemetry.rc_rssi}` : ""}${telemetry.rc_channels ? ` · ${telemetry.rc_channels} ch` : ""}`}
          >
            <Radio size={10} /> RC{telemetry.rc_rssi != null && telemetry.rc_rssi >= 0 ? ` ${telemetry.rc_rssi}` : ""}
          </span>
        ) : telemetry.rc_age_s != null && telemetry.rc_age_s >= 0 ? (
          <span className="badge badge-armed" title={`Transmitter signal lost ${Math.round(telemetry.rc_age_s)}s ago`}>
            <Radio size={10} /> RC LOST
          </span>
        ) : (
          <span className="badge" style={{ opacity: 0.6 }} title="No RC transmitter stream — MAVLink/autonomous control">
            <Radio size={10} /> NO RC
          </span>
        )}

        {telemetry.armed && <span className="badge badge-armed">ARMED</span>}
        {charging?.charging && (
          <span className="badge badge-charging">
            <Zap size={10} fill="currentColor" /> CHARGING
          </span>
        )}
      </div>

      <div className="tb-right">
        <div className="tb-stat" title="Battery">
          <BattIcon size={15} color={battColor} />
          <b style={{ color: battColor }}>
            {batt >= 0 ? `${batt}%` : "—"}
            <i>{telemetry.battery_voltage_v > 0 ? `${telemetry.battery_voltage_v.toFixed(1)}V` : ""}</i>
          </b>
        </div>

        <div className="tb-stat" title="GNSS">
          <Satellite size={14} color={telemetry.satellites >= 8 ? "#34d399" : "#f59e0b"} />
          <b>{telemetry.connected ? `${telemetry.satellites} SAT` : "—"}</b>
        </div>

        <div className="tb-stat" title="Data link">
          {isConnected ? <Wifi size={14} color="#34d399" /> : <WifiOff size={14} color="#ef4444" />}
        </div>

        <button className="btn btn-danger btn-sm tb-emergency" onClick={onEmergency}>
          <AlertOctagon size={14} />
          EMERGENCY
        </button>
      </div>
    </header>
  );
};
