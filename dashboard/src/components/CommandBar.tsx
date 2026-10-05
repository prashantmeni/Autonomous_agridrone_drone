import React, { useState } from "react";
import { Play, ArrowDown, Crosshair, Home, ShieldCheck, ShieldAlert, ChevronUp, ChevronDown } from "lucide-react";
import { TelemetryData } from "../types";
import { api } from "../services/api";
import { ConfirmModal } from "./ConfirmModal";

interface CommandBarProps {
  telemetry: TelemetryData;
  chargingDocked: boolean;
  onTakeoff: (alt: number) => void;
  onLand: () => void;
  onRTL: () => void;
  onPrecisionLand: () => void;
  notify: (text: string, type?: "success" | "warning" | "error") => void;
}

export const CommandBar: React.FC<CommandBarProps> = ({
  telemetry,
  chargingDocked,
  onTakeoff,
  onLand,
  onRTL,
  onPrecisionLand,
  notify,
}) => {
  const [takeoffAlt, setTakeoffAlt] = useState<number>(15);
  const [armModal, setArmModal] = useState<boolean>(false);
  const [altOpen, setAltOpen] = useState<boolean>(false);

  const airborne =
    telemetry.relative_alt_m > 1.0 ||
    ["TAKEOFF", "MISSION", "SURVEY", "RETURN_HOME", "PRECISION_LANDING", "LANDING"].includes(
      telemetry.fsm_state || ""
    );

  const handleArm = async () => {
    try {
      if (telemetry.armed) {
        await api.disarm();
        notify("Motors disarmed", "success");
      } else {
        await api.arm();
        notify("Motors armed", "success");
      }
    } catch (e: any) {
      notify(`Command failed: ${e.message}`, "error");
    }
  };

  const altOptions = [10, 15, 20, 25, 30];

  return (
    <div className="cmd-bar">
      {/* Arm / disarm */}
      <button
        className={`cmd-btn ${telemetry.armed ? "cmd-warning" : "cmd-ghost"}`}
        onClick={() => setArmModal(true)}
        title={telemetry.armed ? "Disarm motors" : "Arm motors"}
      >
        {telemetry.armed ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
        <span>{telemetry.armed ? "Disarm" : "Arm"}</span>
      </button>

      <div className="cmd-sep" />

      {!airborne ? (
        <div className="cmd-alt">
          <button className="cmd-btn cmd-primary" onClick={() => onTakeoff(takeoffAlt)} title="Auto takeoff">
            <Play size={15} fill="currentColor" />
            <span>Take off</span>
          </button>
          <button className="cmd-alt-chip" onClick={() => setAltOpen(!altOpen)} title="Takeoff altitude">
            {takeoffAlt} m
            <ChevronUp size={11} style={{ transform: altOpen ? "rotate(180deg)" : "none" }} />
          </button>
          {altOpen && (
            <div className="cmd-alt-menu">
              {altOptions.map((a) => (
                <button
                  key={a}
                  className={a === takeoffAlt ? "active" : ""}
                  onClick={() => {
                    setTakeoffAlt(a);
                    setAltOpen(false);
                  }}
                >
                  {a} m
                </button>
              ))}
            </div>
          )}
        </div>
      ) : (
        <button className="cmd-btn cmd-ghost" onClick={() => onLand()} title="Auto land at current position">
          <ArrowDown size={16} />
          <span>Land</span>
        </button>
      )}

      <button
        className="cmd-btn cmd-cyan"
        onClick={onPrecisionLand}
        disabled={!airborne}
        title="Vision-guided precision landing on the charging pad"
      >
        <Crosshair size={16} />
        <span>Precision land</span>
      </button>

      <button className="cmd-btn cmd-amber" onClick={onRTL} title="Return to home" disabled={!telemetry.connected}>
        <Home size={16} />
        <span>Return home</span>
      </button>

      {chargingDocked && (
        <>
          <div className="cmd-sep" />
          <div className="cmd-charging">
            <ChevronDown size={12} />
            Docked
          </div>
        </>
      )}

      <ConfirmModal
        isOpen={armModal}
        title={telemetry.armed ? "Disarm motors" : "Arm motors"}
        message={
          telemetry.armed
            ? "Disarming cuts motor power immediately. Only do this once the drone is safely on the ground."
            : "Arming will spin the propellers. Confirm the takeoff area is clear of people and obstacles."
        }
        checklist={
          !telemetry.armed
            ? [
                `GPS ${telemetry.gps_fix >= 3 ? "3D confirmed" : "optional (not required)"}`,
                `Battery ${battText(telemetry.battery_remaining_pct)}`,
                "Geofence loaded",
                "Takeoff area clear",
              ]
            : undefined
        }
        confirmText={telemetry.armed ? "Disarm" : "Arm"}
        confirmStyle={telemetry.armed ? "warning" : "danger"}
        onConfirm={handleArm}
        onClose={() => setArmModal(false)}
      />
    </div>
  );
};

function battText(pct: number) {
  if (pct < 0) return "unknown";
  return `${pct}%`;
}
