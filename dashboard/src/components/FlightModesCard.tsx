import React, { useCallback, useEffect, useState } from "react";
import { Compass, Layers, MapPinned, Plane, RefreshCw, ScanEye } from "lucide-react";
import { api } from "../services/api";
import { ModesStatus } from "../types";

/**
 * Farm flight modes: SURVEY, MONITOR, MAPPING and RTL.
 *
 * These wrap the raw PX4 mode switch with the sequencing each mode needs — the
 * RTL altitude is pushed to the flight controller first and the vehicle climbs
 * to its working altitude. A disarmed request is refused by the backend and
 * reported here rather than silently accepted.
 */

const MODE_META: Record<string, { label: string; icon: React.ReactNode; blurb: string }> = {
  SURVEY: {
    label: "SURVEY",
    icon: <Layers size={13} />,
    blurb: "Fly the uploaded coverage mission",
  },
  MAPPING: {
    label: "MAPPING",
    icon: <MapPinned size={13} />,
    blurb: "Survey flown for mapping data",
  },
  MONITOR: {
    label: "MONITOR",
    icon: <ScanEye size={13} />,
    blurb: "Hold altitude and scan crops",
  },
  RTL: {
    label: "RTL",
    icon: <Plane size={13} />,
    blurb: "Return to launch point",
  },
};

export const FlightModesCard: React.FC<{ notify?: (t: string, k?: "success" | "warning" | "error") => void }> = ({
  notify,
}) => {
  const [modes, setModes] = useState<ModesStatus | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.getModes().then(setModes).catch(() => {});
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
  }, [refresh]);

  const enter = async (mode: string) => {
    setBusy(mode);
    try {
      const res = await api.enterMode(mode);
      notify?.(
        res.status === "ok"
          ? `${mode} active (PX4 ${res.px4_mode}${res.altitude_m ? `, ${res.altitude_m} m` : ""})`
          : `${mode}: ${res.error || res.status}`,
        res.status === "ok" ? "success" : "warning"
      );
    } catch (e: any) {
      notify?.(`${mode} refused: ${e.message}`, "error");
    } finally {
      setBusy(null);
      refresh();
    }
  };

  const available = modes?.available ?? ["SURVEY", "MAPPING", "MONITOR", "RTL"];

  return (
    <div className="card" style={{ padding: 16 }}>
      <div className="card-header" style={{ marginBottom: 8 }}>
        <span className="card-title" style={{ fontSize: "0.88rem" }}>
          <Compass size={14} /> Flight modes
        </span>
        <button className="btn btn-sm btn-ghost" onClick={refresh} title="Refresh">
          <RefreshCw size={11} />
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 8 }}>
        {available.map((mode) => {
          const meta = MODE_META[mode];
          const active = modes?.current === mode;
          return (
            <button
              key={mode}
              data-testid={`mode-${mode}`}
              className={`btn btn-sm ${active ? "btn-primary" : "btn-ghost"}`}
              disabled={busy !== null || !modes?.armed}
              onClick={() => enter(mode)}
              title={
                modes?.armed
                  ? meta?.blurb || mode
                  : "Vehicle is disarmed — take off before entering a mode"
              }
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                gap: 2,
                padding: "8px 10px",
                opacity: modes?.armed ? 1 : 0.55,
              }}
            >
              <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                {meta?.icon}
                {meta?.label || mode}
              </span>
              <span style={{ fontSize: "0.62rem", color: "var(--text-dim)", fontWeight: 400 }}>
                {busy === mode ? "sending…" : meta?.blurb || ""}
              </span>
            </button>
          );
        })}
      </div>

      {modes && (
        <div
          style={{
            marginTop: 10,
            display: "grid",
            gridTemplateColumns: "repeat(2, 1fr)",
            gap: 4,
            fontSize: "0.68rem",
            color: "var(--text-dim)",
            fontFamily: "var(--font-mono)",
          }}
        >
          <span>survey {modes.survey_altitude_m} m</span>
          <span>cruise {modes.cruise_speed_mps} m/s</span>
          <span>RTH alt {modes.rth_altitude_m} m</span>
          <span>battery reserve {modes.rth_battery_reserve_pct}%</span>
        </div>
      )}
    </div>
  );
};