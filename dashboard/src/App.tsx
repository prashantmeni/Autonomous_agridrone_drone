import React, { useCallback, useEffect, useState } from "react";
import { NavRail, ViewId } from "./components/NavRail";
import { TopBar } from "./components/TopBar";
import { ConfirmModal } from "./components/ConfirmModal";
import { DroneBehaviourModal } from "./components/DroneBehaviourModal";
import { DiseaseMarkingModal } from "./components/DiseaseMarkingModal";
import { LiveView } from "./pages/LiveView";
import { DroneView } from "./pages/DroneView";
import { MissionPlanner } from "./pages/MissionPlanner";
import { PlantHealth } from "./pages/PlantHealth";
import { FlightHealth } from "./pages/FlightHealth";
import { Logs } from "./pages/Logs";
import { useTelemetry } from "./hooks/useTelemetry";
import { usePolling } from "./hooks/usePolling";
import { api } from "./services/api";

const VALID_VIEWS: ViewId[] = ["drone", "live", "survey", "crop", "health", "logs"];

const initialView = (): ViewId => {
  try {
    const saved = localStorage.getItem("agridrone_view");
    if (saved && (VALID_VIEWS as string[]).includes(saved)) return saved as ViewId;
  } catch {
    /* ignore storage errors */
  }
  return "drone";
};

export default function App() {
  const { telemetry, isConnected } = useTelemetry();
  const [view, setView] = useState<ViewId>(initialView);
  const changeView = useCallback((v: ViewId) => {
    setView(v);
    try {
      localStorage.setItem("agridrone_view", v);
    } catch {
      /* ignore storage errors */
    }
  }, []);
  const [emergencyOpen, setEmergencyOpen] = useState(false);
  const [behaviourOpen, setBehaviourOpen] = useState(false);
  const [apiReachable, setApiReachable] = useState(true);
  const [notification, setNotification] = useState<{
    text: string;
    type: "success" | "warning" | "error";
  } | null>(null);

  const [marking, setMarking] = useState<{
    isOpen: boolean;
    disease: string;
    confidence: number;
  }>({ isOpen: false, disease: "", confidence: 0 });

  const notify = useCallback((text: string, type: "success" | "warning" | "error" = "success") => {
    setNotification({ text, type });
    setTimeout(() => setNotification(null), 4000);
  }, []);

  // Companion API reachability
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        await api.getHealth();
        if (alive) setApiReachable(true);
      } catch {
        if (alive) setApiReachable(false);
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  // Slow-status polls (charge dock, obstacles, precision landing)
  const { data: charging } = usePolling(() => api.getChargingStatus(), 5000);
  const { data: obstacles, error: obstacleErr } = usePolling(() => api.getObstacleStatus(), 3000);
  const { data: precision } = usePolling(() => api.precisionLandingStatus(), 2000);
  const { data: detections, refresh: refreshDetections } = usePolling(() => api.getDetections(), 15000);

  // Command handlers
  const handleTakeoff = async (alt: number) => {
    try {
      await api.takeoff(alt);
      notify(`Auto takeoff to ${alt} m commanded`, "success");
    } catch (e: any) {
      notify(`Takeoff rejected: ${e.message}`, "error");
    }
  };

  const handleLand = async () => {
    try {
      const res = await api.land();
      if (res.skipped) notify("Land ignored — vehicle is disarmed", "warning");
      else notify("Auto landing at current position", "warning");
    } catch (e: any) {
      notify(`Land failed: ${e.message}`, "error");
    }
  };

  const handleRTL = async () => {
    try {
      const res = await api.rtl();
      if (res.skipped) notify("RTL ignored — vehicle is disarmed", "warning");
      else notify("Return-to-home engaged", "warning");
    } catch (e: any) {
      notify(`RTH failed: ${e.message}`, "error");
    }
  };

  const handlePrecisionLand = async () => {
    try {
      const res = await api.precisionLand();
      notify(`Precision landing: ${res.phase || "searching for pad"}`, "warning");
    } catch (e: any) {
      notify(`Precision landing: ${e.message}`, "error");
    }
  };

  const openMarking = (disease: string, confidence: number) => {
    setMarking({ isOpen: true, disease, confidence });
  };

  return (
    <div className="app-shell">
      <NavRail
        active={view}
        onSelect={changeView}
        onOpenBehaviour={() => setBehaviourOpen(true)}
        apiReachable={apiReachable}
      />

      <div className="app-body">
        <TopBar
          telemetry={telemetry}
          isConnected={isConnected}
          charging={charging}
          onEmergency={() => setEmergencyOpen(true)}
        />

        <main className="app-content">
          {view === "drone" && (
            <DroneView
              telemetry={telemetry}
              charging={charging}
              obstacles={obstacles}
              obstaclesUnreachable={!!obstacleErr}
              detections={detections || []}
              onTakeoff={handleTakeoff}
              onLand={handleLand}
              onRTL={handleRTL}
              onPrecisionLand={handlePrecisionLand}
              notify={notify}
            />
          )}
          {view === "live" && (
            <LiveView
              telemetry={telemetry}
              charging={charging}
              precision={precision}
              onMarkDisease={openMarking}
              notify={notify}
            />
          )}
          {view === "survey" && <MissionPlanner telemetry={telemetry} />}
          {view === "crop" && <PlantHealth telemetry={telemetry} onMarkDetection={openMarking} />}
          {view === "health" && <FlightHealth telemetry={telemetry} />}
          {view === "logs" && <Logs />}
        </main>
      </div>

      {notification && (
        <div className={`toast toast-${notification.type}`}>{notification.text}</div>
      )}

      <ConfirmModal
        isOpen={emergencyOpen}
        title="EMERGENCY"
        message="Triggers the failsafe state machine: immediate return-to-home / emergency descent. Use only in a critical situation."
        confirmText="EXECUTE EMERGENCY"
        confirmStyle="danger"
        onConfirm={handleRTL}
        onClose={() => setEmergencyOpen(false)}
      />

      <DroneBehaviourModal
        isOpen={behaviourOpen}
        onClose={() => setBehaviourOpen(false)}
        onSave={() => notify("Drone behaviour rules saved to the companion computer", "success")}
      />

      <DiseaseMarkingModal
        isOpen={marking.isOpen}
        initialDisease={marking.disease}
        initialConfidence={marking.confidence}
        latitude={telemetry.lat ?? 0}
        longitude={telemetry.lon ?? 0}
        onClose={() => setMarking((prev) => ({ ...prev, isOpen: false }))}
        onSuccess={() => {
          notify("Disease zone plotted on the map", "success");
          refreshDetections();
        }}
      />
    </div>
  );
}
