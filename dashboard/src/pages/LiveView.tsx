import React from "react";
import { VideoPanel } from "../components/VideoPanel";
import { TelemetryData, ChargingStatus, PrecisionLandingStatus } from "../types";

interface LiveViewProps {
  telemetry: TelemetryData;
  charging: ChargingStatus | null;
  precision: PrecisionLandingStatus | null;
  onMarkDisease: (disease: string, confidence: number) => void;
  notify: (text: string, type?: "success" | "warning" | "error") => void;
}

export const LiveView: React.FC<LiveViewProps> = ({ telemetry, charging, precision, onMarkDisease, notify }) => {
  return (
    <div className="live-layout">
      <div className="live-main">
        <VideoPanel
          telemetry={telemetry}
          precision={precision}
          charging={!!charging?.charging}
          docked={!!charging?.docked}
          onMarkDisease={onMarkDisease}
          notify={notify}
        />
      </div>
    </div>
  );
};
