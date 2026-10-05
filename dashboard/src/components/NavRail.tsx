import React from "react";
import { Video, Map, Leaf, Activity, Sliders, Radio, Drone } from "lucide-react";

export type ViewId = "drone" | "live" | "survey" | "crop" | "health" | "logs";

interface NavRailProps {
  active: ViewId;
  onSelect: (v: ViewId) => void;
  onOpenBehaviour: () => void;
  apiReachable: boolean;
}

const ITEMS: { id: ViewId; label: string; icon: React.ReactNode }[] = [
  { id: "drone", label: "Drone", icon: <Drone size={20} /> },
  { id: "live", label: "Live", icon: <Video size={20} /> },
  { id: "survey", label: "Survey", icon: <Map size={20} /> },
  { id: "crop", label: "Crop AI", icon: <Leaf size={20} /> },
  { id: "health", label: "Health", icon: <Activity size={20} /> },
  { id: "logs", label: "Logs", icon: <Radio size={20} /> },
];

export const NavRail: React.FC<NavRailProps> = ({ active, onSelect, onOpenBehaviour, apiReachable }) => {
  return (
    <nav className="rail">
      <div className="rail-brand" title="AgriDrone">
        <span className="rail-brand-mark">A</span>
      </div>

      <div className="rail-items">
        {ITEMS.map((item) => (
          <button
            key={item.id}
            className={`rail-btn ${active === item.id ? "active" : ""}`}
            onClick={() => onSelect(item.id)}
            title={item.label}
          >
            {item.icon}
            <span className="rail-btn-label">{item.label}</span>
          </button>
        ))}
      </div>

      <div className="rail-footer">
        <button className="rail-btn" onClick={onOpenBehaviour} title="Drone behaviour settings">
          <Sliders size={20} />
          <span className="rail-btn-label">Rules</span>
        </button>
        <div
          className={`rail-link ${apiReachable ? "ok" : "down"}`}
          title={apiReachable ? "Companion API reachable" : "Companion API unreachable"}
        >
          <span className="pulse-dot" />
        </div>
      </div>
    </nav>
  );
};
