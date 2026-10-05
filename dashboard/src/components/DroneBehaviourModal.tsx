import React, { useEffect, useState } from "react";
import { Sliders, Shield, Home, Navigation, X, Check } from "lucide-react";
import { api } from "../services/api";
import { BehaviourRules } from "../types";

interface DroneBehaviourModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave?: (settings: BehaviourRules) => void;
}

export const DroneBehaviourModal: React.FC<DroneBehaviourModalProps> = ({
  isOpen,
  onClose,
  onSave,
}) => {
  const [rules, setRules] = useState<BehaviourRules>({
    survey_altitude_m: 20,
    cruise_speed_mps: 5.5,
    overlap_pct: 70,
    rth_altitude_m: 25,
    rth_battery_reserve_pct: 25,
    obstacle_brake_distance_m: 3,
    auto_charge_on_landing: true,
    auto_nadir_lock: true,
    geofence_enabled: true,
    landing_pad_id: "PAD_01",
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    api
      .getBehaviour()
      .then(setRules)
      .catch((e) => setError(`Rules unavailable: ${e.message}`))
      .finally(() => setLoading(false));
  }, [isOpen]);

  if (!isOpen) return null;

  const set = <K extends keyof BehaviourRules>(key: K, value: BehaviourRules[K]) =>
    setRules((prev) => ({ ...prev, [key]: value }));

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const saved = await api.setBehaviour(rules);
      if (onSave) onSave(saved);
      onClose();
    } catch (e: any) {
      setError(`Save failed: ${e.message}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal-card"
        style={{ maxWidth: 580, width: "95%" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: "var(--radius-md)",
              background: "rgba(16, 185, 129, 0.15)",
              color: "#10b981",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Sliders size={22} />
          </div>
          <div>
            <h3 style={{ fontSize: "1.15rem", fontWeight: 700 }}>Autonomous Drone Behaviour & Rules</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Autopilot mission heuristics, obstacle limits & induction charging rules
            </p>
          </div>
          <button
            onClick={onClose}
            style={{ marginLeft: "auto", background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
          >
            <X size={20} />
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 18, maxHeight: "65vh", overflowY: "auto", paddingRight: 4 }}>
          {loading && <p className="muted-text">Loading rules from companion computer…</p>}
          {error && <p className="muted-text" style={{ color: "#f87171" }}>{error}</p>}

          {/* Autonomous Surveying Behaviour */}
          <div className="card" style={{ padding: 14 }}>
            <span className="card-title" style={{ fontSize: "0.88rem", marginBottom: 12 }}>
              <Navigation size={16} color="#10b981" />
              Field surveying
            </span>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Survey altitude: {rules.survey_altitude_m} m</label>
                <input
                  type="range"
                  min={10}
                  max={45}
                  value={rules.survey_altitude_m}
                  onChange={(e) => set("survey_altitude_m", Number(e.target.value))}
                  style={{ accentColor: "#10b981" }}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Cruise speed: {rules.cruise_speed_mps} m/s</label>
                <input
                  type="range"
                  min={2}
                  max={10}
                  step={0.5}
                  value={rules.cruise_speed_mps}
                  onChange={(e) => set("cruise_speed_mps", Number(e.target.value))}
                  style={{ accentColor: "#10b981" }}
                />
              </div>
            </div>

            <div className="toggle-row">
              <span>Auto-lock camera to nadir during survey</span>
              <input
                type="checkbox"
                checked={rules.auto_nadir_lock}
                onChange={(e) => set("auto_nadir_lock", e.target.checked)}
                style={{ width: 18, height: 18, accentColor: "#10b981" }}
              />
            </div>
          </div>

          {/* Smart Return To Home (RTH) Logic */}
          <div className="card" style={{ padding: 14 }}>
            <span className="card-title" style={{ fontSize: "0.88rem", marginBottom: 12 }}>
              <Home size={16} color="#f59e0b" />
              Return to home
            </span>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Climb altitude: {rules.rth_altitude_m} m</label>
                <input
                  type="range"
                  min={15}
                  max={60}
                  value={rules.rth_altitude_m}
                  onChange={(e) => set("rth_altitude_m", Number(e.target.value))}
                  style={{ accentColor: "#f59e0b" }}
                />
                <span style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>Clears trees, lines and structures</span>
              </div>

              <div className="form-group">
                <label className="form-label">Battery reserve: {rules.rth_battery_reserve_pct}%</label>
                <input
                  type="range"
                  min={15}
                  max={35}
                  value={rules.rth_battery_reserve_pct}
                  onChange={(e) => set("rth_battery_reserve_pct", Number(e.target.value))}
                  style={{ accentColor: "#f59e0b" }}
                />
                <span style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>Auto-return triggered at this level</span>
              </div>
            </div>
          </div>

          {/* Obstacle Avoidance & Wireless Charging */}
          <div className="card" style={{ padding: 14 }}>
            <span className="card-title" style={{ fontSize: "0.88rem", marginBottom: 12 }}>
              <Shield size={16} color="#06b6d4" />
              Obstacles &amp; charging
            </span>

            <div className="form-group">
              <label className="form-label">Emergency brake distance: {rules.obstacle_brake_distance_m} m</label>
              <input
                type="range"
                min={1.5}
                max={6.0}
                step={0.5}
                value={rules.obstacle_brake_distance_m}
                onChange={(e) => set("obstacle_brake_distance_m", Number(e.target.value))}
                style={{ accentColor: "#06b6d4" }}
              />
            </div>

            <div className="toggle-row">
              <div>
                <div style={{ fontSize: "0.82rem", color: "var(--text-main)", fontWeight: 600 }}>
                  Auto wireless charging after landing
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>
                  Engages induction charging once docked on the pad
                </div>
              </div>
              <input
                type="checkbox"
                checked={rules.auto_charge_on_landing}
                onChange={(e) => set("auto_charge_on_landing", e.target.checked)}
                style={{ width: 18, height: 18, accentColor: "#10b981" }}
              />
            </div>

            <div className="form-group" style={{ marginTop: 10 }}>
              <label className="form-label">Landing pad ID</label>
              <input
                type="text"
                className="form-control"
                value={rules.landing_pad_id}
                onChange={(e) => set("landing_pad_id", e.target.value)}
              />
            </div>
          </div>
        </div>

        <div className="modal-actions" style={{ marginTop: 20 }}>
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={handleSave} disabled={saving || loading}>
            <Check size={16} />
            {saving ? "Saving…" : "Save rules"}
          </button>
        </div>
      </div>
    </div>
  );
};
