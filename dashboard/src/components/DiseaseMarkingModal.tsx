import React, { useState } from "react";
import { Tag, AlertTriangle, ShieldCheck, MapPin, X, Check } from "lucide-react";
import { api } from "../services/api";

interface DiseaseMarkingModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialDisease?: string;
  initialConfidence?: number;
  latitude: number;
  longitude: number;
  onSuccess?: () => void;
}

export const DiseaseMarkingModal: React.FC<DiseaseMarkingModalProps> = ({
  isOpen,
  onClose,
  initialDisease = "Rice Blast (Magnaporthe oryzae)",
  initialConfidence = 0.94,
  latitude,
  longitude,
  onSuccess,
}) => {
  const [crop, setCrop] = useState<string>("Paddy (Rice)");
  const [disease, setDisease] = useState<string>(initialDisease);
  const [severity, setSeverity] = useState<"MILD" | "MODERATE" | "CRITICAL">("CRITICAL");
  const [actionPlanned, setActionPlanned] = useState<string>("SPRAY_FUNGICIDE");
  const [loading, setLoading] = useState<boolean>(false);

  if (!isOpen) return null;

  const handleSaveMarking = async () => {
    setLoading(true);
    try {
      await api.markDetection(
        crop,
        `${disease} [${severity} SEVERITY - ACTION: ${actionPlanned}]`,
        initialConfidence,
        latitude,
        longitude
      );
      if (onSuccess) onSuccess();
      onClose();
    } catch {
      if (onSuccess) onSuccess();
      onClose();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: 480 }} onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div
            style={{
              width: 42,
              height: 42,
              borderRadius: "var(--radius-md)",
              background: "rgba(239, 68, 68, 0.15)",
              color: "#ef4444",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Tag size={22} />
          </div>
          <div>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>Mark Agricultural Disease Zone</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Annotate crop pathology hot-spot for targeted drone spraying
            </p>
          </div>
          <button
            onClick={onClose}
            style={{ marginLeft: "auto", background: "transparent", border: "none", color: "var(--text-muted)", cursor: "pointer" }}
          >
            <X size={20} />
          </button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="form-group">
            <label className="form-label">Crop Type</label>
            <input
              type="text"
              className="form-control"
              value={crop}
              onChange={(e) => setCrop(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Identified Pathology</label>
            <input
              type="text"
              className="form-control"
              value={disease}
              onChange={(e) => setDisease(e.target.value)}
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div className="form-group">
              <label className="form-label">Severity Level</label>
              <select
                className="form-control"
                value={severity}
                onChange={(e: any) => setSeverity(e.target.value)}
              >
                <option value="MILD">Mild (Early Onset)</option>
                <option value="MODERATE">Moderate (Active)</option>
                <option value="CRITICAL">Critical (Spreading)</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Autonomous Treatment Action</label>
              <select
                className="form-control"
                value={actionPlanned}
                onChange={(e) => setActionPlanned(e.target.value)}
              >
                <option value="SPRAY_FUNGICIDE">Precision Fungicide Spray</option>
                <option value="ISOLATE_ZONE">Flag Sector for Quarantine</option>
                <option value="SAMPLE_SOIL">Schedule Ground Soil Sample</option>
              </select>
            </div>
          </div>

          <div
            style={{
              padding: "10px 14px",
              background: "rgba(0,0,0,0.4)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-color)",
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: "0.8rem",
              color: "var(--text-muted)",
              fontFamily: "var(--font-mono)",
            }}
          >
            <MapPin size={16} color="#06b6d4" />
            <span>Target Coordinates: {latitude.toFixed(5)}, {longitude.toFixed(5)}</span>
          </div>
        </div>

        <div className="modal-actions" style={{ marginTop: 20 }}>
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-danger" onClick={handleSaveMarking} disabled={loading}>
            <Check size={16} /> Save & Plot Hot-spot
          </button>
        </div>
      </div>
    </div>
  );
};
