import React from "react";
import { AlertTriangle, X } from "lucide-react";

interface ConfirmModalProps {
  isOpen: boolean;
  title: string;
  message: string;
  checklist?: string[];
  confirmText?: string;
  confirmStyle?: "danger" | "warning" | "primary";
  onConfirm: () => void;
  onClose: () => void;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  title,
  message,
  checklist = [],
  confirmText = "Confirm Command",
  confirmStyle = "danger",
  onConfirm,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className={confirmStyle === "danger" ? "modal-icon-danger" : "modal-icon-danger"} style={{
            background: confirmStyle === "danger" ? "rgba(239, 68, 68, 0.15)" : "rgba(245, 158, 11, 0.15)",
            color: confirmStyle === "danger" ? "#ef4444" : "#f59e0b"
          }}>
            <AlertTriangle size={24} />
          </div>
          <div>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700 }}>{title}</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>Safety-Critical Flight Interlock</p>
          </div>
          <button
            onClick={onClose}
            style={{
              marginLeft: "auto",
              background: "transparent",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
            }}
          >
            <X size={20} />
          </button>
        </div>

        <p style={{ fontSize: "0.9rem", color: "var(--text-main)", marginBottom: 16 }}>
          {message}
        </p>

        {checklist.length > 0 && (
          <div style={{
            background: "rgba(0, 0, 0, 0.35)",
            borderRadius: "var(--radius-md)",
            padding: "12px 16px",
            border: "1px solid var(--border-color)",
            marginBottom: 20
          }}>
            <div style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-dim)", textTransform: "uppercase", marginBottom: 8 }}>
              Automated Safety Verifications
            </div>
            {checklist.map((item, idx) => (
              <div key={idx} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: "0.82rem", color: "#34d399", marginBottom: 4 }}>
                <span style={{ fontSize: "0.9rem" }}>✓</span> {item}
              </div>
            ))}
          </div>
        )}

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className={`btn btn-${confirmStyle}`}
            onClick={() => {
              onConfirm();
              onClose();
            }}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
};
