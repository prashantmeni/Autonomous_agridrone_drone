import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { Detection, TelemetryData, DiseaseAnalysis } from "../types";
import { Leaf, Scan, RefreshCw, MapPin, Tag, Database, Upload, Cpu, CheckCircle2, AlertTriangle } from "lucide-react";
import { DiseaseModelStatus } from "../types";

interface PlantHealthProps {
  telemetry: TelemetryData;
  onMarkDetection?: (disease: string, confidence: number) => void;
}

export const PlantHealth: React.FC<PlantHealthProps> = ({ telemetry, onMarkDetection }) => {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [analysis, setAnalysis] = useState<DiseaseAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [autoMark, setAutoMark] = useState(false);
  const [modelInfo, setModelInfo] = useState<DiseaseModelStatus | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [dragging, setDragging] = useState(false);

  const fetchModelInfo = async () => {
    try {
      setModelInfo(await api.getDiseaseModel());
    } catch {
      /* keep previous */
    }
  };

  const uploadModel = async (file: File) => {
    setUploading(true);
    setUploadMsg(null);
    try {
      const res = await api.uploadDiseaseModel(file);
      setModelInfo(res);
      setUploadMsg({
        kind: res.status === "OK" ? "ok" : "err",
        text:
          res.status === "OK"
            ? `${file.name} loaded via ${res.runtime} (input ${res.input_size?.join("×")})`
            : `Uploaded, but not usable — ${res.detail || res.status}`,
      });
    } catch (e: any) {
      setUploadMsg({ kind: "err", text: e.message || "upload failed" });
    } finally {
      setUploading(false);
    }
  };

  // Optional: attach class names so results read as diseases, not class_N.
  const uploadLabels = async (file: File) => {
    const target = modelInfo?.model_name || modelInfo?.name;
    if (!target) {
      setUploadMsg({ kind: "err", text: "Upload the model file first, then its labels." });
      return;
    }
    setUploading(true);
    setUploadMsg(null);
    try {
      const res = await api.uploadDiseaseLabels(target, file);
      setModelInfo(res);
      setUploadMsg({
        kind: res.status === "OK" ? "ok" : "err",
        text:
          res.status === "OK"
            ? `Labels attached — ${res.labels} classes named`
            : `Labels uploaded, but the model is not usable — ${res.detail || res.status}`,
      });
    } catch (e: any) {
      setUploadMsg({ kind: "err", text: e.message || "label upload failed" });
    } finally {
      setUploading(false);
    }
  };

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) uploadModel(f);
    e.target.value = "";
  };

  const fetchDetections = async () => {
    setLoading(true);
    try {
      setDetections(await api.getDetections());
    } catch {
      /* offline — keep previous */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDetections();
    fetchModelInfo();
  }, []);

  const runAnalysis = async () => {
    setAnalyzing(true);
    try {
      const res = await api.analyzeDisease(autoMark);
      setAnalysis(res);
      if (res.marked) fetchDetections();
    } catch (e: any) {
      setAnalysis({ status: "ERROR", note: e.message });
    } finally {
      setAnalyzing(false);
    }
  };

  // Analyze a still image from disk, so the pipeline can be exercised without
  // relying on the camera.
  const analyzeFile = async (file: File) => {
    setAnalyzing(true);
    setUploadMsg(null);
    try {
      const res = await api.analyzeDiseaseImage(file, autoMark);
      setAnalysis(res);
      if (res.marked) fetchDetections();
    } catch (e: any) {
      setAnalysis({ status: "ERROR", note: e.message });
    } finally {
      setAnalyzing(false);
    }
  };

  const modelReady = analysis?.status === "OK";
  const q = analysis?.image_quality;
  const t = analysis?.timing;
  const poorFrame = analysis && analysis.success === false && !!analysis.error_code
    && analysis.error_code.startsWith("IMAGE_");

  return (
    <div className="page-scroll">
      <div className="page-head">
        <div>
          <h2>
            <Leaf size={22} color="#10b981" />
            Crop health analysis
          </h2>
          <p>Run the edge classifier on the live frame, review findings, and mark affected zones.</p>
        </div>
        <div className="page-actions">
          <label className="switch-label">
            <input type="checkbox" checked={autoMark} onChange={(e) => setAutoMark(e.target.checked)} />
            Auto-mark above threshold
          </label>
          <button className="btn btn-secondary btn-sm" onClick={fetchDetections} disabled={loading}>
            <RefreshCw size={13} className={loading ? "spin" : ""} /> Refresh
          </button>
          <label className="btn btn-secondary btn-sm" style={{ cursor: analyzing ? "wait" : "pointer" }}>
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) analyzeFile(f);
                e.target.value = "";
              }}
              disabled={analyzing}
              style={{ display: "none" }}
            />
            <Upload size={13} /> Test on image
          </label>
          <button className="btn btn-primary" onClick={runAnalysis} disabled={analyzing}>
            <Scan size={15} />
            {analyzing ? "Analyzing frame…" : "Analyze live frame"}
          </button>
        </div>
      </div>

      <div className="crop-grid">
        {/* Analysis result */}
        <div className={`card ${analysis ? "" : "card-muted"}`}>
          <div className="card-header">
            <span className="card-title">
              <Scan size={16} color="#06b6d4" />
              Latest inference
            </span>
            <span
              className={`status-pill ${
                analysis?.status === "OK" ? "pill-live" : analysis ? "pill-idle" : ""
              }`}
            >
              {analysis ? analysis.status : "NOT RUN"}
            </span>
          </div>

          {!analysis && (
            <p className="muted-text">
              No analysis yet. This runs the configured disease model against the current camera
              frame, or an image you upload.
            </p>
          )}

          {poorFrame && (
            <div
              style={{
                marginTop: 6,
                padding: "12px 14px",
                borderRadius: 9,
                background: "rgba(245,158,11,0.10)",
                border: "1px solid rgba(245,158,11,0.35)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 7, color: "#f59e0b", fontWeight: 700, fontSize: 14 }}>
                <AlertTriangle size={16} />
                {analysis!.message || "Image cannot be used for detection."}
              </div>
              <div className="muted-text" style={{ marginTop: 5 }}>
                {analysis!.suggestion}
              </div>
              <div className="muted-text mono" style={{ marginTop: 6, fontSize: 11 }}>
                {analysis!.error_code}
                {q?.metrics?.brightness != null && ` · brightness ${q.metrics.brightness}`}
                {q?.metrics?.blur_score != null && ` · blur ${q.metrics.blur_score}`}
              </div>
            </div>
          )}

          {analysis?.success === true && (
            <>
              <div className="analysis-big">{analysis.disease}</div>
              <div className="muted-text">
                {analysis.crop} · score {analysis.confidence?.toFixed(3)}
                {analysis.marked ? " · plotted to map" : ""}
              </div>
              <div style={{ display: "flex", gap: 6, marginTop: 9, flexWrap: "wrap" }}>
                <span
                  className="conf-pill"
                  style={{
                    background:
                      analysis.confidence_level === "HIGH"
                        ? "rgba(16,185,129,0.18)"
                        : analysis.confidence_level === "MEDIUM"
                        ? "rgba(245,158,11,0.18)"
                        : "rgba(239,68,68,0.18)",
                    color:
                      analysis.confidence_level === "HIGH"
                        ? "#10b981"
                        : analysis.confidence_level === "MEDIUM"
                        ? "#f59e0b"
                        : "#ef4444",
                  }}
                >
                  {analysis.confidence_level}
                </span>
                <span className={`status-pill ${analysis.reliable ? "pill-live" : "pill-idle"}`}>
                  {analysis.reliable ? "reliable" : "not reliable"}
                </span>
                {analysis.source && <span className="status-pill pill-idle">{analysis.source}</span>}
              </div>
              {!analysis.reliable && (
                <p className="muted-text" style={{ marginTop: 8, fontSize: 12 }}>
                  Low confidence — treat this as a hint, not a confirmed finding.
                </p>
              )}
              {onMarkDetection && analysis.reliable && (
                <button
                  className="btn btn-sm btn-secondary"
                  style={{ marginTop: 12 }}
                  onClick={() => onMarkDetection(analysis.disease || "Anomaly", analysis.confidence || 0.5)}
                >
                  <Tag size={13} /> Mark zone at current GPS
                </button>
              )}
            </>
          )}

          {analysis && !analysis.success && !poorFrame && (
            <p className="muted-text">
              {analysis.message || analysis.note || analysis.status}
              {analysis.suggestion ? ` ${analysis.suggestion}` : ""}
            </p>
          )}
          <div className="analysis-foot" style={{ flexDirection: "column", alignItems: "flex-start", gap: 3 }}>
            <span className={modelReady ? "ok" : ""}>
              Model:{" "}
              {modelInfo
                ? modelInfo.status === "OK"
                  ? modelInfo.name
                  : modelInfo.status
                : "unknown until first run"}
            </span>
            {q?.valid && (
              <span className="muted-text">
                Image quality: ok
                {q.metrics?.brightness != null && ` · brightness ${q.metrics.brightness}`}
                {q.metrics?.blur_score != null && ` · blur score ${q.metrics.blur_score}`}
              </span>
            )}
            {q?.skipped && <span className="muted-text">Image quality check skipped</span>}
            {t?.total_ms != null && (
              <span className="muted-text mono">
                {t.preprocess_ms}ms prep · {t.inference_ms}ms infer · {t.postprocess_ms}ms post ·{" "}
                {t.total_ms}ms total
              </span>
            )}
            <span>
              GPS: {telemetry.lat?.toFixed(5) ?? "—"}, {telemetry.lon?.toFixed(5) ?? "—"}
            </span>
          </div>
        </div>

        {/* Model upload */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <Cpu size={16} color="#f59e0b" />
              Disease model
            </span>
            <span
              className={`status-pill ${
                modelInfo?.status === "OK"
                  ? "pill-live"
                  : modelInfo && modelInfo.status !== "MODEL_NOT_AVAILABLE"
                  ? "pill-warn"
                  : "pill-idle"
              }`}
            >
              {modelInfo?.status ?? "…"}
            </span>
          </div>

          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) uploadModel(f);
            }}
            style={{
              display: "block",
              border: `1.5px dashed ${dragging ? "#10b981" : "rgba(148,163,184,0.35)"}`,
              borderRadius: 10,
              padding: "18px 14px",
              textAlign: "center",
              cursor: uploading ? "wait" : "pointer",
              background: dragging ? "rgba(16,185,129,0.08)" : "transparent",
            }}
          >
            <input
              type="file"
              accept=".onnx,.tflite"
              onChange={onPick}
              disabled={uploading}
              style={{ display: "none" }}
            />
            <Upload size={20} color="#94a3b8" style={{ marginBottom: 6 }} />
            <div style={{ fontSize: 13, fontWeight: 600 }}>
              {uploading ? "Uploading…" : "Drop a model file or click to browse"}
            </div>
            <div className="muted-text" style={{ fontSize: 11, marginTop: 3 }}>
              .onnx recommended · .tflite only if its runtime is installed · max 512 MB
            </div>
          </label>

          {uploadMsg && (
            <div
              className="muted-text"
              style={{
                marginTop: 10,
                fontSize: 12,
                color: uploadMsg.kind === "ok" ? "#10b981" : "#f87171",
              }}
            >
              {uploadMsg.kind === "ok" ? <CheckCircle2 size={12} /> : <AlertTriangle size={12} />} {uploadMsg.text}
            </div>
          )}

          <div className="analysis-foot" style={{ marginTop: 10, flexDirection: "column", alignItems: "flex-start", gap: 3 }}>
            {modelInfo?.status === "OK" ? (
              <>
                <span className="ok">
                  {modelInfo.name} · {modelInfo.runtime} ·{" "}
                  {modelInfo.input_width}×{modelInfo.input_height}
                  {modelInfo.num_classes ? ` · ${modelInfo.num_classes} classes` : ""}
                  {modelInfo.model_version ? ` · v${modelInfo.model_version}` : ""}
                </span>
                {modelInfo.validation && (
                  <span className={modelInfo.validation.valid ? "ok" : ""} style={{ fontSize: 11 }}>
                    validation:{" "}
                    {modelInfo.validation.valid ? "passed" : `failed (${modelInfo.validation.code})`}
                    {modelInfo.validation.test_inference
                      ? ` · test inference ${modelInfo.validation.test_inference}`
                      : ""}
                  </span>
                )}
                {modelInfo.labels === 0 ? (
                  <>
                    <span className="muted-text">
                      No label file — results read as class_0, class_1…
                    </span>
                    <label
                      className="btn btn-sm btn-secondary"
                      style={{ marginTop: 6, cursor: uploading ? "wait" : "pointer" }}
                    >
                      <input
                        type="file"
                        accept=".txt,.json,.labels"
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) uploadLabels(f);
                          e.target.value = "";
                        }}
                        disabled={uploading}
                        style={{ display: "none" }}
                      />
                      <Tag size={12} /> Attach labels file
                    </label>
                  </>
                ) : (
                  <span className="muted-text">{modelInfo.labels} classes named</span>
                )}
              </>
            ) : (
              <>
                <span className="muted-text">{modelInfo?.detail ?? "No model loaded."}</span>
                {modelInfo?.model_name && (
                  <span className="muted-text">Configured: {modelInfo.model_name}</span>
                )}
              </>
            )}
          </div>
        </div>

        {/* Summary */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">
              <Database size={16} color="#a855f7" />
              Marked zones
            </span>
            <span className="status-pill pill-idle">{detections.length} stored</span>
          </div>
          <div className="analysis-big">{detections.length}</div>
          <p className="muted-text">Geotagged pathology records in the on-board database.</p>
        </div>
      </div>

      {/* Detections table */}
      <div className="card table-card">
        <div className="card-header" style={{ padding: "14px 18px", marginBottom: 0 }}>
          <span className="card-title">
            <MapPin size={16} color="#f87171" />
            Disease zones
          </span>
        </div>
        {detections.length === 0 ? (
          <div className="empty-state">No disease zones marked yet.</div>
        ) : (
          <div className="table-container" style={{ border: "none", borderRadius: 0 }}>
            <table className="drone-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Crop</th>
                  <th>Disease</th>
                  <th>Confidence</th>
                  <th>Coordinates</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((det, idx) => (
                  <tr key={idx}>
                    <td className="mono muted">{new Date(det.timestamp * (det.timestamp < 1e12 ? 1 : 1000)).toLocaleString()}</td>
                    <td style={{ fontWeight: 600 }}>{det.crop}</td>
                    <td style={{ color: "#f87171", fontWeight: 600 }}>{det.disease}</td>
                    <td>
                      <span className="conf-pill">{(det.confidence * 100).toFixed(0)}%</span>
                    </td>
                    <td className="mono muted">
                      {det.latitude.toFixed(5)}, {det.longitude.toFixed(5)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
