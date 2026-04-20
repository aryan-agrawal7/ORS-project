import React, { useCallback, useEffect, useRef, useState } from "react";
import { drawCA, drawProbabilityHeatmap } from "./caRenderer";

const WS_URL = "ws://localhost:8000/ws";

export default function App() {
  const caCanvasRef = useRef(null);
  const probCanvasRef = useRef(null);
  const wsRef = useRef(null);
  const cellSizeRef = useRef(5);

  const [status, setStatus] = useState("disconnected");
  const [step, setStep] = useState(0);
  const [gridSize, setGridSize] = useState({ h: 0, w: 0 });
  const [cellSize, setCellSize] = useState(5);
  const [meta, setMeta] = useState(null);
  const [showProb, setShowProb] = useState(true);
  const [exportsInfo, setExportsInfo] = useState({});

  useEffect(() => {
    cellSizeRef.current = cellSize;
  }, [cellSize]);

  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    setStatus("connecting");
    setStep(0);
    setMeta(null);
    setExportsInfo({});

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === "completed" ? prev : "disconnected"));
    };

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);

      if (msg.type === "error") {
        setStatus("error");
        setMeta((prev) => ({ ...(prev || {}), runtime_error: msg.message }));
        return;
      }

      if (msg.type === "meta") {
        setMeta(msg);
        setExportsInfo({
          lssvm_pc_projected_tif: msg.lssvm_pc_projected_tif,
          lssvm_pc_geographic_tif: msg.lssvm_pc_geographic_tif,
        });
        return;
      }

      if (msg.type === "probability") {
        const { height, width, data } = msg;
        const canvas = probCanvasRef.current;
        if (canvas) {
          const drawSize = cellSizeRef.current;
          canvas.width = width * drawSize;
          canvas.height = height * drawSize;
          const ctx = canvas.getContext("2d");
          drawProbabilityHeatmap(ctx, data, height, width, drawSize);
        }
        return;
      }

      if (msg.type === "frame") {
        const { step: s, height, width, cells } = msg;
        setStep(s);
        setGridSize({ h: height, w: width });

        const canvas = caCanvasRef.current;
        if (canvas) {
          const drawSize = cellSizeRef.current;
          canvas.width = width * drawSize;
          canvas.height = height * drawSize;
          const ctx = canvas.getContext("2d");
          drawCA(ctx, cells, height, width, drawSize);
        }
        return;
      }

      if (msg.type === "completed") {
        setStatus("completed");
        setExportsInfo((prev) => ({
          ...prev,
          ca_final_projected_tif: msg.ca_final_projected_tif,
          ca_final_geographic_tif: msg.ca_final_geographic_tif,
        }));
      }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return (
    <div style={{ fontFamily: "system-ui", padding: 16, maxWidth: 1300 }}>
      <h2 style={{ marginTop: 0 }}>LSSVM-CA Forest Fire Spread Simulation</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <span><b>Status:</b> {status}</span>
        <span><b>Step:</b> {step}</span>
        <span><b>Grid:</b> {gridSize.h} x {gridSize.w}</span>
        {meta && (
          <>
            <span title="LSSVM training time"><b>Train:</b> {meta.train_time_s}s ({meta.train_samples} samples)</span>
            <span title="Overall LSSVM accuracy on training set">
              <b>Train Acc:</b> {(meta.train_accuracy * 100).toFixed(1)}%
              <span style={{ fontSize: 11, color: "#666" }}>
                {" "}(fire: {(meta.fire_accuracy * 100).toFixed(1)}%, non-fire: {(meta.nofire_accuracy * 100).toFixed(1)}%)
              </span>
            </span>
            {meta.val_accuracy !== null && meta.val_accuracy !== undefined && (
              <span title="Holdout validation metrics (20% stratified split)">
                <b>Val:</b> acc {(meta.val_accuracy * 100).toFixed(1)}% | rec {(meta.val_recall * 100).toFixed(1)}% | f1 {(meta.val_f1 * 100).toFixed(1)}% | auc {meta.val_roc_auc?.toFixed(3)}
              </span>
            )}
            <span><b>Pc range:</b> [{meta.Pc_min?.toFixed(3)}, {meta.Pc_max?.toFixed(3)}]</span>
            <span><b>Model:</b> {meta.cache}</span>
            <span><b>Projected CRS:</b> {meta.projected_crs}</span>
            <span><b>Geographic CRS:</b> {meta.geographic_crs}</span>
            <span><b>Ignition:</b> ({meta.ignition_lat}, {meta.ignition_lon})</span>
          </>
        )}
      </div>

      {meta?.runtime_error && (
        <div style={{ marginBottom: 12, color: "#8b0000" }}>
          <b>Runtime error:</b> {meta.runtime_error}
        </div>
      )}

      <div style={{ marginBottom: 12, border: "1px solid #ccc", borderRadius: 8, padding: 12 }}>
        <div style={{ marginBottom: 8, fontWeight: "bold" }}>Simulation Controls</div>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <button onClick={connect} style={btnStyle}>Start / Restart</button>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={showProb} onChange={(e) => setShowProb(e.target.checked)} />
            Show Pc heatmap
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            Cell size
            <input
              type="range"
              min="2"
              max="10"
              value={cellSize}
              onChange={(e) => setCellSize(parseInt(e.target.value, 10))}
            />
            <span>{cellSize}px</span>
          </label>
        </div>
      </div>

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div>
          <h4 style={{ margin: "0 0 4px 0" }}>CA Fire Spread</h4>
          <div style={canvasContainerStyle}>
            <canvas ref={caCanvasRef} />
          </div>
        </div>

        {showProb && (
          <div>
            <h4 style={{ margin: "0 0 4px 0" }}>LSSVM Ignition Probability (Pc)</h4>
            <div style={canvasContainerStyle}>
              <canvas ref={probCanvasRef} />
            </div>
          </div>
        )}
      </div>

      <div style={{ marginTop: 10, color: "#555", fontSize: 13 }}>
        <div><b>GeoTIFF exports:</b></div>
        <div>Pc projected: {exportsInfo.lssvm_pc_projected_tif || "pending"}</div>
        <div>Pc geographic: {exportsInfo.lssvm_pc_geographic_tif || "pending"}</div>
        <div>CA final projected: {exportsInfo.ca_final_projected_tif || "pending"}</div>
        <div>CA final geographic: {exportsInfo.ca_final_geographic_tif || "pending"}</div>
      </div>
    </div>
  );
}

const btnStyle = {
  padding: "6px 16px",
  borderRadius: 6,
  border: "1px solid #888",
  background: "#2d6a2d",
  color: "#fff",
  cursor: "pointer",
  fontWeight: "bold",
};

const canvasContainerStyle = {
  border: "1px solid #ddd",
  borderRadius: 10,
  overflow: "auto",
  maxWidth: "100%",
  maxHeight: "70vh",
  padding: 8,
};
