import React, { useEffect, useRef, useState, useCallback } from "react";
import { drawCA, drawProbabilityHeatmap } from "./caRenderer";

const DEFAULT_CONFIG = {
  grid_h: 120,
  grid_w: 160,
  lssvm_gamma: 100,
  lssvm_sigma: 1.0,
  n_train_fire: 600,
  n_train_nofire: 600,
  ca_alpha: 2.0,
  ca_beta: 1.0,
  ca_seed: 42,
  wind_speed: 3.0,
  wind_direction: 60.0,
  terrain_seed: 42,
  ignition_row_frac: 0.5,
  ignition_col_frac: 0.65,
};

export default function App() {
  const caCanvasRef = useRef(null);
  const probCanvasRef = useRef(null);
  const wsRef = useRef(null);

  const [status, setStatus] = useState("disconnected");
  const [step, setStep] = useState(0);
  const [gridSize, setGridSize] = useState({ h: 0, w: 0 });
  const [cellSize, setCellSize] = useState(5);
  const [meta, setMeta] = useState(null);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [showProb, setShowProb] = useState(true);

  const connect = useCallback(() => {
    // close existing
    if (wsRef.current) {
      wsRef.current.close();
    }

    setStatus("connecting");
    setStep(0);
    setMeta(null);

    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      // send config as first message
      ws.send(JSON.stringify(config));
    };

    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("error");

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);

      if (msg.type === "meta") {
        setMeta(msg);
        return;
      }

      if (msg.type === "probability") {
        const { height, width, data } = msg;
        const canvas = probCanvasRef.current;
        if (canvas) {
          canvas.width = width * cellSize;
          canvas.height = height * cellSize;
          const ctx = canvas.getContext("2d");
          drawProbabilityHeatmap(ctx, data, height, width, cellSize);
        }
        return;
      }

      if (msg.type === "frame") {
        const { step: s, height, width, cells } = msg;
        setStep(s);
        setGridSize({ h: height, w: width });

        const canvas = caCanvasRef.current;
        if (canvas) {
          canvas.width = width * cellSize;
          canvas.height = height * cellSize;
          const ctx = canvas.getContext("2d");
          drawCA(ctx, cells, height, width, cellSize);
        }
      }
    };
  }, [config, cellSize]);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const handleConfigChange = (key, value) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div style={{ fontFamily: "system-ui", padding: 16, maxWidth: 1200 }}>
      <h2 style={{ marginTop: 0 }}>🔥 LSSVM-CA Forest Fire Spread Simulation</h2>

      {/* ---- Status bar ---- */}
      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
        <span><b>Status:</b> {status}</span>
        <span><b>Step:</b> {step}</span>
        <span><b>Grid:</b> {gridSize.h} × {gridSize.w}</span>
        {meta && (
          <>
            <span title="LSSVM training time"><b>Train:</b> {meta.train_time_s}s</span>
            <span><b>Pc range:</b> [{meta.Pc_min?.toFixed(3)}, {meta.Pc_max?.toFixed(3)}]</span>
            <span><b>LSSVM b:</b> {meta.lssvm_b}</span>
          </>
        )}
      </div>

      {/* ---- Configuration panel ---- */}
      <details open style={{ marginBottom: 12, border: "1px solid #ccc", borderRadius: 8, padding: 12 }}>
        <summary style={{ cursor: "pointer", fontWeight: "bold" }}>⚙️ Simulation Parameters</summary>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 10, marginTop: 10 }}>
          <NumInput label="LSSVM γ (regularisation)" value={config.lssvm_gamma}
            onChange={(v) => handleConfigChange("lssvm_gamma", v)} min={1} max={10000} step={10} />
          <NumInput label="LSSVM σ (kernel width)" value={config.lssvm_sigma}
            onChange={(v) => handleConfigChange("lssvm_sigma", v)} min={0.01} max={10} step={0.1} />
          <NumInput label="Training fire samples" value={config.n_train_fire}
            onChange={(v) => handleConfigChange("n_train_fire", v)} min={50} max={2000} step={50} />
          <NumInput label="Training non-fire samples" value={config.n_train_nofire}
            onChange={(v) => handleConfigChange("n_train_nofire", v)} min={50} max={2000} step={50} />
          <NumInput label="Wind speed (m/s)" value={config.wind_speed}
            onChange={(v) => handleConfigChange("wind_speed", v)} min={0} max={20} step={0.5} />
          <NumInput label="Wind direction (°)" value={config.wind_direction}
            onChange={(v) => handleConfigChange("wind_direction", v)} min={0} max={359} step={5} />
          <NumInput label="CA α" value={config.ca_alpha}
            onChange={(v) => handleConfigChange("ca_alpha", v)} min={0.1} max={10} step={0.1} />
          <NumInput label="CA β" value={config.ca_beta}
            onChange={(v) => handleConfigChange("ca_beta", v)} min={0.1} max={5} step={0.1} />
          <NumInput label="Terrain seed" value={config.terrain_seed}
            onChange={(v) => handleConfigChange("terrain_seed", v)} min={0} max={9999} step={1} />
          <NumInput label="Ignition row %" value={config.ignition_row_frac}
            onChange={(v) => handleConfigChange("ignition_row_frac", v)} min={0} max={1} step={0.05} />
          <NumInput label="Ignition col %" value={config.ignition_col_frac}
            onChange={(v) => handleConfigChange("ignition_col_frac", v)} min={0} max={1} step={0.05} />
        </div>
        <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
          <button onClick={connect} style={btnStyle}>▶ Start / Restart</button>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input type="checkbox" checked={showProb} onChange={(e) => setShowProb(e.target.checked)} />
            Show Pc heatmap
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            Cell size
            <input type="range" min="2" max="10" value={cellSize}
              onChange={(e) => setCellSize(parseInt(e.target.value, 10))} />
            <span>{cellSize}px</span>
          </label>
        </div>
      </details>

      {/* ---- Visualizations ---- */}
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

      <p style={{ marginTop: 10, color: "#555", fontSize: 13 }}>
        Based on: Xu et al. (2022) — <em>"Modeling Forest Fire Spread Using Machine Learning-Based
        Cellular Automata in a GIS Environment"</em>, Forests 13(12), 1974.
      </p>
    </div>
  );
}

/* ---- Small helper components / styles ---- */

function NumInput({ label, value, onChange, min, max, step }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", padding: 4, marginTop: 2 }}
      />
    </label>
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