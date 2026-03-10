import React, { useEffect, useRef, useState } from "react";
import { drawCA } from "./caRenderer";

export default function App() {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);

  const [status, setStatus] = useState("disconnected");
  const [step, setStep] = useState(0);
  const [gridSize, setGridSize] = useState({ h: 0, w: 0 });
  const [cellSize, setCellSize] = useState(5);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    setStatus("connecting");
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;

    ws.onopen = () => setStatus("connected");
    ws.onclose = () => setStatus("disconnected");
    ws.onerror = () => setStatus("error");

    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      const { step, height, width, cells } = msg;

      setStep(step);
      setGridSize({ h: height, w: width });

      canvas.width = width * cellSize;
      canvas.height = height * cellSize;

      // Optional: draw a background map image first (placeholder hook)
      // ctx.drawImage(mapImg, 0, 0, canvas.width, canvas.height);

      drawCA(ctx, cells, height, width, cellSize);
    };

    return () => ws.close();
  }, [cellSize]);

  return (
    <div style={{ fontFamily: "system-ui", padding: 16 }}>
      <h2 style={{ marginTop: 0 }}>Forest Fire CA — Live Visualization</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <div><b>Status:</b> {status}</div>
        <div><b>Step:</b> {step}</div>
        <div><b>Grid:</b> {gridSize.h} × {gridSize.w}</div>

        <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <b>Cell size</b>
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

      <div style={{
        marginTop: 12,
        border: "1px solid #ddd",
        borderRadius: 10,
        overflow: "auto",
        maxWidth: "100%",
        maxHeight: "80vh",
        padding: 8
      }}>
        <canvas ref={canvasRef} />
      </div>

      <p style={{ marginTop: 10, color: "#555" }}>
        Next step: replace the placeholder ignition probability raster <code>p</code> in the Python backend
        with your LSSVM-trained output (or a GIS-derived layer), and optionally draw a real basemap image behind the grid.
      </p>
    </div>
  );
}