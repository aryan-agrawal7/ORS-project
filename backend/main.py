from __future__ import annotations
import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from ca_model import ForestFireCA, CAConfig, Wind, UNBURNABLE, UNIGNITED, BURNING, BURNED

app = FastAPI()

# Allow local dev front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def make_demo_grid(h=120, w=160) -> tuple[np.ndarray, np.ndarray]:
    """
    Demo setup:
      - a circular unburnable "lake"
      - everything else burnable
      - ignition probability p is higher in the center (just a placeholder)
    Replace p with your own LSSVM-derived raster (or a GIS layer) when ready.
    """
    grid = np.full((h, w), UNIGNITED, dtype=np.uint8)

    # Create an unburnable circle
    rr, cc = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 3
    lake = (rr - cy) ** 2 + (cc - cx) ** 2 < (min(h, w) * 0.12) ** 2
    grid[lake] = UNBURNABLE

    # Initial ignition point
    grid[cy, cx + 40] = BURNING

    # Placeholder ignition probability surface in [0,1]
    dist = np.sqrt((rr - cy) ** 2 + (cc - (cx + 40)) ** 2)
    p = np.exp(-(dist / (min(h, w) * 0.35)) ** 2).astype(np.float32)
    p = np.clip(p, 0.05, 0.95)

    # unburnable cells shouldn't ignite
    p[grid == UNBURNABLE] = 0.0
    return grid, p

@app.get("/health")
def health():
    return {"ok": True}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    grid, p = make_demo_grid()
    cfg = CAConfig(
        alpha=2.0,
        beta=1.0,
        seed=42,
        wind=Wind(speed_mps=3.0, direction_deg=60.0)  # tweak freely
    )
    ca = ForestFireCA(grid, p, cfg)

    # stream loop
    step_idx = 0
    try:
        while True:
            frame = ca.step()
            payload = {
                "step": step_idx,
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                # send as flat list for easy JS rendering
                "cells": frame.flatten().tolist(),
            }
            await ws.send_text(json.dumps(payload))
            step_idx += 1
            await asyncio.sleep(0.08)  # ~12.5 FPS
    except Exception:
        # client disconnected, etc.
        return