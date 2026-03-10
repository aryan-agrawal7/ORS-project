from __future__ import annotations
import asyncio
import json
import time
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ca_model import ForestFireCA, CAConfig, Wind, UNBURNABLE, UNIGNITED, BURNING, BURNED
from lssvm_model import LSSVM
from data_generator import generate_terrain, generate_training_data

app = FastAPI()

# Allow local dev front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Pydantic models for the config API
# ------------------------------------------------------------------

class SimConfig(BaseModel):
    grid_h: int = 120
    grid_w: int = 160
    # LSSVM hyper-parameters
    lssvm_gamma: float = 100.0
    lssvm_sigma: float = 1.0
    n_train_fire: int = 600
    n_train_nofire: int = 600
    # CA parameters
    ca_alpha: float = 2.0
    ca_beta: float = 1.0
    ca_seed: int = 42
    # Wind
    wind_speed: float = 3.0
    wind_direction: float = 60.0
    # Terrain seed
    terrain_seed: int = 42
    # Ignition point (row, col)  — relative fractions of grid size
    ignition_row_frac: float = 0.5
    ignition_col_frac: float = 0.65


# ------------------------------------------------------------------
# Build the full LSSVM → CA pipeline
# ------------------------------------------------------------------

def build_simulation(cfg: SimConfig):
    """
    1. Generate synthetic terrain rasters
    2. Sample training data
    3. Train LSSVM
    4. Compute Pc probability surface
    5. Assemble initial CA grid
    6. Return CA instance + metadata
    """
    h, w = cfg.grid_h, cfg.grid_w

    # --- 1. terrain ---
    terrain = generate_terrain(h, w, seed=cfg.terrain_seed)
    features = terrain["features"]          # (5, H, W)
    unburnable = terrain["unburnable_mask"]  # (H, W) bool
    slope = terrain["slope"]                # (H, W) float32

    # --- 2. training samples ---
    X_train, y_train = generate_training_data(
        features, unburnable,
        n_fire=cfg.n_train_fire,
        n_nofire=cfg.n_train_nofire,
        seed=cfg.terrain_seed + 1,
    )

    # --- 3. train LSSVM ---
    t0 = time.time()
    model = LSSVM(gamma=cfg.lssvm_gamma, sigma=cfg.lssvm_sigma)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # --- 4. Pc probability surface (Eq. 13) ---
    Pc = model.compute_probability_surface(features)
    Pc[unburnable] = 0.0

    # --- 5. initial grid ---
    grid = np.full((h, w), UNIGNITED, dtype=np.uint8)
    grid[unburnable] = UNBURNABLE

    # Ignition point
    iy = int(cfg.ignition_row_frac * h)
    ix = int(cfg.ignition_col_frac * w)
    iy = max(0, min(h - 1, iy))
    ix = max(0, min(w - 1, ix))
    grid[iy, ix] = BURNING

    # --- 6. CA ---
    ca_cfg = CAConfig(
        alpha=cfg.ca_alpha,
        beta=cfg.ca_beta,
        seed=cfg.ca_seed,
        wind=Wind(speed_mps=cfg.wind_speed, direction_deg=cfg.wind_direction),
    )
    ca = ForestFireCA(grid, Pc, ca_cfg, slope_deg=slope)

    meta = {
        "train_samples": len(y_train),
        "train_time_s": round(train_time, 3),
        "lssvm_b": round(model.b, 4),
        "Pc_min": float(np.min(Pc[~unburnable])),
        "Pc_max": float(np.max(Pc[~unburnable])),
        "Pc_mean": float(np.mean(Pc[~unburnable])),
    }

    return ca, Pc, meta


# ------------------------------------------------------------------
# REST endpoints
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/config-check")
def config_check(cfg: SimConfig):
    """Validate config without running simulation — returns meta info."""
    _, Pc, meta = build_simulation(cfg)
    return {"ok": True, **meta}


# ------------------------------------------------------------------
# WebSocket — streaming simulation
# ------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    # Wait for initial config message (or use defaults)
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=2.0)
        cfg = SimConfig(**json.loads(raw))
    except (asyncio.TimeoutError, Exception):
        cfg = SimConfig()

    ca, Pc, meta = build_simulation(cfg)

    # Send metadata as first message
    await ws.send_text(json.dumps({"type": "meta", **meta}))

    # Send Pc surface so frontend can show the heatmap
    await ws.send_text(json.dumps({
        "type": "probability",
        "height": int(Pc.shape[0]),
        "width": int(Pc.shape[1]),
        "data": Pc.flatten().tolist(),
    }))

    # Stream CA steps
    step_idx = 0
    try:
        while True:
            frame = ca.step()
            payload = {
                "type": "frame",
                "step": step_idx,
                "height": int(frame.shape[0]),
                "width": int(frame.shape[1]),
                "cells": frame.flatten().tolist(),
            }
            await ws.send_text(json.dumps(payload))
            step_idx += 1
            await asyncio.sleep(0.08)  # ~12.5 FPS
    except Exception:
        return