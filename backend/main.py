from __future__ import annotations
import asyncio
import hashlib
import json
import time
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ca_model import ForestFireCA, CAConfig, Wind, UNBURNABLE, UNIGNITED, BURNING, BURNED
from lssvm_model import LSSVM
from data_generator import generate_terrain, generate_training_data
from gee_data_loader import load_gee_data

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
    # Path to GEE-exported data folder (empty string = use synthetic data)
    gee_data_dir: str = "../data/"
    # Multi-step burning: how many CA steps a cell stays BURNING
    burn_duration: int = 3
    # Ignition block radius (0 = single cell, 3 = 7×7 block)
    ignition_radius: int = 3


# ------------------------------------------------------------------
# Model cache — avoids retraining when only CA params change
# ------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / ".model_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _model_cache_key(cfg: SimConfig) -> str:
    """
    Deterministic hash of the parameters that affect LSSVM training.
    If any of these change the model must be retrained; otherwise
    we can reuse the saved .npz and Pc surface.
    """
    parts = (
        cfg.grid_h,
        cfg.grid_w,
        cfg.lssvm_gamma,
        cfg.lssvm_sigma,
        cfg.n_train_fire,
        cfg.n_train_nofire,
        cfg.terrain_seed,
        cfg.gee_data_dir,
    )
    raw = json.dumps(parts, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# In-memory cache so repeated WebSocket reconnects are instant
_mem_cache: dict[str, tuple[LSSVM, np.ndarray, dict, np.ndarray, np.ndarray]] = {}


# ------------------------------------------------------------------
# Build the full LSSVM → CA pipeline
# ------------------------------------------------------------------

def build_simulation(cfg: SimConfig):
    """
    1. Generate / load terrain rasters
    2. Sample / load training data
    3. Train LSSVM  (or load from cache)
    4. Compute Pc probability surface
    5. Assemble initial CA grid
    6. Return CA instance + metadata
    """
    h, w = cfg.grid_h, cfg.grid_w
    cache_key = _model_cache_key(cfg)
    model_path = CACHE_DIR / f"lssvm_{cache_key}.npz"
    pc_path    = CACHE_DIR / f"pc_{cache_key}.npy"

    # ── Try in-memory cache first ──────────────────────────────
    if cache_key in _mem_cache:
        model, Pc, base_meta, features_unburnable_mask, slope = (
            _mem_cache[cache_key][0],
            _mem_cache[cache_key][1],
            _mem_cache[cache_key][2],
            _mem_cache[cache_key][3],
            _mem_cache[cache_key][4],
        )
        unburnable = features_unburnable_mask.astype(bool)
        cache_src = "memory"
    # ── Try disk cache ─────────────────────────────────────────
    elif model_path.exists() and pc_path.exists():
        t0 = time.time()
        model = LSSVM.load(model_path)
        Pc = np.load(str(pc_path)).astype(np.float32)

        # We still need terrain for slope + unburnable mask
        if cfg.gee_data_dir:
            terrain, _, _ = load_gee_data(cfg.gee_data_dir, target_shape=(h, w))
        else:
            terrain = generate_terrain(h, w, seed=cfg.terrain_seed)
        unburnable = terrain["unburnable_mask"]
        slope      = terrain["slope"]
        train_time = time.time() - t0

        # Recompute accuracy from cached model
        if model.X_train is not None and model.y_train is not None:
            y_pred = model.predict(model.X_train)
            train_acc = float(np.mean(y_pred == model.y_train))
            n_fire = int((model.y_train == 1).sum())
            n_nofire = int((model.y_train == -1).sum())
            fire_acc = float(np.mean(y_pred[model.y_train == 1] == 1)) if n_fire > 0 else 0.0
            nofire_acc = float(np.mean(y_pred[model.y_train == -1] == -1)) if n_nofire > 0 else 0.0
        else:
            train_acc = 0.0
            n_fire = n_nofire = 0
            fire_acc = nofire_acc = 0.0

        base_meta = {
            "train_samples": len(model.y_train) if model.y_train is not None else 0,
            "train_fire": n_fire,
            "train_nofire": n_nofire,
            "train_time_s": round(train_time, 3),
            "train_accuracy": round(train_acc, 4),
            "fire_accuracy": round(fire_acc, 4),
            "nofire_accuracy": round(nofire_acc, 4),
            "lssvm_b": round(float(model.b or 0), 4),
            "Pc_min": float(np.min(Pc[~unburnable])),
            "Pc_max": float(np.max(Pc[~unburnable])),
            "Pc_mean": float(np.mean(Pc[~unburnable])),
        }
        # Store in memory for next time
        _mem_cache[cache_key] = (model, Pc, base_meta, unburnable.astype(np.uint8), slope)
        cache_src = "disk"
    # ── Train from scratch ─────────────────────────────────────
    else:
        if cfg.gee_data_dir:
            terrain, X_train, y_train = load_gee_data(
                cfg.gee_data_dir, target_shape=(h, w)
            )
        else:
            terrain = generate_terrain(h, w, seed=cfg.terrain_seed)
            X_train, y_train = generate_training_data(
                terrain["features"], terrain["unburnable_mask"],
                n_fire=cfg.n_train_fire,
                n_nofire=cfg.n_train_nofire,
                seed=cfg.terrain_seed + 1,
            )

        features   = terrain["features"]
        unburnable = terrain["unburnable_mask"]
        slope      = terrain["slope"]

        t0 = time.time()
        model = LSSVM(gamma=cfg.lssvm_gamma, sigma=cfg.lssvm_sigma)
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        Pc = model.compute_probability_surface(features)
        Pc[unburnable] = 0.0

        # --- Compute accuracy metrics ---
        y_pred = model.predict(X_train)
        train_accuracy = float(np.mean(y_pred == y_train))
        n_fire = int((y_train == 1).sum())
        n_nofire = int((y_train == -1).sum())
        fire_acc = float(np.mean(y_pred[y_train == 1] == 1)) if n_fire > 0 else 0.0
        nofire_acc = float(np.mean(y_pred[y_train == -1] == -1)) if n_nofire > 0 else 0.0

        # Persist to disk
        model.save(model_path)
        np.save(str(pc_path), Pc)

        base_meta = {
            "train_samples": len(y_train),
            "train_fire": n_fire,
            "train_nofire": n_nofire,
            "train_time_s": round(train_time, 3),
            "train_accuracy": round(train_accuracy, 4),
            "fire_accuracy": round(fire_acc, 4),
            "nofire_accuracy": round(nofire_acc, 4),
            "lssvm_b": round(float(model.b or 0), 4),
            "Pc_min": float(np.min(Pc[~unburnable])),
            "Pc_max": float(np.max(Pc[~unburnable])),
            "Pc_mean": float(np.mean(Pc[~unburnable])),
        }
        _mem_cache[cache_key] = (model, Pc, base_meta, unburnable.astype(np.uint8), slope)
        cache_src = "trained"

    # ── Build CA (always fresh — depends on wind, ignition, etc.) ──
    grid = np.full((h, w), UNIGNITED, dtype=np.uint8)
    grid[unburnable.astype(bool)] = UNBURNABLE

    iy = max(0, min(h - 1, int(cfg.ignition_row_frac * h)))
    ix = max(0, min(w - 1, int(cfg.ignition_col_frac * w)))

    # Multi-cell ignition block: set a (2r+1)×(2r+1) patch to BURNING
    r = cfg.ignition_radius
    r0 = max(0, iy - r)
    r1 = min(h, iy + r + 1)
    c0 = max(0, ix - r)
    c1 = min(w, ix + r + 1)
    patch = grid[r0:r1, c0:c1]
    patch[patch == UNIGNITED] = BURNING  # only ignite burnable cells

    ca_cfg = CAConfig(
        alpha=cfg.ca_alpha,
        beta=cfg.ca_beta,
        seed=cfg.ca_seed,
        wind=Wind(speed_mps=cfg.wind_speed, direction_deg=cfg.wind_direction),
        burn_duration=cfg.burn_duration,
        ignition_radius=cfg.ignition_radius,
    )
    ca = ForestFireCA(grid, Pc, ca_cfg, slope_deg=slope)

    meta = {**base_meta, "cache": cache_src}
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