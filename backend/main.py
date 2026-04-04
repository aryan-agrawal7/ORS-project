from __future__ import annotations
import asyncio
import hashlib
import json
import time
import os
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from ca_model import ForestFireCA, CAConfig, UNBURNABLE, UNIGNITED, BURNING, BURNED
from lssvm_model import LSSVM
from data_generator import generate_terrain, generate_training_data
from gee_data_loader import load_gee_data
from wind_data_processor import process_wind_data

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
        "metrics_v2",
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


def _stratified_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Deterministic stratified split for labels in {-1, +1}."""
    if val_ratio <= 0.0 or val_ratio >= 1.0 or len(y) < 4:
        return X, y, None, None

    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == -1)[0]

    def _split_class(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(idx) < 2:
            return idx, np.array([], dtype=np.int64)
        idx = idx.copy()
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_ratio))
        n_val = max(1, min(n_val, len(idx) - 1))
        return idx[n_val:], idx[:n_val]

    tr_pos, va_pos = _split_class(idx_pos)
    tr_neg, va_neg = _split_class(idx_neg)

    train_idx = np.concatenate([tr_pos, tr_neg])
    val_idx = np.concatenate([va_pos, va_neg])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    if len(val_idx) == 0:
        return X[train_idx], y[train_idx], None, None

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _binary_roc_auc_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Compute ROC-AUC for positive class (+1) using rank statistics."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = (y_true == 1)
    neg = (y_true == -1)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)

    i = 0
    n = len(sorted_scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> dict:
    """Return confusion-matrix metrics for fire(+1) vs non-fire(-1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    fp = int(np.sum((y_true == -1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == -1)))
    n = int(len(y_true))

    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    balanced_acc = 0.5 * (recall + specificity)
    auc = _binary_roc_auc_from_scores(y_true, y_score) if y_score is not None else None

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc": None if auc is None else float(auc),
    }


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
    backend_dir = Path(__file__).parent
    gee_data_dir = Path(cfg.gee_data_dir)
    if not gee_data_dir.is_absolute():
        gee_data_dir = (backend_dir / gee_data_dir).resolve()
    gee_data_dir_str = str(gee_data_dir)
    wind_generation_status = "not_requested"
    if cfg.gee_data_dir and not (gee_data_dir / "wind_t0").is_dir():
        grib_path = backend_dir.parent / "downloadeduwindvwind.grib"
        slope_ref = gee_data_dir / "slope.tif"
        if grib_path.is_file() and slope_ref.is_file():
            try:
                process_wind_data(
                    grib_path=str(grib_path),
                    ref_path=str(slope_ref),
                    output_dir=gee_data_dir_str,
                )
                wind_generation_status = "generated"
            except Exception as exc:
                print(f"Warning: wind data processing failed: {exc}")
                wind_generation_status = "generation_failed"
        else:
            wind_generation_status = "missing_grib_or_slope"

    wind_data_dir = gee_data_dir_str if (gee_data_dir / "wind_t0").is_dir() else None

    cache_key = _model_cache_key(cfg)
    model_path = CACHE_DIR / f"lssvm_{cache_key}.npz"
    pc_path    = CACHE_DIR / f"pc_{cache_key}.npy"
    metrics_path = CACHE_DIR / f"metrics_{cache_key}.json"

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
            terrain, _, _ = load_gee_data(gee_data_dir_str, target_shape=(h, w))
        else:
            terrain = generate_terrain(h, w, seed=cfg.terrain_seed)
        unburnable = terrain["unburnable_mask"]
        slope      = terrain["slope"]
        train_time = time.time() - t0

        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    base_meta = json.load(f)
            except Exception:
                base_meta = {}
        else:
            base_meta = {}

        if not base_meta:
            # Fallback for old caches without metrics metadata.
            if model.X_train is not None and model.y_train is not None:
                y_pred = model.predict(model.X_train)
                y_score = model.predict_proba(model.X_train)
                m = _classification_metrics(model.y_train, y_pred, y_score)
                n_fire = int((model.y_train == 1).sum())
                n_nofire = int((model.y_train == -1).sum())
            else:
                m = _classification_metrics(np.array([], dtype=np.int8), np.array([], dtype=np.int8), None)
                n_fire = n_nofire = 0

            base_meta = {
                "total_samples": len(model.y_train) if model.y_train is not None else 0,
                "train_samples": len(model.y_train) if model.y_train is not None else 0,
                "val_samples": 0,
                "train_fire": n_fire,
                "train_nofire": n_nofire,
                "val_fire": 0,
                "val_nofire": 0,
                "train_time_s": None,
                "train_accuracy": round(m["accuracy"], 4),
                "fire_accuracy": round(m["recall"], 4),
                "nofire_accuracy": round(m["specificity"], 4),
                "train_precision": round(m["precision"], 4),
                "train_recall": round(m["recall"], 4),
                "train_specificity": round(m["specificity"], 4),
                "train_f1": round(m["f1"], 4),
                "train_balanced_accuracy": round(m["balanced_accuracy"], 4),
                "train_roc_auc": None if m["roc_auc"] is None else round(m["roc_auc"], 4),
                "val_accuracy": None,
                "val_precision": None,
                "val_recall": None,
                "val_specificity": None,
                "val_f1": None,
                "val_balanced_accuracy": None,
                "val_roc_auc": None,
                "val_tp": None,
                "val_tn": None,
                "val_fp": None,
                "val_fn": None,
                "lssvm_b": round(float(model.b or 0), 4),
                "Pc_min": float(np.min(Pc[~unburnable])),
                "Pc_max": float(np.max(Pc[~unburnable])),
                "Pc_mean": float(np.mean(Pc[~unburnable])),
            }

        base_meta["cache_load_time_s"] = round(train_time, 3)
        # Store in memory for next time
        _mem_cache[cache_key] = (model, Pc, base_meta, unburnable.astype(np.uint8), slope)
        cache_src = "disk"
    # ── Train from scratch ─────────────────────────────────────
    else:
        if cfg.gee_data_dir:
            terrain, X_train, y_train = load_gee_data(
                gee_data_dir_str, target_shape=(h, w)
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

        X_fit, y_fit, X_val, y_val = _stratified_train_val_split(
            X_train,
            y_train,
            val_ratio=0.2,
            seed=cfg.ca_seed,
        )

        t0 = time.time()
        model = LSSVM(gamma=cfg.lssvm_gamma, sigma=cfg.lssvm_sigma)
        model.fit(X_fit, y_fit)
        train_time = time.time() - t0

        Pc = model.compute_probability_surface(features)
        Pc[unburnable] = 0.0

        # --- Compute train/validation metrics ---
        y_fit_pred = model.predict(X_fit)
        y_fit_score = model.predict_proba(X_fit)
        m_train = _classification_metrics(y_fit, y_fit_pred, y_fit_score)

        if X_val is not None and y_val is not None and len(y_val) > 0:
            y_val_pred = model.predict(X_val)
            y_val_score = model.predict_proba(X_val)
            m_val = _classification_metrics(y_val, y_val_pred, y_val_score)
            val_samples = int(len(y_val))
            val_fire = int((y_val == 1).sum())
            val_nofire = int((y_val == -1).sum())
        else:
            m_val = None
            val_samples = 0
            val_fire = 0
            val_nofire = 0

        n_fire = int((y_fit == 1).sum())
        n_nofire = int((y_fit == -1).sum())

        # Persist to disk
        model.save(model_path)
        np.save(str(pc_path), Pc)

        base_meta = {
            "total_samples": int(len(y_train)),
            "train_samples": int(len(y_fit)),
            "val_samples": val_samples,
            "train_fire": n_fire,
            "train_nofire": n_nofire,
            "val_fire": val_fire,
            "val_nofire": val_nofire,
            "train_time_s": round(train_time, 3),
            "train_accuracy": round(m_train["accuracy"], 4),
            "fire_accuracy": round(m_train["recall"], 4),
            "nofire_accuracy": round(m_train["specificity"], 4),
            "train_precision": round(m_train["precision"], 4),
            "train_recall": round(m_train["recall"], 4),
            "train_specificity": round(m_train["specificity"], 4),
            "train_f1": round(m_train["f1"], 4),
            "train_balanced_accuracy": round(m_train["balanced_accuracy"], 4),
            "train_roc_auc": None if m_train["roc_auc"] is None else round(m_train["roc_auc"], 4),
            "val_accuracy": None if m_val is None else round(m_val["accuracy"], 4),
            "val_precision": None if m_val is None else round(m_val["precision"], 4),
            "val_recall": None if m_val is None else round(m_val["recall"], 4),
            "val_specificity": None if m_val is None else round(m_val["specificity"], 4),
            "val_f1": None if m_val is None else round(m_val["f1"], 4),
            "val_balanced_accuracy": None if m_val is None else round(m_val["balanced_accuracy"], 4),
            "val_roc_auc": None if m_val is None or m_val["roc_auc"] is None else round(m_val["roc_auc"], 4),
            "val_tp": None if m_val is None else int(m_val["tp"]),
            "val_tn": None if m_val is None else int(m_val["tn"]),
            "val_fp": None if m_val is None else int(m_val["fp"]),
            "val_fn": None if m_val is None else int(m_val["fn"]),
            "lssvm_b": round(float(model.b or 0), 4),
            "Pc_min": float(np.min(Pc[~unburnable])),
            "Pc_max": float(np.max(Pc[~unburnable])),
            "Pc_mean": float(np.mean(Pc[~unburnable])),
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(base_meta, f)
        _mem_cache[cache_key] = (model, Pc, base_meta, unburnable.astype(np.uint8), slope)
        cache_src = "trained"

    # --- Build CA (always fresh — depends on wind, ignition, etc.) ──
    shape = (h, w)
    grid = np.full(shape, UNIGNITED, dtype=np.uint8)
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

    # The Pc surface is the ignition probability
    p_ignite_grid = Pc

    ca_cfg = CAConfig(
        alpha=cfg.ca_alpha,
        beta=cfg.ca_beta,
        seed=cfg.ca_seed,
        burn_duration=cfg.burn_duration,
        ignition_radius=cfg.ignition_radius,
    )

    # Cellular Automaton
    ca = ForestFireCA(
        initial_grid=grid,
        p_ignite=p_ignite_grid,
        cfg=ca_cfg,
        slope_deg=slope,
        wind_data_dir=wind_data_dir,
    )
    
    meta = {
        **base_meta,
        "cache": cache_src,
        "wind_mode": "dynamic_grib_kw" if wind_data_dir else "constant_fallback",
        "wind_generation_status": wind_generation_status,
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
if __name__ == "__main__":
    host = os.getenv("CA_HOST", "0.0.0.0")
    port = int(os.getenv("CA_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
