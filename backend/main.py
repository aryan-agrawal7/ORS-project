from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, rowcol
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform as reproject_points,
    transform_bounds,
)

from ca_model import BURNING, CAConfig, ForestFireCA, UNBURNABLE, UNIGNITED
from gee_data_loader import load_gee_rasters
from lssvm_model import LSSVM
from wind_data_processor import process_wind_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CONFIG_PATH = Path(__file__).parent / "simulation_config.json"


class SimConfig(BaseModel):
    model_config = {"extra": "forbid"}

    gee_data_dir: str = "../data"
    grib_path: str = "../downloadeduwindvwind.grib"

    grid_h: int = Field(default=120, ge=8, le=5000)
    grid_w: int = Field(default=160, ge=8, le=5000)

    projected_crs: str = "EPSG:32648"
    geographic_crs: str = "EPSG:4326"

    ignition_lat: float = Field(default=27.85, ge=-90.0, le=90.0)
    ignition_lon: float = Field(default=102.25, ge=-180.0, le=180.0)
    ignition_radius: int = Field(default=3, ge=0, le=50)

    lssvm_gamma: float = Field(default=100.0, gt=0.0)
    lssvm_sigma: float = Field(default=1.0, gt=0.0)
    lssvm_model_path: str = ".model_cache/lssvm_model.npz"

    ca_alpha: float = Field(default=2.0, gt=0.0)
    ca_beta: float = Field(default=1.0, gt=0.0)
    ca_seed: int = 42
    burn_duration: int = Field(default=3, ge=1, le=100)

    max_steps: int = Field(default=400, ge=1, le=5000)

    output_dir: str = "../outputs"
    output_prefix: str = "wildfire"


def _resolve_path(base_dir: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _get_config_path() -> Path:
    env_path = os.getenv("SIM_CONFIG_PATH", "").strip()
    if env_path:
        return Path(env_path).resolve()
    return CONFIG_PATH


def load_runtime_config() -> SimConfig:
    cfg_path = _get_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    try:
        cfg = SimConfig(**raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {cfg_path}: {exc}") from exc

    try:
        CRS.from_user_input(cfg.projected_crs)
        CRS.from_user_input(cfg.geographic_crs)
    except Exception as exc:  # pragma: no cover - rasterio errors vary by platform
        raise ValueError(f"Invalid CRS value in config: {exc}") from exc

    return cfg


def _model_cache_key(cfg: SimConfig, data_dir: Path, model_path: Path) -> str:
    slope_tif = data_dir / "slope.tif"
    parts = (
        "georef_v2",
        str(data_dir),
        str(model_path),
        model_path.stat().st_mtime_ns,
        cfg.grid_h,
        cfg.grid_w,
        cfg.projected_crs,
        cfg.geographic_crs,
        slope_tif.stat().st_mtime_ns if slope_tif.exists() else 0,
    )
    raw = json.dumps(parts, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _stratified_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == -1
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
        avg_rank = (i + j + 2) / 2.0
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


def _write_geotiff(
    out_path: Path,
    array: np.ndarray,
    transform,
    crs,
    dtype: str,
    nodata: Optional[float] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def _reproject_array(
    src_array: np.ndarray,
    src_transform,
    src_crs,
    dst_crs,
    resampling: Resampling,
    dst_dtype,
) -> tuple[np.ndarray, object]:
    src_h, src_w = src_array.shape
    left, bottom, right, top = array_bounds(src_h, src_w, src_transform)
    dst_transform, dst_w_raw, dst_h_raw = calculate_default_transform(
        src_crs,
        dst_crs,
        src_w,
        src_h,
        left,
        bottom,
        right,
        top,
    )
    if dst_w_raw is None or dst_h_raw is None:
        raise ValueError("Unable to compute destination shape for reprojection")
    dst_w = int(dst_w_raw)
    dst_h = int(dst_h_raw)

    dst = np.empty((dst_h, dst_w), dtype=dst_dtype)
    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst, dst_transform


def _export_probability_geotiffs(
    cfg: SimConfig,
    cache_key: str,
    pc: np.ndarray,
    grid_transform,
    grid_crs,
    backend_dir: Path,
) -> dict:
    out_dir = _resolve_path(backend_dir, cfg.output_dir)
    projected_path = out_dir / f"{cfg.output_prefix}_pc_{cache_key}_projected.tif"
    geographic_path = out_dir / f"{cfg.output_prefix}_pc_{cache_key}_geographic.tif"

    _write_geotiff(
        projected_path,
        pc,
        transform=grid_transform,
        crs=grid_crs,
        dtype="float32",
    )

    geo_pc, geo_transform = _reproject_array(
        src_array=pc.astype(np.float32),
        src_transform=grid_transform,
        src_crs=grid_crs,
        dst_crs=CRS.from_user_input(cfg.geographic_crs),
        resampling=Resampling.bilinear,
        dst_dtype=np.float32,
    )
    _write_geotiff(
        geographic_path,
        geo_pc,
        transform=geo_transform,
        crs=cfg.geographic_crs,
        dtype="float32",
    )

    return {
        "lssvm_pc_projected_tif": str(projected_path),
        "lssvm_pc_geographic_tif": str(geographic_path),
    }


def _export_final_ca_geotiffs(
    cfg: SimConfig,
    cache_key: str,
    step_idx: int,
    grid: np.ndarray,
    grid_transform,
    grid_crs,
    backend_dir: Path,
) -> dict:
    out_dir = _resolve_path(backend_dir, cfg.output_dir)
    projected_path = out_dir / f"{cfg.output_prefix}_ca_final_{cache_key}_s{step_idx:04d}_projected.tif"
    geographic_path = out_dir / f"{cfg.output_prefix}_ca_final_{cache_key}_s{step_idx:04d}_geographic.tif"

    _write_geotiff(
        projected_path,
        grid,
        transform=grid_transform,
        crs=grid_crs,
        dtype="uint8",
        nodata=255,
    )

    geo_grid, geo_transform = _reproject_array(
        src_array=grid.astype(np.uint8),
        src_transform=grid_transform,
        src_crs=grid_crs,
        dst_crs=CRS.from_user_input(cfg.geographic_crs),
        resampling=Resampling.nearest,
        dst_dtype=np.uint8,
    )
    _write_geotiff(
        geographic_path,
        geo_grid,
        transform=geo_transform,
        crs=cfg.geographic_crs,
        dtype="uint8",
        nodata=255,
    )

    return {
        "ca_final_projected_tif": str(projected_path),
        "ca_final_geographic_tif": str(geographic_path),
    }


def _resolve_ignition_cell(
    cfg: SimConfig,
    grid_transform,
    grid_crs,
    shape: tuple[int, int],
    unburnable_mask: np.ndarray,
) -> tuple[int, int, float, float]:
    h, w = shape
    left, bottom, right, top = array_bounds(h, w, grid_transform)

    projected = reproject_points(
        cfg.geographic_crs,
        grid_crs,
        [cfg.ignition_lon],
        [cfg.ignition_lat],
    )
    xs = projected[0]
    ys = projected[1]
    x_ign = float(xs[0])
    y_ign = float(ys[0])

    inside = left <= x_ign <= right and bottom <= y_ign <= top
    if not inside:
        geo_left, geo_bottom, geo_right, geo_top = transform_bounds(
            grid_crs,
            cfg.geographic_crs,
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )

        swapped_projected = reproject_points(
            cfg.geographic_crs,
            grid_crs,
            [cfg.ignition_lat],
            [cfg.ignition_lon],
        )
        x_swapped = float(swapped_projected[0][0])
        y_swapped = float(swapped_projected[1][0])
        swapped_inside = left <= x_swapped <= right and bottom <= y_swapped <= top

        if swapped_inside:
            raise ValueError(
                "Ignition coordinate is outside the modeled raster extent after reprojection. "
                "Coordinates appear swapped. "
                f"Try ignition_lat={cfg.ignition_lon}, ignition_lon={cfg.ignition_lat}. "
                "Grid geographic extent "
                f"(x_min, y_min, x_max, y_max) in {cfg.geographic_crs}: "
                f"({geo_left:.6f}, {geo_bottom:.6f}, {geo_right:.6f}, {geo_top:.6f})."
            )

        raise ValueError(
            "Ignition coordinate is outside the modeled raster extent after reprojection. "
            f"Configured ignition_lat={cfg.ignition_lat}, ignition_lon={cfg.ignition_lon}. "
            "Grid geographic extent "
            f"(x_min, y_min, x_max, y_max) in {cfg.geographic_crs}: "
            f"({geo_left:.6f}, {geo_bottom:.6f}, {geo_right:.6f}, {geo_top:.6f})."
        )

    row, col = rowcol(grid_transform, x_ign, y_ign)
    iy = int(row)
    ix = int(col)

    if iy < 0 or iy >= h or ix < 0 or ix >= w:
        raise ValueError(
            "Ignition coordinate mapped to an invalid grid index after reprojection"
        )

    if bool(unburnable_mask[iy, ix]):
        raise ValueError(
            "Ignition coordinate falls on an unburnable cell (NDVI <= 0)."
        )

    return iy, ix, x_ign, y_ign


def _validate_training_data(X_train: np.ndarray, y_train: np.ndarray) -> None:
    if X_train.ndim != 2 or X_train.shape[1] != 5:
        raise ValueError("Training features must have shape (N, 5)")
    if y_train.ndim != 1 or len(y_train) != len(X_train):
        raise ValueError("Training labels must have shape (N,) matching features")
    if len(y_train) < 10:
        raise ValueError("Insufficient training data; at least 10 samples are required")

    classes = set(np.unique(y_train).tolist())
    if not classes.issubset({-1.0, 1.0}):
        raise ValueError("Training labels must be binary: -1 and +1")
    if len(classes) < 2:
        raise ValueError("Training labels must contain both fire and non-fire classes")


def build_simulation(cfg: SimConfig):
    backend_dir = Path(__file__).parent
    data_dir = _resolve_path(backend_dir, cfg.gee_data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for fname in ("slope.tif", "aspect.tif", "elevation.tif", "ndvi.tif", "humidity.tif"):
        if not (data_dir / fname).is_file():
            raise FileNotFoundError(f"Missing required input file: {data_dir / fname}")

    wind_generation_status = "precomputed"
    wind_t0 = data_dir / "wind_t0"
    if not wind_t0.is_dir():
        grib_path = _resolve_path(backend_dir, cfg.grib_path)
        if not grib_path.is_file():
            raise FileNotFoundError(
                f"Missing wind input: neither {wind_t0} exists nor GRIB file found at {grib_path}"
            )

        slope_ref = data_dir / "slope.tif"
        process_wind_data(
            grib_path=str(grib_path),
            ref_path=str(slope_ref),
            output_dir=str(data_dir),
        )
        wind_generation_status = "generated_from_grib"

    if not wind_t0.is_dir():
        raise FileNotFoundError(f"Wind processing failed; expected directory not found: {wind_t0}")

    terrain = load_gee_rasters(
        data_dir=str(data_dir),
        projected_crs=cfg.projected_crs,
        target_shape=(cfg.grid_h, cfg.grid_w),
    )

    features = terrain["features"]
    unburnable = terrain["unburnable_mask"].astype(bool)
    slope = terrain["slope"]
    grid_transform = terrain["grid_transform"]
    grid_crs = terrain["grid_crs"]

    h, w = int(features.shape[1]), int(features.shape[2])

    model_path = _resolve_path(backend_dir, cfg.lssvm_model_path)
    resolved_model_path = model_path if model_path.exists() else model_path.with_suffix(".npz")
    if not resolved_model_path.is_file():
        raise FileNotFoundError(
            "Configured LSSVM model not found. "
            f"Expected at {resolved_model_path}. "
            "Run train_lssvm.py first to create it."
        )

    t0 = time.time()
    model = LSSVM.load(resolved_model_path)
    model_load_time = time.time() - t0

    Pc = model.compute_probability_surface(features)
    cache_key = _model_cache_key(cfg, data_dir, resolved_model_path)

    metrics_path = resolved_model_path.with_suffix(".metrics.json")
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as fh:
            loaded_meta = json.load(fh)
        base_meta = loaded_meta if isinstance(loaded_meta, dict) else {}
    else:
        base_meta = {}

    base_meta.update(
        {
            "lssvm_model_path": str(resolved_model_path),
            "lssvm_metrics_path": str(metrics_path) if metrics_path.is_file() else None,
            "model_load_time_s": round(model_load_time, 3),
            "lssvm_gamma": float(model.gamma),
            "lssvm_sigma": float(model.sigma),
            "lssvm_b": round(float(model.b or 0.0), 4),
        }
    )
    cache_src = "pretrained"

    Pc = np.asarray(Pc, dtype=np.float32)
    Pc[unburnable] = 0.0

    valid = ~unburnable
    if not np.any(valid):
        raise ValueError("All cells are unburnable after preprocessing")

    pc_meta = {
        "Pc_min": float(np.min(Pc[valid])),
        "Pc_max": float(np.max(Pc[valid])),
        "Pc_mean": float(np.mean(Pc[valid])),
    }

    pc_export_meta = _export_probability_geotiffs(
        cfg=cfg,
        cache_key=cache_key,
        pc=Pc,
        grid_transform=grid_transform,
        grid_crs=grid_crs,
        backend_dir=backend_dir,
    )

    iy, ix, ign_x, ign_y = _resolve_ignition_cell(
        cfg=cfg,
        grid_transform=grid_transform,
        grid_crs=grid_crs,
        shape=(h, w),
        unburnable_mask=unburnable,
    )

    grid = np.full((h, w), UNIGNITED, dtype=np.uint8)
    grid[unburnable] = UNBURNABLE

    r = int(cfg.ignition_radius)
    r0 = max(0, iy - r)
    r1 = min(h, iy + r + 1)
    c0 = max(0, ix - r)
    c1 = min(w, ix + r + 1)

    patch = grid[r0:r1, c0:c1]
    ignitable = patch == UNIGNITED
    if not np.any(ignitable):
        raise ValueError("Ignition area does not overlap any burnable cells")
    patch[ignitable] = BURNING

    ca_cfg = CAConfig(
        alpha=cfg.ca_alpha,
        beta=cfg.ca_beta,
        seed=cfg.ca_seed,
        burn_duration=cfg.burn_duration,
        ignition_radius=cfg.ignition_radius,
    )

    ca = ForestFireCA(
        initial_grid=grid,
        p_ignite=Pc,
        cfg=ca_cfg,
        slope_deg=slope,
        wind_data_dir=str(data_dir),
        grid_transform=grid_transform,
        grid_crs=grid_crs,
    )

    meta = {
        **base_meta,
        **pc_meta,
        **pc_export_meta,
        "cache": cache_src,
        "wind_mode": "dynamic_grib_kw",
        "wind_generation_status": wind_generation_status,
        "projected_crs": str(grid_crs),
        "geographic_crs": cfg.geographic_crs,
        "ignition_lat": cfg.ignition_lat,
        "ignition_lon": cfg.ignition_lon,
        "ignition_x_m": round(ign_x, 3),
        "ignition_y_m": round(ign_y, 3),
        "ignition_row": iy,
        "ignition_col": ix,
        "grid_h": h,
        "grid_w": w,
        "config_path": str(_get_config_path()),
    }

    export_ctx = {
        "cache_key": cache_key,
        "grid_transform": grid_transform,
        "grid_crs": grid_crs,
    }

    return ca, Pc, meta, export_ctx


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/config")
def get_config():
    try:
        cfg = load_runtime_config()
        return {"ok": True, "config": cfg.model_dump(), "config_path": str(_get_config_path())}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/config-check")
def config_check():
    try:
        cfg = load_runtime_config()
        _, _, meta, _ = build_simulation(cfg)
        return {"ok": True, "config": cfg.model_dump(), **meta}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    try:
        cfg = load_runtime_config()
        ca, Pc, meta, export_ctx = build_simulation(cfg)
    except Exception as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        await ws.close(code=1011)
        return

    await ws.send_text(json.dumps({"type": "meta", **meta}))
    await ws.send_text(
        json.dumps(
            {
                "type": "probability",
                "height": int(Pc.shape[0]),
                "width": int(Pc.shape[1]),
                "data": Pc.flatten().tolist(),
            }
        )
    )

    step_idx = 0
    final_frame = ca.grid.copy()

    try:
        while step_idx < cfg.max_steps:
            frame = ca.step()
            final_frame = frame.copy()

            await ws.send_text(
                json.dumps(
                    {
                        "type": "frame",
                        "step": step_idx,
                        "height": int(frame.shape[0]),
                        "width": int(frame.shape[1]),
                        "cells": frame.flatten().tolist(),
                    }
                )
            )
            step_idx += 1

            if not np.any(frame == BURNING):
                break

            await asyncio.sleep(0.08)
    except Exception:
        return

    try:
        ca_exports = _export_final_ca_geotiffs(
            cfg=cfg,
            cache_key=export_ctx["cache_key"],
            step_idx=step_idx,
            grid=final_frame,
            grid_transform=export_ctx["grid_transform"],
            grid_crs=export_ctx["grid_crs"],
            backend_dir=Path(__file__).parent,
        )
        await ws.send_text(
            json.dumps(
                {
                    "type": "completed",
                    "steps": step_idx,
                    **ca_exports,
                }
            )
        )
    except Exception as exc:
        await ws.send_text(
            json.dumps(
                {
                    "type": "warning",
                    "message": f"Final CA GeoTIFF export failed: {exc}",
                }
            )
        )


if __name__ == "__main__":
    host = os.getenv("CA_HOST", "0.0.0.0")
    port = int(os.getenv("CA_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
