from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from ca_model import BURNED, BURNING, CAConfig, ForestFireCA, UNBURNABLE, UNIGNITED
from gee_data_loader import load_gee_rasters
from lssvm_model import LSSVM
from main import _get_config_path, _resolve_ignition_cell, _resolve_path, load_runtime_config
from wind_data_processor import process_wind_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune CA hyperparameters in simulation_config.json so final burned area "
            "matches Final_burnt.geojson as closely as possible (IoU objective)."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to simulation_config.json. Defaults to backend/simulation_config.json.",
    )
    parser.add_argument(
        "--target-geojson",
        type=str,
        default=None,
        help="Path to the target burnt-area GeoJSON. Defaults to ../data/Final_burnt.geojson.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of random-search trials (baseline config is evaluated in addition).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for hyperparameter sampling.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps for each CA simulation. Defaults to config max_steps.",
    )
    parser.add_argument(
        "--tune-ca-seed",
        action="store_true",
        help="Also search over ca_seed. Otherwise keeps the current config seed fixed.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tuning_results.json",
        help="Path to write tuning results JSON.",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best hyperparameters back into simulation_config.json.",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize target polygons with all_touched=True for more permissive pixel coverage.",
    )
    return parser.parse_args()


def _extract_geometries(geojson_obj: dict[str, Any]) -> list[dict[str, Any]]:
    gtype = geojson_obj.get("type")
    if gtype == "FeatureCollection":
        geoms = [
            f.get("geometry")
            for f in geojson_obj.get("features", [])
            if isinstance(f, dict) and f.get("geometry")
        ]
    elif gtype == "Feature":
        geom = geojson_obj.get("geometry")
        geoms = [geom] if geom else []
    elif gtype in {"Polygon", "MultiPolygon", "GeometryCollection"}:
        geoms = [geojson_obj]
    else:
        geoms = []

    valid = [g for g in geoms if isinstance(g, dict) and g.get("type")]
    if not valid:
        raise ValueError("No valid geometries found in target GeoJSON")
    return valid


def _infer_geojson_crs(geojson_obj: dict[str, Any], fallback: str) -> str:
    crs_obj = geojson_obj.get("crs")
    if isinstance(crs_obj, dict):
        props = crs_obj.get("properties")
        if isinstance(props, dict):
            name = str(props.get("name", "")).upper()
            if "CRS84" in name or "EPSG:4326" in name:
                return "EPSG:4326"
            if "EPSG:" in name:
                idx = name.find("EPSG:")
                return name[idx:]
    return fallback


def _rasterize_target_mask(
    target_geojson_path: Path,
    target_shape: tuple[int, int],
    grid_transform,
    dst_crs: CRS,
    fallback_src_crs: str,
    all_touched: bool,
) -> np.ndarray:
    with target_geojson_path.open("r", encoding="utf-8") as fh:
        target_obj = json.load(fh)

    src_crs = _infer_geojson_crs(target_obj, fallback=fallback_src_crs)
    geoms = _extract_geometries(target_obj)

    projected_geoms: list[dict[str, Any]] = []
    for geom in geoms:
        projected_geoms.append(
            transform_geom(
                src_crs=src_crs,
                dst_crs=str(dst_crs),
                geom=geom,
                precision=6,
            )
        )

    mask = rasterize(
        [(g, 1) for g in projected_geoms],
        out_shape=target_shape,
        transform=grid_transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    return mask.astype(bool)


def _binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(pred & target))
    tn = int(np.sum((~pred) & (~target)))
    fp = int(np.sum(pred & (~target)))
    fn = int(np.sum((~pred) & target))

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(accuracy),
    }


def _sample_candidate(
    rng: np.random.Generator,
    base_cfg,
    best: dict[str, Any] | None,
    tune_ca_seed: bool,
) -> dict[str, float | int]:
    alpha_lo = max(0.5, float(base_cfg.ca_alpha) - 1.5)
    alpha_hi = min(6.0, float(base_cfg.ca_alpha) + 1.5)
    beta_lo = max(0.1, float(base_cfg.ca_beta) - 0.6)
    beta_hi = min(3.0, float(base_cfg.ca_beta) + 0.8)
    dur_lo = max(1, int(base_cfg.burn_duration) - 2)
    dur_hi = min(20, int(base_cfg.burn_duration) + 5)
    rad_lo = max(0, int(base_cfg.ignition_radius) - 3)
    rad_hi = min(20, int(base_cfg.ignition_radius) + 8)

    if best is not None and rng.random() < 0.35:
        alpha = float(np.clip(rng.normal(float(best["alpha"]), 0.25), alpha_lo, alpha_hi))
        beta = float(np.clip(rng.normal(float(best["beta"]), 0.15), beta_lo, beta_hi))
        burn_duration = int(np.clip(round(rng.normal(int(best["burn_duration"]), 1.2)), dur_lo, dur_hi))
        ignition_radius = int(np.clip(round(rng.normal(int(best["ignition_radius"]), 1.4)), rad_lo, rad_hi))
    else:
        alpha = float(rng.uniform(alpha_lo, alpha_hi))
        beta = float(rng.uniform(beta_lo, beta_hi))
        burn_duration = int(rng.integers(dur_lo, dur_hi + 1))
        ignition_radius = int(rng.integers(rad_lo, rad_hi + 1))

    if tune_ca_seed:
        ca_seed = int(rng.integers(1, 10000))
    else:
        ca_seed = int(base_cfg.ca_seed)

    return {
        "alpha": round(alpha, 4),
        "beta": round(beta, 4),
        "burn_duration": int(burn_duration),
        "ignition_radius": int(ignition_radius),
        "ca_seed": int(ca_seed),
    }


def _build_inputs(cfg, backend_dir: Path) -> dict[str, Any]:
    data_dir = _resolve_path(backend_dir, cfg.gee_data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for fname in ("slope.tif", "aspect.tif", "elevation.tif", "ndvi.tif", "humidity.tif"):
        fpath = data_dir / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing required input file: {fpath}")

    wind_t0 = data_dir / "wind_t0"
    if not wind_t0.is_dir():
        grib_path = _resolve_path(backend_dir, cfg.grib_path)
        if not grib_path.is_file():
            raise FileNotFoundError(
                f"Missing wind input: neither {wind_t0} exists nor GRIB file found at {grib_path}"
            )
        process_wind_data(
            grib_path=str(grib_path),
            ref_path=str(data_dir / "slope.tif"),
            output_dir=str(data_dir),
        )

    terrain = load_gee_rasters(
        data_dir=str(data_dir),
        projected_crs=cfg.projected_crs,
        target_shape=(cfg.grid_h, cfg.grid_w),
    )

    model_path = _resolve_path(backend_dir, cfg.lssvm_model_path)
    resolved_model_path = model_path if model_path.exists() else model_path.with_suffix(".npz")
    if not resolved_model_path.is_file():
        raise FileNotFoundError(
            "Configured LSSVM model not found. "
            f"Expected at {resolved_model_path}. "
            "Run train_lssvm.py first to create it."
        )

    model = LSSVM.load(resolved_model_path)
    pc = model.compute_probability_surface(terrain["features"]).astype(np.float32)
    pc[terrain["unburnable_mask"]] = 0.0

    return {
        "data_dir": data_dir,
        "pc": pc,
        "unburnable_mask": terrain["unburnable_mask"].astype(bool),
        "slope": terrain["slope"],
        "grid_transform": terrain["grid_transform"],
        "grid_crs": terrain["grid_crs"],
        "shape": terrain["grid_shape"],
    }


def _simulate_with_params(
    cfg,
    inputs: dict[str, Any],
    target_mask: np.ndarray,
    params: dict[str, float | int],
    max_steps: int,
) -> dict[str, Any]:
    h, w = inputs["shape"]
    unburnable = inputs["unburnable_mask"]

    iy, ix, _, _ = _resolve_ignition_cell(
        cfg=cfg,
        grid_transform=inputs["grid_transform"],
        grid_crs=inputs["grid_crs"],
        shape=(h, w),
        unburnable_mask=unburnable,
    )

    grid = np.full((h, w), UNIGNITED, dtype=np.uint8)
    grid[unburnable] = UNBURNABLE

    radius = int(params["ignition_radius"])
    r0 = max(0, iy - radius)
    r1 = min(h, iy + radius + 1)
    c0 = max(0, ix - radius)
    c1 = min(w, ix + radius + 1)

    patch = grid[r0:r1, c0:c1]
    ignitable = patch == UNIGNITED
    if not np.any(ignitable):
        raise ValueError("Ignition area does not overlap any burnable cells")
    patch[ignitable] = BURNING

    ca_cfg = CAConfig(
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        seed=int(params["ca_seed"]),
        burn_duration=int(params["burn_duration"]),
        ignition_radius=int(params["ignition_radius"]),
    )

    ca = ForestFireCA(
        initial_grid=grid,
        p_ignite=inputs["pc"],
        cfg=ca_cfg,
        slope_deg=inputs["slope"],
        wind_data_dir=str(inputs["data_dir"]),
        grid_transform=inputs["grid_transform"],
        grid_crs=inputs["grid_crs"],
    )

    final_grid = ca.grid.copy()
    steps_done = 0
    for step in range(max_steps):
        frame = ca.step()
        final_grid = frame.copy()
        steps_done = step + 1
        if not np.any(frame == BURNING):
            break

    pred_burned = (final_grid == BURNED) | (final_grid == BURNING)
    stats = _binary_metrics(pred_burned, target_mask)

    return {
        "alpha": float(params["alpha"]),
        "beta": float(params["beta"]),
        "burn_duration": int(params["burn_duration"]),
        "ignition_radius": int(params["ignition_radius"]),
        "ca_seed": int(params["ca_seed"]),
        "steps": int(steps_done),
        "burn_fraction": float(pred_burned.mean()),
        **stats,
    }


def _result_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(result["iou"]),
        float(result["f1"]),
        float(result["recall"]),
        float(result["precision"]),
    )


def _apply_best_to_config(cfg_path: Path, best: dict[str, Any]) -> None:
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    raw["ca_alpha"] = float(best["alpha"])
    raw["ca_beta"] = float(best["beta"])
    raw["burn_duration"] = int(best["burn_duration"])
    raw["ignition_radius"] = int(best["ignition_radius"])
    raw["ca_seed"] = int(best["ca_seed"])

    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2)
        fh.write("\n")


def main() -> None:
    args = _parse_args()

    if args.config:
        os.environ["SIM_CONFIG_PATH"] = str(Path(args.config).resolve())

    cfg = load_runtime_config()
    cfg_path = _get_config_path()
    backend_dir = Path(__file__).parent

    target_geojson = (
        Path(args.target_geojson).resolve()
        if args.target_geojson
        else _resolve_path(backend_dir, "../data/Final_burnt.geojson")
    )
    if not target_geojson.is_file():
        raise FileNotFoundError(f"Target GeoJSON not found: {target_geojson}")

    if args.trials < 1:
        raise ValueError("--trials must be >= 1")

    start_time = time.time()
    rng = np.random.default_rng(args.seed)

    inputs = _build_inputs(cfg, backend_dir=backend_dir)
    target_mask = _rasterize_target_mask(
        target_geojson_path=target_geojson,
        target_shape=inputs["shape"],
        grid_transform=inputs["grid_transform"],
        dst_crs=inputs["grid_crs"],
        fallback_src_crs=cfg.geographic_crs,
        all_touched=bool(args.all_touched),
    )
    if not np.any(target_mask):
        raise ValueError("Target mask is empty after rasterization; check target GeoJSON and CRS")

    max_steps = int(args.max_steps) if args.max_steps is not None else int(cfg.max_steps)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    baseline_params = {
        "alpha": float(cfg.ca_alpha),
        "beta": float(cfg.ca_beta),
        "burn_duration": int(cfg.burn_duration),
        "ignition_radius": int(cfg.ignition_radius),
        "ca_seed": int(cfg.ca_seed),
    }
    baseline = _simulate_with_params(cfg, inputs, target_mask, baseline_params, max_steps=max_steps)

    best = baseline
    trials: list[dict[str, Any]] = []
    seen: set[tuple[float, float, int, int, int]] = {
        (
            round(float(baseline["alpha"]), 4),
            round(float(baseline["beta"]), 4),
            int(baseline["burn_duration"]),
            int(baseline["ignition_radius"]),
            int(baseline["ca_seed"]),
        )
    }

    print(
        "Baseline: "
        f"iou={baseline['iou']:.4f}, "
        f"f1={baseline['f1']:.4f}, "
        f"recall={baseline['recall']:.4f}, "
        f"precision={baseline['precision']:.4f}"
    )

    for i in range(args.trials):
        for _ in range(25):
            params = _sample_candidate(
                rng=rng,
                base_cfg=cfg,
                best=best,
                tune_ca_seed=bool(args.tune_ca_seed),
            )
            key = (
                round(float(params["alpha"]), 4),
                round(float(params["beta"]), 4),
                int(params["burn_duration"]),
                int(params["ignition_radius"]),
                int(params["ca_seed"]),
            )
            if key not in seen:
                seen.add(key)
                break

        result = _simulate_with_params(cfg, inputs, target_mask, params, max_steps=max_steps)
        trials.append(result)

        if _result_key(result) > _result_key(best):
            best = result

        print(
            f"[{i + 1:02d}/{args.trials:02d}] "
            f"iou={result['iou']:.4f}, best_iou={best['iou']:.4f}, "
            f"alpha={result['alpha']:.3f}, beta={result['beta']:.3f}, "
            f"dur={result['burn_duration']}, rad={result['ignition_radius']}, seed={result['ca_seed']}"
        )

    sorted_trials = sorted(trials, key=_result_key, reverse=True)
    elapsed = time.time() - start_time

    report = {
        "runtime_seconds": round(elapsed, 3),
        "config_path": str(cfg_path),
        "target_geojson": str(target_geojson),
        "target_pixels": int(target_mask.sum()),
        "grid_shape": [int(inputs["shape"][0]), int(inputs["shape"][1])],
        "max_steps": int(max_steps),
        "trials_requested": int(args.trials),
        "trials_completed": int(len(trials)),
        "search_seed": int(args.seed),
        "tune_ca_seed": bool(args.tune_ca_seed),
        "baseline": baseline,
        "best": best,
        "improvement": {
            "delta_iou": float(best["iou"] - baseline["iou"]),
            "delta_f1": float(best["f1"] - baseline["f1"]),
            "delta_recall": float(best["recall"] - baseline["recall"]),
            "delta_precision": float(best["precision"] - baseline["precision"]),
        },
        "top_trials": sorted_trials[:10],
        "all_trials": sorted_trials,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (backend_dir / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    config_updated = False
    if args.apply_best:
        _apply_best_to_config(cfg_path, best)
        config_updated = True

    print(f"Results written to: {output_path}")
    print(
        "Best: "
        f"iou={best['iou']:.4f}, "
        f"f1={best['f1']:.4f}, "
        f"alpha={best['alpha']:.3f}, beta={best['beta']:.3f}, "
        f"burn_duration={best['burn_duration']}, "
        f"ignition_radius={best['ignition_radius']}, "
        f"ca_seed={best['ca_seed']}"
    )
    if config_updated:
        print(f"Updated config: {cfg_path}")


if __name__ == "__main__":
    main()
