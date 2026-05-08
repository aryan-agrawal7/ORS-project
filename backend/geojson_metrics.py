from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import transform_geom


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute precision, recall, F1, and IoU between two GeoJSON files "
            "(ground truth vs prediction)."
        )
    )
    parser.add_argument("--ground-truth", required=True, help="Path to ground-truth GeoJSON")
    parser.add_argument("--prediction", required=True, help="Path to predicted GeoJSON")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=256,
        help="Longest grid side used for rasterization (default: 2048)",
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Use all_touched=True during rasterization",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print output as JSON",
    )
    return parser.parse_args()


def _read_geojson(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _infer_crs(geojson_obj: dict[str, Any]) -> str:
    crs_obj = geojson_obj.get("crs")
    if isinstance(crs_obj, dict):
        props = crs_obj.get("properties")
        if isinstance(props, dict):
            name = str(props.get("name", "")).upper()
            if "CRS84" in name or "EPSG:4326" in name:
                return "EPSG:4326"
            idx = name.find("EPSG:")
            if idx >= 0:
                return name[idx:]
    return "EPSG:4326"


def _extract_geometries(geojson_obj: dict[str, Any]) -> list[dict[str, Any]]:
    gtype = geojson_obj.get("type")
    if gtype == "FeatureCollection":
        geoms = [
            feature.get("geometry")
            for feature in geojson_obj.get("features", [])
            if isinstance(feature, dict) and feature.get("geometry")
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
        raise ValueError("No valid geometries found")
    return valid


def _coords_bounds(coords: Any) -> tuple[float, float, float, float]:
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        x = float(coords[0])
        y = float(coords[1])
        return x, y, x, y

    mins_x: list[float] = []
    mins_y: list[float] = []
    maxs_x: list[float] = []
    maxs_y: list[float] = []
    for child in coords:
        x0, y0, x1, y1 = _coords_bounds(child)
        mins_x.append(x0)
        mins_y.append(y0)
        maxs_x.append(x1)
        maxs_y.append(y1)
    return min(mins_x), min(mins_y), max(maxs_x), max(maxs_y)


def _geom_bounds(geom: dict[str, Any]) -> tuple[float, float, float, float]:
    gtype = geom.get("type")
    if gtype == "GeometryCollection":
        items = geom.get("geometries", [])
        if not items:
            raise ValueError("Empty GeometryCollection found")
        bounds = [_geom_bounds(g) for g in items]
        return (
            min(b[0] for b in bounds),
            min(b[1] for b in bounds),
            max(b[2] for b in bounds),
            max(b[3] for b in bounds),
        )

    coords = geom.get("coordinates")
    if coords is None:
        raise ValueError("Geometry has no coordinates")
    return _coords_bounds(coords)


def _combined_bounds(geoms: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    bounds = [_geom_bounds(g) for g in geoms]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


def _compute_grid_shape(
    bounds: tuple[float, float, float, float],
    max_side: int,
) -> tuple[int, int]:
    left, bottom, right, top = bounds
    width = max(right - left, 1e-9)
    height = max(top - bottom, 1e-9)

    if width >= height:
        w = max_side
        h = max(1, int(round(max_side * (height / width))))
    else:
        h = max_side
        w = max(1, int(round(max_side * (width / height))))
    return h, w


def _binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & (~gt)))
    fn = int(np.sum((~pred) & gt))
    tn = int(np.sum((~pred) & (~gt)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
    }


def main() -> None:
    args = _parse_args()
    gt_path = Path(args.ground_truth).resolve()
    pred_path = Path(args.prediction).resolve()

    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth GeoJSON not found: {gt_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction GeoJSON not found: {pred_path}")
    if args.grid_size < 128:
        raise ValueError("--grid-size should be >= 128")

    gt_obj = _read_geojson(gt_path)
    pred_obj = _read_geojson(pred_path)

    gt_crs = _infer_crs(gt_obj)
    pred_crs = _infer_crs(pred_obj)

    gt_geoms = _extract_geometries(gt_obj)
    pred_geoms = _extract_geometries(pred_obj)

    if pred_crs != gt_crs:
        gt_geoms = [transform_geom(gt_crs, pred_crs, g, precision=6) for g in gt_geoms]
        common_crs = pred_crs
    else:
        common_crs = pred_crs

    left1, bottom1, right1, top1 = _combined_bounds(gt_geoms)
    left2, bottom2, right2, top2 = _combined_bounds(pred_geoms)
    bounds = (
        min(left1, left2),
        min(bottom1, bottom2),
        max(right1, right2),
        max(top1, top2),
    )

    # Small padding avoids edge-clipping when bounds align exactly with pixel edges.
    pad_x = max((bounds[2] - bounds[0]) * 0.001, 1e-9)
    pad_y = max((bounds[3] - bounds[1]) * 0.001, 1e-9)
    padded_bounds = (
        bounds[0] - pad_x,
        bounds[1] - pad_y,
        bounds[2] + pad_x,
        bounds[3] + pad_y,
    )

    height, width = _compute_grid_shape(padded_bounds, args.grid_size)
    transform = from_bounds(*padded_bounds, width=width, height=height)

    gt_mask = rasterize(
        [(g, 1) for g in gt_geoms],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=bool(args.all_touched),
    ).astype(bool)

    pred_mask = rasterize(
        [(g, 1) for g in pred_geoms],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=bool(args.all_touched),
    ).astype(bool)

    metrics = _binary_metrics(pred_mask, gt_mask)
    payload = {
        "ground_truth": str(gt_path),
        "prediction": str(pred_path),
        "crs": common_crs,
        "grid": {"height": int(height), "width": int(width)},
        **metrics,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Ground truth: {payload['ground_truth']}")
        print(f"Prediction:   {payload['prediction']}")
        print(f"CRS:          {payload['crs']}")
        print(f"Grid:         {height} x {width}")
        print("")
        print(f"Precision:    {payload['precision']:.6f}")
        print(f"Recall:       {payload['recall']:.6f}")
        print(f"F1:           {payload['f1']:.6f}")
        print(f"IoU:          {payload['iou']:.6f}")
        print("")
        print(
            "Confusion: "
            f"TP={payload['tp']}, FP={payload['fp']}, FN={payload['fn']}, TN={payload['tn']}"
        )


if __name__ == "__main__":
    main()
