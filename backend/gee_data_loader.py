"""Utilities for loading and validating real GEE-exported wildfire inputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import calculate_default_transform, reproject


FEATURE_FILES = (
    "slope.tif",
    "aspect.tif",
    "elevation.tif",
    "ndvi.tif",
    "humidity.tif",
)


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _build_target_grid(
    slope_path: Path,
    dst_crs: CRS,
    target_shape: Optional[Tuple[int, int]],
) -> tuple[object, int, int]:
    """Create the projected modeling grid transform/shape."""
    with rasterio.open(slope_path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS metadata in {slope_path}")

        base_transform, base_w, base_h = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

    if target_shape is None:
        return base_transform, int(base_h), int(base_w)

    dst_h, dst_w = int(target_shape[0]), int(target_shape[1])
    if dst_h <= 0 or dst_w <= 0:
        raise ValueError("target_shape must contain positive values")

    left, bottom, right, top = array_bounds(base_h, base_w, base_transform)
    transform = from_bounds(left, bottom, right, top, dst_w, dst_h)
    return transform, dst_h, dst_w


def _reproject_single_band(
    path: Path,
    dst_transform,
    dst_crs: CRS,
    dst_h: int,
    dst_w: int,
    resampling: Resampling,
) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Missing CRS metadata in {path}")

        dst = np.empty((dst_h, dst_w), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )
    return np.nan_to_num(dst, nan=0.0)


def load_gee_rasters(
    data_dir: str,
    projected_crs: str,
    target_shape: Optional[Tuple[int, int]] = None,
) -> dict:
    """Load required rasters and reproject to a meter-based modeling CRS."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    dst_crs = CRS.from_user_input(projected_crs)

    for fname in FEATURE_FILES:
        fpath = data_path / fname
        if not fpath.is_file():
            raise FileNotFoundError(f"Missing required raster: {fpath}")

    dst_transform, dst_h, dst_w = _build_target_grid(
        data_path / "slope.tif",
        dst_crs,
        target_shape,
    )

    raw: dict[str, np.ndarray] = {}
    for fname in FEATURE_FILES:
        key = fname[:-4]
        raw[key] = _reproject_single_band(
            data_path / fname,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_h=dst_h,
            dst_w=dst_w,
            resampling=Resampling.bilinear,
        )

    burned_mask = None
    burned_path = data_path / "burned_mask.tif"
    if burned_path.is_file():
        burned_mask = _reproject_single_band(
            burned_path,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_h=dst_h,
            dst_w=dst_w,
            resampling=Resampling.nearest,
        ) > 0

    features = np.stack(
        [
            _normalise(raw["slope"]),
            _normalise(raw["aspect"]),
            _normalise(raw["elevation"]),
            _normalise(raw["ndvi"]),
            _normalise(raw["humidity"]),
        ],
        axis=0,
    )

    unburnable = raw["ndvi"] <= 0.0

    return {
        "elevation": raw["elevation"],
        "slope": raw["slope"],
        "aspect": raw["aspect"],
        "ndvi": raw["ndvi"],
        "humidity": raw["humidity"],
        "features": features,
        "unburnable_mask": unburnable,
        "burned_mask": burned_mask,
        "grid_transform": dst_transform,
        "grid_crs": dst_crs,
        "grid_shape": (dst_h, dst_w),
        "grid_bounds": array_bounds(dst_h, dst_w, dst_transform),
    }


def load_gee_training_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate training samples from CSV."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    required_cols = {"slope", "aspect", "elevation", "ndvi", "humidity", "label"}
    rows = []

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                "training_samples.csv must contain columns: slope, aspect, elevation, ndvi, humidity, label"
            )

        for row in reader:
            try:
                vals = [
                    float(row["slope"]),
                    float(row["aspect"]),
                    float(row["elevation"]),
                    float(row["ndvi"]),
                    float(row["humidity"]),
                ]
                label = float(row["label"])
                rows.append(vals + [label])
            except (TypeError, ValueError):
                continue

    if not rows:
        raise ValueError("No valid rows found in training_samples.csv")

    data = np.asarray(rows, dtype=np.float64)
    X = data[:, :5]
    y_raw = data[:, 5]

    unique = set(np.unique(y_raw).tolist())
    if unique.issubset({0.0, 1.0}):
        y = np.where(y_raw >= 0.5, 1.0, -1.0)
    elif unique.issubset({-1.0, 1.0}):
        y = y_raw
    else:
        raise ValueError(
            "training labels must be either {-1, +1} or {0, 1}; found: "
            + ", ".join(str(v) for v in sorted(unique))
        )

    for col in range(5):
        lo, hi = X[:, col].min(), X[:, col].max()
        if hi - lo > 1e-8:
            X[:, col] = (X[:, col] - lo) / (hi - lo)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_gee_data(
    data_dir: str,
    projected_crs: str,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Load validated real-world rasters and training data for modeling."""
    terrain = load_gee_rasters(
        data_dir=data_dir,
        projected_crs=projected_crs,
        target_shape=target_shape,
    )
    X_train, y_train = load_gee_training_csv(str(Path(data_dir) / "training_samples.csv"))
    return terrain, X_train, y_train
