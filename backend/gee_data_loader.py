"""
Loader for GEE-exported training data.

Reads the GeoTIFF rasters and CSV training samples exported by
``gee_export.py`` and converts them into the exact format expected by
the LSSVM → CA pipeline in ``main.py``.

Expected directory layout (Google Drive → local download)
─────────────────────────────────────────────────────────
  <data_dir>/
    slope.tif
    aspect.tif
    elevation.tif
    ndvi.tif
    humidity.tif
    burned_mask.tif          (optional, for visualisation)
    training_samples.csv     (slope, aspect, elevation, ndvi, humidity, label)

Usage
-----
  from gee_data_loader import load_gee_data
  terrain, X_train, y_train = load_gee_data("./lssvm_fire")
  # terrain dict is a drop-in replacement for data_generator.generate_terrain()
"""

from __future__ import annotations

import os
import numpy as np
import csv
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Try to use rasterio (preferred), fall back to a GDAL-free TIFF reader
# ---------------------------------------------------------------------------
_HAS_RASTERIO = False
try:
    import rasterio                   # type: ignore
    _HAS_RASTERIO = True
except ImportError:
    pass

_HAS_TIFFFILE = False
if not _HAS_RASTERIO:
    try:
        import tifffile               # type: ignore
        _HAS_TIFFFILE = True
    except ImportError:
        pass


def _read_tiff(path: str) -> np.ndarray:
    """Read a single-band GeoTIFF and return it as a 2-D float32 array."""
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
        return arr
    if _HAS_TIFFFILE:
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 3:             # (bands, H, W) or (H, W, bands)
            arr = arr[0] if arr.shape[0] <= arr.shape[-1] else arr[..., 0]
        return arr

    raise ImportError(
        "Neither 'rasterio' nor 'tifffile' is installed.\n"
        "  pip install rasterio       # recommended\n"
        "  pip install tifffile       # lightweight alternative"
    )


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gee_rasters(data_dir: str, target_shape: Optional[Tuple[int, int]] = None):
    """
    Load the five feature rasters exported by ``gee_export.py``.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing the GeoTIFFs.
    target_shape : (H, W) or None
        If given, resample all rasters to this shape (nearest neighbour).
        Useful to match the CA grid size when the raw rasters are larger.

    Returns
    -------
    dict  — same schema as ``data_generator.generate_terrain()``:
        'slope', 'aspect', 'elevation', 'ndvi', 'humidity' : (H, W) raw
        'features'        : (5, H, W) normalised [0, 1]
        'unburnable_mask' : (H, W) bool — True where NDVI ≤ 0 (water / barren)
    """
    required = ["slope.tif", "aspect.tif", "elevation.tif", "ndvi.tif", "humidity.tif"]
    for fname in required:
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"Missing required raster: {fpath}")

    raw = {}
    for fname in required:
        name = fname.replace(".tif", "")
        arr  = _read_tiff(os.path.join(data_dir, fname))
        # Replace NaN / nodata with sensible defaults
        arr = np.nan_to_num(arr, nan=0.0)
        raw[name] = arr

    # Optional: resample to a common target shape
    if target_shape is not None:
        from scipy.ndimage import zoom          # type: ignore
        for name, arr in raw.items():
            if arr.shape != target_shape:
                factors = (target_shape[0] / arr.shape[0],
                           target_shape[1] / arr.shape[1])
                raw[name] = zoom(arr, factors, order=1).astype(np.float32)

    # Ensure all rasters are the same shape (use the first one as reference)
    ref_shape = raw["slope"].shape
    for name, arr in raw.items():
        if arr.shape != ref_shape:
            raise ValueError(
                f"Raster shape mismatch: {name} is {arr.shape}, "
                f"expected {ref_shape} (same as slope.tif). "
                f"Use target_shape= to resample."
            )

    # Build normalised feature stack:  [slope, aspect, elevation, ndvi, humidity]
    features = np.stack([
        _normalise(raw["slope"]),
        _normalise(raw["aspect"]),
        _normalise(raw["elevation"]),
        _normalise(raw["ndvi"]),
        _normalise(raw["humidity"]),
    ], axis=0)   # (5, H, W)

    # Unburnable mask: water / barren where NDVI ≤ 0
    unburnable = raw["ndvi"] <= 0.0

    # If burned_mask.tif exists, load it too
    burned_path = os.path.join(data_dir, "burned_mask.tif")
    burned_mask = None
    if os.path.isfile(burned_path):
        burned_mask = _read_tiff(burned_path).astype(bool)
        if target_shape and burned_mask.shape != ref_shape:
            from scipy.ndimage import zoom
            burned_mask = zoom(burned_mask.astype(np.float32),
                               (ref_shape[0] / burned_mask.shape[0],
                                ref_shape[1] / burned_mask.shape[1]),
                               order=0).astype(bool)

    return {
        "elevation":      raw["elevation"],
        "slope":          raw["slope"],
        "aspect":         raw["aspect"],
        "ndvi":           raw["ndvi"],
        "humidity":       raw["humidity"],
        "features":       features,
        "unburnable_mask": unburnable,
        "burned_mask":    burned_mask,
    }


def load_gee_training_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the training CSV exported by ``gee_export.py``.

    Parameters
    ----------
    csv_path : str
        Path to ``training_samples.csv``.

    Returns
    -------
    X : (N, 5) float64 — columns: [slope, aspect, elevation, ndvi, humidity]
    y : (N,)   float64 — labels: +1 (fire) / -1 (non-fire)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
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
            except (KeyError, ValueError):
                continue   # skip malformed rows

    if not rows:
        raise ValueError("No valid rows found in training CSV")

    data = np.array(rows, dtype=np.float64)
    X = data[:, :5]
    y = data[:, 5]

    # Normalise features to [0, 1] per column (matching raster normalisation)
    for col in range(5):
        lo, hi = X[:, col].min(), X[:, col].max()
        if hi - lo > 1e-8:
            X[:, col] = (X[:, col] - lo) / (hi - lo)

    # Shuffle
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def load_gee_data(
    data_dir: str,
    target_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Convenience wrapper — loads both rasters and training CSV.

    Parameters
    ----------
    data_dir : str
        Folder with the GEE exports.
    target_shape : (H, W) or None
        Resample rasters to this size.

    Returns
    -------
    terrain : dict       — same as generate_terrain()
    X_train : (N, 5)
    y_train : (N,)
    """
    terrain = load_gee_rasters(data_dir, target_shape=target_shape)
    csv_path = os.path.join(data_dir, "training_samples.csv")
    X_train, y_train = load_gee_training_csv(csv_path)
    return terrain, X_train, y_train
