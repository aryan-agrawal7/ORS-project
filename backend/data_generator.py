"""
Synthetic environmental data generator for the LSSVM-CA fire spread model.

Generates raster layers for the five driving factors used in the paper:
  1. Slope (degrees)
  2. Aspect (degrees, 0–360)
  3. Elevation (metres)
  4. NDVI (normalised, -1 to 1)
  5. Relative Humidity (%, 0–100)

Also provides a utility to generate labelled training samples (fire / non-fire)
from these rasters and a known burned-area mask.

When real GIS data (DEM, Landsat, etc.) is available, replace these
functions with actual raster loaders (e.g. rasterio).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple


def _perlin_like(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Quick smooth noise via bilinear-interpolated random grid."""
    gh = max(2, int(h / scale) + 2)
    gw = max(2, int(w / scale) + 2)
    base = rng.random((gh, gw), dtype=np.float32)

    row_idx = np.linspace(0, gh - 1, h, dtype=np.float32)
    col_idx = np.linspace(0, gw - 1, w, dtype=np.float32)

    from scipy.ndimage import map_coordinates

    coords = np.meshgrid(row_idx, col_idx, indexing="ij")
    return map_coordinates(base, coords, order=1, mode="reflect").astype(np.float32)


def generate_elevation(h: int, w: int, rng: np.random.Generator,
                       base: float = 1500.0, amplitude: float = 800.0) -> np.ndarray:
    """
    Elevation raster in metres.
    Creates a smooth terrain with a ridge in the center.
    """
    noise = _perlin_like(h, w, scale=30, rng=rng)
    rr, cc = np.ogrid[:h, :w]
    # Broad ridge along center
    ridge = amplitude * 0.5 * np.exp(-((rr - h / 2) / (h * 0.4)) ** 2).astype(np.float32)
    elevation = base + amplitude * noise + ridge
    return elevation


def generate_slope(elevation: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
    """
    Slope in degrees, derived from elevation via gradient.
    cell_size is the spatial resolution in metres (30 m for Landsat/ASTER).
    """
    dy, dx = np.gradient(elevation, cell_size)
    slope_rad = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    return np.degrees(slope_rad).astype(np.float32)


def generate_aspect(elevation: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
    """
    Aspect in degrees (0–360), derived from elevation via gradient.
    """
    dy, dx = np.gradient(elevation, cell_size)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = (aspect + 360.0) % 360.0
    return aspect.astype(np.float32)


def generate_ndvi(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """
    NDVI raster in roughly [0.2, 0.85] — typical for forested areas.
    """
    noise = _perlin_like(h, w, scale=25, rng=rng)
    ndvi = 0.2 + 0.65 * noise
    return ndvi.astype(np.float32)


def generate_humidity(h: int, w: int, rng: np.random.Generator,
                      base_rh: float = 35.0) -> np.ndarray:
    """
    Relative humidity raster (%). Paper uses this as an influence factor.
    In real usage, this would come from meteorological reanalysis data.
    """
    noise = _perlin_like(h, w, scale=40, rng=rng)
    rh = base_rh + 30.0 * noise  # roughly 35–65 %
    return np.clip(rh, 5.0, 95.0).astype(np.float32)


# ------------------------------------------------------------------
# Full terrain stack
# ------------------------------------------------------------------

def generate_terrain(h: int, w: int, seed: int = 42) -> dict:
    """
    Generate all five feature rasters plus a feature stack.

    Returns
    -------
    dict with keys:
      'elevation', 'slope', 'aspect', 'ndvi', 'humidity'  — each (H, W) float32
      'features'  — shape (5, H, W)  suitable for LSSVM.compute_probability_surface
      'unburnable_mask'  — bool (H, W), True where cell cannot burn (e.g. water)
    """
    rng = np.random.default_rng(seed)

    elevation = generate_elevation(h, w, rng)
    slope = generate_slope(elevation)
    aspect = generate_aspect(elevation)
    ndvi = generate_ndvi(h, w, rng)
    humidity = generate_humidity(h, w, rng)

    # Normalise features to [0, 1] for LSSVM training stability
    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    features = np.stack([
        _norm(slope),
        _norm(aspect),
        _norm(elevation),
        _norm(ndvi),
        _norm(humidity),
    ], axis=0)  # (5, H, W)

    # Create an unburnable zone (simulated lake / river)
    rr, cc = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 3
    lake_radius = min(h, w) * 0.08
    unburnable = ((rr - cy) ** 2 + (cc - cx) ** 2) < lake_radius ** 2

    return {
        "elevation": elevation,
        "slope": slope,
        "aspect": aspect,
        "ndvi": ndvi,
        "humidity": humidity,
        "features": features,
        "unburnable_mask": unburnable.astype(bool),
    }


# ------------------------------------------------------------------
# Training sample generation
# ------------------------------------------------------------------

def generate_training_data(
    features: np.ndarray,
    unburnable_mask: np.ndarray,
    n_fire: int = 800,
    n_nofire: int = 800,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create labelled training samples.

    In the paper, real fire-point and non-fire-point data were extracted
    from historical fire events. Here we use a synthetic rule:

      fire likelihood ∝ high NDVI (dense vegetation) + steep slope
                        + low humidity  + moderate elevation

    Parameters
    ----------
    features : (5, H, W)
    unburnable_mask : (H, W) bool
    n_fire, n_nofire : numbers of positive / negative samples

    Returns
    -------
    X : (N, 5)  feature vectors
    y : (N,)    labels in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    _, H, W = features.shape
    burnable = ~unburnable_mask

    # synthetic fire likelihood score based on factors
    slope_n, aspect_n, elev_n, ndvi_n, hum_n = features  # each (H, W)

    # Higher fire probability with: steep slope, high NDVI, low humidity
    score = (
        0.30 * slope_n
        + 0.05 * aspect_n
        + 0.15 * (1.0 - np.abs(elev_n - 0.5) * 2)  # peak at mid-elevation
        + 0.30 * ndvi_n
        + 0.20 * (1.0 - hum_n)
    )
    score[~burnable] = 0.0

    # Flatten burnable cells
    flat_mask = burnable.ravel()
    flat_score = score.ravel()

    burnable_idx = np.where(flat_mask)[0]
    burnable_scores = flat_score[burnable_idx]

    # Sample fire points — bias towards high-score cells
    fire_probs = burnable_scores / burnable_scores.sum()
    fire_sel = rng.choice(burnable_idx, size=n_fire, replace=False, p=fire_probs)

    # Sample non-fire points — bias towards low-score cells
    inv_probs = (1.0 - burnable_scores)
    inv_probs = inv_probs / inv_probs.sum()
    nofire_sel = rng.choice(burnable_idx, size=n_nofire, replace=False, p=inv_probs)

    # Build X, y
    all_idx = np.concatenate([fire_sel, nofire_sel])
    X_all = features.reshape(5, -1)[:, all_idx].T  # (N, 5)
    y_all = np.concatenate([np.ones(n_fire), -np.ones(n_nofire)])

    # Shuffle
    perm = rng.permutation(len(y_all))
    return X_all[perm].astype(np.float64), y_all[perm].astype(np.float64)
