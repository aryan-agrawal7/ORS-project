from __future__ import annotations
import math
import numpy as np
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

# States (as in the paper, Section 2.1)
UNBURNABLE = 0  # cannot be burned  (e.g. water, rock)
UNIGNITED  = 1  # has not been ignited
BURNING    = 2  # burning
BURNED     = 3  # has been burned


@dataclass
class Wind:
    """
    Wind settings.  The paper (Section 2.2) decomposes wind to 8 directions
    using projected components V·cos(δ).
    direction_deg: compass direction wind blows TOWARDS (0 = N, 90 = E).
    """
    speed_mps: float
    direction_deg: float


def _dir_to_unit_vectors_8() -> Tuple[Tuple[int, int], ...]:
    """
    Moore neighborhood offsets in an order that matches 8 compass directions.
    We'll use (di, dj) where i = row (down), j = col (right).

    Mapping chosen:
      N  = (-1, 0)
      NE = (-1, +1)
      E  = (0, +1)
      SE = (+1, +1)
      S  = (+1, 0)
      SW = (+1, -1)
      W  = (0, -1)
      NW = (-1, -1)
    """
    return ((-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1))


WIND_DIR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _offset_to_angle_deg(di: int, dj: int) -> float:
    """
    Convert neighbor offset to compass angle in degrees where:
      0 = North, 90 = East, 180 = South, 270 = West.
    """
    # row decreases -> North, col increases -> East
    angle = math.degrees(math.atan2(dj, -di))  # atan2(x, y) but we want 0 at North
    return (angle + 360.0) % 360.0


def compute_wind_weights(wind: Wind) -> Dict[str, float]:
    """
    Paper Section 2.2 decomposes wind to 8 directions using projections V*cos(delta).
    Then uses Kw = exp(0.1783 * V_proj) for each direction.

    We compute per-neighbor directional weights Kw(dir).
    """
    offsets = _dir_to_unit_vectors_8()
    wind_to = wind.direction_deg % 360.0
    weights = []
    for (di, dj) in offsets:
        neighbor_angle = _offset_to_angle_deg(di, dj)  # direction from cell to neighbor
        # Angular difference between wind direction and this neighbor direction
        delta = math.radians((neighbor_angle - wind_to + 540.0) % 360.0 - 180.0)  # [-pi, pi]
        v_proj = wind.speed_mps * math.cos(delta)
        kw = math.exp(0.1783 * v_proj)  # paper Eq.(2) applied to projected wind
        weights.append(kw)
    w = np.array(weights, dtype=np.float32)
    return w


def random_threshold(alpha: float, beta: float, rng: np.random.Generator, shape: Tuple[int, int]) -> np.ndarray:
    """
    Paper Eq.(1):
      e^t = beta * 1/(1 + (-ln(gamma))^alpha)
    gamma uniform in (0,1)
    """
    gamma = rng.random(shape, dtype=np.float32)
    # prevent ln(0)
    gamma = np.clip(gamma, 1e-7, 1.0 - 1e-7)
    e = beta * (1.0 / (1.0 + np.power(-np.log(gamma), alpha)))
    return e.astype(np.float32)


def compute_theta(grid: np.ndarray, wind_weights: np.ndarray) -> np.ndarray:
    """
    Paper Eq.(3):
      theta(i,j) = sum(Kw * c(neighbor)) / sum(Kw)
    where c(neighbor) = 1 if neighbor is burning else 0.

    We'll do it via shifts for the 8 neighbors.
    """
    h, w = grid.shape
    burning = (grid == BURNING).astype(np.float32)

    offsets = _dir_to_unit_vectors_8()
    num = np.zeros((h, w), dtype=np.float32)
    if wind_weights.ndim == 1:
        den = float(np.sum(wind_weights))
    else:
        den = np.sum(wind_weights, axis=0).astype(np.float32)

    for k, (di, dj) in enumerate(offsets):
        kw = wind_weights[k]
        shifted = np.roll(np.roll(burning, di, axis=0), dj, axis=1)
        num += kw * shifted

    if isinstance(den, float):
        theta = num / den if den > 0 else np.zeros_like(num)
    else:
        theta = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return theta


def any_burning_neighbor(grid: np.ndarray) -> np.ndarray:
    """Boolean mask: cell has at least one burning neighbor (Moore)."""
    burning = (grid == BURNING)
    offsets = _dir_to_unit_vectors_8()
    acc = np.zeros_like(burning, dtype=bool)
    for di, dj in offsets:
        acc |= np.roll(np.roll(burning, di, axis=0), dj, axis=1)
    return acc


from dataclasses import dataclass, field

@dataclass
class CAConfig:
    alpha: float = 2.0
    beta: float = 1.0
    seed: int = 123
    wind: Wind = field(default_factory=lambda: Wind(speed_mps=2.0, direction_deg=90.0))
    burn_duration: int = 2          # how many steps a cell stays BURNING before → BURNED
    ignition_radius: int = 3        # radius of the initial ignition block (0 = single cell)


def compute_slope_factor(slope_deg: np.ndarray) -> np.ndarray:
    """
    Paper Eq. (18): K_phi = exp(3.533 * (tan(phi))^1.2)
    where phi is the terrain slope angle in degrees.

    In the paper, K_phi modulates the rate-of-spread R (Eq. 17: t = L/R).
    Steeper slopes → higher K_phi → faster fire advance.
    We normalise so that flat terrain = 1.0 and steep slopes > 1.0.
    """
    phi_rad = np.radians(np.clip(slope_deg, 0.0, 75.0))
    tan_phi = np.tan(phi_rad)
    k_phi = np.exp(3.533 * np.power(np.clip(tan_phi, 0.0, 5.0), 1.2))
    # Normalise so flat terrain has factor 1.0
    k_phi = k_phi / k_phi.min() if k_phi.min() > 0 else k_phi
    return k_phi.astype(np.float32)


class ForestFireCA:
    """
    Implements the LSSVM-CA update rules from the paper:
      - 4 states (Section 2.1)
      - Moore neighbourhood
      - Ignition probability  P(i,j) = Pc * theta  (Eq. 15-16)
        where Pc is the LSSVM-derived probability and theta the adjacent wind effect
      - Slope factor K_phi modulates rate-of-spread (Eq. 17-18):
        steeper terrain → longer burn duration → fire front advances faster
      - Random threshold e^t  (Eq. 1)

    Parameters
    ----------
    initial_grid : ndarray (H, W) uint8
        Cell states.
    p_ignite : ndarray (H, W) float32
        LSSVM-derived ignition probability Pc for each cell.
    cfg : CAConfig
    slope_deg : ndarray (H, W) float32, optional
        Slope raster in degrees.  If provided, the slope factor K_phi
        from Eq. 18 modulates per-cell burn duration (rate of spread).
    """
    def __init__(self, initial_grid: np.ndarray, p_ignite: np.ndarray,
                 cfg: Optional[CAConfig] = None,
                 slope_deg: Optional[np.ndarray] = None,
                 wind_data_dir: Optional[str] = None,
                 grid_transform=None,
                 grid_crs=None):
        if initial_grid.shape != p_ignite.shape:
            raise ValueError("initial_grid and p_ignite must have same shape")
        self.grid = initial_grid.astype(np.uint8)
        self.p = np.clip(p_ignite.astype(np.float32), 0.0, 1.0)
        self.cfg = cfg or CAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.wind_data_dir = wind_data_dir
        self.grid_transform = grid_transform
        self.grid_crs = grid_crs
        self.step_index = 0
        self.dynamic_wind_enabled = False
        self.wind_weights = compute_wind_weights(self.cfg.wind)

        if self.wind_data_dir is not None:
            loaded = self._load_wind_weights_for_step(0)
            self.dynamic_wind_enabled = loaded

        # Slope factor (Eq. 18) → modulates per-cell burn duration
        # Paper Eq. 17:  t = L / R  where R ∝ K_phi * K_w
        # Higher K_phi → faster spread → longer effective burn window
        if slope_deg is not None:
            self.k_phi = compute_slope_factor(slope_deg)
        else:
            self.k_phi = np.ones(initial_grid.shape, dtype=np.float32)

        # Per-cell burn duration: base_duration * K_phi (capped at a max)
        self.cell_burn_dur = np.clip(
            (self.cfg.burn_duration * self.k_phi).astype(np.int32),
            self.cfg.burn_duration,
            self.cfg.burn_duration * 5,
        )

        # Burn timer: how many more steps each cell stays BURNING
        self.burn_timer = np.zeros(initial_grid.shape, dtype=np.int32)
        # Initialise timers for any cells that start as BURNING
        mask = (self.grid == BURNING)
        self.burn_timer[mask] = self.cell_burn_dur[mask]

    def _wind_step_dir(self, step: int) -> str:
        return os.path.join(self.wind_data_dir, f"wind_t{step}")

    def _load_wind_weights_for_step(self, step: int) -> bool:
        """Load per-cell 8-direction wind coefficients for a step; fallback to t0."""
        if self.wind_data_dir is None:
            return False

        candidate = self._wind_step_dir(step)
        if not os.path.isdir(candidate):
            candidate = self._wind_step_dir(0)
            if not os.path.isdir(candidate):
                return False

        weights = []
        for name in WIND_DIR_NAMES:
            fp = os.path.join(candidate, f"kw_{name}.tif")
            if not os.path.isfile(fp):
                return False
            with rasterio.open(fp) as src:
                if self.grid_transform is not None and self.grid_crs is not None:
                    if src.crs is None:
                        return False
                    arr = np.empty(self.grid.shape, dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=self.grid_transform,
                        dst_crs=self.grid_crs,
                        resampling=Resampling.bilinear,
                    )
                else:
                    if (src.height, src.width) == self.grid.shape:
                        arr = src.read(1).astype(np.float32)
                    else:
                        arr = src.read(
                            1,
                            out_shape=self.grid.shape,
                            resampling=Resampling.bilinear,
                        ).astype(np.float32)
            weights.append(arr)

        self.wind_weights = np.stack(weights, axis=0)
        return True

    def step(self) -> np.ndarray:
        g = self.grid
        next_g = g.copy()

        if self.dynamic_wind_enabled:
            self._load_wind_weights_for_step(self.step_index)

        # --- Burning → Burned only after per-cell burn_duration steps ---
        burning_mask = (g == BURNING)
        self.burn_timer[burning_mask] -= 1
        expired = burning_mask & (self.burn_timer <= 0)
        next_g[expired] = BURNED

        # --- Ignition of unignited cells ---
        candidates = (g == UNIGNITED)
        has_fire_neighbor = any_burning_neighbor(g)

        # compute theta from wind-weighted burning neighbors  — Eq. (16)
        theta = compute_theta(g, self.wind_weights)  # [0..1]

        # P(i,j) = Pc * theta   — Eq. (15-16)
        # NOTE: K_phi is NOT in the transition probability (paper uses it
        # in the time-step Eq. 17, not the ignition probability).
        P = self.p * theta
        P = np.clip(P, 0.0, 1.0)

        # random threshold e^t  — Eq. (1)
        e = random_threshold(self.cfg.alpha, self.cfg.beta, self.rng, g.shape)

        ignite = candidates & has_fire_neighbor & (P > e)
        next_g[ignite] = BURNING
        self.burn_timer[ignite] = self.cell_burn_dur[ignite]

        self.grid = next_g
        self.step_index += 1
        return self.grid