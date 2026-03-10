from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

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


def _offset_to_angle_deg(di: int, dj: int) -> float:
    """
    Convert neighbor offset to compass angle in degrees where:
      0 = North, 90 = East, 180 = South, 270 = West.
    """
    # row decreases -> North, col increases -> East
    angle = math.degrees(math.atan2(dj, -di))  # atan2(x, y) but we want 0 at North
    return (angle + 360.0) % 360.0


def compute_wind_weights(wind: Wind) -> np.ndarray:
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
    den = float(np.sum(wind_weights))

    for k, (di, dj) in enumerate(offsets):
        kw = float(wind_weights[k])
        shifted = np.roll(np.roll(burning, di, axis=0), dj, axis=1)
        num += kw * shifted

    theta = num / den
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


def compute_slope_factor(slope_deg: np.ndarray) -> np.ndarray:
    """
    Paper Eq. (18): K_phi = exp(3.533 * (tan(phi))^1.2)
    where phi is the terrain slope angle in degrees.
    Clamp to avoid extreme values on very steep terrain.
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
      - Ignition probability  P(i,j) = Pc * theta  (Eq. 16)
        where Pc is the LSSVM-derived probability and theta the adjacent wind effect
      - Slope factor K_phi modifies effective spread rate
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
        from Eq. 18 modulates the transition probability.
    """
    def __init__(self, initial_grid: np.ndarray, p_ignite: np.ndarray,
                 cfg: Optional[CAConfig] = None,
                 slope_deg: Optional[np.ndarray] = None):
        if initial_grid.shape != p_ignite.shape:
            raise ValueError("initial_grid and p_ignite must have same shape")
        self.grid = initial_grid.astype(np.uint8)
        self.p = np.clip(p_ignite.astype(np.float32), 0.0, 1.0)
        self.cfg = cfg or CAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.wind_weights = compute_wind_weights(self.cfg.wind)

        # Slope factor (Eq. 18) — default to 1.0 if no slope data
        if slope_deg is not None:
            self.k_phi = compute_slope_factor(slope_deg)
        else:
            self.k_phi = np.ones(initial_grid.shape, dtype=np.float32)

    def step(self) -> np.ndarray:
        g = self.grid
        # base deterministic transitions
        next_g = g.copy()
        next_g[g == BURNING] = BURNED  # burning -> burned
        # burned stays burned; unburnable stays unburnable

        # candidates: unignited cells
        candidates = (g == UNIGNITED)

        # need at least one burning neighbor
        has_fire_neighbor = any_burning_neighbor(g)

        # compute theta from wind-weighted burning neighbors  — Eq. (3)
        theta = compute_theta(g, self.wind_weights)  # [0..1]

        # P(i,j) = Pc * theta * K_phi   — Eq. (16) extended with slope
        P = self.p * theta * self.k_phi

        # Clamp to [0, 1]
        P = np.clip(P, 0.0, 1.0)

        # random threshold e^t  — Eq. (1)
        e = random_threshold(self.cfg.alpha, self.cfg.beta, self.rng, g.shape)

        ignite = candidates & has_fire_neighbor & (P > e)
        next_g[ignite] = BURNING

        self.grid = next_g
        return self.grid