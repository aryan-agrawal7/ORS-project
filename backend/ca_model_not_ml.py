import numpy as np

# -----------------------------
# STATES
# -----------------------------
UNBURNT = 0
EARLY = 1
FULL = 2
EXTINGUISHING = 3
EXTINGUISHED = 4

# Frontend/common states used by ca_model.py and caRenderer.js.
FRONTEND_UNBURNABLE = 0
FRONTEND_UNIGNITED = 1
FRONTEND_BURNING = 2
FRONTEND_BURNED = 3


# -----------------------------
# MODEL (Eq. 1–5)
# -----------------------------
def wind_level(v):
    return np.floor((v / 0.836) ** (2/3))


def compute_R(T, v, RH, phi, slope, g, Ks, Kr):
    W = wind_level(v)
    R0 = 0.03*T + 0.05*W + 0.01*(100 - RH) - 0.3

    Kphi = np.exp(0.1783 * v * np.cos(phi))
    Ktheta = np.exp(3.553 * g * np.tan(1.2 * slope))

    return np.maximum(R0, 0) * Kphi * Ktheta * Ks * Kr


# -----------------------------
# GRID SHIFT
# -----------------------------
def shift(grid, di, dj):
    H, W = grid.shape
    out = np.zeros_like(grid)

    for i in range(H):
        for j in range(W):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                out[i, j] = grid[ni, nj]

    return out


# -----------------------------
# Eq. (6): CONTINUOUS ACCUMULATION
# FIXED: true accumulation into S_cont
# -----------------------------
def compute_increment(grid, R, dt_grid, L):
    inc = np.zeros_like(R)

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue

            nb_state = shift(grid, di, dj)
            nb_R = shift(R, di, dj)

            # Only FULL cells ignite neighbors (paper semantics)
            contrib = np.where(nb_state == FULL, nb_R, 0.0)
            inc += contrib

    inc = inc * (dt_grid * 60) / L

    # UNBURNT and EARLY cells both accumulate. If EARLY does not accumulate,
    # it can never reach FULL and the simulation appears static.
    inc = np.where((grid == UNBURNT) | (grid == EARLY), inc, 0.0)

    return inc


# -----------------------------
# Eq. (7)
# -----------------------------
def compute_a(R_prev, R_next):
    a = R_next - R_prev
    a[np.isclose(R_next, R_prev)] = 0
    return a


# -----------------------------
# Eq. (8)
# -----------------------------
def compute_dt(a, dt0):
    return dt0 / np.exp(a)


# -----------------------------
# CONTINUOUS → DISCRETE STATE
# -----------------------------
def discretize_state(S_cont, grid):
    new = grid.copy()
    H, W = grid.shape

    for i in range(H):
        for j in range(W):

            if grid[i, j] == UNBURNT:
                if S_cont[i, j] > 0:
                    new[i, j] = EARLY

            elif grid[i, j] == EARLY:
                if S_cont[i, j] >= 1:
                    new[i, j] = FULL

    return new


# -----------------------------
# CORRECT CA TRANSITIONS
# -----------------------------
def ca_transition(grid):
    new = grid.copy()
    H, W = grid.shape

    for i in range(H):
        for j in range(W):

            if grid[i, j] == FULL:
                all_neighbors_full = True

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj

                        if 0 <= ni < H and 0 <= nj < W:
                            if grid[ni, nj] < FULL:
                                all_neighbors_full = False

                if all_neighbors_full:
                    new[i, j] = EXTINGUISHING

            elif grid[i, j] == EXTINGUISHING:
                new[i, j] = EXTINGUISHED

    return new


# -----------------------------
# ENVIRONMENT (time varying)
# -----------------------------
def environment_update(T, v, RH, phi, t):
    v_t = v * (1 + 0.2 * np.sin(0.1 * t))
    T_t = T + 2 * np.sin(0.05 * t)
    RH_t = RH + 5 * np.cos(0.07 * t)
    phi_t = phi + 0.1 * np.sin(0.03 * t)

    return T_t, v_t, RH_t, phi_t


# -----------------------------
# FULL SIMULATION (FIXED)
# -----------------------------
def simulate(
    grid,
    T, v, RH, phi, slope, g, Ks,
    Kr=1/25,
    L=30,
    dt0=1.0,
    T_end=10
):
    t = 0

    # continuous state field (critical fix)
    S_cont = np.zeros_like(grid, dtype=float)

    R = compute_R(T, v, RH, phi, slope, g, Ks, Kr)

    while t < T_end:

        T_t, v_t, RH_t, phi_t = environment_update(T, v, RH, phi, t)
        R_next = compute_R(T_t, v_t, RH_t, phi_t, slope, g, Ks, Kr)

        a = compute_a(R, R_next)
        dt_grid = compute_dt(a, dt0)

        # Eq. (6) accumulation
        inc = compute_increment(grid, R, dt_grid, L)
        S_cont += inc   # CRITICAL FIX

        # discretize based on accumulated state
        grid = discretize_state(S_cont, grid)

        # apply CA transitions
        grid = ca_transition(grid)

        t += np.mean(dt_grid)
        R = R_next

    return grid


class ContinuousFireCA:
    """
    Step-wise wrapper around the non-ML CA model.

    The original simulate() function above returns only the final grid. The
    FastAPI visualizer needs one frame at a time, so this class preserves the
    same equations while exposing a step() method like ForestFireCA.
    """

    def __init__(
        self,
        grid,
        T,
        v,
        RH,
        phi,
        slope,
        g,
        Ks,
        burnable_mask=None,
        Kr=1/25,
        L=30,
        dt0=1.0,
    ):
        self.grid = grid.astype(np.uint8)
        self.shape = self.grid.shape
        self.T = self._as_grid(T)
        self.v = self._as_grid(v)
        self.RH = self._as_grid(RH)
        self.phi = self._as_grid(phi)
        self.slope = self._as_grid(slope)
        self.g = self._as_grid(g)
        self.Ks = self._as_grid(Ks)
        self.Kr = Kr
        self.L = L
        self.dt0 = dt0
        self.t = 0.0

        if burnable_mask is None:
            self.burnable_mask = np.ones(self.shape, dtype=bool)
        else:
            self.burnable_mask = burnable_mask.astype(bool)

        self.grid[~self.burnable_mask] = UNBURNT
        self.S_cont = np.zeros(self.shape, dtype=float)
        self.full_steps = np.zeros(self.shape, dtype=np.int32)
        self.R = compute_R(
            self.T,
            self.v,
            self.RH,
            self.phi,
            self.slope,
            self.g,
            self.Ks,
            self.Kr,
        )

    def _as_grid(self, value):
        arr = np.asarray(value, dtype=float)
        if arr.shape == self.shape:
            return arr
        if arr.shape == ():
            return np.full(self.shape, float(arr), dtype=float)
        return np.broadcast_to(arr, self.shape).astype(float)

    def step(self):
        T_t, v_t, RH_t, phi_t = environment_update(
            self.T,
            self.v,
            self.RH,
            self.phi,
            self.t,
        )
        R_next = compute_R(
            T_t,
            v_t,
            RH_t,
            phi_t,
            self.slope,
            self.g,
            self.Ks,
            self.Kr,
        )

        a = compute_a(self.R, R_next)
        dt_grid = compute_dt(a, self.dt0)

        inc = compute_increment(self.grid, self.R, dt_grid, self.L)
        inc = np.where(self.burnable_mask, inc, 0.0)
        self.S_cont += inc
        self.S_cont[~self.burnable_mask] = 0.0

        self.grid = discretize_state(self.S_cont, self.grid)
        self.grid = ca_transition(self.grid)

        self.grid[~self.burnable_mask] = UNBURNT

        self.t += float(np.mean(dt_grid))
        self.R = R_next
        return self.grid

    def to_frontend_grid(self):
        """
        Convert 5-state non-ML CA output into the 4-state renderer palette.
        EARLY, FULL, and EXTINGUISHING are shown as active fire.
        """
        out = np.full(self.shape, FRONTEND_UNIGNITED, dtype=np.uint8)
        out[~self.burnable_mask] = FRONTEND_UNBURNABLE
        out[self.grid == EXTINGUISHED] = FRONTEND_BURNED
        active = (
            (self.grid == EARLY)
            | (self.grid == FULL)
            | (self.grid == EXTINGUISHING)
        )
        out[active] = FRONTEND_BURNING
        return out
