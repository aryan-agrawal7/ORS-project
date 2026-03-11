"""
Least Squares Support Vector Machine (LSSVM) classifier for fire ignition
probability, following the paper:

  Xu et al. "Modeling Forest Fire Spread Using Machine Learning-Based
  Cellular Automata in a GIS Environment", Forests 2022, 13, 1974.

Key equations implemented:
  - RBF (Gaussian) kernel:  K(x_k, x_i) = exp(-||x_k - x_i||^2 / (2*sigma^2))   [Eq. 14]
  - LSSVM dual system:      [0, -y^T; y, Omega + gamma^{-1}*I] [b; alpha] = [0; 1]  [Eq. 10]
  - Fire probability:       Pc = 1 / (1 + exp(-(sum_i alpha_i * y_i * K(x, x_i) + b)))  [Eq. 13]
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from pathlib import Path


class LSSVM:
    """
    Least Squares Support Vector Machine for binary classification.

    The classifier learns Lagrangian multipliers alpha and bias b
    from training data, then outputs a continuous probability via
    a sigmoid applied to the decision function.

    Parameters
    ----------
    gamma : float
        Regularisation parameter (larger = less regularisation).
    sigma : float
        Width of the Gaussian RBF kernel.
    """

    def __init__(self, gamma: float = 100.0, sigma: float = 1.0):
        self.gamma = gamma
        self.sigma = sigma
        # Learned parameters
        self.alpha: Optional[np.ndarray] = None
        self.b: Optional[float] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Gaussian RBF kernel matrix between rows of X1 and X2.

        K(x_k, x_i) = exp(-||x_k - x_i||^2 / (2 * sigma^2))

        Returns shape (n1, n2).
        """
        # ||x_k - x_i||^2  = ||x_k||^2 + ||x_i||^2 - 2 * x_k . x_i
        sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)  # (n1, 1)
        sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)  # (n2, 1)
        dist2 = sq1 + sq2.T - 2.0 * X1 @ X2.T        # (n1, n2)
        return np.exp(-dist2 / (2.0 * self.sigma ** 2))

    # ------------------------------------------------------------------
    # Training — solving the dual linear system (Eq. 10)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSSVM":
        """
        Train LSSVM on data.

        Parameters
        ----------
        X : ndarray of shape (N, n_features)
            Feature matrix (e.g. slope, aspect, elevation, NDVI, humidity).
        y : ndarray of shape (N,)
            Labels in {-1, +1}.  Fire point = +1, non-fire = -1.
            (The paper uses {0,1} externally; we convert internally.)
        """
        N = X.shape[0]
        self.X_train = X.copy()
        self.y_train = y.copy().astype(np.float64)

        K = self._rbf_kernel(X, X)  # (N, N)

        # Omega_kj = y_k * y_j * K(x_k, x_j)   — Eq. 9
        yy = self.y_train[:, None] * self.y_train[None, :]  # (N, N)
        Omega = yy * K

        # Build the bordered system   — Eq. 10
        # [ 0      -y^T   ] [ b     ]   [ 0 ]
        # [ y   Omega+I/g ] [ alpha ] = [ 1 ]
        A = np.zeros((N + 1, N + 1), dtype=np.float64)
        A[0, 1:] = -self.y_train
        A[1:, 0] = self.y_train
        A[1:, 1:] = Omega + np.eye(N) / self.gamma

        rhs = np.zeros(N + 1, dtype=np.float64)
        rhs[1:] = 1.0

        # Solve  A @ [b, alpha]^T = rhs
        solution = np.linalg.solve(A, rhs)

        self.b = float(solution[0])
        self.alpha = solution[1:]

        return self

    # ------------------------------------------------------------------
    # Decision function & probability
    # ------------------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        f(x) = sum_i alpha_i * y_i * K(x, x_i) + b

        Returns shape (M,) for M query points.
        """
        K = self._rbf_kernel(X, self.X_train)          # (M, N)
        ay = self.alpha * self.y_train                  # (N,)
        return K @ ay + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Fire ignition probability  — Eq. 13:
          Pc = 1 / (1 + exp(-f(x)))

        Returns values in (0, 1) for each query point.
        """
        f = self.decision_function(X)
        # Clip to avoid overflow in exp
        f = np.clip(f, -500, 500)
        return 1.0 / (1.0 + np.exp(-f))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard classification: +1 if Pc >= 0.5, else -1."""
        return np.where(self.predict_proba(X) >= 0.5, 1, -1)

    # ------------------------------------------------------------------
    # Convenience: compute full probability raster
    # ------------------------------------------------------------------
    def compute_probability_surface(
        self,
        feature_rasters: np.ndarray,
    ) -> np.ndarray:
        """
        Given a stack of feature rasters of shape (n_features, H, W),
        compute the fire ignition probability Pc for every cell.

        Returns shape (H, W) with values in (0, 1).
        """
        n_features, H, W = feature_rasters.shape
        # Reshape to (H*W, n_features)
        X_flat = feature_rasters.reshape(n_features, -1).T
        Pc_flat = self.predict_proba(X_flat)
        return Pc_flat.reshape(H, W).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence — save / load trained model
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """
        Save the trained model to a .npz file.
        Stores all state needed to reconstruct the model and run predictions.
        """
        if self.alpha is None:
            raise RuntimeError("Cannot save an untrained model — call fit() first.")
        np.savez_compressed(
            str(path),
            gamma=np.array(self.gamma),
            sigma=np.array(self.sigma),
            b=np.array(self.b),
            alpha=self.alpha,
            X_train=self.X_train,
            y_train=self.y_train,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LSSVM":
        """
        Load a previously saved model from a .npz file.
        Returns a fully‐initialised LSSVM ready for prediction.
        """
        path = Path(path)
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        data = np.load(str(path))
        model = cls(
            gamma=float(data["gamma"]),
            sigma=float(data["sigma"]),
        )
        model.b       = float(data["b"])
        model.alpha   = data["alpha"]
        model.X_train = data["X_train"]
        model.y_train = data["y_train"]
        return model
