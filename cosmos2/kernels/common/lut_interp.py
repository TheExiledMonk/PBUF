"""LUT interpolation helpers for PBUF (Numba-safe)."""

import numba
import numpy as np


@numba.njit
def _find_interval(x: float, grid: np.ndarray) -> int:
    """Return index i such that grid[i] <= x < grid[i+1] (clamped)."""
    n = grid.shape[0]
    if x <= grid[0]:
        return 0
    for i in range(n - 1):
        if x < grid[i + 1]:
            return i
    return n - 2


@numba.njit
def lut_interp_scalar(x: float, lut_x: np.ndarray, lut_y: np.ndarray) -> float:
    """Linear interpolation for monotonic LUTs."""
    n = lut_x.shape[0]
    if n == 0:
        return 0.0
    if n == 1 or x <= lut_x[0]:
        return lut_y[0]
    if x >= lut_x[n - 1]:
        return lut_y[n - 1]

    idx = _find_interval(x, lut_x)
    x0 = lut_x[idx]
    x1 = lut_x[idx + 1]
    y0 = lut_y[idx]
    y1 = lut_y[idx + 1]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


@numba.njit
def lut_eval_all(
    T: float,
    lut_T: np.ndarray,
    lut_eps0: np.ndarray,
    lut_alpha: np.ndarray,
    lut_gstar: np.ndarray,
    lut_gstarS: np.ndarray,
    lut_meta: np.ndarray,
):
    """
    Evaluate all thermal LUT fields at temperature T.

    Returns tuple (eps0, alpha, gstar, gstarS, meta_interp).
    """
    eps0 = lut_interp_scalar(T, lut_T, lut_eps0)
    alpha = lut_interp_scalar(T, lut_T, lut_alpha)
    gstar = lut_interp_scalar(T, lut_T, lut_gstar)
    gstarS = lut_interp_scalar(T, lut_T, lut_gstarS)
    meta_val = lut_interp_scalar(T, lut_T, lut_meta)
    return eps0, alpha, gstar, gstarS, meta_val
