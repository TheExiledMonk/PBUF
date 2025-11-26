"""Fixed-step integrators for cosmos2 kernels (Numba-safe)."""

import numba
import numpy as np


@numba.njit
def trapz(x: np.ndarray, y: np.ndarray) -> float:
    """Uniform-step trapezoidal integral."""
    n = y.shape[0]
    if n < 2:
        return 0.0

    total = 0.0
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        total += (y[i - 1] + y[i]) * dx
    return 0.5 * total


@numba.njit
def cumulative_trapz(x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    """
    Cumulative trapezoid integral stored in ``out`` (same length as ``x``).
    Assumes uniform spacing; caller preallocates ``out``.
    """
    n = y.shape[0]
    out[0] = 0.0
    if n < 2:
        return

    for i in range(1, n):
        dx = x[i] - x[i - 1]
        out[i] = out[i - 1] + 0.5 * (y[i - 1] + y[i]) * dx
