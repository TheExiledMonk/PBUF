"""Numba-backed helpers for building/interpolating PBUF growth tables."""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=True)
def rk4_step_njit(rhs, a: float, y: np.ndarray, h: float) -> np.ndarray:
    """Single RK4 step for y' = rhs(a, y)."""

    k1 = rhs(a, y)
    k2 = rhs(a + 0.5 * h, y + 0.5 * h * k1)
    k3 = rhs(a + 0.5 * h, y + 0.5 * h * k2)
    k4 = rhs(a + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@numba.njit(cache=True)
def build_growth_table_njit(rhs, a_min: float, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate growth RHS over log-spaced a-grid and return normalized D and D' (nopython)."""

    log_start = math.log(a_min)
    log_end = 0.0
    log_grid = np.linspace(log_start, log_end, steps, dtype=np.float64)
    a_grid = np.exp(log_grid)

    d_values = np.empty_like(a_grid)
    d_prime_values = np.empty_like(a_grid)

    y = np.asarray([a_grid[0], 1.0], dtype=np.float64)
    d_values[0] = y[0]
    d_prime_values[0] = y[1]

    for idx in range(steps - 1):
        a = a_grid[idx]
        h = a_grid[idx + 1] - a
        y = rk4_step_njit(rhs, a, y, h)
        d_values[idx + 1] = y[0]
        d_prime_values[idx + 1] = y[1]

    scale = 1.0 / d_values[-1]
    return a_grid, d_values * scale, d_prime_values * scale


__all__ = ["rk4_step_njit", "build_growth_table_njit"]
