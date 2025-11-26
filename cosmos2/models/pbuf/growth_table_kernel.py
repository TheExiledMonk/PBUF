"""Python fallback growth table builder (used when njit rhs unavailable)."""

from __future__ import annotations

from typing import Callable

import numpy as np

GrowthRHS = Callable[[float, np.ndarray], np.ndarray]


def build_growth_table_kernel(rhs: GrowthRHS, a_min: float, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate growth RHS over log-spaced a-grid and return normalized D and D'."""

    log_start = np.log(a_min)
    log_end = 0.0
    log_grid = np.linspace(log_start, log_end, steps, dtype=float)
    a_grid = np.exp(log_grid)

    d_values = np.empty_like(a_grid)
    d_prime_values = np.empty_like(a_grid)

    y = np.asarray([a_grid[0], 1.0], dtype=float)
    d_values[0] = y[0]
    d_prime_values[0] = y[1]

    for idx in range(steps - 1):
        a = a_grid[idx]
        h = a_grid[idx + 1] - a
        k1 = rhs(a, y)
        k2 = rhs(a + 0.5 * h, y + 0.5 * h * k1)
        k3 = rhs(a + 0.5 * h, y + 0.5 * h * k2)
        k4 = rhs(a + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        d_values[idx + 1] = y[0]
        d_prime_values[idx + 1] = y[1]

    scale = 1.0 / d_values[-1]
    return a_grid, d_values * scale, d_prime_values * scale


__all__ = ["build_growth_table_kernel"]
