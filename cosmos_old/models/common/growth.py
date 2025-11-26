"""Shared growth integrator that caches D(a) solutions for probing fσ₈."""

from __future__ import annotations

from typing import Callable

import numpy as np

GrowthRHS = Callable[[float, np.ndarray], np.ndarray]


def _ensure_positive(value: float) -> float:
    if value <= 0.0:
        raise ValueError("Scale factor must be strictly positive.")
    return float(value)


class GrowthTable:
    """Cache of D(a) and D′(a) built by integrating a general growth RHS."""

    def __init__(self, rhs: GrowthRHS, *, a_min: float = 1e-5, steps: int = 4096):
        self._rhs = rhs
        self._a_min = float(max(a_min, 1e-8))
        self._steps = max(steps, 64)
        self._a_grid, self._d_grid, self._d_prime_grid = self._build_table()
        self._log_grid = np.log(self._a_grid)

    def _build_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_start = np.log(_ensure_positive(self._a_min))
        log_end = 0.0
        log_grid = np.linspace(log_start, log_end, self._steps, dtype=float)
        a_grid = np.exp(log_grid)

        d_values = np.empty_like(a_grid)
        d_prime_values = np.empty_like(a_grid)

        y = np.asarray([a_grid[0], 1.0], dtype=float)
        d_values[0] = y[0]
        d_prime_values[0] = y[1]

        for idx in range(self._steps - 1):
            a = a_grid[idx]
            h = a_grid[idx + 1] - a
            y = self._rk4_step(a, y, h)
            d_values[idx + 1] = y[0]
            d_prime_values[idx + 1] = y[1]

        scale = 1.0 / d_values[-1]
        return a_grid, d_values * scale, d_prime_values * scale

    def _rk4_step(self, a: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = self._rhs(a, y)
        k2 = self._rhs(a + 0.5 * h, y + 0.5 * h * k1)
        k3 = self._rhs(a + 0.5 * h, y + 0.5 * h * k2)
        k4 = self._rhs(a + h, y + h * k3)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _interpolate(self, a: float, table: np.ndarray) -> float:
        a_clamped = min(max(a, self._a_grid[0]), self._a_grid[-1])
        if a_clamped == self._a_grid[-1]:
            return float(table[-1])
        return float(np.interp(np.log(a_clamped), self._log_grid, table))

    def growth_factor(self, a: float) -> float:
        return self._interpolate(a, self._d_grid)

    def growth_rate(self, a: float) -> float:
        d = self.growth_factor(a)
        d_prime = self._interpolate(a, self._d_prime_grid)
        if d == 0.0:
            return 0.0
        return float(a * d_prime / d)

    def sigma8(self, a: float, sigma8_today: float) -> float:
        return sigma8_today * self.growth_factor(a)

    def fs8(self, a: float, sigma8_today: float) -> float:
        return self.growth_rate(a) * self.sigma8(a, sigma8_today)
