"""GrowthTable helper copied from the legacy cosmos PBUF implementation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from cosmos2.kernels.pbuf_growth_table import build_growth_table_njit
from .growth_table_kernel import build_growth_table_kernel

GrowthRHS = Callable[[float, np.ndarray], np.ndarray]
_GROWTH_TABLE_ENV = __import__("os").environ.get("PBUF_GROWTH_TABLE", "").strip().lower()
_FORCE_PYTHON_TABLE = _GROWTH_TABLE_ENV in {"python", "py", "force_python"}


def _ensure_positive(value: float) -> float:
    if value <= 0.0:
        raise ValueError("Scale factor must be strictly positive.")
    return float(value)


class GrowthTable:
    """Cache of D(a) and D′(a) built by integrating a general growth RHS."""

    def __init__(
        self,
        rhs: GrowthRHS,
        *,
        rhs_njit: GrowthRHS | None = None,
        a_min: float = 1e-5,
        steps: int = 4096,
    ):
        self._rhs = rhs
        self._rhs_njit = rhs_njit
        self._a_min = float(max(a_min, 1e-8))
        self._steps = max(steps, 64)
        self._a_grid, self._d_grid, self._d_prime_grid = self._build_table()
        self._log_grid = np.log(self._a_grid)

    def _build_table(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a_min = _ensure_positive(self._a_min)
        if self._rhs_njit is not None and not _FORCE_PYTHON_TABLE:
            try:
                return build_growth_table_njit(self._rhs_njit, a_min, self._steps)
            except Exception:
                pass
        return build_growth_table_kernel(self._rhs, a_min, self._steps)

    def _rk4_step(self, a: float, y: np.ndarray, h: float) -> np.ndarray:
        return rk4_step_kernel(self._rhs, a, y, h)

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


__all__ = ["GrowthTable"]
