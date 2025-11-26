"""Numba kernels for Phase-6a curvature diagnostics."""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True)
def curvature_stats_kernel(a_grid: np.ndarray, H_values: np.ndarray, curv_eps: float):
    """
    Compute first/second derivatives and |H''/H'| ratios on a log-spaced grid.
    """

    n = a_grid.size
    Hp = np.zeros(n - 2, dtype=np.float64)
    Hpp = np.zeros(n - 2, dtype=np.float64)
    ratio = np.zeros(n - 2, dtype=np.float64)
    valid_mask = np.zeros(n - 2, dtype=np.bool_)

    for i in range(n - 2):
        a_prev = a_grid[i]
        a_curr = a_grid[i + 1]
        a_next = a_grid[i + 2]
        H_prev = H_values[i]
        H_curr = H_values[i + 1]
        H_next = H_values[i + 2]

        Hp_prev = (H_curr - H_prev) / (a_curr - a_prev)
        Hp_next = (H_next - H_curr) / (a_next - a_curr)
        Hp_val = (H_next - H_prev) / (a_next - a_prev)
        Hp[i] = Hp_val
        Hpp[i] = (Hp_next - Hp_prev) / (a_next - a_prev)

        if abs(Hp_val) > curv_eps and np.isfinite(Hpp[i]) and np.isfinite(Hp_val):
            valid_mask[i] = True
            ratio[i] = Hpp[i] / Hp_val
        else:
            ratio[i] = 0.0

    finite_mask = np.isfinite(ratio) & valid_mask
    if finite_mask.any():
        max_abs_ratio = float(np.nanmax(np.abs(ratio[finite_mask])))
        min_abs_ratio = float(np.nanmin(np.abs(ratio[finite_mask])))
    else:
        max_abs_ratio = 0.0
        min_abs_ratio = 0.0

    return Hp, Hpp, ratio, valid_mask, max_abs_ratio, min_abs_ratio


__all__ = ["curvature_stats_kernel"]
