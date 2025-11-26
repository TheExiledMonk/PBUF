"""Numba helpers for thermal table interpolation."""

from __future__ import annotations

import math

import numba
import numpy as np

# Field ids for numba interpolation (consistent with ThermalTable.numba_field_id)
FIELD_T = 0
FIELD_EPS = 1
FIELD_ALPHA = 2
FIELD_DLN_EPS = 3
FIELD_DLN_ALPHA = 4
FIELD_GSTAR = 5
FIELD_GSTARS = 6


@numba.njit(cache=True)
def _interp_linear(a_val: float, a_grid: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(a_val, a_grid, values))


@numba.njit(cache=True)
def _interp_log(a_val: float, log_a: np.ndarray, values: np.ndarray) -> float:
    safe = np.clip(values, 1e-50, None)
    log_vals = np.log(safe)
    return float(math.exp(np.interp(math.log(a_val), log_a, log_vals)))


@numba.njit(cache=True)
def interp_field_njit(
    a_val: float,
    a_grid: np.ndarray,
    log_a: np.ndarray,
    field_id: int,
    T: np.ndarray,
    eps: np.ndarray,
    alpha: np.ndarray,
    dln_eps: np.ndarray,
    dln_alpha: np.ndarray,
    g_star: np.ndarray,
    g_starS: np.ndarray,
) -> float:
    """Interpolate a known thermal field at scale factor a_val using field ids."""

    a_clamped = a_val
    if a_clamped < a_grid[0]:
        a_clamped = a_grid[0]
    elif a_clamped > a_grid[-1]:
        a_clamped = a_grid[-1]

    if field_id == FIELD_T:
        return _interp_log(a_clamped, log_a, T)
    if field_id == FIELD_EPS:
        return _interp_linear(a_clamped, a_grid, eps)
    if field_id == FIELD_ALPHA:
        return _interp_linear(a_clamped, a_grid, alpha)
    if field_id == FIELD_DLN_EPS:
        return _interp_linear(a_clamped, a_grid, dln_eps)
    if field_id == FIELD_DLN_ALPHA:
        return _interp_linear(a_clamped, a_grid, dln_alpha)
    if field_id == FIELD_GSTAR:
        return _interp_linear(a_clamped, a_grid, g_star)
    if field_id == FIELD_GSTARS:
        return _interp_linear(a_clamped, a_grid, g_starS)
    return math.nan


__all__ = [
    "FIELD_T",
    "FIELD_EPS",
    "FIELD_ALPHA",
    "FIELD_DLN_EPS",
    "FIELD_DLN_ALPHA",
    "FIELD_GSTAR",
    "FIELD_GSTARS",
    "interp_field_njit",
]
