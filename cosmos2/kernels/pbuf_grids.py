"""Numba kernel to build PBUF H/DM grids."""

from __future__ import annotations

import numba
import numpy as np

from .pbuf_distances import C_LIGHT, H_kernel


@numba.njit(cache=True)
def omega_sigma_interp_njit(
    a_val: float,
    a_min: float,
    a_max: float,
    a_grid: np.ndarray,
    alpha_grid: np.ndarray,
    eps_grid: np.ndarray,
    Rmax: float,
    sigma_rescale: float,
    mode_flag: int,
) -> float:
    """Interpolate ωσ(a) using alpha/epsilon grids; mode_flag: 0=free, 1=flat_today."""

    a_clamped = a_val
    if a_clamped < a_min:
        a_clamped = a_min
    elif a_clamped > a_max:
        a_clamped = a_max

    alpha_val = np.interp(a_clamped, a_grid, alpha_grid)
    eps_val = np.interp(a_clamped, a_grid, eps_grid)
    kmax_val = eps_val - alpha_val
    decay = np.exp(-a_clamped / Rmax)
    S = 1.0 - (1.0 - kmax_val) * decay
    omega_sigma = alpha_val * (1.0 - decay) * S
    if mode_flag == 1:
        omega_sigma = sigma_rescale * omega_sigma
    return float(omega_sigma)


@numba.njit(cache=True)
def build_grids_njit(
    a_grid: np.ndarray,
    H0: float,
    Omega_m0: float,
    Omega_r0: float,
    alpha_curv: float,
    Rmax: float,
    sigma_rescale: float,
    mode_flag: int,
    a_table: np.ndarray,
    alpha_table: np.ndarray,
    eps_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute H(a) and D_M(a) grids nopython."""

    n = a_grid.size
    H_vals = np.empty(n, dtype=np.float64)
    integrand = np.empty(n, dtype=np.float64)

    a_min = a_table[0]
    a_max = a_table[-1]

    for i in range(n):
        a_val = a_grid[i]
        omega_sigma = omega_sigma_interp_njit(
            a_val,
            a_min,
            a_max,
            a_table,
            alpha_table,
            eps_table,
            Rmax,
            sigma_rescale,
            mode_flag,
        )
        H_val = H_kernel(a_val, H0, Omega_m0, Omega_r0, alpha_curv, omega_sigma)
        H_vals[i] = H_val
        integrand[i] = C_LIGHT / (a_val * a_val * max(H_val, 1e-12))

    cumulative = np.empty(n, dtype=np.float64)
    cumulative[0] = 0.0
    for i in range(n - 1):
        dx = a_grid[i + 1] - a_grid[i]
        cumulative[i + 1] = cumulative[i] + dx * 0.5 * (integrand[i] + integrand[i + 1])

    total = cumulative[-1]
    DM_grid = total - cumulative
    return H_vals, DM_grid


__all__ = ["build_grids_njit"]
