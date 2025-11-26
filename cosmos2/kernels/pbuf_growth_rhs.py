"""Numba kernels for growth RHS with inline thermal interpolation."""

from __future__ import annotations

import math

import numba
import numpy as np

from .pbuf_distances import E_kernel
from .pbuf_elastic import kmax_from_eps_alpha, omega_sigma_raw_kernel


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
    decay = math.exp(-a_clamped / Rmax)
    S = 1.0 - (1.0 - kmax_val) * decay
    omega_sigma = alpha_val * (1.0 - decay) * S
    if mode_flag == 1:
        omega_sigma = sigma_rescale * omega_sigma
    return float(omega_sigma)


@numba.njit(cache=True)
def _E_from_table_njit(
    a_val: float,
    Omega_m0: float,
    Omega_r0: float,
    alpha_curv: float,
    Rmax: float,
    sigma_rescale: float,
    mode_flag: int,
    a_min: float,
    a_max: float,
    a_grid: np.ndarray,
    alpha_grid: np.ndarray,
    eps_grid: np.ndarray,
) -> float:
    """
    Match the Python E(a) evaluation used in growth_ode_rhs, including thermal lookups.
    """

    a_safe = a_val if a_val > 1.0e-12 else 1.0e-12

    # Inline omega_sigma_of_a with the same clamp + normalization as the Python path.
    a_clamped = a_safe
    if a_clamped < a_min:
        a_clamped = a_min
    elif a_clamped > a_max:
        a_clamped = a_max
    eps_val = np.interp(a_clamped, a_grid, eps_grid)
    alpha_val = np.interp(a_clamped, a_grid, alpha_grid)
    kmax_val = kmax_from_eps_alpha(eps_val, alpha_val)
    omega_sigma = omega_sigma_raw_kernel(a_clamped, Rmax, alpha_val, kmax_val)
    if mode_flag == 1:
        omega_sigma = sigma_rescale * omega_sigma

    return E_kernel(a_safe, Omega_m0, Omega_r0, alpha_curv, omega_sigma)


@numba.njit(cache=True)
def growth_rhs_njit(
    a_val: float,
    y: np.ndarray,
    Omega_m0: float,
    Omega_r0: float,
    alpha_curv: float,
    Rmax: float,
    sigma_rescale: float,
    mode_flag: int,
    a_min: float,
    a_max: float,
    a_grid: np.ndarray,
    alpha_grid: np.ndarray,
    eps_grid: np.ndarray,
) -> np.ndarray:
    """Nopython growth RHS that mirrors the Python growth_ode_rhs bit-for-bit."""

    D = y[0]
    D_prime = y[1]

    a_safe = a_val if a_val > 1.0e-12 else 1.0e-12
    eps_fd = 1.0e-5

    E_a = _E_from_table_njit(
        a_safe, Omega_m0, Omega_r0, alpha_curv, Rmax, sigma_rescale, mode_flag, a_min, a_max, a_grid, alpha_grid, eps_grid
    )
    E_a_plus = _E_from_table_njit(
        a_safe + eps_fd,
        Omega_m0,
        Omega_r0,
        alpha_curv,
        Rmax,
        sigma_rescale,
        mode_flag,
        a_min,
        a_max,
        a_grid,
        alpha_grid,
        eps_grid,
    )
    E_a_minus = _E_from_table_njit(
        a_safe - eps_fd,
        Omega_m0,
        Omega_r0,
        alpha_curv,
        Rmax,
        sigma_rescale,
        mode_flag,
        a_min,
        a_max,
        a_grid,
        alpha_grid,
        eps_grid,
    )
    dE_da = (E_a_plus - E_a_minus) / (2.0 * eps_fd)

    term1 = -(3.0 / a_safe + dE_da / E_a) * D_prime
    term2 = 1.5 * Omega_m0 / (a_safe ** 5 * E_a ** 2) * D

    return np.array([D_prime, term1 + term2], dtype=np.float64)


__all__ = ["omega_sigma_interp_njit", "growth_rhs_njit"]
