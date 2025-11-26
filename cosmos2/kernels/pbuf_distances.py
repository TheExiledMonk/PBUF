"""Numba kernels for PBUF background expansion."""

from __future__ import annotations

import math

import numba
import numpy as np

C_LIGHT = 299_792.458  # km/s


@numba.njit(cache=True)
def omega_total_kernel(a_val: float, Omega_m0: float, Omega_r0: float, alpha: float, omega_sigma: float) -> float:
    """Total density parameter at scale factor a."""

    a_safe = a_val if a_val > 1.0e-12 else 1.0e-12
    Om = Omega_m0 / (a_safe * a_safe * a_safe)
    Or = Omega_r0 / (a_safe * a_safe * a_safe * a_safe)
    Ok = alpha / (a_safe * a_safe)
    return Om + Or + Ok + omega_sigma


@numba.njit(cache=True)
def E_squared_kernel(a_val: float, Omega_m0: float, Omega_r0: float, alpha: float, omega_sigma: float) -> float:
    """Dimensionless expansion rate squared."""

    return omega_total_kernel(a_val, Omega_m0, Omega_r0, alpha, omega_sigma)


@numba.njit(cache=True)
def E_kernel(a_val: float, Omega_m0: float, Omega_r0: float, alpha: float, omega_sigma: float) -> float:
    """Dimensionless expansion rate."""

    return math.sqrt(max(E_squared_kernel(a_val, Omega_m0, Omega_r0, alpha, omega_sigma), 0.0))


@numba.njit(cache=True)
def H_kernel(a_val: float, H0: float, Omega_m0: float, Omega_r0: float, alpha: float, omega_sigma: float) -> float:
    """Hubble rate in km/s/Mpc."""

    return H0 * E_kernel(a_val, Omega_m0, Omega_r0, alpha, omega_sigma)


@numba.njit(cache=True)
def comoving_integrand_kernel(a_val: float, H0: float, Omega_m0: float, Omega_r0: float, alpha: float, omega_sigma: float) -> float:
    """c/H(a) integrand used for comoving distance."""

    H_val = H_kernel(a_val, H0, Omega_m0, Omega_r0, alpha, omega_sigma)
    return C_LIGHT / H_val if H_val > 0.0 else math.inf


@numba.njit(cache=True)
def comoving_distance_njit(
    z_target: float,
    H0: float,
    Omega_m0: float,
    Omega_r0: float,
    alpha: float,
    Rmax: float,
    sigma_rescale: float,
    mode_flag: int,
    a_grid: np.ndarray,
    alpha_grid: np.ndarray,
    eps_grid: np.ndarray,
    steps: int = 4096,
) -> float:
    """
    Nopython comoving distance ∫ c/H dz using inline omega_sigma interpolation.

    mode_flag: 0=free, 1=flat_today (uses sigma_rescale).
    """

    if z_target <= 0.0:
        return 0.0
    n = max(steps, 2 * (steps // 2))
    h = z_target / n
    result = 0.0

    a_min = a_grid[0]
    a_max = a_grid[-1]

    for i in range(n + 1):
        z_val = i * h
        a_val = 1.0 / (1.0 + z_val)
        # Inline omega_sigma interpolation to avoid circular imports.
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
        hubble = H_kernel(a_val, H0, Omega_m0, Omega_r0, alpha, omega_sigma)
        integrand = C_LIGHT / hubble if hubble > 0.0 else math.inf
        weight = 4.0 if i % 2 == 1 else 2.0
        if i == 0 or i == n:
            weight = 1.0
        result += weight * integrand

    return result * h / 3.0


__all__ = [
    "C_LIGHT",
    "omega_total_kernel",
    "E_squared_kernel",
    "E_kernel",
    "H_kernel",
    "comoving_integrand_kernel",
    "comoving_distance_njit",
]
