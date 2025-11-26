"""Numba kernels for PBUF CMB/drag-era helpers."""

from __future__ import annotations

import math
import os

import numba
import numpy as np

from .pbuf_distances import C_LIGHT, H_kernel
from .pbuf_elastic import kmax_from_eps_alpha, omega_sigma_raw_kernel

PHOTON_G_DEGREES = 2.0


@numba.njit(cache=True)
def photon_density_parameter_kernel(g_today: float, Omega_r0: float) -> float:
    """Photon density parameter today from g_star(a=1)."""

    return Omega_r0 * PHOTON_G_DEGREES / g_today


@numba.njit(cache=True)
def R_b_kernel(a_val: float, Og0: float, Omega_b0: float, T0: float, Tz: float) -> float:
    """Baryon-to-photon momentum density ratio."""

    if a_val <= 0.0 or Og0 <= 0.0 or T0 <= 0.0 or Tz <= 0.0:
        return math.inf
    baryon_term = 1.0 / (a_val * a_val * a_val)
    temperature_term = (T0 / Tz) ** 4
    return 0.75 * (Omega_b0 / Og0) * baryon_term * temperature_term


@numba.njit(cache=True)
def c_s_kernel(R_b_val: float) -> float:
    """Sound speed of the photon-baryon fluid."""

    return C_LIGHT / math.sqrt(3.0 * (1.0 + R_b_val))


@numba.njit(cache=True)
def z_star_hu_sugiyama_kernel(Obh2: float, Omh2: float) -> float:
    g1 = (0.0783 * Obh2 ** -0.238) / (1.0 + 39.5 * Obh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * Obh2 ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * Obh2 ** -0.738) * (1.0 + g1 * Omh2 ** g2)


@numba.njit(cache=True)
def z_drag_eh_kernel(Obh2: float, Omh2: float) -> float:
    b1 = 0.313 * Omh2 ** -0.419 * (1.0 + 0.607 * Obh2 ** 0.674)
    b2 = 0.238 * Omh2 ** 0.223
    numerator = 1291.0 * Omh2 ** 0.251
    denominator = 1.0 + 0.659 * Omh2 ** 0.828
    return numerator / denominator * (1.0 + b1 * Obh2 ** b2)


@numba.njit(cache=True)
def sound_integrand_kernel(
    a_val: float,
    H0: float,
    Omega_m0: float,
    Omega_r0: float,
    alpha: float,
    omega_sigma: float,
    R_b_val: float,
) -> float:
    """Integrand c_s / (a^2 H(a)) used for r_s."""

    if a_val <= 0.0:
        return math.inf
    hubble = H_kernel(a_val, H0, Omega_m0, Omega_r0, alpha, omega_sigma)
    if hubble <= 0.0:
        return math.inf
    c_s = c_s_kernel(R_b_val)
    return c_s / (a_val * a_val * hubble)


@numba.njit(cache=True)
def _interp_linear(a_val: float, a_grid: np.ndarray, values: np.ndarray) -> float:
    return float(np.interp(a_val, a_grid, values))


@numba.njit(cache=True)
def _interp_log(a_val: float, log_a: np.ndarray, log_values: np.ndarray) -> float:
    return math.exp(np.interp(math.log(a_val), log_a, log_values))


@numba.njit(cache=True)
def sound_horizon_njit(
    z_target: float,
    H0: float,
    Omega_m0: float,
    Omega_r0: float,
    Omega_b0: float,
    alpha: float,
    Rmax: float,
    sigma_rescale: float,
    mode_flag: int,
    a_grid: np.ndarray,
    log_a_grid: np.ndarray,
    T_grid: np.ndarray,
    eps_grid: np.ndarray,
    alpha_grid: np.ndarray,
    g_star_grid: np.ndarray,
    steps: int = 4096,
) -> float:
    """
    Nopython sound horizon ∫ c_s/(a^2 H) da from a=0 to a=1/(1+z_target).

    mode_flag: 0=free, 1=flat_today (uses sigma_rescale).
    """

    if z_target <= 0.0:
        return 0.0

    n = max(steps, 2 * (steps // 2))
    a_upper = 1.0 / (1.0 + z_target)
    h = a_upper / n

    a_min = a_grid[0]
    a_max = a_grid[-1]

    log_T_grid = np.log(np.clip(T_grid, 1e-50, None))

    a_today = 1.0
    a_today_clamped = a_today
    if a_today_clamped < a_min:
        a_today_clamped = a_min
    elif a_today_clamped > a_max:
        a_today_clamped = a_max
    g_today = _interp_linear(a_today_clamped, a_grid, g_star_grid)
    if g_today <= 0.0:
        return math.inf
    T0 = _interp_log(a_today_clamped, log_a_grid, log_T_grid)
    if T0 <= 0.0:
        return math.inf
    Og0 = photon_density_parameter_kernel(g_today, Omega_r0)
    if Og0 <= 0.0:
        return math.inf

    result = 0.0
    for i in range(n + 1):
        a_val = i * h
        if a_val <= 0.0:
            continue
        a_clamped = a_val
        if a_clamped < a_min:
            a_clamped = a_min
        elif a_clamped > a_max:
            a_clamped = a_max

        eps_val = _interp_linear(a_clamped, a_grid, eps_grid)
        alpha_val = _interp_linear(a_clamped, a_grid, alpha_grid)
        kmax_val = kmax_from_eps_alpha(eps_val, alpha_val)
        omega_sigma = omega_sigma_raw_kernel(a_clamped, Rmax, alpha_val, kmax_val)
        if mode_flag == 1:
            omega_sigma = sigma_rescale * omega_sigma

        T_interp = _interp_log(a_clamped, log_a_grid, log_T_grid)
        if T_interp <= 0.0:
            continue

        Rb = R_b_kernel(a_val, Og0, Omega_b0, T0, T_interp)
        integrand = sound_integrand_kernel(
            a_val,
            H0,
            Omega_m0,
            Omega_r0,
            alpha,
            omega_sigma,
            Rb,
        )
        weight = 4.0 if i % 2 == 1 else 2.0
        if i == 0 or i == n:
            weight = 1.0
        result += weight * integrand

    return result * h / 3.0


@numba.njit(cache=True)
def integrate_callable_njit(func, lower: float, upper: float, steps: int) -> float:
    """
    Nopython Simpson integration for numba-callable integrands.
    """

    if upper == lower:
        return 0.0
    n = max(steps, 2 * (steps // 2))
    h = (upper - lower) / n
    result = func(lower) + func(upper)
    i = 1
    while i < n:
        weight = 4.0 if i % 2 == 1 else 2.0
        result += weight * func(lower + i * h)
        i += 1
    return result * h / 3.0


def integrate_callable_kernel(func, lower: float, upper: float, steps: int) -> float:
    """
    Simpson integration for a Python callable (kept in Python to avoid numba jit dependency).
    """

    if upper == lower:
        return 0.0
    n = max(steps, 2 * (steps // 2))
    h = (upper - lower) / n
    result = func(lower) + func(upper)
    i = 1
    while i < n:
        weight = 4.0 if i % 2 == 1 else 2.0
        result += weight * func(lower + i * h)
        i += 1
    return result * h / 3.0


__all__ = [
    "PHOTON_G_DEGREES",
    "photon_density_parameter_kernel",
    "R_b_kernel",
    "c_s_kernel",
    "z_star_hu_sugiyama_kernel",
    "z_drag_eh_kernel",
    "sound_integrand_kernel",
    "sound_horizon_njit",
    "integrate_callable_kernel",
    "integrate_callable_njit",
]
