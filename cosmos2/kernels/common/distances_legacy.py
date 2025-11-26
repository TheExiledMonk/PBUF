"""Legacy-style CMB distance helpers (cosmos-old compatible, non-Numba)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

C_LIGHT = 299_792.458  # km/s
T_CMB = 2.7255


def _simpson_integral(func: Callable[[float], float], lower: float, upper: float, steps: int) -> float:
    steps = max(int(steps), 2)
    if steps % 2 == 1:
        steps += 1
    h = (upper - lower) / steps
    total = func(lower) + func(upper)
    for i in range(1, steps):
        x = lower + i * h
        weight = 4.0 if i % 2 == 1 else 2.0
        total += weight * func(x)
    return total * h / 3.0


def comoving_distance_to_z(z_star: float, a_grid: np.ndarray, H_of_a: np.ndarray, *, steps: int = 4096) -> float:
    """Line-of-sight χ(z*) via c/H(z), independent of the precomputed χ grid."""
    if z_star <= 0.0:
        return 0.0
    a_grid = np.asarray(a_grid, dtype=float)
    H_of_a = np.asarray(H_of_a, dtype=float)

    def H_of_z(z: float) -> float:
        a = 1.0 / (1.0 + z)
        return float(np.interp(a, a_grid, H_of_a))

    def integrand(z: float) -> float:
        H = H_of_z(z)
        if H <= 0.0:
            return 0.0
        return C_LIGHT / H

    return float(_simpson_integral(integrand, 0.0, float(z_star), steps))


def _omega_gamma0(H0: float, T_cmb: float = T_CMB) -> float:
    """Photon density today (Ω_γ) from temperature."""
    h = H0 / 100.0
    Omega_gamma_h2 = 2.469e-5 * (T_cmb / 2.7255) ** 4
    return Omega_gamma_h2 / (h * h)


def sound_horizon_to_z(
    z_target: float,
    a_grid: np.ndarray,
    H_of_a: np.ndarray,
    H0: float,
    Omega_b0: float,
    Omega_r0: float,
    *,
    steps: int = 4096,
    T_cmb: float = T_CMB,
) -> float:
    """Sound horizon r_s integrated to an explicit redshift using legacy c_s/(a^2 H) in a-space."""
    if z_target <= 0.0:
        return 0.0
    a_target = 1.0 / (1.0 + float(z_target))
    a_grid = np.asarray(a_grid, dtype=float)
    H_of_a = np.asarray(H_of_a, dtype=float)
    Og0 = _omega_gamma0(H0, T_cmb=T_cmb)

    def c_s_from_a(a: float) -> float:
        z = 1.0 / a - 1.0
        R_b = (3.0 * Omega_b0) / (4.0 * Og0) / (1.0 + z)
        return C_LIGHT / math.sqrt(3.0 * (1.0 + R_b))

    def integrand(a: float) -> float:
        if a <= 0.0:
            return 0.0
        H = float(np.interp(a, a_grid, H_of_a))
        if H <= 0.0:
            return 0.0
        return c_s_from_a(a) / (a * a * H)

    return float(_simpson_integral(integrand, 0.0, a_target, steps))


__all__ = [
    "comoving_distance_to_z",
    "sound_horizon_to_z",
]
