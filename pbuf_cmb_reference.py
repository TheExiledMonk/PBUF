#!/usr/bin/env python3
"""
Reference, self-contained PBUF CMB calculator (thermal off).
"""

from math import pi, sqrt

import numpy as np


def compute_pbuf_cmb(params: dict) -> tuple[float, float, float]:
    """
    Compute (R, lA, theta_star) using the standalone PBUF background recipe.
    """
    H0 = float(params["H0"])
    h = H0 / 100.0
    Om0 = float(params["Omega_m0"])
    Ob0 = float(params["Omega_b0"])
    Ok0 = float(params.get("Omega_k0", 0.0))
    Or0 = float(params.get("Omega_r0", 9.0e-5))
    alpha = float(params["alpha"])
    Rmax = float(params["Rmax"])
    k_sat = float(params["k_sat"])

    z_star = 1089.92
    a_star = 1.0 / (1.0 + z_star)

    T_CMB = 2.7255
    Omega_gamma_h2 = 2.469e-5 * (T_CMB / 2.7255) ** 4
    Og0 = Omega_gamma_h2 / (h * h)

    N = 20000
    a_min = 1.0e-6
    a_grid = np.logspace(np.log10(a_min), 0.0, N, dtype=float)

    def omega_sigma(a_arr: np.ndarray) -> np.ndarray:
        return alpha * (1.0 - np.exp(-a_arr * Rmax)) + k_sat * a_arr

    Omega_sigma = omega_sigma(a_grid)
    E_grid = np.sqrt(Om0 / (a_grid ** 3) + Or0 / (a_grid ** 4) + Ok0 / (a_grid ** 2) + Omega_sigma)

    c = 299_792.458  # km/s
    mask_rs = a_grid <= a_star
    a_rs = a_grid[mask_rs]
    E_rs = E_grid[mask_rs]
    R_b = 3.0 * Ob0 / (4.0 * Og0) * a_rs
    c_s = c / np.sqrt(3.0 * (1.0 + R_b))
    integrand_rs = c_s / (a_rs ** 2 * E_rs)
    r_s = (1.0 / H0) * np.trapz(integrand_rs, a_rs)

    mask_chi = a_grid >= a_star
    a_chi = a_grid[mask_chi]
    E_chi = E_grid[mask_chi]
    integrand_chi = 1.0 / (a_chi ** 2 * E_chi)
    D_M = (c / H0) * np.trapz(integrand_chi, a_chi)

    R_cmb = sqrt(Om0) * (H0 * D_M / c)
    lA = pi * D_M / r_s if r_s != 0.0 else float("inf")
    theta_star = r_s / D_M if D_M != 0.0 else float("inf")

    return R_cmb, lA, theta_star


__all__ = ["compute_pbuf_cmb"]
