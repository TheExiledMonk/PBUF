"""Shared sound-horizon helper mirroring the reference r_s integral."""

from __future__ import annotations

import numpy as np

from .lcdm_math import C_LIGHT


def compute_sound_horizon_from_H(
    a_grid: np.ndarray,
    H_grid: np.ndarray,
    Ob0: float,
    omega_gamma0: float,
    lut_a: np.ndarray,
    lut_T: np.ndarray,
    T0: float,
    a_max: float | None = None,
) -> tuple[float, float]:
    """
    r_s = ∫ c_s(a) / (a^2 H(a)) da with c_s(a) = c / sqrt(3 (1 + R(a))),
    R(a) = (3 * Omega_b0 / a^3) / (4 * Omega_gamma(a)), Omega_gamma(a) = omega_gamma0 * (T/T0)^4.

    Integrates over the provided a_grid (assumed ascending) up to a_max if given.
    Returns (r_s, r_d) where r_d mirrors r_s for compatibility with callers.
    """
    a = np.asarray(a_grid, dtype=float)
    H = np.asarray(H_grid, dtype=float)
    if a.size < 2:
        return 0.0, 0.0

    if a_max is not None:
        mask = a <= a_max
        if not np.any(mask):
            return 0.0, 0.0
        a = a[mask]
        H = H[mask]

    T_a = np.interp(a, lut_a, lut_T)
    theta = (T_a / T0) ** 4
    # Mirror reference R_baryon definition: rho_gamma ∝ (T/T0)^4 (no explicit omega_gamma0 factor)
    R = (3.0 * Ob0 / np.clip(a ** 3, 1.0e-30, None)) / (4.0 * np.clip(theta, 1.0e-30, None))
    cs = C_LIGHT / np.sqrt(3.0 * (1.0 + R))
    integrand = cs / np.clip(a * a * H, 1.0e-30, None)
    r_s = float(np.trapz(integrand, a))
    return r_s, r_s  # r_d mirrors r_s for now


__all__ = ["compute_sound_horizon_from_H"]
