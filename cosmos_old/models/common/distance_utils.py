"""Shared distance helpers for cosmology models."""

from __future__ import annotations

import math

import numpy as np

C_LIGHT = 299_792.458  # km/s


def transverse_comoving_distance(chi: float, H0: float, Omega_k: float, *, c_light: float = C_LIGHT) -> float:
    """
    Convert the line-of-sight comoving distance χ(z) into the transverse comoving
    distance D_M(z) accounting for curvature.
    """

    if not math.isfinite(chi) or chi <= 0.0 or abs(Omega_k) < 1.0e-12:
        return chi

    sqrt_abs = math.sqrt(abs(Omega_k))
    prefactor = c_light / (H0 * sqrt_abs)
    arg = sqrt_abs * H0 * chi / c_light

    if Omega_k > 0.0:
        return prefactor * math.sinh(arg)
    return prefactor * math.sin(arg)


def luminosity_distance(D_M: float, z: float) -> float:
    """Compute the luminosity distance D_L = (1 + z) D_M."""

    if not math.isfinite(D_M) or D_M <= 0.0 or z < -0.999999:
        return math.inf
    return D_M * (1.0 + z)


def distance_modulus_from_luminosity_distance(D_L: float) -> float:
    """Return μ(z) = 5 log10(D_L/Mpc) + 25."""

    if not math.isfinite(D_L) or D_L <= 0.0:
        return math.inf
    return 5.0 * (math.log10(D_L) + 5.0)


def luminosity_distance_non_decreasing(z: np.ndarray, dL: np.ndarray, *, tol: float = 1e-8) -> bool:
    """
    Ensure that the luminosity distance is monotonically increasing with redshift.
    """

    if z.size < 2 or dL.size < 2:
        return True

    sort_idx = np.argsort(z)
    sorted_dL = dL[sort_idx]

    diffs = np.diff(sorted_dL)
    diffs = np.where(np.isfinite(diffs), diffs, -tol * 2)

    return np.all(diffs >= -tol)
