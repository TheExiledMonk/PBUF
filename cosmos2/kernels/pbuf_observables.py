"""Numba kernels for PBUF distance observables."""

from __future__ import annotations

import math

import numba

from .pbuf_distances import C_LIGHT


@numba.njit(cache=True)
def distance_modulus_from_DM_kernel(DM: float, z_val: float) -> float:
    """Compute distance modulus μ from transverse comoving distance DM."""

    if not math.isfinite(DM) or DM <= 0.0 or z_val < -0.999999:
        return math.inf
    return 5.0 * math.log10(DM) + 25.0


@numba.njit(cache=True)
def dv_from_DM_H_kernel(z_val: float, DM_val: float, H_val: float) -> float:
    """Compute volume-averaged distance DV for a single redshift."""

    if z_val < 0.0 or not math.isfinite(DM_val) or not math.isfinite(H_val) or H_val <= 0.0:
        return math.inf
    denom = 1.0 + z_val
    if denom <= 0.0:
        return math.inf
    da = DM_val / denom
    factor = z_val * (1.0 + z_val) * (1.0 + z_val) * da * da * C_LIGHT / H_val
    return math.inf if factor <= 0.0 else math.pow(factor, 1.0 / 3.0)


@numba.njit(cache=True)
def dh_from_H_kernel(H_val: float) -> float:
    """Compute D_H = c / H."""

    return C_LIGHT / H_val if H_val > 0.0 else math.inf


__all__ = ["distance_modulus_from_DM_kernel", "dv_from_DM_H_kernel", "dh_from_H_kernel"]
