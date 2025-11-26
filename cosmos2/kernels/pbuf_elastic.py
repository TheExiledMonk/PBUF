"""Numba kernels for the PBUF elastic sector."""

from __future__ import annotations

import math

import numba


@numba.njit(cache=True)
def kmax_from_eps_alpha(eps: float, alpha: float) -> float:
    """Compute k_max(T) = ε₀ − α."""

    return eps - alpha


@numba.njit(cache=True)
def omega_sigma_raw_kernel(a_val: float, Rmax: float, alpha_val: float, kmax_val: float) -> float:
    """Unnormalized elastic density fraction helper."""

    decay = math.exp(-a_val / Rmax)
    S = 1.0 - (1.0 - kmax_val) * decay
    return alpha_val * (1.0 - decay) * S


__all__ = ["kmax_from_eps_alpha", "omega_sigma_raw_kernel"]
