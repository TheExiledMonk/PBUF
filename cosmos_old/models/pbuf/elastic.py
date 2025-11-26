"""Elastic sector helpers sourced from the temperature table."""

from __future__ import annotations

import numpy as np

from cosmos.models.pbuf.params import PBUFParams
from cosmos.models.pbuf.thermal_table import ThermalTable


def epsilon_of_a(a: float, table: ThermalTable) -> float:
    """Return ε₀(T(a))."""

    return table.get("epsilon0_T", at_scale_factor=a)


def alpha_of_a(a: float, table: ThermalTable) -> float:
    """Return α(T(a))."""

    return table.get("alpha_T", at_scale_factor=a)


def kmax_of_a(a: float, table: ThermalTable) -> float:
    """Return k_max(T(a)) = ε₀ − α."""

    eps = epsilon_of_a(a, table)
    alpha = alpha_of_a(a, table)
    return float(eps - alpha)


def omega_sigma_raw_of_a(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Unnormalized elastic density fraction."""

    a_val = table.clamp_scale_factor(a)
    alpha_val = alpha_of_a(a_val, table)
    kmax_val = kmax_of_a(a_val, table)
    decay = np.exp(-a_val / params.Rmax)
    S = 1.0 - (1.0 - kmax_val) * decay
    omega_sigma = alpha_val * (1.0 - decay) * S
    return float(omega_sigma)


def omega_sigma_of_a(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Normalized elastic density fraction used in the background solution."""

    raw = omega_sigma_raw_of_a(a, params, table)
    mode = getattr(params, "omega_normalization", "flat_today")
    if mode == "free":
        return raw

    if mode == "flat_today":
        rescale = getattr(params, "sigma_rescale", 1.0)
        return rescale * raw

    raise ValueError(f"Unknown omega_normalization mode '{mode}'.")


__all__ = ["alpha_of_a", "epsilon_of_a", "kmax_of_a", "omega_sigma_raw_of_a", "omega_sigma_of_a"]
