"""Elastic sector helpers sourced from the temperature table (ported from cosmos_old)."""

from __future__ import annotations

from cosmos2.kernels.pbuf_elastic import kmax_from_eps_alpha, omega_sigma_raw_kernel

from .params import PBUFParams
from .thermal_table import ThermalTable


def _fast_lookup(field: str, a: float, table: ThermalTable) -> float:
    """Use fast_get when available and fall back to Python interpolation."""

    try:
        return table.fast_get(field, at_scale_factor=a)
    except Exception:
        return table.get(field, at_scale_factor=a)


def epsilon_of_a(a: float, table: ThermalTable) -> float:
    """Return ε₀(T(a))."""

    return _fast_lookup("epsilon0_T", a, table)


def alpha_of_a(a: float, table: ThermalTable) -> float:
    """Return α(T(a))."""

    return _fast_lookup("alpha_T", a, table)


def kmax_of_a(a: float, table: ThermalTable) -> float:
    """Return k_max(T(a)) = ε₀ − α."""

    eps = epsilon_of_a(a, table)
    alpha = alpha_of_a(a, table)
    return float(kmax_from_eps_alpha(eps, alpha))


def omega_sigma_raw_of_a(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Unnormalized elastic density fraction."""

    a_val = table.clamp_scale_factor(a)
    alpha_val = alpha_of_a(a_val, table)
    kmax_val = kmax_of_a(a_val, table)
    return float(omega_sigma_raw_kernel(a_val, params.Rmax, alpha_val, kmax_val))


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
