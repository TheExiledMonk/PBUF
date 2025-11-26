"""Expansion history helpers for the DESI modified ΛCDM model."""

from __future__ import annotations

import math

from cosmos.models.desi_mod.parameters import DESIModParams

_MIN_SCALE_FACTOR = 1.0e-12


def _safe_scale_factor(a: float) -> float:
    return max(float(a), _MIN_SCALE_FACTOR)


def _omega_de_0(params: DESIModParams) -> float:
    return max(params.Omega_DE0, 0.0)


def omega_de(a: float, params: DESIModParams) -> float:
    """Dark-energy density fraction evolving according to the DESI w0-wa law."""

    a_val = _safe_scale_factor(a)
    omega_de0 = _omega_de_0(params)
    exponent = -3.0 * (1.0 + params.w0 + params.wa)
    decay = math.exp(-3.0 * params.wa * (1.0 - a_val))
    return omega_de0 * (a_val**exponent) * decay


def E_squared(a: float, params: DESIModParams) -> float:
    a_val = _safe_scale_factor(a)
    matter = params.Omega_m0 / a_val**3
    radiation = params.Omega_r0 / a_val**4
    curvature = params.Omega_k0 / a_val**2
    total = matter + radiation + curvature + omega_de(a_val, params)
    return max(total, 0.0)


def E(a: float, params: DESIModParams) -> float:
    """Dimensionless Hubble expansion rate."""

    return math.sqrt(E_squared(a, params))


def H(a: float, params: DESIModParams) -> float:
    """Hubble expansion rate in km/s/Mpc."""

    return params.H0 * E(a, params)


def dE_da(a: float, params: DESIModParams) -> float:
    a_val = _safe_scale_factor(a)
    derivative = _dE_squared_da(a_val, params)
    h = E(a_val, params)
    if h <= 0.0:
        return 0.0
    return 0.5 * derivative / h


def _dE_squared_da(a: float, params: DESIModParams) -> float:
    matter_term = -3.0 * params.Omega_m0 / a**4
    radiation_term = -4.0 * params.Omega_r0 / a**5
    curvature_term = -2.0 * params.Omega_k0 / a**3
    omega_de_val = omega_de(a, params)
    exponent = -3.0 * (1.0 + params.w0 + params.wa)
    domega_de = omega_de_val * (exponent / a + 3.0 * params.wa)
    return matter_term + radiation_term + curvature_term + domega_de
