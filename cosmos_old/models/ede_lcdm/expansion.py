"""Expansion history helpers for the Early Dark Energy LCDM variant."""

from __future__ import annotations

import math

from cosmos.models.ede_lcdm.parameters import EDELCDMParams

_MIN_SCALE_FACTOR = 1.0e-12
_MIN_CRITICAL_A = 1.0e-15


def _safe_scale_factor(a: float) -> float:
    return max(float(a), _MIN_SCALE_FACTOR)


def _safe_critical_a(params: EDELCDMParams) -> float:
    return max(float(params.a_c), _MIN_CRITICAL_A)


def omega_ede(a: float, params: EDELCDMParams) -> float:
    """Early dark energy fraction evolving as the Hilltop phenomenology."""

    a_val = _safe_scale_factor(a)
    ac = _safe_critical_a(params)
    exponent = float(params.n)
    ratio = (a_val / ac) ** exponent
    value = params.Omega_EDE_0 / (1.0 + ratio)
    return max(value, 0.0)


def omega_lambda(params: EDELCDMParams) -> float:
    """Late-time Λ fraction derived from the closure condition."""

    closure_head = params.Omega_m0 + params.Omega_r0 + params.Omega_k0 + omega_ede(1.0, params)
    value = 1.0 - closure_head
    return max(value, 0.0)


def E_squared(a: float, params: EDELCDMParams) -> float:
    """Dimensionless expansion rate squared, including Ω_EDE(a)."""

    a_val = _safe_scale_factor(a)
    matter = params.Omega_m0 / a_val**3
    radiation = params.Omega_r0 / a_val**4
    curvature = params.Omega_k0 / a_val**2
    lambda_ = omega_lambda(params)
    ede = omega_ede(a_val, params)
    total = matter + radiation + curvature + lambda_ + ede
    return max(total, 0.0)


def E(a: float, params: EDELCDMParams) -> float:
    """Dimensionless Hubble expansion rate."""

    return math.sqrt(E_squared(a, params))


def H(a: float, params: EDELCDMParams) -> float:
    """Hubble expansion rate in km/s/Mpc."""

    return params.H0 * E(a, params)


def H_z(z: float, params: EDELCDMParams) -> float:
    return H(1.0 / (1.0 + z), params)


def _omega_ede_derivative(a: float, params: EDELCDMParams) -> float:
    a_val = _safe_scale_factor(a)
    ac = _safe_critical_a(params)
    exponent = float(params.n)
    ratio = (a_val / ac) ** exponent
    numerator = -params.Omega_EDE_0 * exponent * ratio
    denominator = a_val * (1.0 + ratio) ** 2
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def dE_squared_da(a: float, params: EDELCDMParams) -> float:
    a_val = _safe_scale_factor(a)
    matter_term = -3.0 * params.Omega_m0 / a_val**4
    radiation_term = -4.0 * params.Omega_r0 / a_val**5
    curvature_term = -2.0 * params.Omega_k0 / a_val**3
    lambda_term = 0.0
    ede_derivative = _omega_ede_derivative(a_val, params)
    return matter_term + radiation_term + curvature_term + lambda_term + ede_derivative


def dE_da(a: float, params: EDELCDMParams) -> float:
    derivative = dE_squared_da(a, params)
    e_val = E(a, params)
    if e_val <= 0.0:
        return 0.0
    return 0.5 * derivative / e_val
