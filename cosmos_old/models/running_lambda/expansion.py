"""Expansion history helpers for the running-Λ model."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from cosmos.models.running_lambda.parameters import RunningLambdaParams

_MIN_SCALE_FACTOR = 1.0e-12
_MIN_DENOMINATOR = 1.0e-12


def _ensure_array(a: float | Iterable[float]) -> tuple[np.ndarray, bool]:
    arr = np.asarray(a, dtype=float)
    scalar = arr.ndim == 0
    safe = np.maximum(arr, _MIN_SCALE_FACTOR)
    return safe, scalar


def _one_minus_nu(params: RunningLambdaParams) -> float:
    value = 1.0 - float(params.nu_lambda)
    if abs(value) < _MIN_DENOMINATOR:
        raise ValueError("RunningΛ: 1 - νΛ is too close to zero.")
    return value


def _omega_lambda(params: RunningLambdaParams) -> float:
    return float(params.Omega_lambda)


def E_squared(a: float | Iterable[float], params: RunningLambdaParams) -> float | np.ndarray:
    """Dimensionless expansion rate squared for the running vacuum."""

    arr, scalar = _ensure_array(a)
    one_minus_nu = _one_minus_nu(params)
    nu = float(params.nu_lambda)
    matter_exponent = -3.0 * (1.0 - nu)

    matter = params.Omega_m0 * arr ** matter_exponent
    radiation = params.Omega_r0 * arr ** -4.0
    curvature = params.Omega_k0 * arr ** -2.0
    constant = _omega_lambda(params) - nu

    total = matter + radiation + curvature + constant
    values = total / one_minus_nu

    finite = np.asarray(values, dtype=float)
    if np.any(finite <= 0.0):
        minimum = float(np.min(finite))
        raise ValueError(f"RunningΛ: E(a)^2 became non-positive (min={minimum}).")

    if scalar:
        return float(finite.item())
    return finite


def E(a: float | Iterable[float], params: RunningLambdaParams) -> float | np.ndarray:
    """Effective E(a) for the running vacuum background."""

    squared = E_squared(a, params)
    return np.sqrt(squared)


def H(a: float | Iterable[float], params: RunningLambdaParams) -> float | np.ndarray:
    """Hubble rate in km/s/Mpc for the running vacuum."""

    return params.H0 * E(a, params)


def H_z(z: float, params: RunningLambdaParams) -> float:
    """Hubble rate as a function of redshift."""

    a = 1.0 / (1.0 + max(float(z), -0.999999))
    return float(H(a, params))


def dE_squared_da(a: float | Iterable[float], params: RunningLambdaParams) -> float | np.ndarray:
    arr, scalar = _ensure_array(a)
    one_minus_nu = _one_minus_nu(params)
    nu = float(params.nu_lambda)
    matter_exponent = -3.0 * (1.0 - nu)
    matter_term = params.Omega_m0 * matter_exponent * arr ** (matter_exponent - 1.0)
    radiation_term = -4.0 * params.Omega_r0 * arr ** -5.0
    curvature_term = -2.0 * params.Omega_k0 * arr ** -3.0
    total = (matter_term + radiation_term + curvature_term) / one_minus_nu
    if scalar:
        return float(total.item())
    return total


def dE_da(a: float | Iterable[float], params: RunningLambdaParams) -> float | np.ndarray:
    derivative = dE_squared_da(a, params)
    e_val = E(a, params)
    if isinstance(e_val, np.ndarray):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 0.5 * derivative / e_val
        result = np.where(np.isfinite(result), result, 0.0)
        return result
    if e_val <= 0.0:
        return 0.0
    return 0.5 * float(derivative) / float(e_val)
