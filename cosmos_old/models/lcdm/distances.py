"""Expansion/distance helpers for LCDM."""

from __future__ import annotations

from typing import Callable

import numpy as np

from cosmos.models.lcdm.params import LCDMParams
from cosmos.models.lcdm import utils


def closure_residual(Omega_lambda: float, params: LCDMParams) -> float:
    """Residual for Ω_total = 1 closure in LCDM."""

    return params.Omega_m0 + params.Omega_r0 + params.Omega_k0 + Omega_lambda - 1.0


def E_squared(a: float, params: LCDMParams) -> float:
    """Dimensionless expansion rate squared."""

    Om = params.Omega_m0 / a**3
    Or = params.Omega_r0 / a**4
    Ok = params.Omega_k0 / a**2
    Ol = (params.Omega_lambda0 or (1.0 - params.Omega_m0 - params.Omega_r0 - params.Omega_k0))
    return Om + Or + Ok + Ol


def E(a: float, params: LCDMParams) -> float:
    return np.sqrt(E_squared(a, params))


def H(a: float, params: LCDMParams) -> float:
    return params.H0 * E(a, params)


def H_z(z: float, params: LCDMParams) -> float:
    return H(1.0 / (1.0 + z), params)


def comoving_distance(z: float, params: LCDMParams, integrator: Callable) -> float:
    def integrand(zp: float) -> float:
        return utils.C_LIGHT / H_z(zp, params)

    return integrator(integrand, 0.0, z)


def angular_diameter_distance(z: float, params: LCDMParams, integrator: Callable) -> float:
    chi = comoving_distance(z, params, integrator)
    return chi / (1.0 + z)
