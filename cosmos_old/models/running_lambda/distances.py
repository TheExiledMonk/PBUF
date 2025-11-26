"""Distance helpers for the running-Λ background."""

from __future__ import annotations

from typing import Callable

from cosmos.models.common.distance_utils import (
    C_LIGHT,
    luminosity_distance as _luminosity_distance,
    transverse_comoving_distance,
)
from cosmos.models.running_lambda import expansion
from cosmos.models.running_lambda.parameters import RunningLambdaParams

Integrator = Callable[[Callable[[float], float], float, float], float]


def _scale_factor(z: float) -> float:
    safe_z = max(float(z), -0.999999)
    return 1.0 / (1.0 + safe_z)


def H_z(z: float, params: RunningLambdaParams) -> float:
    return expansion.H_z(z, params)


def comoving_distance(z: float, params: RunningLambdaParams, integrator: Integrator) -> float:
    def integrand(zp: float) -> float:
        return C_LIGHT / H_z(zp, params)

    return integrator(integrand, 0.0, max(float(z), 0.0))


def luminosity_distance(z: float, params: RunningLambdaParams, integrator: Integrator) -> float:
    chi = comoving_distance(z, params, integrator)
    D_M = transverse_comoving_distance(chi, params.H0, params.Omega_k0)
    return _luminosity_distance(D_M, z)


def angular_diameter_distance(z: float, params: RunningLambdaParams, integrator: Integrator) -> float:
    chi = comoving_distance(z, params, integrator)
    D_M = transverse_comoving_distance(chi, params.H0, params.Omega_k0)
    return D_M / (1.0 + max(float(z), -0.999999))
