"""Distance helpers for the Early Dark Energy ΛCDM variant."""

from __future__ import annotations

from typing import Callable

from cosmos.models.common.distance_utils import C_LIGHT
from cosmos.models.ede_lcdm.expansion import H_z
from cosmos.models.ede_lcdm.parameters import EDELCDMParams


def comoving_distance(z: float, params: EDELCDMParams, integrator: Callable) -> float:
    def integrand(zp: float) -> float:
        return C_LIGHT / H_z(zp, params)

    return integrator(integrand, 0.0, z)


def angular_diameter_distance(z: float, params: EDELCDMParams, integrator: Callable) -> float:
    chi = comoving_distance(z, params, integrator)
    return chi / (1.0 + z)


def luminosity_distance(z: float, params: EDELCDMParams, integrator: Callable) -> float:
    chi = comoving_distance(z, params, integrator)
    return chi * (1.0 + z)
