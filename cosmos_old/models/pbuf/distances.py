"""Background expansion helpers for the PBUF model."""

from __future__ import annotations

import math
from typing import Callable

from cosmos.models.pbuf import utils
from cosmos.models.pbuf.elastic import omega_sigma_of_a
from cosmos.models.pbuf.params import PBUFParams
from cosmos.models.pbuf.thermal_table import ThermalTable


def omega_total_at_a(
    a: float, params: PBUFParams, table: ThermalTable, *, alpha: float | None = None
) -> float:
    """Total density parameter evaluated at the supplied scale factor."""

    a_val = max(float(a), 1.0e-12)
    Om = params.Omega_m0 / a_val**3
    Or = params.Omega_r0 / a_val**4
    curvature = alpha if alpha is not None else params.alpha
    Ok = curvature / a_val**2
    Os = omega_sigma_of_a(a_val, params, table)
    return Om + Or + Ok + Os


def E_squared(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Dimensionless expansion rate squared."""

    return omega_total_at_a(a, params, table)


def E(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Dimensionless expansion rate."""

    return math.sqrt(max(E_squared(a, params, table), 0.0))


def H(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Hubble rate in km/s/Mpc."""

    return params.H0 * E(a, params, table)


def H_z(z: float, params: PBUFParams, table: ThermalTable) -> float:
    """Hubble rate at redshift z."""

    return H(utils.as_scale_factor(z), params, table)


def comoving_distance(
    z: float,
    params: PBUFParams,
    table: ThermalTable,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    """Line-of-sight comoving distance in Mpc."""

    def integrand(zp: float) -> float:
        return utils.C_LIGHT / H_z(zp, params, table)

    return integrator(integrand, 0.0, z)


def angular_diameter_distance(
    z: float,
    params: PBUFParams,
    table: ThermalTable,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    """Angular diameter distance in Mpc (flat-only for now)."""

    chi = comoving_distance(z, params, table, integrator)
    return chi / (1.0 + z)
