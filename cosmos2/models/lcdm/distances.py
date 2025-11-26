"""Background expansion helpers for the LCDM model (mirrors PBUF layout)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from cosmos2.kernels.common import distances as dist_common
from cosmos2.kernels import lcdm_math
from . import utils
from .params import LCDMParams


def omega_total_at_a(a: float, params: LCDMParams) -> float:
    """Total density parameter evaluated at the supplied scale factor."""

    a_val = max(float(a), 1.0e-12)
    Om = params.Omega_m0 / a_val**3
    Or = params.Omega_r0 / a_val**4
    Ok = params.Omega_k0 / a_val**2
    Ol = 1.0 - params.Omega_m0 - params.Omega_r0 - params.Omega_k0
    return Om + Or + Ok + Ol


def E_squared(a: float, params: LCDMParams) -> float:
    """Dimensionless expansion rate squared."""

    return omega_total_at_a(a, params)


def E(a: float, params: LCDMParams) -> float:
    """Dimensionless expansion rate."""

    return math.sqrt(max(E_squared(a, params), 0.0))


def H(a: float, params: LCDMParams) -> float:
    """Hubble rate in km/s/Mpc."""

    return params.H0 * E(a, params)


def H_z(z: float, params: LCDMParams) -> float:
    """Hubble rate at redshift z."""

    return H(utils.as_scale_factor(z), params)


def comoving_distance(
    z: float,
    params: LCDMParams,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    """Line-of-sight comoving distance in Mpc."""

    def integrand(zp: float) -> float:
        return utils.C_LIGHT / H_z(zp, params)

    return integrator(integrand, 0.0, z)


def angular_diameter_distance(
    z: float,
    params: LCDMParams,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    """Angular diameter distance in Mpc (curvature-aware)."""

    chi = comoving_distance(z, params, integrator)
    curvature = params.Omega_k0
    return dist_common.transverse_comoving_distance(chi, params.H0, curvature) / (1.0 + z)


def comoving_distance_grid(a_grid: np.ndarray, params: LCDMParams) -> np.ndarray:
    """High-accuracy comoving distance grid matching legacy LCDMModel."""

    chi = dist_common.comoving_distance_simpson_z(a_grid, np.array([E(a, params) for a in a_grid]), params.H0)
    DM = np.empty_like(chi)
    for i, val in enumerate(chi):
        DM[i] = dist_common.transverse_comoving_distance(val, params.H0, params.Omega_k0)
    return DM


__all__ = [
    "omega_total_at_a",
    "E_squared",
    "E",
    "H",
    "H_z",
    "comoving_distance",
    "angular_diameter_distance",
    "comoving_distance_grid",
]
