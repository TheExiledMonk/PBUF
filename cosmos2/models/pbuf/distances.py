"""Background expansion helpers for the PBUF model (ported from cosmos_old)."""

from __future__ import annotations

import math
import os
from typing import Callable

from cosmos2.kernels.pbuf_distances import (
    E_kernel,
    E_squared_kernel,
    H_kernel,
    comoving_distance_njit,
    comoving_integrand_kernel,
    omega_total_kernel,
)

from . import utils
from .elastic import omega_sigma_of_a
from .params import PBUFParams
from .thermal_table import ThermalTable

_COMOVING_ENV = os.environ.get("PBUF_COMOVING_DISTANCE", "").strip().lower()
_FORCE_PYTHON_COMOVING = _COMOVING_ENV in {"python", "py", "force_python"}


def omega_total_at_a(
    a: float, params: PBUFParams, table: ThermalTable, *, alpha: float | None = None
) -> float:
    """Total density parameter evaluated at the supplied scale factor."""

    a_val = max(float(a), 1.0e-12)
    curvature = alpha if alpha is not None else params.alpha
    Os = omega_sigma_of_a(a_val, params, table)
    return omega_total_kernel(a_val, params.Omega_m0, params.Omega_r0, curvature, Os)


def E_squared(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Dimensionless expansion rate squared."""

    a_val = max(float(a), 1.0e-12)
    Os = omega_sigma_of_a(a_val, params, table)
    return E_squared_kernel(a_val, params.Omega_m0, params.Omega_r0, params.alpha, Os)


def E(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Dimensionless expansion rate."""

    a_val = max(float(a), 1.0e-12)
    Os = omega_sigma_of_a(a_val, params, table)
    return E_kernel(a_val, params.Omega_m0, params.Omega_r0, params.alpha, Os)


def H(a: float, params: PBUFParams, table: ThermalTable) -> float:
    """Hubble rate in km/s/Mpc."""

    a_val = max(float(a), 1.0e-12)
    Os = omega_sigma_of_a(a_val, params, table)
    return H_kernel(a_val, params.H0, params.Omega_m0, params.Omega_r0, params.alpha, Os)


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

    mode = getattr(params, "omega_normalization", "flat_today")
    mode_flag = 0 if mode == "free" else 1 if mode == "flat_today" else -1
    if not _FORCE_PYTHON_COMOVING and mode_flag >= 0:
        sigma_rescale = float(getattr(params, "sigma_rescale", 1.0))
        a_grid, log_a, T_arr, eps_arr, alpha_arr, dln_eps, dln_alpha, g_star, g_starS = table.numba_payload()
        try:
            return comoving_distance_njit(
                float(z),
                float(params.H0),
                float(params.Omega_m0),
                float(params.Omega_r0),
                float(params.alpha),
                float(params.Rmax),
                sigma_rescale,
                mode_flag,
                a_grid,
                alpha_arr,
                eps_arr,
                steps=4096,
            )
        except Exception:
            pass

    def integrand(zp: float) -> float:
        a_val = utils.as_scale_factor(zp)
        a_val = max(float(a_val), 1.0e-12)
        Os = omega_sigma_of_a(a_val, params, table)
        return comoving_integrand_kernel(a_val, params.H0, params.Omega_m0, params.Omega_r0, params.alpha, Os)

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


__all__ = [
    "omega_total_at_a",
    "E_squared",
    "E",
    "H",
    "H_z",
    "comoving_distance",
    "angular_diameter_distance",
]
