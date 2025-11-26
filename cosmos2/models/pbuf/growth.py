"""Linear growth helpers (ported from cosmos_old)."""

from __future__ import annotations

import importlib
import math

import numpy as np

from cosmos2.kernels.pbuf_growth_rhs import growth_rhs_njit
from cosmos2.kernels.pbuf_growth_table import build_growth_table_njit

from .elastic import omega_sigma_of_a
from .params import PBUFParams
from .thermal_table import ThermalTable


def growth_ode_rhs(a: float, y, params: PBUFParams, table: ThermalTable):
    """
    Right-hand side of the growth equation:
        y = [D(a), D'(a)]
    """

    y = np.asarray(y)
    D, D_prime = y

    a_val = max(float(a), 1.0e-12)
    omega_sigma_val = omega_sigma_of_a(a_val, params, table)

    eps = 1e-5

    def _E_val(a_in: float) -> float:
        a_safe = max(float(a_in), 1.0e-12)
        omega_total = (
            params.Omega_m0 / (a_safe ** 3)
            + params.Omega_r0 / (a_safe ** 4)
            + params.alpha / (a_safe ** 2)
            + omega_sigma_of_a(a_safe, params, table)
        )
        return math.sqrt(max(omega_total, 0.0))

    E_a = _E_val(a_val)
    E_a_plus = _E_val(a_val + eps)
    E_a_minus = _E_val(a_val - eps)
    dE_da = (E_a_plus - E_a_minus) / (2.0 * eps)

    term1 = -(3.0 / a_val + dE_da / E_a) * D_prime
    term2 = 1.5 * params.Omega_m0 / (a_val ** 5 * E_a ** 2) * D

    return np.array([D_prime, term1 + term2])


def make_growth_rhs_njit(params: PBUFParams, table: ThermalTable):
    """
    Build a nopython growth RHS that inlines omega_sigma interpolation.
    Returns None if params use unsupported normalization mode.
    """

    mode = getattr(params, "omega_normalization", "flat_today")
    if mode not in {"free", "flat_today"}:
        return None
    mode_flag = 0 if mode == "free" else 1
    sigma_rescale = float(getattr(params, "sigma_rescale", 1.0))
    a_grid, log_a, T_arr, eps_grid, alpha_grid, dln_eps, dln_alpha, g_star, g_starS = table.numba_payload()
    a_min = float(a_grid[0])
    a_max = float(a_grid[-1])

    Omega_m0 = float(params.Omega_m0)
    Omega_r0 = float(params.Omega_r0)
    alpha_curv = float(params.alpha)
    Rmax = float(params.Rmax)

    def rhs(a: float, y_arr: np.ndarray) -> np.ndarray:
        return growth_rhs_njit(
            a,
            y_arr,
            Omega_m0,
            Omega_r0,
            alpha_curv,
            Rmax,
            sigma_rescale,
            mode_flag,
            a_min,
            a_max,
            a_grid,
            alpha_grid,
            eps_grid,
        )

    try:
        numba = importlib.import_module("numba")
        return numba.njit(cache=True)(rhs)
    except Exception:
        return None


__all__ = ["growth_ode_rhs", "make_growth_rhs_njit"]
