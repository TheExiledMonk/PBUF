"""CMB distance priors for the PBUF model (ported from cosmos_old)."""

from __future__ import annotations

import math
import os
from typing import Callable

from cosmos2.kernels.pbuf_cmb import (
    PHOTON_G_DEGREES,
    R_b_kernel,
    c_s_kernel,
    integrate_callable_kernel,
    integrate_callable_njit,
    photon_density_parameter_kernel,
    sound_horizon_njit,
    sound_integrand_kernel,
    z_drag_eh_kernel,
    z_star_hu_sugiyama_kernel,
)
from cosmos2.models.lcdm.common import CMBOutput

from . import utils
from .distances import H as H_of_a
from .distances import angular_diameter_distance, comoving_distance
from .elastic import alpha_of_a, epsilon_of_a, kmax_of_a, omega_sigma_of_a
from .params import PBUFParams
from .temperature import T_of_a, T_of_z
from .thermal_table import ThermalTable

DEFAULT_DISTANCE_STEPS = 4096
DEFAULT_SOUND_STEPS = 4096
DEBUG_HEAT_TABLE = os.getenv("COSMOS_PBUF_DEBUG_HEAT_TABLE", "").strip().lower() in {"1", "true", "yes", "on"}
_SOUND_ENV = os.getenv("PBUF_SOUND_HORIZON", "").strip().lower()
_FORCE_PYTHON_SOUND = _SOUND_ENV in {"python", "py", "force_python"}


def h(params: PBUFParams) -> float:
    """Reduced Hubble parameter."""

    return params.H0 / 100.0


def Omega_b_h2(params: PBUFParams) -> float:
    """Ω_b h^2."""

    return params.Omega_b0 * h(params) ** 2


def Omega_m_h2(params: PBUFParams) -> float:
    """Ω_m h^2."""

    return params.Omega_m0 * h(params) ** 2


def _photon_density_parameter(params: PBUFParams, table: ThermalTable) -> float:
    """
    Photon density parameter today derived from the thermal table.

    Uses the ratio of photon degrees of freedom to the total relativistic
    content recorded in g_star at a = 1.
    """

    try:
        g_today = table.fast_get("g_star", at_scale_factor=1.0)
    except Exception:
        g_today = table.get("g_star", at_scale_factor=1.0)
    if g_today <= 0.0:
        raise ValueError("Thermal table reports non-positive g_star at a=1.0.")
    return photon_density_parameter_kernel(g_today, params.Omega_r0)


def R_b(z: float, params: PBUFParams, table: ThermalTable) -> float:
    """Baryon-to-photon momentum density ratio."""

    Og0 = _photon_density_parameter(params, table)
    if Og0 <= 0.0:
        raise ValueError("Photon density parameter must be positive.")

    a = utils.as_scale_factor(z)
    T0 = T_of_a(1.0, table)
    Tz = T_of_z(z, table)
    if T0 <= 0.0 or Tz <= 0.0:
        raise ValueError("Thermal table returned non-physical temperatures.")

    return R_b_kernel(a, Og0, params.Omega_b0, T0, Tz)


def c_s(z: float, params: PBUFParams, table: ThermalTable) -> float:
    """Sound speed of the photon-baryon fluid."""

    return c_s_kernel(R_b(z, params, table))


def z_star_hu_sugiyama(params: PBUFParams) -> float:
    """Hu & Sugiyama fitting formula."""

    Obh2 = Omega_b_h2(params)
    Omh2 = Omega_m_h2(params)

    return z_star_hu_sugiyama_kernel(Obh2, Omh2)


def z_drag_eh(params: PBUFParams) -> float:
    """Eisenstein & Hu drag epoch redshift tailored to PBUF parameters."""

    Obh2 = Omega_b_h2(params)
    Omh2 = Omega_m_h2(params)
    return z_drag_eh_kernel(Obh2, Omh2)


def sound_horizon(
    z_star: float,
    params: PBUFParams,
    table: ThermalTable,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    return _sound_horizon_from_redshift(z_star, params, table, integrator)


def _sound_horizon_from_redshift(
    z_target: float,
    params: PBUFParams,
    table: ThermalTable,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    a_target = 1.0 / (1.0 + z_target)

    # Try the njit path first if allowed and supported by normalization mode.
    mode = getattr(params, "omega_normalization", "flat_today")
    mode_flag = 0 if mode == "free" else 1 if mode == "flat_today" else -1
    if not _FORCE_PYTHON_SOUND and mode_flag >= 0:
        try:
            sigma_rescale = float(getattr(params, "sigma_rescale", 1.0))
            a_grid, log_a, T_arr, eps_arr, alpha_arr, dln_eps, dln_alpha, g_star, g_starS = table.numba_payload()
            return float(
                sound_horizon_njit(
                    float(z_target),
                    float(params.H0),
                    float(params.Omega_m0),
                    float(params.Omega_r0),
                    float(params.Omega_b0),
                    float(params.alpha),
                    float(params.Rmax),
                    sigma_rescale,
                    mode_flag,
                    a_grid,
                    log_a,
                    T_arr,
                    eps_arr,
                    alpha_arr,
                    g_star,
                    steps=DEFAULT_SOUND_STEPS,
                )
            )
        except Exception:
            pass

    def integrand(a: float) -> float:
        a = max(a, 1e-8)
        z = 1.0 / a - 1.0
        a_val = max(float(a), 1e-8)
        # Compute R_b(z) pieces in Python, then delegate the heavy math to the kernel.
        Og0 = _photon_density_parameter(params, table)
        if Og0 <= 0.0:
            return math.inf
        T0 = T_of_a(1.0, table)
        Tz = T_of_z(z, table)
        if T0 <= 0.0 or Tz <= 0.0:
            return math.inf
        Rb = R_b_kernel(a_val, Og0, params.Omega_b0, T0, Tz)
        omega_sigma = omega_sigma_of_a(a_val, params, table)
        return sound_integrand_kernel(
            a_val,
            params.H0,
            params.Omega_m0,
            params.Omega_r0,
            params.alpha,
            omega_sigma,
            Rb,
        )

    return integrator(integrand, 0.0, a_target)


def sound_horizon_drag(
    params: PBUFParams,
    table: ThermalTable,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    z_drag = z_drag_eh(params)
    return _sound_horizon_from_redshift(z_drag, params, table, integrator)


def _integrate(func: Callable[[float], float], lower: float, upper: float, *, steps: int) -> float:
    # Prefer njit integrator when the callable is numba-backed; fall back to Python otherwise.
    try:
        return integrate_callable_njit(func, float(lower), float(upper), int(max(steps, 2 * (steps // 2))))
    except Exception:
        return integrate_callable_kernel(func, float(lower), float(upper), int(max(steps, 2 * (steps // 2))))


def _maybe_print_diagnostics(params: PBUFParams, table: ThermalTable, z_star: float) -> None:
    """Optional diagnostic dump for manual validation."""

    if not DEBUG_HEAT_TABLE:
        return

    a_star = utils.as_scale_factor(z_star)
    print("Ωσ(a*) =", omega_sigma_of_a(a_star, params, table))
    print("T(a*) =", T_of_a(a_star, table))
    print("alpha(a*) =", alpha_of_a(a_star, params, table))
    print("eps(a*) =", epsilon_of_a(a_star, params, table))
    print("kmax(a*) =", kmax_of_a(a_star, params, table))


def compute_cmb_output(params: PBUFParams, table: ThermalTable) -> CMBOutput:
    """Compute the CMB distance prior observables for the supplied parameters."""

    integrator = lambda f, a, b: _integrate(f, a, b, steps=DEFAULT_DISTANCE_STEPS)

    z_star = z_star_hu_sugiyama(params)
    D_M = comoving_distance(z_star, params, table, integrator)
    D_A = angular_diameter_distance(z_star, params, table, integrator)

    sound_integrator = lambda f, a, b: _integrate(f, a, b, steps=DEFAULT_SOUND_STEPS)
    r_s = sound_horizon(z_star, params, table, sound_integrator)

    theta_star = r_s / D_M
    l_A = math.pi * D_M / r_s

    _maybe_print_diagnostics(params, table, z_star)

    shift_R = math.sqrt(params.Omega_m0) * (params.H0 * D_M / utils.C_LIGHT)

    extras = {
        "model": "pbuf",
        "Omega_m_h2": Omega_m_h2(params),
        "thermal_metadata": table.metadata_summary(),
    }

    return CMBOutput(
        R=shift_R,
        l_A=l_A,
        Omega_b_h2=Omega_b_h2(params),
        theta_star=theta_star,
        z_star=z_star,
        D_M_Mpc=D_M,
        D_A_Mpc=D_A,
        r_s_Mpc=r_s,
        extras=extras,
    )


__all__ = [
    "h",
    "Omega_b_h2",
    "Omega_m_h2",
    "R_b",
    "c_s",
    "z_star_hu_sugiyama",
    "z_drag_eh",
    "sound_horizon",
    "sound_horizon_drag",
    "compute_cmb_output",
]
