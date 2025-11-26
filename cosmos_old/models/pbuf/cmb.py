"""CMB distance priors for the PBUF model."""

from __future__ import annotations

import math
import os
from typing import Callable

from cosmos.interfaces import CMBOutput
from cosmos.models.pbuf import utils
from cosmos.models.pbuf.distances import H as H_of_a
from cosmos.models.pbuf.distances import angular_diameter_distance, comoving_distance
from cosmos.models.pbuf.elastic import alpha_of_a, epsilon_of_a, kmax_of_a, omega_sigma_of_a
from cosmos.models.pbuf.params import PBUFParams
from cosmos.models.pbuf.temperature import T_of_a, T_of_z
from cosmos.models.pbuf.thermal_table import ThermalTable

DEFAULT_DISTANCE_STEPS = 4096
DEFAULT_SOUND_STEPS = 4096
PHOTON_G_DEGREES = 2.0  # Photon energy degrees of freedom
DEBUG_HEAT_TABLE = os.getenv("COSMOS_PBUF_DEBUG_HEAT_TABLE", "").strip().lower() in {"1", "true", "yes", "on"}


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

    g_today = table.get("g_star", at_scale_factor=1.0)
    if g_today <= 0.0:
        raise ValueError("Thermal table reports non-positive g_star at a=1.0.")
    photon_fraction = PHOTON_G_DEGREES / g_today
    return params.Omega_r0 * photon_fraction


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

    baryon_term = (1.0 / a ** 3)  # scales as (1 + z)^3
    temperature_term = (T0 / Tz) ** 4
    return 0.75 * (params.Omega_b0 / Og0) * baryon_term * temperature_term


def c_s(z: float, params: PBUFParams, table: ThermalTable) -> float:
    """Sound speed of the photon-baryon fluid."""

    return utils.C_LIGHT / math.sqrt(3.0 * (1.0 + R_b(z, params, table)))


def z_star_hu_sugiyama(params: PBUFParams) -> float:
    """Hu & Sugiyama fitting formula."""

    Obh2 = Omega_b_h2(params)
    Omh2 = Omega_m_h2(params)

    g1 = (0.0783 * Obh2 ** -0.238) / (1.0 + 39.5 * Obh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * Obh2 ** 1.81)

    return 1048.0 * (1.0 + 0.00124 * Obh2 ** -0.738) * (1.0 + g1 * Omh2 ** g2)


def z_drag_eh(params: PBUFParams) -> float:
    """Eisenstein & Hu drag epoch redshift tailored to PBUF parameters."""

    Obh2 = Omega_b_h2(params)
    Omh2 = Omega_m_h2(params)
    b1 = 0.313 * Omh2 ** -0.419 * (1.0 + 0.607 * Obh2 ** 0.674)
    b2 = 0.238 * Omh2 ** 0.223
    numerator = 1291.0 * Omh2 ** 0.251
    denominator = 1.0 + 0.659 * Omh2 ** 0.828
    return numerator / denominator * (1.0 + b1 * Obh2 ** b2)


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

    def integrand(a: float) -> float:
        a = max(a, 1e-8)
        z = 1.0 / a - 1.0
        return c_s(z, params, table) / (a * a * H_of_a(a, params, table))

    return integrator(integrand, 0.0, a_target)


def sound_horizon_drag(
    params: PBUFParams,
    table: ThermalTable,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    z_drag = z_drag_eh(params)
    return _sound_horizon_from_redshift(z_drag, params, table, integrator)


def _integrate(func: Callable[[float], float], lower: float, upper: float, *, steps: int) -> float:
    return utils.simpson_integral(func, lower, upper, n=max(steps, 2 * (steps // 2)))


def _maybe_print_diagnostics(params: PBUFParams, table: ThermalTable, z_star: float) -> None:
    """Optional diagnostic dump for manual validation."""

    if not DEBUG_HEAT_TABLE:
        return

    a_star = utils.as_scale_factor(z_star)
    print("Ωσ(a*) =", omega_sigma_of_a(a_star, params, table))
    print("T(a*) =", T_of_a(a_star, table))
    print("alpha(a*) =", alpha_of_a(a_star, table))
    print("eps(a*) =", epsilon_of_a(a_star, table))
    print("kmax(a*) =", kmax_of_a(a_star, table))


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
