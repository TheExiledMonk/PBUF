"""CMB helpers wired to the running-Λ background."""

from __future__ import annotations

import math
from typing import Callable

from cosmos.interfaces import CMBOutput
from cosmos.models.lcdm import utils
from cosmos.models.running_lambda.distances import angular_diameter_distance, comoving_distance
from cosmos.models.running_lambda.expansion import H as H_of_a
from cosmos.models.running_lambda.parameters import RunningLambdaParams

DEFAULT_DISTANCE_STEPS = 4096
DEFAULT_SOUND_STEPS = 4096


def h(params: RunningLambdaParams) -> float:
    return params.H0 / 100.0


def Omega_b_h2(params: RunningLambdaParams) -> float:
    return params.Omega_b0 * h(params) ** 2


def Omega_m_h2(params: RunningLambdaParams) -> float:
    return params.Omega_m0 * h(params) ** 2


def Omega_gamma_h2(T_cmb: float = 2.7255) -> float:
    return 2.469e-5 * (T_cmb / 2.7255) ** 4


def Omega_nu_h2_massless() -> float:
    return 1.688e-5


def R_b(z: float, params: RunningLambdaParams, T_cmb: float = 2.7255) -> float:
    Og_h2 = Omega_gamma_h2(T_cmb)
    Og0 = Og_h2 / h(params) ** 2
    return 0.75 * (params.Omega_b0 / Og0) / (1.0 + z)


def c_s(z: float, params: RunningLambdaParams, T_cmb: float = 2.7255) -> float:
    return utils.C_LIGHT / math.sqrt(3.0 * (1.0 + R_b(z, params, T_cmb=T_cmb)))


def z_star_hu_sugiyama(params: RunningLambdaParams) -> float:
    Obh2 = Omega_b_h2(params)
    Omh2 = Omega_m_h2(params)

    g1 = (0.0783 * Obh2 ** -0.238) / (1.0 + 39.5 * Obh2 ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * Obh2 ** 1.81)

    return 1048.0 * (1.0 + 0.00124 * Obh2 ** -0.738) * (1.0 + g1 * Omh2 ** g2)


def z_drag_eh(params: RunningLambdaParams) -> float:
    Obh2 = Omega_b_h2(params)
    Omh2 = Omega_m_h2(params)
    b1 = 0.313 * Omh2 ** -0.419 * (1.0 + 0.607 * Obh2 ** 0.674)
    b2 = 0.238 * Omh2 ** 0.223
    numerator = 1291.0 * Omh2 ** 0.251
    denominator = 1.0 + 0.659 * Omh2 ** 0.828
    return numerator / denominator * (1.0 + b1 * Obh2 ** b2)


def _sound_horizon_from_redshift(
    z_target: float,
    params: RunningLambdaParams,
    integrator: Callable[[Callable[[float], float], float, float], float],
    *,
    T_cmb: float = 2.7255,
) -> float:
    a_target = 1.0 / (1.0 + z_target)

    def integrand(a: float) -> float:
        a = max(a, 1e-8)
        z = 1.0 / a - 1.0
        return c_s(z, params, T_cmb=T_cmb) / (a * a * H_of_a(a, params))

    return integrator(integrand, 0.0, a_target)


def sound_horizon(
    z_star: float,
    params: RunningLambdaParams,
    integrator: Callable[[Callable[[float], float], float, float], float],
    *,
    T_cmb: float = 2.7255,
) -> float:
    return _sound_horizon_from_redshift(z_star, params, integrator, T_cmb=T_cmb)


def sound_horizon_drag(
    params: RunningLambdaParams,
    integrator: Callable[[Callable[[float], float], float, float], float],
    *,
    T_cmb: float = 2.7255,
) -> float:
    z_drag = z_drag_eh(params)
    return _sound_horizon_from_redshift(z_drag, params, integrator, T_cmb=T_cmb)


def _integrate(func: Callable[[float], float], lower: float, upper: float, *, steps: int) -> float:
    return utils.simpson_integral(func, lower, upper, n=max(steps, 2 * (steps // 2)))


def compute_cmb_output(params: RunningLambdaParams) -> CMBOutput:
    integrator = lambda f, a, b: _integrate(f, a, b, steps=DEFAULT_DISTANCE_STEPS)
    sound_integrator = lambda f, a, b: _integrate(f, a, b, steps=DEFAULT_SOUND_STEPS)

    z_star = z_star_hu_sugiyama(params)
    D_M = comoving_distance(z_star, params, integrator)
    D_A = angular_diameter_distance(z_star, params, integrator)
    r_s = sound_horizon(z_star, params, sound_integrator)

    theta_star = r_s / D_M
    l_A = math.pi * D_M / r_s
    shift_R = math.sqrt(params.Omega_m0) * (params.H0 * D_M / utils.C_LIGHT)

    extras = {
        "model": "running_lambda",
        "Omega_m_h2": Omega_m_h2(params),
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
