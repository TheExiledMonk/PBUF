"""CMB distance priors for the LCDM model (structured like PBUF CMB)."""

from __future__ import annotations

import math
from typing import Callable

from cosmos2.models.lcdm.common import CMBOutput
from cosmos2.kernels import lcdm_math

from . import utils
from .distances import comoving_distance
from .params import LCDMParams

DEFAULT_DISTANCE_STEPS = 4096
DEFAULT_SOUND_STEPS = 200000


def h(params: LCDMParams) -> float:
    """Reduced Hubble parameter."""

    return params.H0 / 100.0


def Omega_b_h2(params: LCDMParams) -> float:
    """Ω_b h^2."""

    return params.Omega_b0 * h(params) ** 2


def Omega_m_h2(params: LCDMParams) -> float:
    """Ω_m h^2."""

    return params.Omega_m0 * h(params) ** 2


def z_star_hu_sugiyama(params: LCDMParams) -> float:
    """Hu & Sugiyama fitting formula (delegates to kernel helper)."""

    return lcdm_math.z_star_hu_sugiyama(
        float(params.Omega_m0),
        float(params.Omega_b0),
        float(params.H0),
    )


def sound_horizon(
    z_star: float,
    params: LCDMParams,
    integrator: Callable[[Callable[[float], float], float, float], float],
) -> float:
    """Sound horizon r_s integrated up to z_star."""

    if z_star <= 0.0:
        return 0.0
    a_target = 1.0 / (1.0 + z_star)

    def integrand(a: float) -> float:
        a_val = max(a, 1e-8)
        z = 1.0 / a_val - 1.0
        R_b = 0.75 * (params.Omega_b0 / _omega_gamma0(params)) / (1.0 + z)
        c_s = utils.C_LIGHT / math.sqrt(3.0 * (1.0 + R_b))
        return c_s / (a_val * a_val * _H_of_a(a_val, params))

    return integrator(integrand, 0.0, a_target)


def sound_horizon_drag(params: LCDMParams) -> float:
    """Drag-epoch sound horizon using the kernel high-res integrator."""

    Om0 = float(params.Omega_m0)
    Ob0 = float(params.Omega_b0)
    Or0 = float(params.Omega_r0)
    Ok0 = float(params.Omega_k0)
    H0 = float(params.H0)
    Ol0 = 1.0 - Om0 - Or0 - Ok0
    Og0 = _omega_gamma0(params)
    return lcdm_math.sound_horizon_high_z(
        lcdm_math.z_star_hu_sugiyama(Om0, Ob0, H0),
        1.0e5,
        Ob0,
        Om0,
        Or0,
        Og0,
        Ok0,
        H0,
        steps=DEFAULT_SOUND_STEPS,
    )


def compute_cmb_output(params: LCDMParams) -> CMBOutput:
    """Compute the CMB distance prior observables for the supplied parameters."""

    z_star = z_star_hu_sugiyama(params)
    H0 = float(params.H0)
    Om0 = float(params.Omega_m0)
    Ob0 = float(params.Omega_b0)
    Or0 = float(params.Omega_r0)
    Ok0 = float(params.Omega_k0)
    Ol0 = 1.0 - Om0 - Or0 - Ok0

    DM = lcdm_math.comoving_distance_to_z(float(z_star), Om0, Or0, Ok0, Ol0, H0, steps=DEFAULT_DISTANCE_STEPS)
    r_s = sound_horizon(float(z_star), params, lambda f, a, b: utils.simpson_integral(f, a, b, n=DEFAULT_SOUND_STEPS))

    D_A = DM / (1.0 + float(z_star))
    R = math.sqrt(params.Omega_m0) * (params.H0 * DM / utils.C_LIGHT)
    l_A = math.pi * DM / r_s if r_s > 0.0 else float("inf")
    theta_star = r_s / DM if DM > 0.0 else float("inf")

    return CMBOutput(
        R=R,
        l_A=l_A,
        Omega_b_h2=Omega_b_h2(params),
        theta_star=theta_star,
        z_star=float(z_star),
        D_M_Mpc=DM,
        D_A_Mpc=D_A,
        r_s_Mpc=r_s,
        extras={"engine": "cosmos2"},
    )


def _omega_gamma0(params: LCDMParams) -> float:
    """Photon density today from kernel helper."""

    return lcdm_math.omega_gamma0_from_Tcmb(float(params.H0), T_cmb=lcdm_math.T_CMB)


def _H_of_a(a: float, params: LCDMParams) -> float:
    """Local H(a) mirroring LCDMModel._build_grids computation."""

    Om0 = float(params.Omega_m0)
    Or0 = float(params.Omega_r0)
    Ok0 = float(params.Omega_k0)
    Ol0 = 1.0 - Om0 - Or0 - Ok0
    inv_a = 1.0 / max(a, 1.0e-12)
    return float(params.H0) * math.sqrt(max(Om0 * inv_a**3 + Or0 * inv_a**4 + Ok0 * inv_a**2 + Ol0, 0.0))


__all__ = [
    "h",
    "Omega_b_h2",
    "Omega_m_h2",
    "z_star_hu_sugiyama",
    "sound_horizon",
    "sound_horizon_drag",
    "compute_cmb_output",
]
