"""Simple sanity guards for LCDM."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from cosmos.models.common.phase6a import curvature_check, load_phase6a_config
from cosmos.models.lcdm.distances import H as lcdm_H_of_a
from cosmos.models.lcdm.model import LCDMModel
from cosmos.optim.sanity_base import SanityResult
from cosmos.optim.sanity_utils import make_a_grid

ModelParams = Dict[str, float]


_LCDM_PHASE6A_CONFIG = load_phase6a_config("lcdm")


def check_lcdm_sanity(
    params: ModelParams,
    model: LCDMModel,
    cmb_solver: Callable[[ModelParams, LCDMModel], dict] | None = None,
) -> SanityResult:
    result = SanityResult()
    result.merge(check_closure_lcdm(params, model))
    result.merge(check_expansion_lcdm(params, model))

    if cmb_solver is not None:
        result.merge(check_lcdm_cmb_derived(params, model, cmb_solver))

    return result


def check_closure_lcdm(params: ModelParams, model: LCDMModel) -> SanityResult:
    result = SanityResult()
    Omega_m0 = float(params["Omega_m0"])
    Omega_r0 = float(params["Omega_r0"])
    Omega_k0 = float(params["Omega_k0"])
    Omega_b0 = float(params.get("Omega_b0", 0.0))
    Omega_total_nolambda = Omega_m0 + Omega_r0 + Omega_k0 + Omega_b0
    Omega_lambda = 1.0 - Omega_total_nolambda

    if Omega_lambda < 0.0:
        result.add_error(f"LCDM closure: Omega_Lambda < 0 ({Omega_lambda})")

    if abs(Omega_total_nolambda + Omega_lambda - 1.0) > 1e-6:
        result.add_error("LCDM closure: Omega_total(a=1) != 1")

    return result


def check_expansion_lcdm(params: ModelParams, model: LCDMModel) -> SanityResult:
    result = SanityResult()
    a_grid = make_a_grid(n=200)
    H_vals = np.array([lcdm_H_of_a(a, model.params) for a in a_grid])

    dH = np.diff(H_vals)
    if np.any(dH > 1e-8):
        result.add_error("LCDM: H(a) not strictly decreasing with a")

    a_small = np.logspace(-9, -6, 20)
    H_small = np.array([lcdm_H_of_a(a, model.params) for a in a_small])
    test = H_small * (a_small**2)
    if np.mean(np.abs(test)) == 0.0:
        result.add_error("LCDM: Early-time H(a) test produced zero mean")
        return result

    if np.ptp(test) / np.mean(test) > 0.1:
        result.add_error("LCDM: H(a) not approximately ~ a^-2 in radiation era")

    curvature_ok, curvature_reason, _ = curvature_check(
        lambda a: lcdm_H_of_a(a, model.params),
        _LCDM_PHASE6A_CONFIG,
    )
    if not curvature_ok:
        result.add_error(f"LCDM curvature: {curvature_reason}")

    return result


def check_lcdm_cmb_derived(
    params: ModelParams,
    model: LCDMModel,
    cmb_solver: Callable[[ModelParams, LCDMModel], dict],
) -> SanityResult:
    result = SanityResult()
    cmb = cmb_solver(params, model)

    Omega_b_h2 = float(cmb["Omega_b_h2"])
    H0 = float(cmb["H0"])
    r_s = float(cmb["r_s"])

    if not (0.01 <= Omega_b_h2 <= 0.04):
        result.add_error(f"LCDM: Omega_b h^2 out of sane range ({Omega_b_h2})")

    if not (40.0 <= H0 <= 90.0):
        result.add_error(f"LCDM: H0 out of broad sanity range ({H0})")

    if not (80.0 <= r_s <= 200.0):
        result.add_error(f"LCDM: r_s out of broad sanity range ({r_s})")

    return result
