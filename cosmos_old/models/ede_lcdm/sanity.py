"""Sanity guards for the Early Dark Energy LCDM expansion."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from cosmos.models.common.phase6a import curvature_check, load_phase6a_config
from cosmos.models.ede_lcdm import expansion, parameters
from cosmos.optim.sanity_base import SanityResult
from cosmos.optim.sanity_utils import make_a_grid

_REQUIRED_KEYS = {
    "H0",
    "Omega_b0",
    "Omega_m0",
    "Omega_k0",
    "Omega_r0",
    "Omega_EDE_0",
    "a_c",
    "n",
}

_PHASE6A_CONFIG = load_phase6a_config("ede_lcdm")


def _build_params(raw: Mapping[str, float]) -> parameters.EDELCDMParams:
    missing = sorted(key for key in _REQUIRED_KEYS if key not in raw)
    if missing:
        raise KeyError(f"EDE sanity missing parameters: {missing}")
    coerced = {key: float(raw[key]) for key in _REQUIRED_KEYS}
    return parameters.EDELCDMParams(**coerced)


def sanity_checks(raw_params: Mapping[str, float]) -> SanityResult:
    params = _build_params(raw_params)
    result = SanityResult()

    omega_lambda = expansion.omega_lambda(params)
    omega_ede_today = expansion.omega_ede(1.0, params)
    closure = params.Omega_m0 + params.Omega_r0 + params.Omega_k0 + omega_ede_today + omega_lambda

    if abs(closure - 1.0) > 1e-6:
        result.add_error("EDE closure: Ω_total(a=1) deviates from 1")

    if omega_lambda < 0.0:
        result.add_error("EDE closure: Ω_Λ < 0")

    if omega_ede_today < 0.0:
        result.add_error("EDE closure: Ω_EDE(a=1) < 0")

    a_grid = make_a_grid(n=200)
    E_vals = np.array([expansion.E(a, params) for a in a_grid])
    if np.any(E_vals <= 0.0):
        result.add_error("EDE expansion: E(a) non-positive")

    H_vals = np.array([expansion.H(a, params) for a in a_grid])
    if np.any(np.diff(H_vals) >= 0.0):
        result.add_error("EDE expansion: H(a) not strictly decreasing")

    omega_ede_vals = np.array([expansion.omega_ede(a, params) for a in a_grid])
    if np.any(omega_ede_vals < 0.0):
        result.add_error("EDE expansion: Ω_EDE(a) went negative")

    curvature_ok, curvature_reason, _ = curvature_check(
        lambda a: expansion.H(a, params),
        _PHASE6A_CONFIG,
    )
    if not curvature_ok:
        result.add_error(f"EDE curvature: {curvature_reason}")

    return result
