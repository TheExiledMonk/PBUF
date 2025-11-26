"""Sanity guards for the running-Λ helpers."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from cosmos.optim.sanity_base import SanityResult
from cosmos.models.running_lambda import expansion, parameters

_REQUIRED_KEYS = {
    "H0",
    "Omega_b0",
    "Omega_m0",
    "Omega_k0",
    "Omega_r0",
    "nu_lambda",
}


def _build_params(raw: Mapping[str, float]) -> parameters.RunningLambdaParams:
    missing = sorted(key for key in _REQUIRED_KEYS if key not in raw)
    if missing:
        raise KeyError(f"RunningΛ sanity missing parameters: {missing}")
    coerced = {
        key: float(raw[key])
        for key in _REQUIRED_KEYS
    }
    params = parameters.RunningLambdaParams(**coerced)
    return params.with_lambda(params.Omega_lambda)


def sanity_checks(raw_params: Mapping[str, float]) -> SanityResult:
    result = SanityResult()
    params = _build_params(raw_params)

    if abs(params.nu_lambda) > 0.3:
        result.add_error("RunningΛ: |nu_lambda| must be ≤ 0.3")

    omega_total = params.Omega_m0 + params.Omega_r0 + params.Omega_k0 + params.Omega_lambda
    if abs(omega_total - 1.0) > 1e-6:
        result.add_error("RunningΛ closure: Ω_total(a=1) != 1")
    if params.Omega_lambda < 0.0:
        result.add_error("RunningΛ: Ω_Λ(a=1) must be non-negative")

    a_grid = np.logspace(-5, 0, num=200)
    try:
        h_vals = np.array([expansion.H(a, params) for a in a_grid])
        e_vals = h_vals / params.H0
    except ValueError as exc:
        result.add_error(f"RunningΛ expansion error: {exc}")
        return result

    if np.any(e_vals <= 0.0):
        result.add_error("RunningΛ: H(a) becomes non-positive")

    dh = np.diff(h_vals)
    if np.any(dh >= 0.0):
        result.add_error("RunningΛ: H(a) not strictly decreasing with a")

    return result
