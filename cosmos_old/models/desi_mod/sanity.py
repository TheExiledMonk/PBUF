"""Sanity guards for the DESI modified expansion history."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from cosmos.optim.sanity_base import SanityResult
from cosmos.models.desi_mod import expansion, parameters

_REQUIRED_KEYS = {
    "H0",
    "Omega_m0",
    "Omega_b0",
    "Omega_k0",
    "Omega_r0",
    "w0",
    "wa",
}


def _build_params(raw: Mapping[str, float]) -> parameters.DESIModParams:
    missing = [_ for _ in _REQUIRED_KEYS if _ not in raw]
    if missing:
        raise KeyError(f"DESI_mod sanity missing parameters: {missing}")
    coerced = {key: float(raw[key]) for key in _REQUIRED_KEYS}
    return parameters.DESIModParams(**coerced)


def sanity_checks(raw_params: Mapping[str, float]) -> SanityResult:
    params = _build_params(raw_params)
    result = SanityResult()

    omega_sum = params.Omega_m0 + params.Omega_r0 + params.Omega_k0 + params.Omega_DE0
    if abs(omega_sum - 1.0) > 1e-6:
        result.add_error("DESI_mod closure: sum of fractions at a=1 deviates from 1")
    if params.Omega_DE0 <= 0.0:
        result.add_error("DESI_mod closure: Omega_DE0 must be positive")

    a_grid = np.logspace(-5, 0, num=200)
    E_vals = np.array([expansion.E(a, params) for a in a_grid])
    if np.any(E_vals <= 0.0):
        result.add_error("DESI_mod: E(a) non-positive on the a grid")

    H_vals = np.array([expansion.H(a, params) for a in a_grid])
    dH = np.diff(H_vals)
    if np.any(dH >= 0.0):
        result.add_error("DESI_mod: H(a) not strictly decreasing toward a=1")

    Omega_de_vals = np.array([expansion.omega_de(a, params) for a in a_grid])
    if np.any(Omega_de_vals <= 0.0):
        result.add_error("DESI_mod: Omega_DE(a) non-positive for some a")

    return result
