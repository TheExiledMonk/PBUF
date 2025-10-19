"""
PBUF cosmology model sharing the unified API with ΛCDM.

The implementation leverages the evolution hooks in
`core.pbuf_evolution` and the σ-field helpers from `core.pbuf_sigma`
to modify the standard expansion history.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from config.constants import C_LIGHT, KM_TO_M, MPC_TO_M
from core import gr_models
from core.pbuf_evolution import evolve_params
from core.pbuf_sigma import sigma_eff


def _baseline_params(params: Mapping[str, float]) -> Dict[str, float]:
    keys = ("alpha", "Rmax", "eps0", "n_eps")
    return {key: float(params.get(key, 0.0)) for key in keys}


def _policy(params: Mapping[str, float]) -> Dict[str, float]:
    return dict(params.get("evolution_policy", {}))


def _hubble_param(params: Mapping[str, float]) -> float:
    hubble = params.get("H0", 70.0)
    return hubble * KM_TO_M / MPC_TO_M


def _saturation_factor(z: np.ndarray, k_sat: float) -> np.ndarray:
    """
    Phenomenological suppression factor encoding elastic saturation.

    Larger k_sat delays saturation to higher curvature (redshift) while
    smaller values quench σ-field contributions earlier.
    """

    z = np.asarray(z, dtype=float)
    k_sat = max(float(k_sat), 1e-6)
    pivot = 1.0 + 1.0e3 * k_sat
    ratio = z / pivot
    return 1.0 / (1.0 + ratio**4)


def E2(z: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """
    Effective dimensionless Hubble parameter squared for PBUF.

    E²(z) = E²_LCDM(z) + σ_eff(z; k_sat)

    The parameter ``k_sat`` controls how quickly elastic corrections
    saturate as curvature increases; CMB acoustic scale fits calibrate
    its early-time value.
    """

    z = np.asarray(z, dtype=float)
    lcdm = gr_models.E2(z, params)
    base = _baseline_params(params)
    policy = _policy(params)
    evolved = evolve_params(z, base, policy)
    alpha = evolved.get("alpha", np.zeros_like(z))
    rmax = evolved.get("Rmax", np.ones_like(z))
    eps = evolved.get("eps0", np.ones_like(z) * base.get("eps0", 0.0))
    sigma = sigma_eff(alpha, eps, rmax)
    k_sat = float(params.get("k_sat", 1.0))
    sigma *= _saturation_factor(z, k_sat)
    e2 = lcdm + sigma
    or0 = float(params.get("Or0", 0.0))
    if or0 > 0.0:
        rad = or0 * (1.0 + z) ** 4
        modifier = (k_sat - 1.0) / (1.0 + (1.0 + z) / 1.0e3)
        e2 = e2 + rad * modifier
    return np.clip(e2, 1e-6, None)


def H(z: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Hubble parameter H(z) in km s^-1 Mpc^-1."""

    hubble0 = params.get("H0", 70.0)
    return hubble0 * np.sqrt(E2(z, params))


def Dc(z: np.ndarray, params: Mapping[str, float], steps: int = 256) -> np.ndarray:
    """
    Comoving distance for the PBUF model in megaparsecs.

    The integration is evaluated with a simple trapezoidal rule using
    the modified E²(z) function.
    """

    z = np.asarray(z, dtype=float)
    hubble_si = _hubble_param(params)
    distances = np.zeros_like(z, dtype=float)
    for idx, zi in enumerate(z):
        grid = np.linspace(0.0, zi, steps)
        integrand = C_LIGHT / MPC_TO_M / np.sqrt(E2(grid, params)) / hubble_si
        trap = getattr(np, "trapezoid", np.trapz)
        distances[idx] = trap(integrand, grid)
    return distances


def DL(z: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Luminosity distance (Mpc)."""

    return (1.0 + np.asarray(z, dtype=float)) * Dc(z, params)


def mu(z: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Distance modulus μ(z) using the PBUF luminosity distance."""

    dl_mpc = DL(z, params)
    dl_pc = dl_mpc * 1e6
    return 5.0 * np.log10(dl_pc / 10.0)
