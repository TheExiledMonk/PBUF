"""
ΛCDM baseline models implementing the shared cosmology API.

All functions accept vectorised redshift arrays and a parameter dictionary.
Unless stated otherwise, the returned distances are in megaparsecs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from config.constants import C_LIGHT, KM_TO_M, MPC_TO_M


def _hubble_param(params: Dict[str, float]) -> float:
    """Return the present-day Hubble constant in s^-1."""

    hubble = params.get("H0", 70.0)  # km/s/Mpc
    return hubble * KM_TO_M / MPC_TO_M


def E2(z: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Dimensionless Hubble parameter squared, E²(z).

    E²(z) = Ωₘ (1 + z)^3 + Ωᵣ (1 + z)^4 + Ω_k (1 + z)^2 + Ω_Λ

    The radiation term defaults to zero unless `Or0` is provided. If
    `Ol0` is not supplied, flatness is assumed such that Ω_Λ = 1 - Ωₘ - Ωᵣ - Ω_k.
    """

    z = np.asarray(z, dtype=float)
    om0 = params.get("Om0", 0.3)
    or0 = params.get("Or0", 0.0)
    ok0 = params.get("Ok0", 0.0)
    if "Ol0" in params:
        ol0 = params["Ol0"]
    else:
        ol0 = 1.0 - om0 - or0 - ok0
    scale = (1.0 + z)
    return om0 * scale**3 + or0 * scale**4 + ok0 * scale**2 + ol0


# core/gr_models.py
import numpy as np

def H(z, p):
    """
    Hubble rate in km/s/Mpc for flat ΛCDM with optional curvature/radiation terms.

    Parameters expected in p:
      H0  : km/s/Mpc
      Om0 : Ω_m0
      Or0 : Ω_r0  (already includes photons + neutrinos)
      Ok0 : Ω_k0  (default 0)
      Ol0 : Ω_Λ0  (default 1 - Om0 - Or0 - Ok0)
    """
    z = np.asarray(z, dtype=float)
    H0 = float(p["H0"])
    Om = float(p["Om0"])
    Or = float(p.get("Or0", 0.0))
    Ok = float(p.get("Ok0", 0.0))
    Ol = float(p.get("Ol0", 1.0 - Om - Or - Ok))
    zp1 = 1.0 + z
    E2 = Om * zp1**3 + Or * zp1**4 + Ok * zp1**2 + Ol
    return H0 * np.sqrt(E2)



def Dc(z: np.ndarray, params: Dict[str, float], steps: int = 256) -> np.ndarray:
    """
    Line-of-sight comoving distance in megaparsecs.

    The integral is evaluated with a simple trapezoidal rule on a uniform
    grid of `steps` points between 0 and z.
    """

    z = np.asarray(z, dtype=float)
    hz = H(z, params)  # km/s/Mpc
    hz_si = hz * KM_TO_M / MPC_TO_M  # s^-1
    hubble_si = _hubble_param(params)
    distances = np.zeros_like(z, dtype=float)
    for idx, zi in enumerate(z):
        grid = np.linspace(0.0, zi, steps)
        integrand = C_LIGHT / MPC_TO_M / np.sqrt(E2(grid, params)) / hubble_si
        distances[idx] = np.trapz(integrand, grid)
    return distances


def DL(z: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Luminosity distance in megaparsecs: D_L = (1 + z) D_c(z)."""

    return (1.0 + np.asarray(z, dtype=float)) * Dc(z, params)


def mu(z: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Distance modulus μ(z).

    μ(z) = 5 log₁₀(D_L / 10 pc)
    """

    dl_mpc = DL(z, params)
    dl_pc = dl_mpc * 1e6
    return 5.0 * np.log10(dl_pc / 10.0)
