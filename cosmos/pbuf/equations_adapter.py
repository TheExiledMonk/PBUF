"""Adapters for using PBUF background equations in standalone calculations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .equations import E2_pbuf, omega_sigma_raw as _omega_sigma_raw


def E_PBUF(
    a: float,
    *,
    H0: float,
    Om0: float,
    Or0: float,
    Ok0: float,
    alpha: float,
    Rmax: float,
    k_sat: float,
    eps0: float,
    n_alpha: float = 0.0,
    n_eps: float = 0.0,
    n_R: float = 0.0,
) -> float:
    """Return E(a) = H(a)/H0 for the PBUF background.

    Parameters mirror the Path A elastic model; additional keys can be supplied
    later by expanding the parameter dictionary passed into ``E2_pbuf``.
    """

    # Guard against numerical probes slightly beyond the physical domain.
    a = float(np.clip(a, 1e-6, 1.0))

    params: Dict[str, Any] = {
        "H0": H0,
        "Om0": Om0,
        "Or0": Or0,
        "Ok0": Ok0,
        "alpha": alpha,
        "Rmax": Rmax,
        "k_sat": k_sat,
        "eps0": eps0,
        "n_alpha": n_alpha,
        "n_eps": n_eps,
        "n_R": n_R,
    }

    E2 = E2_pbuf(a, params)
    return float(np.sqrt(E2))


def omega_sigma_raw(a: float, **params: Any) -> float:
    """Expose the raw elastic reservoir for diagnostic use."""

    return float(_omega_sigma_raw(a, params))
