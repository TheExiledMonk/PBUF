"""Lightweight Halofit wrapper (with a linear fallback when a backend is unavailable)."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np

_SMALL = 1e-12


def _try_external_halofit(k: np.ndarray, z: np.ndarray, pk_lin: np.ndarray, omega_m0: float, h: float) -> np.ndarray | None:
    """
    Attempt to use an external halofit provider if installed.

    This keeps the internal dependency surface minimal while allowing users
    to plug in a more accurate non-linear correction via `pyhalofit` or `pyccl`.
    """

    # pyhalofit API: halofit(z, k, pk_lin, h, om_m)
    try:
        import pyhalofit  # type: ignore

        return np.asarray(pyhalofit.halofit(z, k, pk_lin, h, omega_m0), dtype=float)
    except Exception:
        pass

    # pyccl fallback (if available) via its halofit engine
    try:
        import pyccl  # type: ignore

        cosmo = pyccl.Cosmology(
            Omega_c=omega_m0 - 0.0,
            Omega_b=0.0,
            h=h,
            sigma8=1.0,
            n_s=1.0,
        )
        delta_nl = pyccl.nonlin_matter_power(cosmo, k, 1.0 / (1.0 + z))
        return np.asarray(delta_nl, dtype=float)
    except Exception:
        return None


def apply_halofit(
    k: Iterable[float] | float | np.ndarray,
    z: Iterable[float] | float | np.ndarray,
    pk_lin: Iterable[float] | float | np.ndarray,
    *,
    omega_m0: float,
    h: float,
) -> np.ndarray:
    """
    Return P_nl(k, z) using Halofit where available, falling back to linear P(k, z).
    """

    k_arr = np.asarray(k, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    pk_arr = np.asarray(pk_lin, dtype=float)

    external = _try_external_halofit(k_arr, z_arr, pk_arr, omega_m0, h)
    if external is not None:
        return np.clip(external, 0.0, np.inf)

    warnings.warn("Halofit backend unavailable; returning linear power spectrum.", RuntimeWarning, stacklevel=2)
    return np.clip(pk_arr, 0.0, np.inf)


__all__ = ["apply_halofit"]
