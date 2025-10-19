"""
σ-field helpers and Planck-bound limiters for the PBUF model.

The σ-field encodes deviations from standard ΛCDM expansion and must
respect theoretical ceiling values tied to the Planck scale. The
utilities provided here ensure that evolutionary policies cannot push
parameters beyond those limits.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from config.constants import H_BAR


PLANCK_ENERGY = np.sqrt(H_BAR)  # Arbitrary proxy for a theoretical bound.


def enforce_planck_bound(values: np.ndarray, max_scale: float = PLANCK_ENERGY) -> np.ndarray:
    """
    Clamp values element-wise to respect a Planck-inspired upper bound.

    Parameters
    ----------
    values : np.ndarray
        Raw σ-field amplitudes.
    max_scale : float
        Maximum admissible magnitude. Defaults to a proxy derived from ħ.
    """

    return np.clip(values, -max_scale, max_scale)


def sigma_eff(alpha: np.ndarray, epsilon: np.ndarray, rmax: np.ndarray) -> np.ndarray:
    """
    Compute the effective σ-field strength.

    Notes
    -----
    The placeholder formulation combines the evolved α(z), ε(z), and R_max(z)
    parameters. The result is clamped via :func:`enforce_planck_bound`.
    """

    # Avoid division by zero, retain vectorised evaluation.
    safe_rmax = np.where(rmax == 0.0, 1.0, rmax)
    sigma = alpha * epsilon / np.sqrt(np.abs(safe_rmax))
    return enforce_planck_bound(sigma)
