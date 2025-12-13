"""FFTLog-inspired Hankel transforms for WL ξ± with Bessel fallback."""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Tuple

import numpy as np

from .bessel import bessel_j0, bessel_jn

_TWO_PI = 2.0 * math.pi


def _hankel_integral(cl: np.ndarray, ell: np.ndarray, theta: float, order: int) -> float:
    """
    Compute ∫ dℓ ℓ C_ℓ J_order(ℓ θ) / (2π) using log-friendly trapezoidal weighting.
    """

    arg = ell * theta
    if order == 0:
        bessel = bessel_j0(arg)
    else:
        bessel = bessel_jn(order, arg)
    integrand = ell * cl * bessel
    return float(np.trapezoid(integrand, ell) / _TWO_PI)


def xi_from_cls_fftlog(
    cls: np.ndarray,
    ell_grid: np.ndarray,
    theta_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hankel-transform C_ℓ → ξ± using FFTLog-friendly integration on a log ℓ grid.

    Falls back to the analytic Bessel helper internally; callers should keep a
    Bessel-only route as a safety net for constrained environments.
    """

    xi_plus = np.zeros((cls.shape[0], cls.shape[1], theta_bins.size), dtype=float)
    xi_minus = np.zeros_like(xi_plus)
    ell = np.asarray(ell_grid, dtype=float)
    for t_idx, theta in enumerate(theta_bins):
        for i in range(cls.shape[0]):
            for j in range(cls.shape[1]):
                cl_ij = np.asarray(cls[i, j], dtype=float)
                xi_plus[i, j, t_idx] = _hankel_integral(cl_ij, ell, float(theta), order=0)
                xi_minus[i, j, t_idx] = _hankel_integral(cl_ij, ell, float(theta), order=4)
    return xi_plus, xi_minus


def safe_xi_fftlog(
    cls: np.ndarray,
    ell_grid: np.ndarray,
    theta_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Guarded FFTLog transform that falls back to a warning on failure.
    """

    try:
        return xi_from_cls_fftlog(cls, ell_grid, theta_bins)
    except Exception as exc:  # pragma: no cover - defensive guard
        warnings.warn(f"FFTLog xi± transform failed, consider Bessel fallback: {exc}", RuntimeWarning, stacklevel=2)
        raise


__all__ = ["xi_from_cls_fftlog", "safe_xi_fftlog"]
