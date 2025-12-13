"""Lightweight Bessel function helpers (J0, J1, Jn) without SciPy."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def _series_j0(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    x4 = x2 * x2
    x6 = x4 * x2
    return 1.0 - 0.25 * x2 + (1.0 / 64.0) * x4 - (1.0 / 2304.0) * x6


def _series_j1(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    x7 = x5 * x2
    return 0.5 * x - (1.0 / 16.0) * x3 + (1.0 / 384.0) * x5 - (1.0 / 18432.0) * x7


def bessel_j0(x: Iterable[float] | float | np.ndarray) -> np.ndarray:
    """Approximate J0 using a power-series for small |x| and asymptotics otherwise."""
    arr = np.asarray(x, dtype=float)
    mask_small = np.abs(arr) < 3.0
    out = np.empty_like(arr, dtype=float)
    if np.any(mask_small):
        out[mask_small] = _series_j0(arr[mask_small])
    if np.any(~mask_small):
        x_big = arr[~mask_small]
        out[~mask_small] = _SQRT_2_OVER_PI * np.cos(x_big - math.pi / 4.0) / np.sqrt(x_big)
    return out


def bessel_j1(x: Iterable[float] | float | np.ndarray) -> np.ndarray:
    """Approximate J1 using a power-series for small |x| and asymptotics otherwise."""
    arr = np.asarray(x, dtype=float)
    mask_small = np.abs(arr) < 3.0
    out = np.empty_like(arr, dtype=float)
    if np.any(mask_small):
        out[mask_small] = _series_j1(arr[mask_small])
    if np.any(~mask_small):
        x_big = arr[~mask_small]
        out[~mask_small] = _SQRT_2_OVER_PI * np.cos(x_big - 3.0 * math.pi / 4.0) / np.sqrt(x_big)
    return out


def bessel_jn(n: int, x: Iterable[float] | float | np.ndarray) -> np.ndarray:
    """
    Compute J_n via upward recurrence seeded by J0/J1 approximations.

    This avoids heavy dependencies; accuracy is sufficient for WL Hankel-like
    transforms used in coarse diagnostic pipelines.
    """
    if n < 0:
        raise ValueError("Order n must be non-negative.")
    arr = np.asarray(x, dtype=float)
    if n == 0:
        return bessel_j0(arr)
    if n == 1:
        return bessel_j1(arr)

    jnm2 = bessel_j0(arr)
    jnm1 = bessel_j1(arr)
    for m in range(1, n):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (2.0 * m / np.where(arr == 0.0, np.inf, arr)) * jnm1
        jn = term - jnm2
        jn = np.where(arr == 0.0, 0.0, jn)
        jnm2, jnm1 = jnm1, jn
    return jnm1


__all__ = ["bessel_j0", "bessel_j1", "bessel_jn"]
