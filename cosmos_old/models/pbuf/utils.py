"""PBUF-only numerical helpers."""

from __future__ import annotations

import math
from typing import Callable

C_LIGHT = 299_792.458  # km/s


def simpson_integral(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    n: int = 2048,
) -> float:
    """
    Integrate `func` from `lower` to `upper` via Simpson's rule.

    The integrator is intentionally local to the model to avoid shared physics
    code. Increase `n` if higher precision is required.
    """

    if upper == lower:
        return 0.0

    if n <= 0 or n % 2 != 0:
        raise ValueError("Simpson integrator expects a positive even number of steps")

    h = (upper - lower) / n
    result = func(lower) + func(upper)

    for i in range(1, n):
        weight = 4.0 if i % 2 == 1 else 2.0
        result += weight * func(lower + i * h)

    return result * h / 3.0


def bisection_root(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> float:
    """Simple bisection solver for monotonic functions."""

    f_low = func(lower)
    f_high = func(upper)

    if f_low == 0.0:
        return lower

    if f_high == 0.0:
        return upper

    if f_low * f_high > 0:
        raise ValueError("Bisection solver requires a bracketing interval")

    a, b = lower, upper
    fa, fb = f_low, f_high

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = func(mid)

        if abs(fm) < tol or abs(b - a) < tol:
            return mid

        if fa * fm < 0.0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    raise RuntimeError("Bisection solver did not converge within max iterations")


def as_scale_factor(z: float) -> float:
    """Convert redshift to scale factor."""

    return 1.0 / (1.0 + z)


def as_redshift(a: float) -> float:
    """Convert scale factor to redshift."""

    return 1.0 / a - 1.0
