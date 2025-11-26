"""Numba-backed utility kernels for PBUF."""

from __future__ import annotations

import numba


@numba.njit(cache=True)
def simpson_integral_njit(func, lower: float, upper: float, n: int = 2048) -> float:
    """
    Nopython Simpson integrator. Expects `func` to be a numba-callable taking a float and returning a float.
    """

    if upper == lower:
        return 0.0
    if n <= 0 or n % 2 != 0:
        return 0.0

    h = (upper - lower) / n
    result = func(lower) + func(upper)
    for i in range(1, n):
        weight = 4.0 if i % 2 == 1 else 2.0
        result += weight * func(lower + i * h)
    return result * h / 3.0


@numba.njit(cache=True)
def bisection_root_njit(func, lower: float, upper: float, tol: float = 1e-8, max_iter: int = 10_000) -> float:
    """
    Nopython bisection solver for monotonic functions. `func` must be numba-callable.
    """

    f_low = func(lower)
    f_high = func(upper)

    if f_low == 0.0:
        return lower
    if f_high == 0.0:
        return upper
    if f_low * f_high > 0:
        return lower

    a = lower
    b = upper
    fa = f_low
    fb = f_high

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = func(mid)
        if abs(fm) < tol or abs(b - a) < tol:
            return mid
        if fa * fm < 0.0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    return 0.5 * (a + b)


__all__ = ["simpson_integral_njit", "bisection_root_njit"]
