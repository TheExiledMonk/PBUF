"""PBUF-only numerical helpers (ported from cosmos_old)."""

from __future__ import annotations

import importlib
import math
from typing import Callable

from cosmos2.kernels.pbuf_utils import bisection_root_njit, simpson_integral_njit

C_LIGHT = 299_792.458  # km/s


def _simpson_integral_python(func: Callable[[float], float], lower: float, upper: float, n: int) -> float:
    if upper == lower:
        return 0.0
    h = (upper - lower) / n
    result = func(lower) + func(upper)
    for i in range(1, n):
        weight = 4.0 if i % 2 == 1 else 2.0
        result += weight * func(lower + i * h)
    return result * h / 3.0


def simpson_integral(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    n: int = 2048,
    assume_numba: bool = False,
) -> float:
    """
    Integrate `func` from `lower` to `upper` via Simpson's rule.

    If `assume_numba` is True, `func` must be numba-callable and the nopython
    integrator will be used; otherwise a pure-Python path is used.
    """

    if n <= 0 or n % 2 != 0:
        raise ValueError("Simpson integrator expects a positive even number of steps")
    dispatcher = _cpu_dispatcher()
    use_njit = assume_numba or (dispatcher is not None and isinstance(func, dispatcher))
    if use_njit:
        try:
            return float(simpson_integral_njit(func, float(lower), float(upper), int(n)))
        except Exception:
            # Fall back to Python path if the numba call fails or the integrand is not njit-compatible.
            if assume_numba:
                return _simpson_integral_python(func, float(lower), float(upper), int(n))
    return _simpson_integral_python(func, float(lower), float(upper), int(n))


def _bisection_root_python(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float,
    max_iter: int,
) -> float:
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


def bisection_root(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tol: float = 1e-8,
    max_iter: int = 10_000,
    assume_numba: bool = False,
) -> float:
    """Simple bisection solver for monotonic functions."""

    dispatcher = _cpu_dispatcher()
    use_njit = assume_numba or (dispatcher is not None and isinstance(func, dispatcher))
    if use_njit:
        try:
            return float(bisection_root_njit(func, float(lower), float(upper), float(tol), int(max_iter)))
        except Exception:
            if assume_numba:
                return float(_bisection_root_python(func, float(lower), float(upper), float(tol), int(max_iter)))
    return float(_bisection_root_python(func, float(lower), float(upper), float(tol), int(max_iter)))


def as_scale_factor(z: float) -> float:
    """Convert redshift to scale factor."""

    return 1.0 / (1.0 + z)


def as_redshift(a: float) -> float:
    """Convert scale factor to redshift."""

    return 1.0 / a - 1.0


def _cpu_dispatcher():
    try:
        module = importlib.import_module("numba.core.registry")
        return getattr(module, "CPUDispatcher", None)
    except Exception:
        return None


__all__ = [
    "C_LIGHT",
    "simpson_integral",
    "simpson_integral_njit",
    "bisection_root",
    "bisection_root_njit",
    "as_scale_factor",
    "as_redshift",
]
