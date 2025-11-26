"""LCDM-local numerical helpers."""

from __future__ import annotations

from typing import Callable

C_LIGHT = 299_792.458


def simpson_integral(func: Callable[[float], float], lower: float, upper: float, *, n: int = 2048) -> float:
    if upper == lower:
        return 0.0

    if n <= 0 or n % 2 != 0:
        raise ValueError("Simpson integrator expects a positive even number of steps")

    h = (upper - lower) / n
    result = func(lower) + func(upper)

    for i in range(1, n):
        weight = 4.0 if i % 2 else 2.0
        result += weight * func(lower + i * h)

    return result * h / 3.0
