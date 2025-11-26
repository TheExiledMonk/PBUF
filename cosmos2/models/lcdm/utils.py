"""LCDM-only numerical helpers (mirrors PBUF layout)."""

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

    Structure matches the PBUF helper; math is standard Simpson.
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


def as_scale_factor(z: float) -> float:
    """Convert redshift to scale factor."""

    return 1.0 / (1.0 + z)


def as_redshift(a: float) -> float:
    """Convert scale factor to redshift."""

    return 1.0 / a - 1.0


__all__ = ["C_LIGHT", "simpson_integral", "as_scale_factor", "as_redshift"]
