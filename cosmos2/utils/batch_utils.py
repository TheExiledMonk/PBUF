"""Batch utilities for cosmos2."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


def clamp_to_bounds(params: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """
    Clamp each parameter to its [low, high] interval if provided.
    """
    clamped = {}
    for key, value in params.items():
        if key in bounds:
            low, high = bounds[key]
            clamped[key] = min(max(value, low), high)
        else:
            clamped[key] = value
    return clamped
