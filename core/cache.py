"""
Simple memoisation helpers for expensive cosmology calculations.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Tuple

import numpy as np


def cached_trapz(func: Callable[..., np.ndarray], maxsize: int = 32) -> Callable[..., np.ndarray]:
    """
    Decorator caching function outputs keyed by array hashes.

    Useful for expensive integral grids that are reused across fits.
    """

    @lru_cache(maxsize=maxsize)
    def _cached(key: Tuple[Tuple[float, ...], ...]) -> np.ndarray:
        args = [np.asarray(component) for component in key]
        return func(*args)

    def wrapper(*arrays: np.ndarray) -> np.ndarray:
        key = tuple(tuple(np.asarray(arr).ravel()) for arr in arrays)
        return _cached(key)

    return wrapper
