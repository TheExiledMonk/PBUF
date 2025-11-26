"""Utility helpers shared by the sanity modules."""

from __future__ import annotations

import numpy as np


def make_a_grid(n: int = 200) -> np.ndarray:
    """Return a log-spaced scale-factor grid from a≈10⁻⁹ to a=1."""

    return np.logspace(-9, 0, n)
