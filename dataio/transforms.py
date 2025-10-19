"""
Data transforms for preparing observational inputs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def to_numpy(values: Iterable[float]) -> np.ndarray:
    """Convert an iterable to a float64 numpy array."""

    return np.asarray(list(values), dtype=float)


def normalise_columns(df, mapping):
    """
    Rename DataFrame columns according to a mapping.

    Provided for future datasets where naming conventions differ.
    """

    return df.rename(columns=mapping)
