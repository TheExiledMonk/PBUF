"""
Utility helpers for LCDM numerical operations.
"""

import numpy as np

def _as_array(value):
    arr = np.asarray(value, dtype=float)
    is_scalar = np.isscalar(value) or arr.shape == ()
    return arr, is_scalar

def _maybe_scalar(value, was_scalar):
    return float(value) if was_scalar else value
