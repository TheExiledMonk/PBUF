"""
Utility helpers for PBUF numerical work.
These are generic array/scalar adapters and should not contain physics.
"""

import numpy as np

def _as_array(value):
    """
    Convert input to numpy array (dtype=float).
    Returns (array, was_scalar_bool).
    """
    arr = np.asarray(value, dtype=float)
    is_scalar = (np.isscalar(value) or arr.shape == ())
    return arr, is_scalar

def _maybe_scalar(value, was_scalar):
    """
    Convert value (array) back to float if original input was scalar.
    """
    if was_scalar:
        return float(value)
    return value
