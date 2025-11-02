"""
Validation and physical sanity checks for ΛCDM cosmological parameters.
"""

import numpy as np
from ..helper.guards import (
    check_scale_factor,
    check_expansion_rate,
    assert_all_params_in_bounds,
)

def validate_lcdm_parameters(omega_m, omega_r, omega_lambda, omega_k, h):
    """
    Ensure ΛCDM parameters are within physical bounds.

    Rules:
    - all Ω_i finite
    - Ω_m > 0
    - Ω_r ≥ 0
    - Ω_Λ ≥ 0
    - |Ω_k| small (not enforced hard here)
    - h > 0
    """

    params = {
        "omega_m": omega_m,
        "omega_r": omega_r,
        "omega_lambda": omega_lambda,
        "omega_k": omega_k,
        "h": h,
    }
    assert_all_params_in_bounds(params)

    if (not np.isfinite(h)) or (h <= 0.0):
        raise ValueError(f"h={h} invalid: must be > 0.")
    if (not np.isfinite(omega_m)) or (omega_m <= 0.0):
        raise ValueError(f"omega_m={omega_m} invalid: must be > 0.")
    if (not np.isfinite(omega_r)) or (omega_r < 0.0):
        raise ValueError(f"omega_r={omega_r} invalid: must be ≥ 0.")
    if (not np.isfinite(omega_lambda)) or (omega_lambda < 0.0):
        raise ValueError(f"omega_lambda={omega_lambda} invalid: must be ≥ 0.")

def validate_scale_and_rate(a, Hz, z=None):
    check_scale_factor(a)
    check_expansion_rate(Hz, (0.0 if z is None else z))
