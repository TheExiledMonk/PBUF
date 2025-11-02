"""
Validation and sanity checks for PBUF parameters and background evolution.
"""

import numpy as np
from ..helper.guards import (
    check_scale_factor,
    check_expansion_rate,
    assert_all_params_in_bounds,
)

def validate_pbuf_parameters(omega_m, omega_k, omega_r,
                             alpha, Rmax, eps0, k_sat,
                             n_alpha, n_eps, n_R,
                             h):
    """
    Validate that all cosmological / elastic parameters are physical.

    Required physical expectations:
    - k_sat > 0 (values slightly above 1 allowed by restored model)
    - Rmax > 0
    - alpha ≥ 0 (elastic sector can be dialled off)
    - eps0 > 0
    - h > 0
    """

    params = {
        "omega_m": omega_m,
        "omega_k": omega_k,
        "omega_r": omega_r,
        "alpha": alpha,
        "Rmax": Rmax,
        "eps0": eps0,
        "k_sat": k_sat,
        "n_alpha": n_alpha,
        "n_eps": n_eps,
        "n_R": n_R,
        "h": h,
    }
    assert_all_params_in_bounds(params)

    if not np.isfinite(k_sat) or (k_sat <= 0.0):
        raise ValueError(f"k_sat={k_sat} is unphysical (require k_sat > 0).")

    if not np.isfinite(Rmax) or (Rmax <= 0.0):
        raise ValueError(f"Rmax={Rmax} must be > 0.")

    if not np.isfinite(alpha) or (alpha < 0.0):
        raise ValueError(f"alpha={alpha} must be ≥ 0.")

    if not np.isfinite(eps0) or (eps0 <= 0.0):
        raise ValueError(f"eps0={eps0} must be > 0.")

    if not np.isfinite(h) or (h <= 0.0):
        raise ValueError(f"h={h} must be > 0.")

    for name, value in (("n_alpha", n_alpha), ("n_eps", n_eps), ("n_R", n_R)):
        if not np.isfinite(value):
            raise ValueError(f"{name}={value} must be finite.")

    # omega_r can be inferred from T_cmb if None upstream; if provided, it must be ≥0
    if not np.isfinite(omega_r) or (omega_r < 0.0):
        raise ValueError(f"omega_r={omega_r} must be ≥ 0.")

    # no explicit bound on omega_m, omega_k beyond assert_all_params_in_bounds
    # (you can tighten later if desired)

def validate_scale_and_rate(a, Hz, z=None):
    """
    Wrapper used by model methods to ensure scale factor + expansion rate are sane.
    """
    check_scale_factor(a)
    # For H(z), we typically validate positivity. z is only for error context.
    check_expansion_rate(Hz, (0.0 if z is None else z))
