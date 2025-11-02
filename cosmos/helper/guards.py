"""
Validation and guard functions for cosmological parameters.
"""

import numpy as np
from .constants import T_CMB0, NEFF, TNU_TRATIO

def check_scale_factor(a):
    """Validate scale factor is in physical range."""
    a_arr = np.asarray(a, dtype=float)
    if np.any((a_arr <= 0.0) | (a_arr > 1.0)):
        raise ValueError(f"Scale factor a={a} out of physical range (0 < a <= 1)")

def check_expansion_rate(H, z=None):
    """Validate expansion rate is positive."""
    H_arr = np.asarray(H, dtype=float)
    if np.any(H_arr <= 0.0):
        context = f" at z={z}" if z is not None else ""
        raise ValueError(f"Expansion rate H={H} must be positive{context}")

def assert_all_params_in_bounds(params):
    """
    Generic parameter bounds checking.

    This is a placeholder - in a full implementation you'd want
    more sophisticated bounds checking based on physical constraints.
    """
    for name, value in params.items():
        if not np.isfinite(value):
            raise ValueError(f"Parameter {name}={value} is not finite")

        # Add specific bounds as needed
        if name == "omega_m" and value < 0.0:
            raise ValueError(f"omega_m={value} must be >= 0")
        if name == "omega_k" and abs(value) > 10.0:  # reasonable curvature bound
            raise ValueError(f"omega_k={value} magnitude too large")
def infer_omega_r(T_cmb, h):
    """
    Infer radiation density parameter today from CMB temperature.

    Standard calculation includes photons and neutrinos.

    Parameters
    ----------
    T_cmb : float
        CMB temperature in Kelvin
    h : float
        Dimensionless Hubble parameter

    Returns
    -------
    float
        omega_r = omega_gamma + omega_nu today
    """
    # Photon contribution: omega_gamma = (pi^2/15) * (kT/hc)^3 / (rho_crit)
    # In practice: omega_gamma = 2.47e-5 * (T_cmb/T_cmb0)^4 * h^-2

    T_ratio = T_cmb / T_CMB0
    omega_gamma = 2.47e-5 * (T_ratio**4) / (h**2)

    # Neutrino contribution (assuming massless neutrinos)
    # omega_nu = (7/8) * (4/11)^{4/3} * N_eff * omega_gamma
    omega_nu = (7.0/8.0) * (TNU_TRATIO**4) * NEFF * omega_gamma
    
    return omega_gamma + omega_nu
