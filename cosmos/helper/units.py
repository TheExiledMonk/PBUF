"""
Unit conversion constants for cosmology.
"""

import numpy as np

# Mpc to meters conversion
MPC_TO_M = 3.0856775814913673e22  # 1 Mpc in meters

# km to meters
KM_TO_M = 1000.0

# Second to megayear conversion (for reference)
SEC_TO_MYR = 3.15576e13  # seconds in a megayear

# Critical density prefactor [kg/m^3 / (m/s)^2]
# rho_crit = RHO_CRIT_PREFAC * H0_SI^2
RHO_CRIT_PREFAC = 3.0 / (8.0 * 3.141592653589793)  # 3/(8π) in SI units

# Hubble distance conversion
# H0 in km/s/Mpc corresponds to H0_SI in s^-1
def h0_to_si(h0_kmsmpc):
    """Convert H0 in km/s/Mpc to SI units (s^-1)."""
    return h0_kmsmpc * KM_TO_M / MPC_TO_M

def h_to_h0(h):
    """Convert dimensionless h to H0 in km/s/Mpc."""
    return 100.0 * h
