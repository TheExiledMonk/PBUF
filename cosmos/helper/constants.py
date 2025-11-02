"""
Physical constants for cosmology calculations.
"""

# CMB temperature today [K]
T_CMB0 = 2.7255

# Effective number of neutrino species
NEFF = 3.046

# Standard neutrino temperature ratio (T_nu / T_gamma)
TNU_TRATIO = (4/11)**(1/3)

# Speed of light [m/s]
C_LIGHT = 299792458.0

# Gravitational constant [m^3 kg^-1 s^-2]
G_GRAVITY = 6.67430e-11

# Distance conversion: 1 Mpc = 10^6 pc
MPC_TO_PC = 1e6

# Speed of light [km/s] (for H(z) and distance integrals)
C_KM_S = C_LIGHT / 1000.0