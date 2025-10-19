"""
Physical constants used across the project.

All values are expressed in SI units unless otherwise noted.
"""

from __future__ import annotations

# Speed of light in vacuum (m s^-1)
C_LIGHT = 299_792_458.0

# Gravitational constant (m^3 kg^-1 s^-2)
G_NEWTON = 6.67430e-11

# Reduced Planck constant (J s)
H_BAR = 1.054_571_817e-34

# Planck constant (J s)
H_PLANCK = 6.626_070_15e-34

# Conversion factors
MPC_TO_M = 3.085_677_581_491_367e22  # megaparsec to meters
KM_TO_M = 1_000.0

# Cosmological units
HUBBLE_100 = 100.0 * KM_TO_M / MPC_TO_M  # 100 km/s/Mpc converted to s^-1

# Derived useful quantities
FOUR_PI = 12.566_370_614_359_172

# CMB and radiation parameters
TCMB = 2.7255  # Cosmic microwave background temperature today [K]
NEFF = 3.046  # Effective number of relativistic neutrino species


def omega_gamma_h2(tcmb: float = TCMB) -> float:
    """
    Photon density parameter multiplied by h^2.

    Parameters
    ----------
    tcmb : float
        CMB temperature today in Kelvin.
    """

    ratio = tcmb / 2.7255
    return 2.469e-5 * ratio**4


def omega_radiation_h2(tcmb: float = TCMB, neff: float = NEFF) -> float:
    """
    Total radiation density parameter (photons + neutrinos) times h^2.
    """

    return omega_gamma_h2(tcmb) * (1.0 + 0.2271 * neff)
