"""
ΛCDM background equations.

Friedmann equation:
    H^2(a) = H0^2 * [
        Ω_m a^-3 +
        Ω_r a^-4 +
        Ω_k a^-2 +
        Ω_Λ
    ]
"""

import numpy as np
from .utils import _as_array, _maybe_scalar

def E_lcdm_a(a, Om0, Or0, Ok0, Ol0):
    """
    Dimensionless expansion rate E(a) = H(a)/H0.
    """
    a_array, was_scalar = _as_array(a)

    if np.any((a_array <= 0.0) | (a_array > 1.0)):
        raise ValueError(f"Invalid scale factor a={a} (must be 0 < a ≤ 1)")

    E2 = (
        Om0 * a_array**(-3.0) +
        Or0 * a_array**(-4.0) +
        Ok0 * a_array**(-2.0) +
        Ol0
    )

    if np.any(E2 <= 0.0):
        raise ValueError(f"Unphysical state at a={a}: H^2/H0^2={E2}")

    return _maybe_scalar(np.sqrt(E2), was_scalar)


def H_lcdm_a(a, H0, Om0, Or0, Ok0, Ol0):
    """
    Hubble parameter as function of scale factor a [km/s/Mpc].
    """
    return H0 * E_lcdm_a(a, Om0, Or0, Ok0, Ol0)


def H_lcdm_z(z, H0, Om0, Or0, Ok0, Ol0):
    """
    Hubble parameter as function of redshift z [km/s/Mpc].
    """
    z_array, was_scalar = _as_array(z)
    one_plus_z = 1.0 + z_array
    if np.any(one_plus_z <= 0.0):
        raise ValueError(f"Invalid z={z} (1+z must be > 0).")

    a_array = 1.0 / one_plus_z
    Hz = H_lcdm_a(a_array, H0, Om0, Or0, Ok0, Ol0)
    return _maybe_scalar(Hz, was_scalar)
