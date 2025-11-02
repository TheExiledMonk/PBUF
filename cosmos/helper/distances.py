"""
Distance calculations for cosmology.

This module provides functions to compute various cosmological distances
and scales needed for CMB analysis and other cosmological calculations.
"""

import numpy as np
from scipy import integrate
from .constants import C_LIGHT


def _as_scalar(value):
    """Convert array-like values to a Python float."""
    arr = np.asarray(value, dtype=float)
    return float(arr.reshape(-1)[0])


def _get_omega_k(model):
    """
    Extract present-day curvature density Ω_k from the model if available.

    Falls back to zero curvature when the attribute cannot be found.
    """
    candidate_attrs = ("omega_k", "Ok0", "Omega_k", "OmegaK0")
    for attr in candidate_attrs:
        if hasattr(model, attr):
            try:
                return float(getattr(model, attr))
            except (TypeError, ValueError):
                continue

    params = getattr(model, "parameters", None)
    if callable(params):
        try:
            param_dict = params()
            for key in candidate_attrs:
                if key in param_dict:
                    return float(param_dict[key])
        except Exception:
            pass

    return 0.0


def _line_of_sight_comoving_distance_array(z_array, model):
    """
    Compute the line-of-sight comoving distance χ(z) for an array of redshifts.
    """
    def integrand(z_prime):
        Hz = model.H(z_prime)
        return C_LIGHT / (1000.0 * Hz)  # Convert to Mpc

    distances = np.empty_like(z_array, dtype=float)
    for idx, z_val in enumerate(z_array):
        result, _ = integrate.quad(integrand, 0.0, float(z_val))
        distances[idx] = result
    return distances


def sound_horizon(model, z_drag=None):
    """
    Compute the sound horizon at recombination or drag epoch.

    The sound horizon is the distance that a sound wave could have traveled
    from the Big Bang until recombination (or drag epoch).

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with H(z) method
    z_drag : float or None
        Redshift of baryon drag epoch. If None, use recombination (z=1089.92)

    Returns
    -------
    float
        Sound horizon in Mpc
    """
    if z_drag is None:
        z_drag = 1089.92  # Planck 2018 convention

    # Integrand: c_s / H(z)
    # Sound speed: cs = c / sqrt(3) during radiation domination
    # Convert: c in m/s, H in km/s/Mpc, result in Mpc
    def integrand(z):
        Hz = model.H(z)  # km/s/Mpc
        cs = 1.0 / np.sqrt(3.0)  # dimensionless
        # C_LIGHT in m/s, convert to km/s by dividing by 1000
        # Then: (km/s) / (km/s/Mpc) = Mpc
        return (C_LIGHT / 1000.0) * cs / Hz

    # Integrate from z_drag to infinity
    result, _ = integrate.quad(integrand, z_drag, np.inf)
    return result


def transverse_comoving_distance(z, model):
    """
    Compute the transverse comoving distance to redshift z.

    For non-flat geometries this applies the appropriate sin/sinh curvature
    factor to the line-of-sight comoving distance.

    Parameters
    ----------
    z : float or array
        Redshift(s)
    model : LCDM or PBUF instance
        Cosmological model with H(z') method

    Returns
    -------
    float or array
        Transverse comoving distance in Mpc
    """
    z_array = np.atleast_1d(np.asarray(z, dtype=float))
    chi = _line_of_sight_comoving_distance_array(z_array, model)

    omega_k = _get_omega_k(model)
    h0 = _as_scalar(model.H(0.0))
    c_km_s = C_LIGHT / 1000.0

    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError("Model returned non-physical H0 when computing distances.")

    if np.abs(omega_k) < 1e-12:
        dm = chi
    else:
        sqrt_ok = np.sqrt(abs(omega_k))
        prefac = c_km_s / (sqrt_ok * h0)
        argument = sqrt_ok * h0 * chi / c_km_s
        if omega_k > 0.0:
            dm = prefac * np.sinh(argument)
        else:
            dm = prefac * np.sin(argument)

    if z_array.size == 1:
        return float(dm[0])
    return dm


def comoving_distance(z, model):
    """
    Compute the line-of-sight comoving distance to redshift z.

    D_C(z) = (c/H0) * ∫_0^z dz'/H(z')

    This is the same as transverse comoving distance for flat universes.

    Parameters
    ----------
    z : float or array
        Redshift(s)
    model : LCDM or PBUF instance
        Cosmological model with H(z') method

    Returns
    -------
    float or array
        Comoving distance in Mpc
    """
    z_array = np.atleast_1d(np.asarray(z, dtype=float))
    chi = _line_of_sight_comoving_distance_array(z_array, model)
    if z_array.size == 1:
        return float(chi[0])
    return chi


def luminosity_distance(z, model):
    """
    Compute the luminosity distance to redshift z.

    D_L(z) = (1 + z) * D_M(z)

    Parameters
    ----------
    z : float or array
        Redshift(s)
    model : LCDM or PBUF instance
        Cosmological model

    Returns
    -------
    float or array
        Luminosity distance in Mpc
    """
    z_array = np.atleast_1d(np.asarray(z, dtype=float))
    dm = transverse_comoving_distance(z_array, model)
    dl = (1.0 + z_array) * dm
    if z_array.size == 1:
        return float(dl[0])
    return dl
