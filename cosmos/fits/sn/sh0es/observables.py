"""
Compute SH0ES supernova distance modulus predictions from cosmological models.

This module provides functions to calculate theoretical distance modulus
predictions μ(z) for cosmological models, which can be compared against
observed SH0ES supernova data, or extract H0 from the model for prior comparison.
"""

import numpy as np
from cosmos.helper.distances import luminosity_distance

def compute_sh0es_mu_model(model, z, M=0.0):
    """
    Compute predicted SN distance modulus μ_th(z) for a given cosmological model.

    The distance modulus is defined as:
    μ(z) = 5 * log10(D_L(z) / 10 pc) = 5 * log10(D_L(z) * 1e6) - 5

    where D_L(z) is the luminosity distance in Mpc.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with H(z) method
    z : array-like
        Redshift(s) at which to evaluate the model
    M : float, optional
        Absolute magnitude offset (nuisance parameter, degenerate with H0)

    Returns
    -------
    np.ndarray
        Model-predicted distance modulus μ(z) values

    Notes
    -----
    The absolute magnitude offset M accounts for calibration uncertainties
    and is degenerate with the Hubble constant H0. For SH0ES, M is typically
    fixed by Cepheid calibration.
    """
    # Compute luminosity distance in Mpc
    D_L = luminosity_distance(z, model)

    # Convert to distance modulus
    # D_L is in Mpc, we need D_L in units where 10 pc = 1
    # 1 Mpc = 1e6 pc, so D_L_Mpc * 1e6 gives distance in units of 10 pc
    mu_th = 5.0 * np.log10(D_L * 1e6) - 5.0 + M

    return mu_th


def extract_model_h0(model, params):
    """
    Extract H0 from model parameters.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model instance
    params : dict
        Model parameters containing H0

    Returns
    -------
    float
        H0 value in km/s/Mpc
    """
    # Prefer direct attributes on the model instance
    if hasattr(model, "h0"):
        value = getattr(model, "h0")
        if value is not None:
            return float(value)

    if hasattr(model, "h"):
        value = getattr(model, "h")
        if value is not None:
            return float(value) * 100.0

    if hasattr(model, "H0"):
        value = getattr(model, "H0")
        if value is not None:
            return float(value)

    # Fall back to parameters dict if provided
    if params is not None and "H0" in params:
        return float(params["H0"])

    raise ValueError("Could not extract H0 from model or parameter dictionary.")
