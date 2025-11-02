"""
Compute Pantheon supernova distance modulus predictions from cosmological models.

This module provides functions to calculate theoretical distance modulus
predictions μ(z) for cosmological models, which can be compared against
observed Pantheon supernova data.
"""

import numpy as np
from cosmos.helper.distances import luminosity_distance
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

def compute_pantheon_mu_model(model, z, M=0.0):
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
    and is degenerate with the Hubble constant H0. It is typically treated
    as a nuisance parameter in supernova fitting.
    """
 
    # Ensure z is a numpy array
    z_array = np.asarray(z, dtype=np.float64)
   
    # Compute luminosity distance in Mpc
    try:
        D_L = luminosity_distance(z_array, model)

    except Exception as e:

        raise

    try:
        # Convert to distance modulus
        # D_L is in Mpc, we need D_L in units where 10 pc = 1
        # 1 Mpc = 1e6 pc, so D_L_Mpc * 1e6 gives distance in units of 10 pc
        mu_th = 5.0 * np.log10(D_L * 1e6) - 5.0 + M

        return mu_th
    except Exception as e:

        raise


def extract_model_h0(model, params):
    """
    Extract the Hubble constant H0 from a cosmological model.
    
    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model
    params : dict
        Parameter dictionary (used as fallback)
        
    Returns
    -------
    float
        Hubble constant H0 in km/s/Mpc
    """
    # Try to get H0 from the model if it has the attribute
    if hasattr(model, 'H0'):
        return model.H0 * 100.0  # Convert from 100 km/s/Mpc to km/s/Mpc if needed
    
    # Fall back to params if H0 is not found in model
    if 'H0' in params:
        return params['H0']
        
    # Default fallback
    return 70.0  # km/s/Mpc

    return mu_th
