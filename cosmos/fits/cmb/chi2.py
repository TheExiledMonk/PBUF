"""
Compute χ² for CMB distance priors.
"""

import numpy as np
from .observables import cmb_observables
from .data_loader import load_planck_priors

def chi_squared_cmb(model, priors=None):
    """
    Compute χ² between model predictions and Planck priors.

    Parameters
    ----------
    model : LCDM or PBUF instance
    priors : dict or None
        Output from load_planck_priors(). If None, load defaults.

    Returns
    -------
    float : χ² value
    """
    if priors is None:
        priors = load_planck_priors()

    obs = cmb_observables(model)
    data = np.array([obs["R"], obs["la"], obs["theta_star"]])
    mean = np.array([
        priors["mean"]["R"],
        priors["mean"]["la"],
        priors["mean"]["theta_star"],
    ])
    diff = data - mean
    cov = priors["cov"]
    cov_inv = np.linalg.inv(0.5 * (cov + cov.T))  # symmetrize
    chi2 = float(diff.T @ cov_inv @ diff)
    return max(chi2, 0.0)
