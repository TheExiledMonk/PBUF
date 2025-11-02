"""
Compute χ² for cosmic chronometer (CC) H(z) data.

This module provides functions to compute the chi-squared statistic
between cosmological model predictions and cosmic chronometer
H(z) measurements using the standardized PBUF data format.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset
from cosmos.fits._dataset_loader import load_cc_dataset
from .observables import compute_cc_hubble_model

def chi_squared_cc(model, data=None):
    """
    Compute χ² between model predictions and cosmic chronometer H(z) data.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with H(z) method
    data : dict or None
        Standardized CC dataset (PBUF Data Object v1). If None, loads default.

    Returns
    -------
    float
        χ² value

    Notes
    -----
    Uses the standardized PBUF data format with keys: name, type, z, obs, err, cov, meta.
    """
    # Load and validate data using standard schema
    if data is None:
        data = load_cc_dataset()

    # Ensure data follows PBUF Data Object v1 schema
    data = ensure_standard_dataset(data, "CC")

    # Extract standardized fields
    z = data["z"]
    obs = data["obs"]  # H(z) measurements
    err = data["err"]
    cov = data["cov"]

    # Compute model predictions
    Hz_model = compute_cc_hubble_model(model, z)

    # Compute residuals
    diff = obs - Hz_model

    # Compute χ²
    if cov is not None:
        # Use full covariance matrix
        cov_inv = np.linalg.inv(cov)
        chi2 = float(diff.T @ cov_inv @ diff)
    elif err is not None:
        # Use diagonal errors
        chi2 = float(np.sum((diff / err) ** 2))
    else:
        raise ValueError("CC dataset must provide either a covariance matrix or 1σ uncertainties.")

    return max(chi2, 0.0)
