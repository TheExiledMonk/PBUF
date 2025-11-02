"""
Compute χ² for isotropic BAO fits.

This module provides functions to compute chi-squared statistics
for isotropic BAO measurements using the standardized PBUF data format.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset
from cosmos.fits._dataset_loader import load_bao_iso_dataset
from .observables import compute_bao_dv_over_rd

def chi_squared_bao_iso(model, data=None):
    """
    Compute χ² for isotropic BAO measurements using standardized format.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with DV_over_rd method
    data : dict or None
        Standardized BAO isotropic dataset (PBUF Data Object v1). If None, loads default.

    Returns
    -------
    float
        χ² value

    Notes
    -----
    Uses the standardized PBUF data format with keys: name, type, z, obs, err, cov, meta.
    Computes D_V(z)/r_d predictions and compares with observations.
    """
    # Load and validate data using standard schema
    if data is None:
        data = load_bao_iso_dataset()

    # Ensure data follows PBUF Data Object v1 schema
    data = ensure_standard_dataset(data, "BAO_ISO")

    # Extract standardized fields
    z = data["z"]
    obs = data["obs"]  # D_V(z)/r_d measurements
    err = data["err"]
    cov = data["cov"]

    # Compute model predictions
    pred = compute_bao_dv_over_rd(model, z)

    # Compute residuals
    diff = obs - pred

    # Compute χ²
    if cov is not None:
        # Use full covariance matrix
        cov_inv = np.linalg.inv(cov)
        chi2 = float(diff.T @ cov_inv @ diff)
    else:
        # Use diagonal errors
        chi2 = float(np.sum((diff / err) ** 2))

    return max(chi2, 0.0)
