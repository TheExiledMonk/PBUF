"""
Compute χ² for supernova data fits.

This module provides functions to calculate chi-squared statistics
between theoretical model predictions and observed supernova data
using the standardized PBUF data format.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset
from cosmos.fits._dataset_loader import load_sn_dataset
from .observables import compute_sn_mu_model

def _fit_absolute_magnitude(diff, weights):
    """
    Fit the absolute magnitude offset (M) analytically using weighted least squares.
    """
    denom = np.sum(weights)
    if denom <= 0:
        return 0.0
    return np.sum(weights * diff) / denom


def _fit_absolute_magnitude_cov(diff, cov_inv):
    """
    Fit the absolute magnitude offset (M) when using a full covariance matrix.
    """
    ones = np.ones_like(diff)
    denom = float(ones.T @ cov_inv @ ones)
    if denom <= 0:
        return 0.0
    numer = float(ones.T @ cov_inv @ diff)
    return numer / denom


def chi_squared_sn(model, M=None, data=None, fit_magnitude=True):
    """
    Compute SN χ² for a model and dataset using standardized format.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with H(z) method
    M : float, optional
        Absolute magnitude offset (nuisance parameter)
    data : dict, optional
        Standardized SN dataset (PBUF Data Object v1). If None, loads default.

    Returns
    -------
    float
        χ² value

    Notes
    -----
    Uses the standardized PBUF data format with keys: name, type, z, obs, err, cov, meta.
    The covariance matrix should include both statistical and systematic uncertainties.
    """
    # Load and validate data using standard schema
    if data is None:
        data = load_sn_dataset()

    # Ensure data follows PBUF Data Object v1 schema
    data = ensure_standard_dataset(data, "SN")

    # Extract standardized fields
    z = data["z"]
    obs = data["obs"]  # Distance modulus measurements
    err = data["err"]
    cov = data["cov"]

    # Compute model predictions
    mu_model = compute_sn_mu_model(model, z, M=0.0 if M is None else M)

    # Compute residuals without the magnitude offset yet
    diff_raw = obs - mu_model

    if (M is None) and fit_magnitude:
        # Solve for the best-fit absolute magnitude offset analytically
        if cov is not None:
            try:
                cov_inv = np.linalg.inv(cov)
                M = _fit_absolute_magnitude_cov(diff_raw, cov_inv)
            except np.linalg.LinAlgError:
                weights = 1.0 / (err**2)
                M = _fit_absolute_magnitude(diff_raw, weights)
        else:
            weights = 1.0 / (err**2)
            M = _fit_absolute_magnitude(diff_raw, weights)
    elif M is None:
        M = 0.0

    diff = diff_raw - M

    # Compute χ² based on whether covariance is available
    if cov is not None:
        try:
            cov_inv = np.linalg.inv(cov)
            chi2 = float(diff.T @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            chi2 = float(np.sum(diff**2 / err**2))
    else:
        chi2 = float(np.sum(diff**2 / err**2))

    return max(chi2, 0.0)
