"""
Compute χ² for SH0ES supernova data fits.

This module provides functions to calculate chi-squared statistics
between theoretical model predictions and observed SH0ES supernova data,
or use SH0ES as a Gaussian prior on H0.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset
from .loader import load_sh0es_data
from .observables import compute_sh0es_mu_model, extract_model_h0

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


def chi2_sn_sh0es(model_func, params):
    """
    Compute SH0ES SN χ² for a model and dataset using standardized format.

    Parameters
    ----------
    model_func : callable
        Function that takes parameter dict and returns model instance
    params : dict
        Cosmological parameters

    Returns
    -------
    dict
        {
            "chi2": chi2_value,
            "status": "valid" or "fail",
            "n_data": N,
            "dataset": "SN_SH0ES",
        }

    Notes
    -----
    This function supports two modes:
    1. Prior mode: Uses SH0ES H0 measurement as Gaussian prior
    2. Local SN mode: Fits local supernovae with fixed absolute magnitude

    Default is prior mode unless actual SN data is available.
    """
    try:
        # Create model instance
        model = model_func(params)

        # Check if we have actual SN data or should use prior
        try:
            # Try to load actual SN data
            data = load_sh0es_data(use_prior=False)
            data = ensure_standard_dataset(data, "SN")
            use_prior_mode = False
            prior_data = None
        except (FileNotFoundError, ValueError):
            # Fall back to prior mode
            prior_data = load_sh0es_data(use_prior=True)
            use_prior_mode = True

        if use_prior_mode:
            # Prior mode: χ² = ((H0_model - H0_obs) / σ_H0)^2
            H0_model = extract_model_h0(model, params)
            H0_obs = prior_data["H0_obs"]
            H0_err = prior_data["H0_err"]

            chi2 = ((H0_model - H0_obs) / H0_err) ** 2

            return {
                "chi2": max(chi2, 0.0),
                "status": "valid",
                "n_data": 1,
                "dataset": "SN_SH0ES",
            }
        else:
            # Local SN mode: fit distance moduli with fixed absolute magnitude
            z = data["z"]
            obs = data["obs"]  # Distance modulus measurements
            err = data["err"]
            cov = data["cov"]

            # Compute model predictions (with M=0, since SH0ES is calibrated)
            mu_model = compute_sh0es_mu_model(model, z, M=0.0)
            diff = obs - mu_model

            # Compute χ² based on whether covariance is available
            if cov is not None:
                try:
                    cov_inv = np.linalg.inv(cov)
                    chi2 = float(diff.T @ cov_inv @ diff)
                except np.linalg.LinAlgError:
                    if err is None:
                        raise ValueError("Covariance inversion failed and no diagonal errors available.")
                    if np.any(err <= 0):
                        raise ValueError("SH0ES SN uncertainties must be positive.")
                    chi2 = float(np.sum(diff**2 / err**2))
            else:
                if err is None:
                    raise ValueError("SH0ES SN data missing uncertainties (mu_err or covariance required).")
                if np.any(err <= 0):
                    raise ValueError("SH0ES SN uncertainties must be positive.")
                chi2 = float(np.sum(diff**2 / err**2))

            return {
                "chi2": max(chi2, 0.0),
                "status": "valid",
                "n_data": len(z),
                "dataset": "SN_SH0ES",
            }

    except Exception as e:
        return {
            "chi2": 1e30,
            "status": "fail",
            "n_data": 0,
            "dataset": "SN_SH0ES",
        }
