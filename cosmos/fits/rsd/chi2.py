"""
Compute χ² for RSD fσ₈(z) data.

This module provides functions to compute chi-squared statistics
for redshift space distortion measurements using the standardized PBUF data format.
When evaluating PBUF models we analytically marginalize over an overall
growth-amplitude nuisance parameter A_rsd so that RSD does not inherit
the fixed Planck σ₈ normalization.
"""

import numpy as np
from data_interface.standardize import ensure_standard_dataset
from cosmos.fits._dataset_loader import load_rsd_dataset

from cosmos.optim.parameter_defaults import SIGMA8_PLANCK
from cosmos.pbuf.model import PBUF
from .observables import compute_rsd_observable


def _is_pbuf_model(model) -> bool:
    """
    Lightweight detection for PBUF models without importing heavy modules elsewhere.
    """
    return isinstance(model, PBUF) or getattr(model, "__class__", None).__name__.lower() == "pbuf"


def _fit_rsd_amplitude(obs, pred, err=None, cov=None):
    """
    Analytically fit the amplitude that minimizes χ² and return (A, variance, chi²).
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if cov is not None:
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(f"RSD covariance inversion failed: {exc}") from exc

        denom = float(pred.T @ cov_inv @ pred)
        numer = float(pred.T @ cov_inv @ obs)

        if denom <= 0.0:
            raise ValueError(f"Non-positive amplitude denominator (value {denom}) during RSD marginalization.")

        amplitude = numer / denom
        variance = 1.0 / denom
        diff = obs - amplitude * pred
        chi2 = float(diff.T @ cov_inv @ diff)
        return amplitude, variance, chi2

    if err is None:
        raise ValueError("RSD amplitude marginalization requires either a covariance matrix or diagonal errors.")

    weights = 1.0 / np.asarray(err, dtype=float) ** 2
    denom = float(np.sum(weights * pred * pred))

    if denom <= 0.0:
        raise ValueError(f"Non-positive amplitude denominator (value {denom}) in diagonal RSD marginalization.")

    numer = float(np.sum(weights * pred * obs))
    amplitude = numer / denom
    variance = 1.0 / denom
    diff = obs - amplitude * pred
    chi2 = float(np.sum(weights * diff * diff))
    return amplitude, variance, chi2


def _record_rsd_diagnostics(model, amplitude, variance):
    """
    Store amplitude diagnostics on the model instance for downstream reporting.
    """
    diagnostics = getattr(model, "diagnostics", None)
    if diagnostics is None:
        diagnostics = {}
        setattr(model, "diagnostics", diagnostics)

    rsd_diag = diagnostics.get("rsd")
    if rsd_diag is None:
        rsd_diag = {}
        diagnostics["rsd"] = rsd_diag

    rsd_diag["A_rsd"] = float(amplitude)
    rsd_diag["A_rsd_var"] = float(variance)


def chi_squared_rsd(model, data=None, sigma8_0=SIGMA8_PLANCK):
    """
    Compute χ² between model predictions and fσ8 observations using standardized format.

    Parameters
    ----------
    model : LCDM or PBUF instance
        Cosmological model with fsigma8 method
    data : dict or None
        Standardized RSD dataset (PBUF Data Object v1). If None, loads default.
    sigma8_0 : float
        σ8 normalization at z=0 (default Planck 2018 value)

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
        data = load_rsd_dataset()

    # Ensure data follows PBUF Data Object v1 schema
    data = ensure_standard_dataset(data, "RSD")

    # Extract standardized fields
    z = data["z"]
    obs = data["obs"]  # fσ8 measurements
    err = data["err"]
    cov = data["cov"]

    is_pbuf = _is_pbuf_model(model)

    # For PBUF we compute the shape with σ₈(a=1)=1 so that A_rsd absorbs the normalization.
    if is_pbuf:
        pred = compute_rsd_observable(model, z, sigma8_0=1.0)
    else:
        sigma8_norm = SIGMA8_PLANCK if sigma8_0 is None else sigma8_0
        pred = compute_rsd_observable(model, z, sigma8_0=sigma8_norm)

    if is_pbuf:
        amplitude, variance, chi2 = _fit_rsd_amplitude(obs, pred, err=err, cov=cov)
        _record_rsd_diagnostics(model, amplitude, variance)
        return max(chi2, 0.0)

    # Non-PBUF models retain the traditional fixed σ₈ normalization.
    diff = obs - pred
    if cov is not None:
        cov_inv = np.linalg.inv(cov)
        chi2 = float(diff.T @ cov_inv @ diff)
    else:
        chi2 = float(np.sum((diff / err) ** 2))

    return max(chi2, 0.0)
