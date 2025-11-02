"""
Compute χ² for Pantheon supernova data fits.

This module provides functions to calculate chi-squared statistics
between theoretical model predictions and observed Pantheon+SH0ES supernova data.
Supports both absolute magnitude (SH0ES-anchored) and relative magnitude modes.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from data_interface.standardize import ensure_standard_dataset
from .loader import load_pantheon_data
from .observables import compute_pantheon_mu_model, extract_model_h0


DEBUG_FLAG = os.environ.get("SN_PANTHEON_DEBUG", "")
DEBUG_ENABLED = DEBUG_FLAG not in {"", "0", "false", "False"}


def _debug(msg: str, *, data: Optional[Dict[str, Any]] = None) -> None:
    if not DEBUG_ENABLED:
        return
    suffix = ""
    if data:
        suffix = " | " + ", ".join(f"{key}={value!r}" for key, value in data.items())
    print(f"[SN_PANTHEON_DEBUG] {msg}{suffix}")

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


def chi2_sn_pantheon_abs(model_func, params, dataset: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute Pantheon+SH0ES χ² using absolute magnitudes (no M marginalization).
    
    Parameters
    ----------
    model_func : callable
        Function that takes parameter dict and returns model instance
    params : dict
        Cosmological parameters
    dataset : dict, optional
        Pre-loaded dataset. If None, loads using load_pantheon_data()
        
    Returns
    -------
    dict
        {
            "chi2": chi2_value,
            "status": "valid" or "fail",
            "n_data": N,
            "dataset": "SN_PANTHEON_ABS",
        }
    """


    try:
        # Get the source file and line number where this function is called from
        import inspect
        frame = inspect.currentframe().f_back
        info = inspect.getframeinfo(frame)




        if dataset is None:
            dataset = load_pantheon_data()

        _debug("Dataset loaded", data={"z_len": len(dataset["z"]), "cov_shape": getattr(dataset["cov"], "shape", None)})
        

        z = dataset["z"]
        mu_obs = dataset["obs_abs"]  # Absolute magnitudes (SH0ES-calibrated)
        C = dataset["cov"]           # Full STAT+SYS covariance
        
      
    except Exception as e:
        error_msg = f"Error in dataset processing: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Ensure we return a properly formatted result even in case of error
        return {
            "chi2": 1e6,
            "status": f"fail: {error_msg}",
            "n_data": 1,  # Ensure n_data is at least 1
            "dataset": "SN_PANTHEON_ABS"
        }
    
    # Get model predictions with M=0 (already absolute)
    model = model_func(params)
    mu_th = compute_pantheon_mu_model(model, z, M=0.0)
    
    # Residuals

    
    r = mu_obs - mu_th

    try:

        L = np.linalg.cholesky(C)
       

        y = np.linalg.solve(L, r)

        
        chi2 = float(y @ y)  # = r^T C^{-1} r

        status = "valid"
        _debug(
            "Cholesky solve succeeded",
            data={"chi2": chi2, "status": status, "residual_min": float(np.min(r)), "residual_max": float(np.max(r))},
        )
    except np.linalg.LinAlgError as e:
        print(f"[WARNING] Cholesky failed, using diagonal approximation: {e}")
        try:
            diag = np.diag(C)

            
            chi2_terms = r**2 / diag
            
            
            chi2 = float(np.sum(chi2_terms))
            status = "diag_approx"
            _debug(
                "Diagonal fallback used",
                data={"chi2": chi2, "status": status, "r2_min": float(np.min(chi2_terms)), "r2_max": float(np.max(chi2_terms))},
            )
        except Exception as diag_err:
            raise


    
    # Ensure z is a numpy array and get its length safely
    try:
        if not isinstance(z, np.ndarray):
            z = np.array(z)
        n_data = z.size

    except Exception as e:
        print(f"[ERROR] Error getting length of z: {e}")
        n_data = 1
    
    if not np.isfinite(chi2):
        _debug(
            "Non-finite chi2 encountered, returning sentinel value",
            data={
                "chi2": chi2,
                "status": status,
                "n_data": n_data,
                "params_subset": {k: params.get(k) for k in ("H0", "Om0", "alpha", "Rmax", "k_sat")},
            },
        )

    return {
        "chi2": float(chi2) if np.isfinite(chi2) else 1e6,
        "status": str(status) if status else "unknown",
        "n_data": int(n_data),
        "dataset": "SN_PANTHEON_ABS",
    }


def chi2_sn_pantheon(model_func, params, dataset: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Compute Pantheon SN χ² with analytic M marginalization (relative magnitudes).
    
    This is for relative (not SH0ES-anchored) supernova samples.
    
    Parameters
    ----------
    model_func : callable
        Function that takes parameter dict and returns model instance
    params : dict
        Cosmological parameters
    dataset : dict, optional
        Pre-loaded dataset. If None, loads using load_pantheon_data()
        
    Returns
    -------
    dict
        {
            "chi2": chi2_value,
            "status": "valid" or "fail",
            "n_data": N,
            "dataset": "SN_PANTHEON",
        }

    Notes
    -----
    Uses analytic marginalization over the absolute magnitude offset M.
    The covariance matrix should include both statistical and systematic uncertainties.
    If covariance matrix inversion fails, falls back to diagonal variance weighting.
    """
    try:
        _debug("chi2_sn_pantheon invoked", data={"params_subset": {k: params.get(k) for k in ("H0", "Om0", "alpha", "Rmax", "k_sat")}})

        # Create model instance
        model = model_func(params)

        # Load and validate data using standard schema
        data = dataset or load_pantheon_data()
        data = ensure_standard_dataset(data, "SN")

        # Extract standardized fields
        z = data["z"]
        obs = data["obs"]  # Distance modulus measurements
        err = data["err"]
        cov = data["cov"]

        # Compute model predictions (without magnitude offset)
        mu_model = compute_pantheon_mu_model(model, z, M=0.0)
        _debug(
            "Model predictions ready",
            data={
                "z_len": len(z),
                "mu_min": float(np.min(mu_model)),
                "mu_max": float(np.max(mu_model)),
            },
        )

        # Compute residuals without the magnitude offset yet
        diff_raw = obs - mu_model

        # Solve for the best-fit absolute magnitude offset analytically
        safe_err = None if err is None else np.clip(err, 1e-12, None)

        if cov is not None:
            try:
                ones = np.ones_like(diff_raw)
                cov_ones = np.linalg.solve(cov, ones)
                denom = float(ones @ cov_ones)
                if denom <= 0.0 or not np.isfinite(denom):
                    raise np.linalg.LinAlgError("Invalid denominator for absolute magnitude fit.")

                cov_diff_raw = np.linalg.solve(cov, diff_raw)
                numer = float(ones @ cov_diff_raw)
                M = numer / denom

                diff = diff_raw - M
                cov_diff = cov_diff_raw - M * cov_ones
                chi2 = float(diff @ cov_diff)
            except np.linalg.LinAlgError:
                if safe_err is None:
                    raise
                weights = 1.0 / (safe_err**2)
                M = _fit_absolute_magnitude(diff_raw, weights)
                diff = diff_raw - M
                chi2 = float(np.sum((diff / safe_err) ** 2))
        else:
            if safe_err is None:
                raise ValueError("Pantheon dataset missing uncertainties.")
            weights = 1.0 / (safe_err**2)
            M = _fit_absolute_magnitude(diff_raw, weights)
            diff = diff_raw - M
            chi2 = float(np.sum((diff / safe_err) ** 2))

        _debug(
            "chi2_sn_pantheon result",
            data={
                "chi2": float(chi2),
                "M": float(M),
                "residual_min": float(np.min(diff)),
                "residual_max": float(np.max(diff)),
            },
        )

        return {
            "chi2": max(chi2, 0.0),
            "status": "valid",
            "n_data": len(obs),
            "dataset": "SN_PANTHEON",
        }

    except Exception as e:
        _debug("chi2_sn_pantheon exception", data={"error": str(e)})
        return {
            "chi2": 1.0e6,  # Large penalty for failure
            "status": f"fail: {str(e)}",
            "n_data": 0,
            "dataset": "SN_PANTHEON",
        }


def chi2_sn_pantheon_prior(model_func, params, H0_prior=None, H0_std=None):
    """
    Compute Pantheon SN χ² using a Gaussian prior on H0.

    Parameters
    ----------
    model_func : callable
        Function that takes parameter dict and returns model instance
    params : dict
        Cosmological parameters
    H0_prior : float, optional
        Central value for H0 prior (km/s/Mpc). If None, uses the model's H0.
    H0_std : float, optional
        Standard deviation of H0 prior (km/s/Mpc). If None, uses 1.0.

    Returns
    -------
    dict
        {
            "chi2": chi2_value,
            "status": "valid" or "fail",
            "n_data": N,
            "dataset": "SN_PANTHEON_PRIOR",
        }
    """
    try:
        # Create model instance
        model = model_func(params)
        
        # Get model's H0 if not provided
        H0_model = extract_model_h0(model, params)
        
        # Use provided prior or model's H0
        if H0_prior is None:
            H0_prior = H0_model
        if H0_std is None:
            H0_std = 1.0  # Default 1 km/s/Mpc uncertainty
            
        # Compute χ² = ((H0_model - H0_prior) / H0_std)^2
        chi2 = ((H0_model - H0_prior) / H0_std) ** 2
        
        return {
            "chi2": max(chi2, 0.0),
            "status": "valid",
            "n_data": 1,  # Counts as 1 data point
            "dataset": "SN_PANTHEON_PRIOR",
        }
        
    except Exception as e:
        return {
            "chi2": 1.0e6,  # Large penalty for failure
            "status": f"fail: {str(e)}",
            "n_data": 1,
            "dataset": "SN_PANTHEON_PRIOR",
        }
