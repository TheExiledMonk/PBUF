"""
Centralized statistical computations for PBUF cosmology fitting.

This module provides consistent χ², AIC, BIC, and degrees of freedom calculations
across all fitters, ensuring identical statistical analysis methodology.
"""

from typing import Dict, List, Any
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None  # Will be handled in implementation
from . import ParameterDict, DatasetDict, MetricsDict, PredictionsDict


def chi2_generic(
    predictions: PredictionsDict, 
    observations: Dict[str, np.ndarray], 
    covariance: np.ndarray
) -> float:
    """
    Compute χ² using consistent matrix operations across all blocks.
    
    Implements χ² = (pred - obs)ᵀ C⁻¹ (pred - obs) with proper error handling
    for matrix inversion and numerical stability.
    
    Args:
        predictions: Dictionary of theoretical predictions
        observations: Dictionary of observational data
        covariance: Covariance matrix for the observations
        
    Returns:
        χ² value
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Validate inputs
    _validate_chi2_inputs(predictions, observations, covariance)
    
    # Convert predictions and observations to consistent arrays
    # Assume both have the same keys and ordering
    pred_keys = sorted(predictions.keys())
    obs_keys = sorted(observations.keys())
    
    if pred_keys != obs_keys:
        raise ValueError(f"Prediction keys {pred_keys} don't match observation keys {obs_keys}")
    
    # Build residual vector
    residuals = []
    for key in pred_keys:
        pred_val = predictions[key]
        obs_val = observations[key]
        
        # Handle both scalar and array values
        if np.isscalar(pred_val):
            residuals.append(pred_val - obs_val)
        else:
            residuals.extend((pred_val - obs_val).flatten())
    
    residuals = np.array(residuals)
    
    # Ensure covariance matrix is the right size
    if covariance.shape[0] != len(residuals):
        raise ValueError(f"Covariance matrix size {covariance.shape} doesn't match residuals size {len(residuals)}")
    
    # Check for positive definiteness and numerical stability
    try:
        # Use Cholesky decomposition for numerical stability
        L = np.linalg.cholesky(covariance)
        # Solve L y = residuals, then compute ||y||²
        y = np.linalg.solve(L, residuals)
        chi2 = np.dot(y, y)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if Cholesky fails
        try:
            cov_inv = np.linalg.pinv(covariance)
            chi2 = np.dot(residuals, np.dot(cov_inv, residuals))
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Covariance matrix inversion failed: {e}")
    
    return float(chi2)


def compute_metrics(
    chi2: float, 
    n_params: int, 
    datasets: List[str]
) -> MetricsDict:
    """
    Compute AIC, BIC, degrees of freedom, and p-value from χ².
    
    Args:
        chi2: Total χ² value
        n_params: Number of free parameters
        datasets: List of datasets included in fit
        
    Returns:
        Dictionary with all statistical metrics
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Compute degrees of freedom
    dof = compute_dof(datasets, n_params)
    
    # Compute information criteria
    aic = chi2 + 2 * n_params
    
    # For BIC, we need total number of data points
    n_data = sum(_get_dataset_size(dataset) for dataset in datasets)
    bic = chi2 + n_params * np.log(n_data)
    
    # Compute p-value
    p_value = compute_p_value(chi2, dof)
    
    # Reduced chi-squared
    chi2_reduced = chi2 / dof if dof > 0 else np.inf
    
    return {
        "chi2": float(chi2),
        "chi2_reduced": float(chi2_reduced),
        "aic": float(aic),
        "bic": float(bic),
        "dof": int(dof),
        "n_params": int(n_params),
        "n_data": int(n_data),
        "p_value": float(p_value)
    }


def compute_dof(datasets: List[str], n_params: int) -> int:
    """
    Compute degrees of freedom across all datasets.
    
    Args:
        datasets: List of dataset names
        n_params: Number of free parameters
        
    Returns:
        Total degrees of freedom (N_data - N_params)
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5
    """
    total_data_points = sum(_get_dataset_size(dataset) for dataset in datasets)
    dof = total_data_points - n_params
    
    if dof < 0:
        raise ValueError(f"Degrees of freedom cannot be negative: {total_data_points} data points - {n_params} parameters = {dof}")
    
    return dof


def delta_aic(aic1: float, aic2: float) -> float:
    """
    Compute ΔAIC for model comparison.
    
    Args:
        aic1: AIC value for first model
        aic2: AIC value for second model
        
    Returns:
        ΔAIC = AIC1 - AIC2
    """
    return float(aic1 - aic2)


def compute_p_value(chi2: float, dof: int) -> float:
    """
    Compute p-value from χ² distribution.
    
    Args:
        chi2: χ² value
        dof: Degrees of freedom
        
    Returns:
        p-value from χ² distribution
    """
    if dof <= 0:
        return np.nan
    
    if chi2 < 0:
        return np.nan
    
    if stats is None:
        # Fallback implementation without scipy
        # Use approximation for large dof: χ² ~ N(dof, 2*dof)
        if dof > 30:
            z_score = (chi2 - dof) / np.sqrt(2 * dof)
            # Approximate p-value using complementary error function
            p_value = 0.5 * (1 + np.sign(z_score) * np.sqrt(1 - np.exp(-2 * z_score**2 / np.pi)))
            return float(1 - p_value)  # Upper tail probability
        else:
            # For small dof, return NaN to indicate scipy is needed
            return np.nan
    else:
        # Use scipy for accurate computation
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        return float(p_value)


def _get_dataset_size(dataset_name: str) -> int:
    """
    Get number of data points for a given dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Number of data points in the dataset
    """
    # Use the DATASET_SIZES mapping, with fallback to datasets module
    if dataset_name in DATASET_SIZES and DATASET_SIZES[dataset_name] is not None:
        return DATASET_SIZES[dataset_name]
    
    # For variable-size datasets, try to load and get actual size
    try:
        from . import datasets
        dataset_info = datasets.get_dataset_info(dataset_name)
        if "n_points" in dataset_info:
            return dataset_info["n_points"]
        
        # Fallback: load dataset and count points
        data = datasets.load_dataset(dataset_name)
        if "observations" in data:
            obs = data["observations"]
            if isinstance(obs, dict):
                # Count total elements across all observation keys
                total_points = 0
                for key, values in obs.items():
                    if np.isscalar(values):
                        total_points += 1
                    else:
                        total_points += len(np.array(values).flatten())
                return total_points
            else:
                return len(np.array(obs).flatten())
        
    except (ImportError, KeyError, AttributeError):
        pass
    
    # Default fallback values based on typical dataset sizes
    fallback_sizes = {
        "cmb": 3,        # R, ℓ_A, θ*
        "bao": 15,       # Typical BAO compilation size
        "bao_ani": 20,   # Anisotropic BAO measurements
        "sn": 1048       # Pantheon+ supernova count
    }
    
    if dataset_name in fallback_sizes:
        return fallback_sizes[dataset_name]
    
    raise ValueError(f"Unknown dataset '{dataset_name}' and cannot determine size")


def _validate_chi2_inputs(
    predictions: PredictionsDict,
    observations: Dict[str, np.ndarray],
    covariance: np.ndarray
) -> bool:
    """
    Validate inputs for χ² computation.
    
    Args:
        predictions: Theoretical predictions
        observations: Observational data
        covariance: Covariance matrix
        
    Returns:
        True if inputs are valid, raises ValueError otherwise
    """
    if not predictions:
        raise ValueError("Predictions dictionary is empty")
    
    if not observations:
        raise ValueError("Observations dictionary is empty")
    
    if covariance is None or covariance.size == 0:
        raise ValueError("Covariance matrix is empty or None")
    
    # Check covariance matrix is square
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"Covariance matrix must be square, got shape {covariance.shape}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(covariance)) or np.any(np.isinf(covariance)):
        raise ValueError("Covariance matrix contains NaN or infinite values")
    
    # Check predictions and observations for NaN/inf
    for key, pred in predictions.items():
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            raise ValueError(f"Predictions for '{key}' contain NaN or infinite values")
    
    for key, obs in observations.items():
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            raise ValueError(f"Observations for '{key}' contain NaN or infinite values")
    
    return True


def validate_chi2_distribution(chi2: float, dof: int, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Validate χ² distribution assumptions and goodness of fit.
    
    Args:
        chi2: Observed χ² value
        dof: Degrees of freedom
        alpha: Significance level for tests (default 0.05)
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "chi2": chi2,
        "dof": dof,
        "chi2_reduced": chi2 / dof if dof > 0 else np.inf,
        "p_value": compute_p_value(chi2, dof),
        "alpha": alpha
    }
    
    # Check if fit is acceptable (p-value > alpha)
    results["fit_acceptable"] = results["p_value"] > alpha if not np.isnan(results["p_value"]) else None
    
    # Check for over/under-fitting based on reduced χ²
    chi2_red = results["chi2_reduced"]
    if not np.isinf(chi2_red):
        # Rule of thumb: reduced χ² should be close to 1
        results["overfitting"] = chi2_red < 0.5  # Suspiciously good fit
        results["underfitting"] = chi2_red > 2.0  # Poor fit
        results["reasonable_fit"] = 0.5 <= chi2_red <= 2.0
    else:
        results["overfitting"] = None
        results["underfitting"] = None
        results["reasonable_fit"] = None
    
    # Confidence intervals for χ² (if scipy available)
    if stats is not None and dof > 0:
        # 95% confidence interval for χ²
        ci_lower = stats.chi2.ppf(0.025, dof)
        ci_upper = stats.chi2.ppf(0.975, dof)
        results["chi2_ci_lower"] = ci_lower
        results["chi2_ci_upper"] = ci_upper
        results["within_ci"] = ci_lower <= chi2 <= ci_upper
    else:
        results["chi2_ci_lower"] = None
        results["chi2_ci_upper"] = None
        results["within_ci"] = None
    
    return results


def compute_confidence_intervals(chi2: float, dof: int, confidence_levels: List[float] = None) -> Dict[str, float]:
    """
    Compute confidence intervals for χ² values.
    
    Args:
        chi2: Observed χ² value
        dof: Degrees of freedom
        confidence_levels: List of confidence levels (default: [0.68, 0.95, 0.99])
        
    Returns:
        Dictionary mapping confidence levels to interval bounds
    """
    if confidence_levels is None:
        confidence_levels = [0.68, 0.95, 0.99]
    
    if stats is None:
        return {f"{cl:.2f}": {"lower": np.nan, "upper": np.nan} for cl in confidence_levels}
    
    if dof <= 0:
        return {f"{cl:.2f}": {"lower": np.nan, "upper": np.nan} for cl in confidence_levels}
    
    intervals = {}
    for cl in confidence_levels:
        alpha = 1 - cl
        lower = stats.chi2.ppf(alpha/2, dof)
        upper = stats.chi2.ppf(1 - alpha/2, dof)
        intervals[f"{cl:.2f}"] = {"lower": lower, "upper": upper}
    
    return intervals


# Dataset size mapping for degrees of freedom calculation
DATASET_SIZES = {
    "cmb": 3,        # R, ℓ_A, θ*
    "bao": None,     # Variable, depends on compilation
    "bao_ani": None, # Variable, depends on compilation  
    "sn": None       # Variable, depends on compilation (typically ~1000+)
}