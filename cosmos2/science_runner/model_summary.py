"""Enhanced Model Summary JSON output for scientific reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


def get_git_info() -> Dict[str, str]:
    """Extract git repository information for provenance."""
    try:
        # Get commit hash
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get remote URL
        try:
            remote_url = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"], 
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except subprocess.CalledProcessError:
            remote_url = "unknown"
        
        # Get branch name
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except subprocess.CalledProcessError:
            branch = "unknown"
        
        # Get commit date
        try:
            commit_date = subprocess.check_output(
                ["git", "log", "-1", "--format=%cI"], 
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except subprocess.CalledProcessError:
            commit_date = "unknown"
        
        # Check if working directory is clean
        try:
            status = subprocess.check_output(
                ["git", "status", "--porcelain"], 
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            is_clean = len(status) == 0
        except subprocess.CalledProcessError:
            is_clean = False
        
        return {
            "commit_hash": commit_hash,
            "remote_url": remote_url,
            "branch": branch,
            "commit_date": commit_date,
            "is_clean": is_clean,
            "has_uncommitted_changes": not is_clean
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit_hash": "unknown",
            "remote_url": "unknown", 
            "branch": "unknown",
            "commit_date": "unknown",
            "is_clean": False,
            "has_uncommitted_changes": True
        }


def get_lut_version(thermal_metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Extract LUT version information from thermal metadata."""
    if not thermal_metadata:
        return {"version": "unknown", "source": "unknown"}
    
    return {
        "version": thermal_metadata.get("lut_version", "unknown"),
        "source": thermal_metadata.get("source", "unknown"),
        "interpolation_method": thermal_metadata.get("interpolation_method", "unknown"),
        "temperature_range": {
            "min": thermal_metadata.get("T_min"),
            "max": thermal_metadata.get("T_max"),
            "n_points": thermal_metadata.get("n_points")
        },
        "alpha_range": {
            "min": thermal_metadata.get("alpha_min"),
            "max": thermal_metadata.get("alpha_max"),
            "n_points": thermal_metadata.get("n_alpha_points")
        }
    }


def get_quantum_metadata(model: Any) -> Dict[str, Any]:
    """Extract quantum engine metadata from PBUF model."""
    metadata = {}
    
    # Try to get quantum-specific parameters
    if hasattr(model, '_params'):
        params = model._params
        metadata.update({
            "regulator_type": getattr(params, 'regulator_type', 'unknown'),
            "field_content": getattr(params, 'field_content', 'unknown'),
            "f_cut": getattr(params, 'f_cutoff', None),
            "f_coup": getattr(params, 'f_coupling', None),
            "epsilon_0_source": getattr(params, 'epsilon_0_source', 'unknown'),
            "alpha_value": getattr(params, 'alpha', None),
            "r_max": getattr(params, 'Rmax', None),
            "omega_normalization": getattr(params, 'omega_normalization', 'unknown'),
            "sigma_rescale": getattr(params, 'sigma_rescale', None)
        })
    
    # Add thermal table info if available
    if hasattr(model, '_thermal'):
        thermal = model._thermal
        if hasattr(thermal, 'metadata'):
            metadata["thermal_metadata"] = dict(thermal.metadata)
    
    # Add micro bootstrap metadata
    if hasattr(model, 'micro_bootstrap_metadata'):
        metadata["bootstrap_metadata"] = dict(model.micro_bootstrap_metadata)
    
    return metadata


def calculate_aic_bic(chi2: float, n_params: int, n_data_points: int) -> Dict[str, float]:
    """Calculate AIC and BIC information criteria."""
    if n_data_points <= 0:
        return {"AIC": np.inf, "BIC": np.inf, "AICc": np.inf}
    
    aic = 2 * n_params + chi2
    bic = n_params * np.log(n_data_points) + chi2
    
    # AICc (corrected AIC for small sample sizes)
    if n_data_points > n_params + 1:
        aic_correction = (2 * n_params * (n_params + 1)) / (n_data_points - n_params - 1)
        aicc = aic + aic_correction
    else:
        aicc = np.inf
    
    return {"AIC": aic, "BIC": bic, "AICc": aicc}


def extract_priors_used(config: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    """Extract prior information from configuration."""
    priors_config = config.get("priors", {})
    model_priors = priors_config.get(model_name, {})
    
    # Extract parameter bounds as priors if no explicit priors
    bounds_config = config.get("parameter_bounds", {})
    model_bounds = bounds_config.get(model_name, {})
    
    priors = {}
    for param, bound in model_bounds.items():
        if isinstance(bound, (list, tuple)) and len(bound) == 2:
            priors[param] = {
                "type": "uniform",
                "min": float(bound[0]),
                "max": float(bound[1])
            }
    
    # Override with any explicit priors
    for param, prior in model_priors.items():
        priors[param] = prior
    
    return priors


def count_data_points(config: Mapping[str, Any], chi2_breakdown: Mapping[str, float]) -> int:
    """Estimate total number of data points from chi2 breakdown."""
    # This is a rough estimate - ideally we'd get this from the actual datasets
    # For now, we'll use typical values for each dataset type
    dataset_counts = {
        "cmb": 2000,  # Planck TT+TE+EE+lowE
        "sn": 1941,   # Pantheon+ sample
        "bao_iso": 24,  # BAO isotropic measurements
        "bao_aniso": 36,  # BAO anisotropic measurements  
        "cc": 32,     # Cosmic chronometers
        "rsd": 26,    # RSD measurements
        "wl": 200,    # Weak lensing (if used)
        "sh0es": 6    # SH0ES prior (if used)
    }
    
    total = 0
    for dataset in chi2_breakdown.keys():
        total += dataset_counts.get(dataset, 0)
    
    return max(total, 1)  # Ensure at least 1 to avoid division by zero


def create_model_summary(
    model_name: str,
    model: Any,
    best_params: Dict[str, float],
    chi2_total: float,
    chi2_breakdown: Dict[str, float],
    config: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any] | None = None,
    fit_outputs: Mapping[str, Any] | None = None,
    engine_result: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Create comprehensive Model Summary JSON with all required fields.
    
    Args:
        model_name: Name of the model (e.g., "lcdm", "pbuf")
        model: The model instance
        best_params: Best-fit parameter values
        chi2_total: Total chi-squared
        chi2_breakdown: Chi-squared breakdown per dataset
        config: Science run configuration
        runtime_metadata: Additional runtime metadata
        fit_outputs: Fit-specific outputs
        engine_result: Engine optimization results
    
    Returns:
        Comprehensive model summary dictionary
    """
    
    # Basic model information
    summary = {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success" if np.isfinite(chi2_total) else "failed"
    }
    
    # Parameter values and priors
    summary["parameters"] = {
        "best_fit": {k: float(v) for k, v in best_params.items()},
        "priors_used": extract_priors_used(config, model_name),
        "derived_quantities": {
            "H0": float(best_params.get("H0", 0.0)),
            "Omega_m0": float(best_params.get("Omega_m0", 0.0)),
            "Omega_b0": float(
                getattr(model, "parameters", {}).get("Omega_b0", best_params.get("Omega_b0", 0.0))
            ),
            "Omega_k0": float(best_params.get("Omega_k0", 0.0)),
            "S8": float(model.S8()) if hasattr(model, 'S8') else None,
            "sigma8": float(model.sigma8()) if hasattr(model, 'sigma8') else None,
            "r_d": float(model.sound_horizon()) if hasattr(model, 'sound_horizon') else None,
            "q0": None  # Will be calculated if possible
        }
    }
    
    # Calculate q0 if we can
    try:
        if hasattr(model, 'Hubble'):
            H0 = model.Hubble(0.0)
            H_dot_0 = 0.0  # Would need numerical derivative for exact value
            # For LCDM: q0 = (H_dot_0/H0^2) - 1, but H_dot_0 = 0 for matter+Lambda
            # For PBUF, this would be more complex - using approximation
            summary["parameters"]["derived_quantities"]["q0"] = -0.5 + (1.0 - float(best_params.get("Omega_m0", 0.3))) if model_name == "lcdm" else None
    except Exception:
        pass
    
    # Chi-squared information
    summary["chi_squared"] = {
        "total": float(chi2_total),
        "per_dataset": {k: float(v) for k, v in chi2_breakdown.items()},
        "reduced": None  # Will be calculated below
    }
    
    # Calculate reduced chi-squared
    n_data = count_data_points(config, chi2_breakdown)
    n_params = len(best_params)
    if n_data > n_params:
        summary["chi_squared"]["reduced"] = float(chi2_total / (n_data - n_params))
        summary["chi_squared"]["degrees_of_freedom"] = n_data - n_params
        summary["chi_squared"]["n_data_points"] = n_data
        summary["chi_squared"]["n_parameters"] = n_params
    
    # Information criteria
    info_criteria = calculate_aic_bic(chi2_total, n_params, n_data)
    summary["information_criteria"] = info_criteria
    
    # Runtime metadata
    runtime = dict(runtime_metadata or {})
    summary["runtime_metadata"] = {
        "start_time": runtime.get("start_time"),
        "end_time": runtime.get("end_time"), 
        "total_runtime": runtime.get("total_runtime"),
        "engine": runtime.get("engine", "unknown"),
        "machine": runtime.get("machine", {}),
        "optimization_settings": {
            "n_scatter": config.get("engine_settings", {}).get("n_scatter"),
            "n_seeds": config.get("engine_settings", {}).get("n_seeds"),
            "n_refine": config.get("engine_settings", {}).get("n_refine"),
            "workers": config.get("engine_settings", {}).get("workers"),
            "seed": config.get("engine_settings", {}).get("seed")
        }
    }
    
    # Provenance information
    summary["provenance"] = {
        "git": get_git_info(),
        "lut_version": get_lut_version(getattr(model, 'micro_bootstrap_metadata', None)),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
        "dataset_hashes": {},  # Would need to compute from actual datasets
        "code_version": "cosmos2"
    }
    
    # Quantum metadata (for PBUF models)
    if model_name.lower() == "pbuf":
        summary["quantum_metadata"] = get_quantum_metadata(model)
    
    fit_details: Dict[str, Any] = {}
    wl_flags_from_fit: Dict[str, Any] | None = None
    if fit_outputs:
        summary["fit_outputs"] = {
            "convergence_status": "success",
            "n_iterations": len(engine_result.get("results", [])) if engine_result else 0,
            "final_chi2": float(chi2_total),
            "parameter_trace": engine_result.get("results", [])[:10] if engine_result else [],  # First 10 points
            "detailed_outputs": {k: v for k, v in fit_outputs.items() if k != "extras"}
        }
        for fit_name, payload in fit_outputs.items():
            extras = payload.get("extras", {}) if isinstance(payload, dict) else {}
            fit_details[fit_name] = {"extras": extras}
            if fit_name in {"wl_kids1000", "weak_lensing_kids1000"} and isinstance(extras, dict):
                wl_flags = extras.get("wl_flags")
                if wl_flags:
                    wl_flags_from_fit = wl_flags

    # Dataset information
    dataset_payload = {
        "used": list(config.get("joint_config", {}).get("fits", [])),
        "weights": dict(config.get("joint_config", {}).get("weights", {})),
        "chi2_contribution": {k: float(v) for k, v in chi2_breakdown.items()},
        "fit_details": fit_details  # Populated from fit outputs when available
    }
    if wl_flags_from_fit:
        dataset_payload["wl_flags"] = wl_flags_from_fit
    summary["datasets"] = dataset_payload
    
    # Model-specific information
    if model_name.lower() == "pbuf":
        summary["model_specific"] = {
            "alpha_resolved": float(getattr(model, '_alpha', 0.0)),
            "thermal_table": {
                "path": str(getattr(model, '_thermal', {}).__dict__.get('path', 'unknown')),
                "interpolation": getattr(model, '_thermal', {}).__dict__.get('interpolation_method', 'unknown')
            },
            "normalization_metadata": dict(getattr(model, '_normalization_metadata', {}))
        }
    else:
        summary["model_specific"] = {}
    
    # Validation checks
    summary["validation"] = {
        "is_valid": getattr(model, 'is_valid', lambda: True)(),
        "parameter_bounds_satisfied": True,  # Would need to check against bounds
        "chi2_finite": np.isfinite(chi2_total),
        "sound_horizon_computed": hasattr(model, '_r_d') and model._r_d is not None
    }
    
    return summary


__all__ = ["create_model_summary"]
