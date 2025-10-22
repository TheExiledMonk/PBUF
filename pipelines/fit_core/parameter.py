"""
Centralized parameter management for PBUF cosmology fitting.

This module provides the single source of truth for all cosmological parameters,
ensuring consistent parameter handling across all fitters and eliminating parameter drift.
"""

from typing import Dict, Optional, Any
from . import ParameterDict
from . import cmb_priors

# Default parameter values for both ΛCDM and PBUF models
DEFAULTS: Dict[str, ParameterDict] = {
    "lcdm": {
        "H0": 67.4,           # Hubble constant (km/s/Mpc)
        "Om0": 0.315,         # Matter density fraction
        "Obh2": 0.02237,      # Physical baryon density
        "ns": 0.9649,         # Scalar spectral index
        "Neff": 3.046,        # Effective neutrino species
        "Tcmb": 2.7255,       # CMB temperature (K)
        "recomb_method": "PLANCK18"  # Recombination calculation method
    },
    "pbuf": {
        # Include all ΛCDM parameters
        "H0": 67.4,
        "Om0": 0.315,
        "Obh2": 0.02237,
        "ns": 0.9649,
        "Neff": 3.046,
        "Tcmb": 2.7255,
        "recomb_method": "PLANCK18",
        # PBUF-specific parameters
        "alpha": 5e-4,        # Elasticity amplitude
        "Rmax": 1e9,          # Saturation length scale
        "eps0": 0.7,          # Elasticity bias term
        "n_eps": 0.0,         # Evolution exponent
        "k_sat": 0.9762       # Saturation coefficient
    }
}


def build_params(model: str, overrides: Optional[ParameterDict] = None) -> ParameterDict:
    """
    Build parameter dictionary for specified model with optional overrides.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        overrides: Optional parameter overrides to apply
        
    Returns:
        Complete parameter dictionary with derived quantities
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    if model not in DEFAULTS:
        raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
    
    # Start with default parameters for the model
    params = get_defaults(model).copy()
    
    # Apply overrides if provided
    if overrides:
        params = apply_overrides(params, overrides)
    
    # Validate the final parameter set
    validate_params(params, model)
    
    # Compute derived parameters using physics modules
    params = cmb_priors.prepare_background_params(params)
    
    # Validate derived parameter consistency
    _validate_derived_params(params, model)
    
    # Add model metadata
    params["model_class"] = model
    
    return params


def get_defaults(model: str) -> ParameterDict:
    """
    Get default parameter values for specified model.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        
    Returns:
        Dictionary of default parameter values
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    if model not in DEFAULTS:
        raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
    
    return DEFAULTS[model].copy()


def validate_params(params: ParameterDict, model: str) -> bool:
    """
    Validate parameter dictionary for physical consistency and completeness.
    
    Args:
        params: Parameter dictionary to validate
        model: Model type for validation rules
        
    Returns:
        True if parameters are valid, raises ValueError otherwise
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    # Define required parameters for each model
    required_lcdm = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb", "recomb_method"}
    required_pbuf = required_lcdm | {"alpha", "Rmax", "eps0", "n_eps", "k_sat"}
    
    required_params = required_pbuf if model == "pbuf" else required_lcdm
    
    # Check for missing required parameters
    missing_params = required_params - set(params.keys())
    if missing_params:
        raise ValueError(f"Missing required parameters for {model} model: {missing_params}")
    
    # Validate physical bounds
    _validate_physical_bounds(params, model)
    
    return True


def _validate_physical_bounds(params: ParameterDict, model: str) -> None:
    """
    Validate parameter values are within physical bounds.
    
    Args:
        params: Parameter dictionary to validate
        model: Model type for validation rules
        
    Raises:
        ValueError: If parameters are outside physical bounds
    """
    # Common ΛCDM parameter bounds
    bounds = {
        "H0": (20.0, 150.0),      # Hubble constant (km/s/Mpc)
        "Om0": (0.01, 0.99),      # Matter density fraction
        "Obh2": (0.005, 0.1),    # Physical baryon density
        "ns": (0.5, 1.5),        # Scalar spectral index
        "Neff": (1.0, 10.0),     # Effective neutrino species
        "Tcmb": (2.0, 3.0),      # CMB temperature (K)
    }
    
    # PBUF-specific parameter bounds
    if model == "pbuf":
        bounds.update({
            "alpha": (1e-6, 1e-2),    # Elasticity amplitude
            "Rmax": (1e6, 1e12),     # Saturation length scale
            "eps0": (0.0, 2.0),      # Elasticity bias term
            "n_eps": (-2.0, 2.0),    # Evolution exponent
            "k_sat": (0.1, 2.0),     # Saturation coefficient
        })
    
    # Check bounds for numeric parameters
    for param, value in params.items():
        if param in bounds and isinstance(value, (int, float)):
            min_val, max_val = bounds[param]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Parameter {param}={value} outside physical bounds "
                    f"[{min_val}, {max_val}] for {model} model"
                )
    
    # Validate string parameters
    if "recomb_method" in params:
        valid_methods = {"PLANCK18", "HS96", "EH98"}
        if params["recomb_method"] not in valid_methods:
            raise ValueError(
                f"Invalid recombination method: {params['recomb_method']}. "
                f"Must be one of {valid_methods}"
            )


def _validate_derived_params(params: ParameterDict, model: str) -> None:
    """
    Validate derived parameters for consistency and physical bounds.
    
    Args:
        params: Parameter dictionary with derived quantities
        model: Model type for validation rules
        
    Raises:
        ValueError: If derived parameters are inconsistent or unphysical
    """
    # Check that all expected derived parameters are present
    expected_derived = {
        "Omh2", "Orh2", "z_recomb", "z_drag", "r_s_drag"
    }
    
    missing_derived = expected_derived - set(params.keys())
    if missing_derived:
        raise ValueError(f"Missing derived parameters: {missing_derived}")
    
    # Validate derived parameter bounds
    derived_bounds = {
        "Omh2": (0.01, 1.0),        # Physical matter density
        "Orh2": (1e-6, 1e-3),       # Physical radiation density
        "z_recomb": (800, 1500),    # Recombination redshift
        "z_drag": (800, 1500),      # Drag epoch redshift
        "r_s_drag": (0.1, 300),     # Sound horizon (Mpc) - allow smaller values for CMB
    }
    
    for param, value in params.items():
        if param in derived_bounds and isinstance(value, (int, float)):
            min_val, max_val = derived_bounds[param]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Derived parameter {param}={value} outside expected bounds "
                    f"[{min_val}, {max_val}] for {model} model"
                )
    
    # Validate physical consistency relationships
    if params["z_recomb"] < params["z_drag"]:
        raise ValueError(
            f"Recombination redshift ({params['z_recomb']}) should be greater than "
            f"drag epoch redshift ({params['z_drag']})"
        )
    
    # Validate density consistency (Omh2 should be consistent with Om0 and H0)
    expected_Omh2 = params["Om0"] * (params["H0"] / 100.0)**2
    if abs(params["Omh2"] - expected_Omh2) > 1e-4:
        raise ValueError(
            f"Inconsistent physical matter density: Omh2={params['Omh2']} "
            f"but Om0*h^2={expected_Omh2}"
        )


def apply_overrides(base_params: ParameterDict, overrides: ParameterDict) -> ParameterDict:
    """
    Apply parameter overrides to base parameter set.
    
    Args:
        base_params: Base parameter dictionary
        overrides: Override values to apply
        
    Returns:
        Updated parameter dictionary
    """
    result = base_params.copy()
    
    for key, value in overrides.items():
        # Validate override parameter types
        if not isinstance(value, (int, float, str)):
            raise ValueError(
                f"Invalid override value type for {key}: {type(value)}. "
                "Must be int, float, or str"
            )
        
        result[key] = value
    
    return result