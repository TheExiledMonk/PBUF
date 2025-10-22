"""
Centralized parameter management for PBUF cosmology fitting.

This module provides the single source of truth for all cosmological parameters,
ensuring consistent parameter handling across all fitters and eliminating parameter drift.
Includes optimization metadata support for selective parameter optimization.
"""

from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass
from . import ParameterDict
from . import cmb_priors


@dataclass
class OptimizationMetadata:
    """Metadata for parameter optimization configuration and results."""
    is_optimizable: bool = False
    optimization_bounds: Optional[Tuple[float, float]] = None
    is_currently_optimized: bool = False
    optimization_source: Optional[str] = None  # e.g., "cmb", "bao", "joint"
    last_optimized: Optional[str] = None  # ISO timestamp
    optimization_method: Optional[str] = None  # e.g., "L-BFGS-B"


@dataclass
class ParameterInfo:
    """Complete parameter information including value and optimization metadata."""
    value: Union[float, int, str]
    metadata: OptimizationMetadata


# Parameter classification by model - defines which parameters can be optimized
OPTIMIZABLE_PARAMETERS: Dict[str, List[str]] = {
    "lcdm": ["H0", "Om0", "Obh2", "ns"],
    "pbuf": ["H0", "Om0", "Obh2", "ns", "alpha", "Rmax", "eps0", "n_eps", "k_sat"]
}

# Physical bounds for optimization - used for bounded optimization algorithms
OPTIMIZATION_BOUNDS: Dict[str, Tuple[float, float]] = {
    # Common ΛCDM parameters
    "H0": (20.0, 150.0),      # Hubble constant (km/s/Mpc)
    "Om0": (0.01, 0.99),      # Matter density fraction
    "Obh2": (0.005, 0.1),    # Physical baryon density
    "ns": (0.5, 1.5),        # Scalar spectral index
    # PBUF-specific parameters
    "alpha": (1e-6, 1e-2),    # Elasticity amplitude
    "Rmax": (1e6, 1e12),     # Saturation length scale
    "eps0": (0.0, 2.0),      # Elasticity bias term
    "n_eps": (-2.0, 2.0),    # Evolution exponent
    "k_sat": (0.1, 2.0),     # Saturation coefficient
}

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


def build_params(
    model: str, 
    overrides: Optional[ParameterDict] = None,
    optimize_params: Optional[List[str]] = None
) -> ParameterDict:
    """
    Build parameter dictionary for specified model with optional overrides and optimization metadata.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        overrides: Optional parameter overrides to apply
        optimize_params: Optional list of parameters to mark for optimization
        
    Returns:
        Complete parameter dictionary with derived quantities and optimization metadata
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
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
    
    # Add optimization metadata
    optimization_metadata = create_parameter_metadata(model, optimize_params)
    params["_optimization_metadata"] = optimization_metadata
    
    # Add parameter classification
    classification = classify_parameters(model, optimize_params)
    params["_parameter_classification"] = classification
    
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


def get_optimizable_parameters(model: str) -> List[str]:
    """
    Get list of parameters that can be optimized for the specified model.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        
    Returns:
        List of parameter names that can be optimized
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    if model not in OPTIMIZABLE_PARAMETERS:
        raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
    
    return OPTIMIZABLE_PARAMETERS[model].copy()


def get_optimization_bounds(model: str, parameter: str) -> Tuple[float, float]:
    """
    Get optimization bounds for a specific parameter.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        parameter: Parameter name
        
    Returns:
        Tuple of (min_value, max_value) for optimization bounds
        
    Raises:
        ValueError: If parameter is not optimizable for the model
        
    Requirements: 1.1, 1.2, 1.3, 1.6
    """
    if model not in OPTIMIZABLE_PARAMETERS:
        raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
    
    if parameter not in OPTIMIZABLE_PARAMETERS[model]:
        raise ValueError(
            f"Parameter '{parameter}' is not optimizable for {model} model. "
            f"Optimizable parameters: {OPTIMIZABLE_PARAMETERS[model]}"
        )
    
    if parameter not in OPTIMIZATION_BOUNDS:
        raise ValueError(f"No optimization bounds defined for parameter '{parameter}'")
    
    return OPTIMIZATION_BOUNDS[parameter]


def validate_optimization_request(model: str, optimize_params: List[str]) -> bool:
    """
    Validate that requested parameters can be optimized for the specified model.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimize_params: List of parameter names to optimize
        
    Returns:
        True if all parameters are valid for optimization
        
    Raises:
        ValueError: If any parameter is invalid for optimization
        
    Requirements: 1.1, 1.2, 1.3, 1.6
    """
    if model not in OPTIMIZABLE_PARAMETERS:
        raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
    
    optimizable = set(OPTIMIZABLE_PARAMETERS[model])
    requested = set(optimize_params)
    
    # Check for invalid parameters
    invalid_params = requested - optimizable
    if invalid_params:
        raise ValueError(
            f"Invalid optimization parameters for {model} model: {invalid_params}. "
            f"Valid parameters: {sorted(optimizable)}"
        )
    
    # Validate bounds exist for all requested parameters
    missing_bounds = []
    for param in optimize_params:
        if param not in OPTIMIZATION_BOUNDS:
            missing_bounds.append(param)
    
    if missing_bounds:
        raise ValueError(f"No optimization bounds defined for parameters: {missing_bounds}")
    
    return True


def create_parameter_metadata(
    model: str, 
    optimize_params: Optional[List[str]] = None
) -> Dict[str, OptimizationMetadata]:
    """
    Create optimization metadata for all parameters in a model.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimize_params: List of parameters to mark as optimizable (optional)
        
    Returns:
        Dictionary mapping parameter names to optimization metadata
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    if model not in DEFAULTS:
        raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
    
    optimize_set = set(optimize_params) if optimize_params else set()
    
    # Validate optimization request if parameters specified
    if optimize_params:
        validate_optimization_request(model, optimize_params)
    
    metadata = {}
    
    for param_name in DEFAULTS[model]:
        # Skip non-numeric parameters
        if isinstance(DEFAULTS[model][param_name], str):
            metadata[param_name] = OptimizationMetadata(
                is_optimizable=False,
                optimization_bounds=None
            )
            continue
        
        # Check if parameter is optimizable for this model
        is_optimizable = param_name in OPTIMIZABLE_PARAMETERS.get(model, [])
        
        # Get bounds if parameter is optimizable
        bounds = None
        if is_optimizable and param_name in OPTIMIZATION_BOUNDS:
            bounds = OPTIMIZATION_BOUNDS[param_name]
        
        # Check if currently being optimized
        is_currently_optimized = param_name in optimize_set
        
        metadata[param_name] = OptimizationMetadata(
            is_optimizable=is_optimizable,
            optimization_bounds=bounds,
            is_currently_optimized=is_currently_optimized
        )
    
    return metadata


def classify_parameters(
    model: str, 
    optimize_params: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Classify parameters into optimizable, fixed, and currently optimized categories.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimize_params: List of parameters to optimize (optional)
        
    Returns:
        Dictionary with keys: 'optimizable', 'fixed', 'currently_optimized'
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
    """
    if model not in DEFAULTS:
        raise ValueError(f"Unknown model type: {model}. Must be 'lcdm' or 'pbuf'")
    
    optimize_set = set(optimize_params) if optimize_params else set()
    
    # Validate optimization request if parameters specified
    if optimize_params:
        validate_optimization_request(model, optimize_params)
    
    all_params = set(DEFAULTS[model].keys())
    optimizable_params = set(OPTIMIZABLE_PARAMETERS.get(model, []))
    
    # Filter out non-numeric parameters from optimizable set
    numeric_params = {
        name for name, value in DEFAULTS[model].items() 
        if isinstance(value, (int, float))
    }
    optimizable_params = optimizable_params & numeric_params
    
    classification = {
        'optimizable': sorted(optimizable_params),
        'fixed': sorted(all_params - optimizable_params),
        'currently_optimized': sorted(optimize_set)
    }
    
    return classification


def get_optimization_bounds_dict(model: str, optimize_params: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Get optimization bounds for multiple parameters as a dictionary.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimize_params: List of parameter names to get bounds for
        
    Returns:
        Dictionary mapping parameter names to (min, max) bounds tuples
        
    Requirements: 1.1, 1.2, 1.3, 1.6
    """
    # Validate the optimization request first
    validate_optimization_request(model, optimize_params)
    
    bounds_dict = {}
    for param in optimize_params:
        bounds_dict[param] = get_optimization_bounds(model, param)
    
    return bounds_dict


def is_parameter_optimizable(model: str, parameter: str) -> bool:
    """
    Check if a parameter can be optimized for the specified model.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        parameter: Parameter name to check
        
    Returns:
        True if parameter can be optimized, False otherwise
        
    Requirements: 1.1, 1.2, 1.3
    """
    if model not in OPTIMIZABLE_PARAMETERS:
        return False
    
    return parameter in OPTIMIZABLE_PARAMETERS[model]