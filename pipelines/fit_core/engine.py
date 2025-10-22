"""
Unified optimization engine for PBUF cosmology fitting.

This module provides the central orchestration and optimization engine used by all fitters.
It handles parameter building, likelihood dispatching, χ² summation, and result compilation.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
from scipy import optimize
from . import ParameterDict, ResultsDict, MetricsDict
from . import parameter
from . import likelihoods
from . import datasets
from . import statistics
from . import logging_utils


def run_fit(
    model: str, 
    datasets_list: List[str], 
    mode: str = "joint",
    overrides: Optional[ParameterDict] = None,
    optimizer_config: Optional[Dict[str, Any]] = None
) -> ResultsDict:
    """
    Central optimization function used by all fitters.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        datasets_list: List of dataset names to include in fit
        mode: Fitting mode ("joint" or "individual")
        overrides: Optional parameter overrides
        optimizer_config: Optional optimizer configuration
        
    Returns:
        Complete results dictionary with parameters, χ² breakdown, and metrics
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
    """
    # Validate inputs
    if model not in ["lcdm", "pbuf"]:
        raise ValueError(f"Invalid model: {model}. Must be 'lcdm' or 'pbuf'")
    
    if not datasets_list:
        raise ValueError("At least one dataset must be specified")
    
    # Build initial parameters using centralized parameter system
    initial_params = parameter.build_params(model, overrides)
    
    # Load and cache all requested datasets
    data_cache = {}
    for dataset_name in datasets_list:
        data_cache[dataset_name] = datasets.load_dataset(dataset_name)
    
    # Set default optimizer configuration if not provided
    if optimizer_config is None:
        optimizer_config = {
            "method": "minimize",
            "algorithm": "L-BFGS-B",
            "options": {"maxiter": 1000, "ftol": 1e-9}
        }
    
    # Build objective function for optimization
    objective_func = _build_objective_function(model, datasets_list, data_cache)
    
    # Execute optimization
    optimization_result = _execute_optimization(objective_func, initial_params, optimizer_config)
    
    # Extract optimized parameters
    optimized_params = optimization_result["params"]
    chi2_breakdown = optimization_result["chi2_breakdown"]
    
    # Compile final results with metrics and diagnostics
    results = _compile_results(optimized_params, chi2_breakdown, datasets_list)
    
    return results


def _build_objective_function(
    model: str,
    datasets_list: List[str],
    data_cache: Dict[str, Any]
) -> Callable:
    """
    Build the objective function for optimization.
    
    Args:
        model: Model type for parameter construction
        datasets_list: List of datasets to include
        data_cache: Cached dataset information
        
    Returns:
        Objective function for scipy.optimize
    """
    # Create likelihood dispatcher mapping
    likelihood_dispatch = {
        "cmb": likelihoods.likelihood_cmb,
        "bao": likelihoods.likelihood_bao,
        "bao_ani": likelihoods.likelihood_bao_ani,
        "sn": likelihoods.likelihood_sn
    }
    
    def objective(param_values: np.ndarray) -> float:
        """
        Objective function that computes total χ² for given parameter values.
        
        Args:
            param_values: Array of parameter values to optimize
            
        Returns:
            Total χ² across all datasets
        """
        try:
            # Convert parameter array back to parameter dictionary
            # This assumes a fixed parameter order - will be handled in task 6.2
            params = _array_to_params(param_values, model)
            
            total_chi2 = 0.0
            
            # Sum χ² contributions from all requested datasets
            for dataset_name in datasets_list:
                if dataset_name not in likelihood_dispatch:
                    raise ValueError(f"Unknown dataset: {dataset_name}")
                
                # Get likelihood function and data
                likelihood_func = likelihood_dispatch[dataset_name]
                data = data_cache[dataset_name]
                
                # Compute χ² for this dataset
                chi2, _ = likelihood_func(params, data)
                total_chi2 += chi2
            
            return total_chi2
            
        except Exception as e:
            # Return large χ² for invalid parameter combinations
            return 1e10
    
    return objective


def _execute_optimization(
    objective_func: Callable,
    initial_params: ParameterDict,
    optimizer_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute the optimization using configured method.
    
    Args:
        objective_func: Function to minimize
        initial_params: Starting parameter values
        optimizer_config: Optimization configuration
        
    Returns:
        Optimization result dictionary
    """
    # Extract model type from initial parameters
    model = "pbuf" if "alpha" in initial_params else "lcdm"
    
    # Convert initial parameters to array for optimization
    initial_array = _params_to_array(initial_params, model)
    
    # Set up parameter bounds based on model
    bounds = _get_parameter_bounds(model)
    
    # Execute optimization based on configured method
    method = optimizer_config.get("method", "minimize")
    
    if method == "minimize":
        algorithm = optimizer_config.get("algorithm", "L-BFGS-B")
        options = optimizer_config.get("options", {"maxiter": 1000, "ftol": 1e-9})
        
        result = optimize.minimize(
            objective_func,
            initial_array,
            method=algorithm,
            bounds=bounds,
            options=options
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        optimized_array = result.x
        final_chi2 = result.fun
        
    elif method == "differential_evolution":
        options = optimizer_config.get("options", {"maxiter": 1000, "atol": 1e-9})
        
        result = optimize.differential_evolution(
            objective_func,
            bounds,
            **options
        )
        
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        optimized_array = result.x
        final_chi2 = result.fun
        
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Convert optimized parameters back to dictionary
    optimized_params = _array_to_params(optimized_array, model)
    
    # Note: χ² breakdown will be computed in _compile_results
    # to avoid circular dependencies with data_cache
    chi2_breakdown = {}
    
    return {
        "params": optimized_params,
        "chi2_breakdown": chi2_breakdown,
        "total_chi2": final_chi2,
        "optimization_result": result
    }


def _compile_results(
    optimized_params: ParameterDict,
    chi2_breakdown: Dict[str, float],
    datasets_list: List[str]
) -> ResultsDict:
    """
    Compile final results dictionary with all metrics and diagnostics.
    
    Args:
        optimized_params: Final parameter values
        chi2_breakdown: χ² values by dataset
        datasets_list: List of fitted datasets
        
    Returns:
        Complete results dictionary
    """
    # Compute total χ² from breakdown
    total_chi2 = sum(chi2_breakdown.values()) if chi2_breakdown else 0.0
    
    # Count number of free parameters
    model = "pbuf" if "alpha" in optimized_params else "lcdm"
    n_params = len(_get_parameter_order(model))
    
    # Compute statistical metrics
    metrics = statistics.compute_metrics(total_chi2, n_params, datasets_list)
    
    # Compute detailed predictions for each dataset
    detailed_results = {}
    data_cache = {}
    
    # Load datasets and compute predictions
    likelihood_dispatch = {
        "cmb": likelihoods.likelihood_cmb,
        "bao": likelihoods.likelihood_bao,
        "bao_ani": likelihoods.likelihood_bao_ani,
        "sn": likelihoods.likelihood_sn
    }
    
    for dataset_name in datasets_list:
        # Load dataset
        data = datasets.load_dataset(dataset_name)
        data_cache[dataset_name] = data
        
        # Rebuild full parameter dictionary with derived quantities
        model = "pbuf" if "alpha" in optimized_params else "lcdm"
        full_params = parameter.build_params(model, optimized_params)
        
        # Compute predictions and χ² for this dataset
        likelihood_func = likelihood_dispatch[dataset_name]
        chi2, predictions = likelihood_func(full_params, data)
        
        detailed_results[dataset_name] = {
            "chi2": chi2,
            "predictions": predictions,
            "observations": data.get("observations", {}),
            "covariance": data.get("covariance", None)
        }
        
        # Update χ² breakdown (always compute for final results)
        chi2_breakdown[dataset_name] = chi2
    
    # Create structured results dictionary
    results = {
        "params": optimized_params,
        "results": detailed_results,
        "metrics": metrics,
        "chi2_breakdown": chi2_breakdown,
        "datasets": datasets_list,
        "model": model
    }
    
    # Log standardized results (basic logging for now)
    _log_basic_results(model, datasets_list, results, metrics)
    
    return results


def _log_basic_results(
    model: str, 
    datasets_list: List[str], 
    results: ResultsDict, 
    metrics: MetricsDict
) -> None:
    """
    Basic logging of results until full logging_utils is implemented.
    
    Args:
        model: Model type
        datasets_list: List of datasets
        results: Results dictionary
        metrics: Metrics dictionary
    """
    print(f"[RUN] model={model} datasets={','.join(datasets_list)} χ²={metrics.get('chi2', 0.0):.3f} AIC={metrics.get('aic', 0.0):.3f}")
    
    # Log key parameters
    params = results["params"]
    param_str = " ".join([f"{k}:{v:.3f}" for k, v in params.items() if isinstance(v, (int, float))])
    print(f"[PARAMS] {param_str}")
    
    # Log χ² breakdown
    breakdown = results["chi2_breakdown"]
    breakdown_str = " ".join([f"{k}:{v:.3f}" for k, v in breakdown.items()])
    print(f"[CHI2] {breakdown_str}")
    
    print(f"[METRICS] AIC={metrics.get('aic', 0.0):.3f} BIC={metrics.get('bic', 0.0):.3f} dof={metrics.get('dof', 0)} p={metrics.get('p_value', 0.0):.4f}")


def _array_to_params(param_values: np.ndarray, model: str) -> ParameterDict:
    """
    Convert parameter array back to parameter dictionary.
    
    Args:
        param_values: Array of parameter values
        model: Model type for parameter structure
        
    Returns:
        Parameter dictionary
    """
    # Get parameter order for the model
    param_order = _get_parameter_order(model)
    
    if len(param_values) != len(param_order):
        raise ValueError(f"Parameter array length {len(param_values)} doesn't match expected {len(param_order)}")
    
    # Build parameter dictionary from array values
    params = {}
    for i, param_name in enumerate(param_order):
        params[param_name] = float(param_values[i])
    
    # Add fixed parameters that aren't optimized
    defaults = parameter.get_defaults(model)
    for key, value in defaults.items():
        if key not in params:
            params[key] = value
    
    return params


def _params_to_array(params: ParameterDict, model: str) -> np.ndarray:
    """
    Convert parameter dictionary to array for optimization.
    
    Args:
        params: Parameter dictionary
        model: Model type for parameter structure
        
    Returns:
        Parameter values as array
    """
    # Get parameter order for the model
    param_order = _get_parameter_order(model)
    
    # Extract values in consistent order
    param_values = []
    for param_name in param_order:
        if param_name not in params:
            raise ValueError(f"Missing required parameter: {param_name}")
        param_values.append(params[param_name])
    
    return np.array(param_values)


def _get_parameter_order(model: str) -> List[str]:
    """
    Get the consistent parameter order for optimization arrays.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        
    Returns:
        List of parameter names in optimization order
    """
    if model == "lcdm":
        return ["H0", "Om0", "Obh2", "ns"]
    elif model == "pbuf":
        return ["H0", "Om0", "Obh2", "ns", "alpha", "Rmax", "eps0", "n_eps", "k_sat"]
    else:
        raise ValueError(f"Unknown model: {model}")


def _get_parameter_bounds(model: str) -> List[Tuple[float, float]]:
    """
    Get parameter bounds for optimization.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        
    Returns:
        List of (min, max) bounds for each parameter
    """
    if model == "lcdm":
        return [
            (50.0, 100.0),    # H0
            (0.1, 0.6),       # Om0
            (0.015, 0.035),   # Obh2
            (0.8, 1.2)        # ns
        ]
    elif model == "pbuf":
        return [
            (50.0, 100.0),    # H0
            (0.1, 0.6),       # Om0
            (0.015, 0.035),   # Obh2
            (0.8, 1.2),       # ns
            (1e-6, 1e-2),     # alpha
            (1e6, 1e12),      # Rmax
            (0.1, 1.0),       # eps0
            (-1.0, 1.0),      # n_eps
            (0.5, 1.5)        # k_sat
        ]
    else:
        raise ValueError(f"Unknown model: {model}")


def _compute_chi2_breakdown(
    params: ParameterDict, 
    datasets_list: List[str], 
    data_cache: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute χ² breakdown by dataset for final results.
    
    Args:
        params: Optimized parameter dictionary
        datasets_list: List of datasets
        data_cache: Cached dataset information
        
    Returns:
        Dictionary mapping dataset names to χ² values
    """
    likelihood_dispatch = {
        "cmb": likelihoods.likelihood_cmb,
        "bao": likelihoods.likelihood_bao,
        "bao_ani": likelihoods.likelihood_bao_ani,
        "sn": likelihoods.likelihood_sn
    }
    
    chi2_breakdown = {}
    
    for dataset_name in datasets_list:
        likelihood_func = likelihood_dispatch[dataset_name]
        data = data_cache[dataset_name]
        chi2, _ = likelihood_func(params, data)
        chi2_breakdown[dataset_name] = chi2
    
    return chi2_breakdown