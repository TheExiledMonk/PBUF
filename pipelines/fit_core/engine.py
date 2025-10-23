"""
Unified optimization engine for PBUF cosmology fitting.

This module provides the central orchestration and optimization engine used by all fitters.
It handles parameter building, likelihood dispatching, χ² summation, and result compilation.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import time
from scipy import optimize
from . import ParameterDict, ResultsDict, MetricsDict
from . import parameter
from . import likelihoods
from . import datasets
from . import statistics
from . import logging_utils
from .optimizer import ParameterOptimizer, optimize_cmb_parameters, OptimizationResult
from .parameter_store import OptimizedParameterStore


def run_fit(
    model: str, 
    datasets_list: List[str], 
    mode: str = "joint",
    overrides: Optional[ParameterDict] = None,
    optimizer_config: Optional[Dict[str, Any]] = None,
    optimize_params: Optional[List[str]] = None,
    covariance_scaling: float = 1.0,
    dry_run: bool = False,
    warm_start: bool = False
) -> ResultsDict:
    """
    Central optimization function used by all fitters with parameter optimization support.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        datasets_list: List of dataset names to include in fit
        mode: Fitting mode ("joint" or "individual")
        overrides: Optional parameter overrides
        optimizer_config: Optional optimizer configuration
        optimize_params: Optional list of parameters to optimize (enables optimization mode)
        covariance_scaling: Scaling factor for covariance matrices (default: 1.0)
        dry_run: If True, perform optimization without persisting results
        warm_start: If True, use recent optimization results as starting point
        
    Returns:
        Complete results dictionary with parameters, χ² breakdown, and metrics
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.2
    """
    # Validate inputs
    if model not in ["lcdm", "pbuf"]:
        raise ValueError(f"Invalid model: {model}. Must be 'lcdm' or 'pbuf'")
    
    if not datasets_list:
        raise ValueError("At least one dataset must be specified")
    
    # Pre-run dataset verification
    print("🔍 Verifying datasets before fitting...")
    if not datasets.verify_all_datasets(datasets_list):
        raise RuntimeError("Dataset verification failed. Cannot proceed with fitting.")
    print("✅ All datasets verified successfully")
    
    # Validate BAO dataset separation to prevent joint use of bao and bao_ani
    try:
        from .bao_aniso_validation import validate_bao_dataset_separation, print_dataset_separation_warning
        validate_bao_dataset_separation(datasets_list)
        print_dataset_separation_warning(datasets_list)
    except ImportError:
        print("⚠️  BAO dataset separation validation not available")
    except Exception as e:
        raise ValueError(f"Dataset configuration error: {e}")
    
    # Initialize parameter store for optimization support
    param_store = OptimizedParameterStore()
    
    # Check if optimization mode is enabled
    if optimize_params is not None and len(optimize_params) > 0:
        return _run_optimization_fit(
            model=model,
            datasets_list=datasets_list,
            optimize_params=optimize_params,
            overrides=overrides,
            covariance_scaling=covariance_scaling,
            dry_run=dry_run,
            warm_start=warm_start,
            param_store=param_store,
            optimizer_config=optimizer_config
        )
    
    # Standard fitting mode - use optimized parameters from store if available
    base_params = param_store.get_model_defaults(model)
    if overrides:
        base_params.update(overrides)
    
    # Build initial parameters using centralized parameter system
    initial_params = parameter.build_params(model, base_params)
    
    # Check degrees of freedom and add datasets if needed
    n_params = len([k for k, v in initial_params.items() if isinstance(v, (int, float))])
    
    # Apply BAO anisotropic validation - no auto-add policy
    if "bao_ani" in datasets_list:
        try:
            from .bao_aniso_validation import validate_no_auto_add_datasets, freeze_weakly_constrained_parameters
            validate_no_auto_add_datasets(datasets_list)
        except ImportError:
            print("⚠️  BAO anisotropic validation not available")
    
    # Check if current datasets have sufficient degrees of freedom
    try:
        from . import statistics
        statistics.compute_dof(datasets_list, n_params)
        datasets_to_use = datasets_list
    except ValueError as e:
        if "negative" in str(e):
            # Handle insufficient DOF based on dataset type
            if "bao_ani" in datasets_list:
                # For BAO anisotropic: freeze parameters instead of adding datasets
                print("🔒 BAO anisotropic detected with low DOF. Freezing weakly-constrained parameters...")
                try:
                    from .bao_aniso_validation import freeze_weakly_constrained_parameters
                    initial_params = freeze_weakly_constrained_parameters(initial_params)
                    # Recompute parameter count
                    n_params = len([k for k, v in initial_params.items() 
                                   if isinstance(v, (int, float)) or 
                                   (isinstance(v, dict) and not v.get("fixed", False))])
                    datasets_to_use = datasets_list
                    print(f"✅ Reduced to {n_params} free parameters for BAO anisotropic fitting")
                except ImportError:
                    print("⚠️  Cannot freeze parameters, validation module unavailable")
                    datasets_to_use = datasets_list
            else:
                # Standard auto-add policy for other datasets
                if len(datasets_list) == 1:
                    # Single dataset case
                    single_dataset = datasets_list[0]
                    if single_dataset == "cmb":
                        datasets_to_use = ["cmb", "bao"]
                        print("Note: Adding BAO data to CMB for sufficient degrees of freedom")
                    elif single_dataset == "bao":
                        datasets_to_use = ["bao", "sn"]
                        print("Note: Adding SN data to BAO for sufficient degrees of freedom")
                    elif single_dataset == "sn":
                        # SN has many data points, this shouldn't happen, but add BAO as backup
                        datasets_to_use = ["sn", "bao"]
                        print("Note: Adding BAO data to SN for sufficient degrees of freedom")
                    else:
                        # For other datasets, add CMB and BAO
                        datasets_to_use = [single_dataset, "cmb", "bao"]
                        print(f"Note: Adding CMB and BAO data to {single_dataset} for sufficient degrees of freedom")
                else:
                    # Multiple datasets still insufficient (likely PBUF model), add all datasets
                    all_datasets = ["cmb", "bao", "sn"]
                    datasets_to_use = list(set(datasets_list + all_datasets))  # Remove duplicates, preserve order
                    print(f"Note: Adding all available datasets for sufficient degrees of freedom (PBUF model needs more data)")
                    
                    # Check if even all datasets are sufficient
                    try:
                        statistics.compute_dof(datasets_to_use, n_params)
                    except ValueError:
                        # Even all datasets insufficient - this is a fundamental problem
                        print(f"Warning: Even with all datasets, DOF may be insufficient for {model} model with {n_params} parameters")
                        # Continue anyway - let the test fail with a clear message
        else:
            # Re-raise the error for other cases
            raise e
    
    # Load and cache all requested datasets
    data_cache = {}
    for dataset_name in datasets_to_use:
        data_cache[dataset_name] = datasets.load_dataset(dataset_name)
    
    # Set default optimizer configuration if not provided
    if optimizer_config is None:
        optimizer_config = {
            "method": "minimize",
            "algorithm": "L-BFGS-B",
            "options": {"maxiter": 1000, "ftol": 1e-9}
        }
    
    # Build objective function for optimization
    objective_func = _build_objective_function(model, datasets_to_use, data_cache)
    
    # Execute optimization
    optimization_result = _execute_optimization(objective_func, initial_params, optimizer_config)
    
    # Extract optimized parameters
    optimized_params = optimization_result["params"]
    chi2_breakdown = optimization_result["chi2_breakdown"]
    
    # Compile final results with metrics and diagnostics
    results = _compile_results(optimized_params, chi2_breakdown, datasets_to_use)
    
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
    # Count number of free parameters
    model = "pbuf" if "alpha" in optimized_params else "lcdm"
    n_params = len(_get_parameter_order(model))
    
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
        
        # Get provenance information for this dataset
        provenance = datasets.get_dataset_provenance(dataset_name)
        
        detailed_results[dataset_name] = {
            "chi2": chi2,
            "predictions": predictions,
            "observations": data.get("observations", {}),
            "covariance": data.get("covariance", None),
            "provenance": provenance
        }
        
        # Update χ² breakdown (always compute for final results)
        chi2_breakdown[dataset_name] = chi2
    
    # Compute total χ² from updated breakdown
    total_chi2 = sum(chi2_breakdown.values()) if chi2_breakdown else 0.0
    
    # Compute statistical metrics with correct total_chi2
    metrics = statistics.compute_metrics(total_chi2, n_params, datasets_list)
    
    # Ensure model_class is included in params for display purposes
    params_with_model = optimized_params.copy()
    params_with_model["model_class"] = model
    
    # Collect dataset provenance information
    dataset_provenance = {}
    for dataset_name in datasets_list:
        provenance = datasets.get_dataset_provenance(dataset_name)
        if provenance:
            dataset_provenance[dataset_name] = provenance
    
    # Create structured results dictionary
    results = {
        "params": params_with_model,
        "results": detailed_results,
        "metrics": metrics,
        "chi2_breakdown": chi2_breakdown,
        "datasets": datasets_list,
        "model": model,
        "diagnostics": {
            "total_datasets": len(datasets_list),
            "total_data_points": sum(len(data_cache[ds].get("observations", [])) if isinstance(data_cache[ds].get("observations"), list) else 1 for ds in datasets_list),
            "convergence_status": "completed"
        },
        "dataset_provenance": dataset_provenance if dataset_provenance else None
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


def _run_optimization_fit(
    model: str,
    datasets_list: List[str],
    optimize_params: List[str],
    overrides: Optional[ParameterDict] = None,
    covariance_scaling: float = 1.0,
    dry_run: bool = False,
    warm_start: bool = False,
    param_store: OptimizedParameterStore = None,
    optimizer_config: Optional[Dict[str, Any]] = None
) -> ResultsDict:
    """
    Execute optimization-enabled fitting with parameter optimization and storage.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        datasets_list: List of dataset names to include in optimization
        optimize_params: List of parameter names to optimize
        overrides: Optional parameter overrides
        covariance_scaling: Scaling factor for covariance matrices
        dry_run: If True, perform optimization without persisting results
        warm_start: If True, use recent optimization results as starting point
        param_store: Parameter store instance for optimization results
        optimizer_config: Optional optimizer configuration
        
    Returns:
        Complete results dictionary with optimization metadata
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    if param_store is None:
        param_store = OptimizedParameterStore()
    
    # Check degrees of freedom and expand datasets if needed
    from . import parameter
    initial_params = parameter.build_params(model, overrides)
    n_params = len(optimize_params)
    
    # Check if current datasets have sufficient degrees of freedom
    try:
        from . import statistics
        statistics.compute_dof(datasets_list, n_params)
        datasets_to_use = datasets_list
    except ValueError as e:
        if "negative" in str(e):
            # Insufficient DOF, add more datasets
            if len(datasets_list) == 1:
                # Single dataset case
                single_dataset = datasets_list[0]
                if single_dataset == "cmb":
                    datasets_to_use = ["cmb", "bao"]
                    print("Note: Adding BAO data to CMB for sufficient degrees of freedom")
                elif single_dataset == "bao":
                    datasets_to_use = ["bao", "sn"]
                    print("Note: Adding SN data to BAO for sufficient degrees of freedom")
                elif single_dataset == "sn":
                    # SN has many data points, this shouldn't happen, but add BAO as backup
                    datasets_to_use = ["sn", "bao"]
                    print("Note: Adding BAO data to SN for sufficient degrees of freedom")
                else:
                    # For other datasets, add CMB and BAO
                    datasets_to_use = [single_dataset, "cmb", "bao"]
                    print(f"Note: Adding CMB and BAO data to {single_dataset} for sufficient degrees of freedom")
            else:
                # Multiple datasets still insufficient (likely PBUF model), add all datasets
                all_datasets = ["cmb", "bao", "sn"]
                datasets_to_use = list(set(datasets_list + all_datasets))  # Remove duplicates, preserve order
                print(f"Note: Adding all available datasets for sufficient degrees of freedom (PBUF model needs more data)")
                
            # Final check with updated datasets
            try:
                statistics.compute_dof(datasets_to_use, n_params)
            except ValueError:
                # Even with additional datasets, DOF is insufficient - this is a fundamental problem
                print(f"Warning: Even with additional datasets {datasets_to_use}, DOF may be insufficient for {model} model with {n_params} parameters")
                # Continue anyway - let the test fail with a clear message
        else:
            # Re-raise the error for other cases
            raise e
    else:
        datasets_to_use = datasets_list
    
    # Get starting parameters with warm-start support
    starting_params = _get_optimization_starting_params(
        model, param_store, overrides, warm_start
    )
    
    # Execute parameter optimization
    optimization_result = _execute_parameter_optimization(
        model=model,
        datasets_list=datasets_to_use,
        optimize_params=optimize_params,
        starting_params=starting_params,
        covariance_scaling=covariance_scaling,
        optimizer_config=optimizer_config
    )
    
    # Process and store optimization results
    final_params = _process_optimization_results(
        optimization_result=optimization_result,
        param_store=param_store,
        dry_run=dry_run
    )
    
    # Propagate optimized parameters to subsequent fits
    if not dry_run:
        _propagate_optimized_parameters(
            model=model,
            optimized_params=optimization_result.optimized_params,
            param_store=param_store
        )
    
    # Build final results with optimization metadata
    results = _build_optimization_results(
        model=model,
        datasets_list=datasets_to_use,
        final_params=final_params,
        optimization_result=optimization_result
    )
    
    return results


def _get_optimization_starting_params(
    model: str,
    param_store: OptimizedParameterStore,
    overrides: Optional[ParameterDict],
    warm_start: bool
) -> ParameterDict:
    """
    Get starting parameters for optimization with warm-start support.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        param_store: Parameter store instance
        overrides: Optional parameter overrides
        warm_start: Whether to use warm-start parameters
        
    Returns:
        Starting parameter dictionary for optimization
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Start with current model defaults (may include previous optimizations)
    starting_params = param_store.get_model_defaults(model)
    
    # Apply warm-start if requested and available
    if warm_start:
        warm_start_params = param_store.get_warm_start_params(model)
        if warm_start_params:
            print(f"[WARM_START] Using recent optimization results for {model} model")
            starting_params.update(warm_start_params)
        else:
            print(f"[WARM_START] No recent optimization results found for {model} model")
    
    # Apply any explicit overrides
    if overrides:
        starting_params.update(overrides)
    
    return starting_params


def _execute_parameter_optimization(
    model: str,
    datasets_list: List[str],
    optimize_params: List[str],
    starting_params: ParameterDict,
    covariance_scaling: float,
    optimizer_config: Optional[Dict[str, Any]]
) -> OptimizationResult:
    """
    Execute parameter optimization using the appropriate optimization routine.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        datasets_list: List of dataset names
        optimize_params: List of parameters to optimize
        starting_params: Starting parameter values
        covariance_scaling: Covariance scaling factor
        optimizer_config: Optional optimizer configuration
        
    Returns:
        OptimizationResult with optimization details
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Use CMB-specific optimization for CMB datasets
    if datasets_list == ["cmb"]:
        return optimize_cmb_parameters(
            model=model,
            optimize_params=optimize_params,
            starting_params=starting_params,
            covariance_scaling=covariance_scaling,
            dry_run=False  # dry_run is handled at higher level
        )
    else:
        # Use general parameter optimizer for other dataset combinations
        optimizer = ParameterOptimizer(covariance_scaling=covariance_scaling)
        return optimizer.optimize_parameters(
            model=model,
            datasets_list=datasets_list,
            optimize_params=optimize_params,
            starting_values=starting_params,
            optimizer_config=optimizer_config
        )


def _process_optimization_results(
    optimization_result: OptimizationResult,
    param_store: OptimizedParameterStore,
    dry_run: bool
) -> ParameterDict:
    """
    Process optimization results and update parameter store if not dry run.
    
    Args:
        optimization_result: Results from parameter optimization
        param_store: Parameter store instance
        dry_run: Whether this is a dry run (no persistence)
        
    Returns:
        Final parameter dictionary with optimized values
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Get current model defaults
    current_params = param_store.get_model_defaults(optimization_result.model)
    
    # Merge optimized parameters with current defaults
    final_params = current_params.copy()
    final_params.update(optimization_result.optimized_params)
    
    # Update parameter store if not dry run
    if not dry_run:
        optimization_metadata = {
            "optimization_timestamp": optimization_result.metadata.get("timestamp"),
            "source_dataset": ",".join(optimization_result.metadata.get("datasets", [])),
            "optimized_params": list(optimization_result.optimized_params.keys()),
            "chi2_improvement": optimization_result.chi2_improvement,
            "convergence_status": optimization_result.convergence_status,
            "optimizer_info": optimization_result.optimizer_info,
            "covariance_scaling": optimization_result.covariance_scaling,
            "final_chi2": optimization_result.final_chi2
        }
        
        param_store.update_model_defaults(
            model=optimization_result.model,
            optimized_params=optimization_result.optimized_params,
            optimization_metadata=optimization_metadata,
            dry_run=False
        )
        
        print(f"[OPTIMIZATION] Updated {optimization_result.model} defaults with optimized parameters")
    else:
        print(f"[DRY_RUN] Optimization completed but results not persisted")
    
    return final_params


def _propagate_optimized_parameters(
    model: str,
    optimized_params: Dict[str, float],
    param_store: OptimizedParameterStore
) -> None:
    """
    Propagate optimized parameters to ensure consistent use across all fitters.
    
    Args:
        model: Model type that was optimized
        optimized_params: Dictionary of optimized parameter values
        param_store: Parameter store instance
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Validate cross-model consistency after parameter updates
    consistency_results = param_store.validate_cross_model_consistency(
        tolerance=1e-3, 
        log_warnings=True
    )
    
    # Log parameter propagation
    param_names = list(optimized_params.keys())
    print(f"[PROPAGATION] Optimized parameters for {model}: {param_names}")
    
    # Check if any shared parameters were optimized
    shared_params = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb"}
    optimized_shared = [p for p in param_names if p in shared_params]
    
    if optimized_shared:
        print(f"[PROPAGATION] Shared parameters optimized: {optimized_shared}")
        print(f"[PROPAGATION] These will be automatically used by all subsequent fits")
    
    # Log consistency validation results
    summary = consistency_results.get("_summary", {})
    if not summary.get("is_fully_consistent", True):
        divergent = summary.get("divergent_params", [])
        print(f"[CONSISTENCY] Warning: Parameter divergence detected in {divergent}")


def _build_optimization_results(
    model: str,
    datasets_list: List[str],
    final_params: ParameterDict,
    optimization_result: OptimizationResult
) -> ResultsDict:
    """
    Build final results dictionary with optimization metadata.
    
    Args:
        model: Model type
        datasets_list: List of datasets used in optimization
        final_params: Final parameter values after optimization
        optimization_result: Optimization result details
        
    Returns:
        Complete results dictionary with optimization metadata
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Build full parameter dictionary with derived quantities
    full_params = parameter.build_params(model, final_params)
    
    # Load datasets and compute final predictions
    data_cache = {}
    detailed_results = {}
    chi2_breakdown = {}
    
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
        
        # Compute predictions and χ² for this dataset
        likelihood_func = likelihood_dispatch[dataset_name]
        chi2, predictions = likelihood_func(full_params, data)
        
        detailed_results[dataset_name] = {
            "chi2": chi2,
            "predictions": predictions,
            "observations": data.get("observations", {}),
            "covariance": data.get("covariance", None)
        }
        
        chi2_breakdown[dataset_name] = chi2
    
    # Compute statistical metrics
    total_chi2 = sum(chi2_breakdown.values())
    n_params = len(optimization_result.optimized_params)
    metrics = statistics.compute_metrics(total_chi2, n_params, datasets_list)
    
    # Add optimization-specific metrics
    metrics["optimization"] = {
        "optimized_parameters": list(optimization_result.optimized_params.keys()),
        "chi2_improvement": optimization_result.chi2_improvement,
        "convergence_status": optimization_result.convergence_status,
        "optimization_time": optimization_result.optimization_time,
        "function_evaluations": optimization_result.n_function_evaluations,
        "bounds_reached": optimization_result.bounds_reached,
        "covariance_scaling": optimization_result.covariance_scaling
    }
    
    # Create structured results dictionary
    results = {
        "params": final_params,
        "results": detailed_results,
        "metrics": metrics,
        "chi2_breakdown": chi2_breakdown,
        "datasets": datasets_list,
        "model": model,
        "optimization_result": optimization_result
    }
    
    # Log optimization results
    _log_optimization_results(model, datasets_list, results, optimization_result)
    
    return results


def _log_optimization_results(
    model: str,
    datasets_list: List[str],
    results: ResultsDict,
    optimization_result: OptimizationResult
) -> None:
    """
    Log optimization results with detailed parameter and performance information.
    
    Args:
        model: Model type
        datasets_list: List of datasets
        results: Results dictionary
        optimization_result: Optimization result details
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    metrics = results["metrics"]
    
    # Log basic run information
    print(f"[OPTIMIZATION_RUN] model={model} datasets={','.join(datasets_list)} χ²={optimization_result.final_chi2:.3f}")
    
    # Log optimized vs fixed parameters
    optimized_params = optimization_result.optimized_params
    all_params = results["params"]
    
    opt_str = " ".join([f"{k}:{v:.3f}*" for k, v in optimized_params.items()])
    fixed_params = {k: v for k, v in all_params.items() 
                   if k not in optimized_params and isinstance(v, (int, float))}
    fixed_str = " ".join([f"{k}:{v:.3f}" for k, v in list(fixed_params.items())[:5]])  # Limit output
    
    print(f"[OPTIMIZED] {opt_str}")
    if fixed_str:
        print(f"[FIXED] {fixed_str}")
    
    # Log χ² breakdown and improvement
    breakdown = results["chi2_breakdown"]
    breakdown_str = " ".join([f"{k}:{v:.3f}" for k, v in breakdown.items()])
    print(f"[CHI2] {breakdown_str} improvement:{optimization_result.chi2_improvement:.3f}")
    
    # Log performance metrics
    print(f"[PERFORMANCE] time={optimization_result.optimization_time:.3f}s evals={optimization_result.n_function_evaluations} status={optimization_result.convergence_status}")
    
    # Log statistical metrics
    print(f"[METRICS] AIC={metrics.get('aic', 0.0):.3f} BIC={metrics.get('bic', 0.0):.3f} dof={metrics.get('dof', 0)} p={metrics.get('p_value', 0.0):.4f}")
    
    # Log bounds information if any were reached
    if optimization_result.bounds_reached:
        print(f"[BOUNDS] reached={','.join(optimization_result.bounds_reached)}")


def run_cmb_optimization_workflow(
    model: str,
    optimize_params: Optional[List[str]] = None,
    covariance_scaling: float = 1.0,
    dry_run: bool = False,
    warm_start: bool = False,
    auto_propagate: bool = True
) -> Dict[str, Any]:
    """
    Execute full CMB optimization workflow for specified model with automatic parameter propagation.
    
    This function provides a complete CMB optimization workflow that:
    1. Validates dataset integrity
    2. Performs parameter optimization
    3. Validates optimization results
    4. Propagates optimized parameters with dataset tagging
    5. Generates comprehensive optimization report
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimize_params: List of parameters to optimize (uses model defaults if None)
        covariance_scaling: Scaling factor for CMB covariance matrix
        dry_run: If True, perform optimization without persisting results
        warm_start: If True, use recent optimization results as starting point
        auto_propagate: If True, automatically propagate results to other models
        
    Returns:
        Dictionary with complete workflow results and metadata
        
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5
    """
    workflow_start_time = time.time()
    
    # Set default optimization parameters if not specified
    if optimize_params is None:
        if model == "lcdm":
            optimize_params = ["H0", "Om0", "Obh2", "ns"]
        elif model == "pbuf":
            optimize_params = ["H0", "Om0", "Obh2", "ns", "k_sat", "alpha"]
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    print(f"[CMB_WORKFLOW] Starting CMB optimization workflow for {model} model")
    print(f"[CMB_WORKFLOW] Optimizing parameters: {optimize_params}")
    
    # Initialize parameter store
    param_store = OptimizedParameterStore()
    
    # Step 1: Validate dataset integrity
    print(f"[CMB_WORKFLOW] Step 1: Validating CMB dataset integrity")
    try:
        cmb_data = datasets.load_dataset("cmb")
        # Basic validation - ensure required fields exist
        required_fields = ["observations", "covariance"]
        for field in required_fields:
            if field not in cmb_data:
                raise ValueError(f"Missing required field: {field}")
        print(f"[CMB_WORKFLOW] Dataset integrity validation passed")
    except Exception as e:
        raise ValueError(f"CMB dataset integrity validation failed: {str(e)}")
    
    # Step 2: Execute CMB optimization
    print(f"[CMB_WORKFLOW] Step 2: Executing CMB parameter optimization")
    
    optimization_results = run_fit(
        model=model,
        datasets_list=["cmb"],
        optimize_params=optimize_params,
        covariance_scaling=covariance_scaling,
        dry_run=dry_run,
        warm_start=warm_start
    )
    
    optimization_result = optimization_results.get("optimization_result")
    if optimization_result is None:
        raise RuntimeError("Optimization failed to produce results")
    
    # Step 3: Validate optimization results
    print(f"[CMB_WORKFLOW] Step 3: Validating optimization results")
    validation_results = _validate_cmb_optimization_results(optimization_result, model)
    
    # Step 4: Automatic parameter propagation with dataset tagging
    propagation_results = {}
    if auto_propagate and not dry_run:
        print(f"[CMB_WORKFLOW] Step 4: Propagating optimized parameters with dataset tagging")
        propagation_results = _propagate_cmb_optimization_results(
            optimization_result=optimization_result,
            param_store=param_store,
            source_model=model
        )
    else:
        print(f"[CMB_WORKFLOW] Step 4: Skipping parameter propagation (auto_propagate={auto_propagate}, dry_run={dry_run})")
    
    # Step 5: Generate optimization report
    print(f"[CMB_WORKFLOW] Step 5: Generating optimization report")
    workflow_time = time.time() - workflow_start_time
    
    workflow_report = {
        "workflow_metadata": {
            "model": model,
            "optimize_params": optimize_params,
            "covariance_scaling": covariance_scaling,
            "dry_run": dry_run,
            "warm_start": warm_start,
            "auto_propagate": auto_propagate,
            "workflow_time": workflow_time,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "optimization_results": optimization_results,
        "validation_results": validation_results,
        "propagation_results": propagation_results,
        "summary": {
            "success": validation_results.get("overall_status") == "success",
            "chi2_improvement": optimization_result.chi2_improvement,
            "convergence_status": optimization_result.convergence_status,
            "optimized_parameters": list(optimization_result.optimized_params.keys()),
            "final_chi2": optimization_result.final_chi2
        }
    }
    
    print(f"[CMB_WORKFLOW] Workflow completed successfully in {workflow_time:.3f}s")
    print(f"[CMB_WORKFLOW] χ² improvement: {optimization_result.chi2_improvement:.6f}")
    
    return workflow_report


def run_dual_model_cmb_optimization(
    optimize_params_lcdm: Optional[List[str]] = None,
    optimize_params_pbuf: Optional[List[str]] = None,
    covariance_scaling: float = 1.0,
    dry_run: bool = False,
    warm_start: bool = False,
    validate_consistency: bool = True
) -> Dict[str, Any]:
    """
    Execute CMB optimization workflow for both ΛCDM and PBUF models with cross-model validation.
    
    Args:
        optimize_params_lcdm: Parameters to optimize for ΛCDM model
        optimize_params_pbuf: Parameters to optimize for PBUF model
        covariance_scaling: Scaling factor for CMB covariance matrix
        dry_run: If True, perform optimization without persisting results
        warm_start: If True, use recent optimization results as starting point
        validate_consistency: If True, validate cross-model parameter consistency
        
    Returns:
        Dictionary with dual model optimization results and cross-model analysis
        
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5
    """
    print(f"[DUAL_CMB_WORKFLOW] Starting dual model CMB optimization workflow")
    
    dual_results = {
        "lcdm": {},
        "pbuf": {},
        "cross_model_analysis": {},
        "summary": {}
    }
    
    # Execute ΛCDM optimization
    print(f"[DUAL_CMB_WORKFLOW] Optimizing ΛCDM model")
    try:
        dual_results["lcdm"] = run_cmb_optimization_workflow(
            model="lcdm",
            optimize_params=optimize_params_lcdm,
            covariance_scaling=covariance_scaling,
            dry_run=dry_run,
            warm_start=warm_start,
            auto_propagate=False  # Handle propagation at dual level
        )
    except Exception as e:
        dual_results["lcdm"] = {"error": str(e), "success": False}
        print(f"[DUAL_CMB_WORKFLOW] ΛCDM optimization failed: {str(e)}")
    
    # Execute PBUF optimization
    print(f"[DUAL_CMB_WORKFLOW] Optimizing PBUF model")
    try:
        dual_results["pbuf"] = run_cmb_optimization_workflow(
            model="pbuf",
            optimize_params=optimize_params_pbuf,
            covariance_scaling=covariance_scaling,
            dry_run=dry_run,
            warm_start=warm_start,
            auto_propagate=False  # Handle propagation at dual level
        )
    except Exception as e:
        dual_results["pbuf"] = {"error": str(e), "success": False}
        print(f"[DUAL_CMB_WORKFLOW] PBUF optimization failed: {str(e)}")
    
    # Cross-model consistency validation
    if validate_consistency and not dry_run:
        print(f"[DUAL_CMB_WORKFLOW] Validating cross-model parameter consistency")
        param_store = OptimizedParameterStore()
        dual_results["cross_model_analysis"] = param_store.validate_cross_model_consistency(
            tolerance=1e-3,
            log_warnings=True
        )
    
    # Generate summary
    lcdm_success = dual_results["lcdm"].get("summary", {}).get("success", False)
    pbuf_success = dual_results["pbuf"].get("summary", {}).get("success", False)
    
    dual_results["summary"] = {
        "lcdm_success": lcdm_success,
        "pbuf_success": pbuf_success,
        "both_successful": lcdm_success and pbuf_success,
        "consistency_validated": validate_consistency and not dry_run,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    
    if lcdm_success and pbuf_success:
        lcdm_chi2 = dual_results["lcdm"]["summary"]["final_chi2"]
        pbuf_chi2 = dual_results["pbuf"]["summary"]["final_chi2"]
        dual_results["summary"]["chi2_comparison"] = {
            "lcdm_chi2": lcdm_chi2,
            "pbuf_chi2": pbuf_chi2,
            "pbuf_improvement": lcdm_chi2 - pbuf_chi2
        }
    
    print(f"[DUAL_CMB_WORKFLOW] Dual model optimization completed")
    print(f"[DUAL_CMB_WORKFLOW] ΛCDM success: {lcdm_success}, PBUF success: {pbuf_success}")
    
    return dual_results


def _validate_cmb_optimization_results(
    optimization_result: OptimizationResult,
    model: str
) -> Dict[str, Any]:
    """
    Validate CMB optimization results for physical consistency and convergence quality.
    
    Args:
        optimization_result: Results from CMB optimization
        model: Model type for validation rules
        
    Returns:
        Dictionary with detailed validation results
        
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5
    """
    validation_results = {
        "convergence_validation": {},
        "parameter_validation": {},
        "physics_validation": {},
        "performance_validation": {},
        "overall_status": "unknown"
    }
    
    # Convergence validation
    convergence_ok = optimization_result.convergence_status == "success"
    improvement_ok = optimization_result.chi2_improvement >= -1e-6  # Allow small numerical errors
    
    validation_results["convergence_validation"] = {
        "convergence_status": optimization_result.convergence_status,
        "convergence_ok": convergence_ok,
        "chi2_improvement": optimization_result.chi2_improvement,
        "improvement_ok": improvement_ok,
        "final_chi2": optimization_result.final_chi2
    }
    
    # Parameter validation
    param_validation = {}
    bounds_violations = []
    
    for param_name, value in optimization_result.optimized_params.items():
        try:
            min_bound, max_bound = parameter.get_optimization_bounds(model, param_name)
            within_bounds = min_bound <= value <= max_bound
            param_validation[param_name] = {
                "value": value,
                "bounds": [min_bound, max_bound],
                "within_bounds": within_bounds
            }
            if not within_bounds:
                bounds_violations.append(param_name)
        except ValueError:
            # Parameter might not have bounds defined
            param_validation[param_name] = {
                "value": value,
                "bounds": None,
                "within_bounds": True
            }
    
    validation_results["parameter_validation"] = {
        "parameters": param_validation,
        "bounds_violations": bounds_violations,
        "all_within_bounds": len(bounds_violations) == 0
    }
    
    # Physics validation
    try:
        # Build full parameter set and validate physics
        full_params = parameter.get_defaults(model)
        full_params.update(optimization_result.optimized_params)
        parameter.validate_params(full_params, model)
        physics_ok = True
        physics_error = None
    except Exception as e:
        physics_ok = False
        physics_error = str(e)
    
    validation_results["physics_validation"] = {
        "physics_ok": physics_ok,
        "physics_error": physics_error
    }
    
    # Performance validation
    reasonable_time = optimization_result.optimization_time < 300  # 5 minutes
    reasonable_evals = optimization_result.n_function_evaluations < 10000
    
    validation_results["performance_validation"] = {
        "optimization_time": optimization_result.optimization_time,
        "function_evaluations": optimization_result.n_function_evaluations,
        "reasonable_time": reasonable_time,
        "reasonable_evaluations": reasonable_evals,
        "bounds_reached": optimization_result.bounds_reached
    }
    
    # Overall status
    all_checks = [
        convergence_ok,
        improvement_ok,
        len(bounds_violations) == 0,
        physics_ok,
        reasonable_time,
        reasonable_evals
    ]
    
    if all(all_checks):
        validation_results["overall_status"] = "success"
    elif convergence_ok and physics_ok:
        validation_results["overall_status"] = "warning"
    else:
        validation_results["overall_status"] = "failure"
    
    return validation_results


def _propagate_cmb_optimization_results(
    optimization_result: OptimizationResult,
    param_store: OptimizedParameterStore,
    source_model: str
) -> Dict[str, Any]:
    """
    Propagate CMB optimization results with dataset tagging and cross-model updates.
    
    Args:
        optimization_result: Results from CMB optimization
        param_store: Parameter store instance
        source_model: Model that was optimized
        
    Returns:
        Dictionary with propagation results and metadata
        
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5
    """
    propagation_results = {
        "source_model": source_model,
        "optimized_params": optimization_result.optimized_params,
        "dataset_tagging": {},
        "cross_model_updates": {},
        "consistency_check": {}
    }
    
    # Dataset tagging - mark parameters as CMB-optimized
    dataset_tag = {
        "source_dataset": "cmb",
        "optimization_timestamp": optimization_result.metadata.get("timestamp"),
        "chi2_improvement": optimization_result.chi2_improvement,
        "convergence_status": optimization_result.convergence_status
    }
    
    propagation_results["dataset_tagging"] = dataset_tag
    
    # Identify shared parameters that should be propagated to other model
    shared_params = {"H0", "Om0", "Obh2", "ns", "Neff", "Tcmb"}
    optimized_shared = {k: v for k, v in optimization_result.optimized_params.items() 
                       if k in shared_params}
    
    if optimized_shared:
        # Determine target model for cross-propagation
        target_model = "pbuf" if source_model == "lcdm" else "lcdm"
        
        print(f"[PROPAGATION] Propagating shared parameters to {target_model} model: {list(optimized_shared.keys())}")
        
        # Get current target model parameters
        target_params = param_store.get_model_defaults(target_model)
        
        # Update shared parameters in target model
        target_params.update(optimized_shared)
        
        # Create metadata for cross-model update
        cross_model_metadata = {
            "source_model": source_model,
            "source_dataset": "cmb",
            "propagated_params": list(optimized_shared.keys()),
            "propagation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "propagation_reason": "cmb_optimization_cross_model_sync"
        }
        
        # Update target model with shared parameters
        param_store.update_model_defaults(
            model=target_model,
            optimized_params=optimized_shared,
            optimization_metadata=cross_model_metadata,
            dry_run=False
        )
        
        propagation_results["cross_model_updates"] = {
            "target_model": target_model,
            "propagated_params": optimized_shared,
            "metadata": cross_model_metadata
        }
        
        print(f"[PROPAGATION] Successfully propagated {len(optimized_shared)} shared parameters to {target_model}")
    else:
        print(f"[PROPAGATION] No shared parameters to propagate")
        propagation_results["cross_model_updates"] = {"message": "no_shared_parameters"}
    
    # Validate cross-model consistency after propagation
    consistency_results = param_store.validate_cross_model_consistency(
        tolerance=1e-6,  # Tighter tolerance after propagation
        log_warnings=False
    )
    
    propagation_results["consistency_check"] = {
        "is_consistent": consistency_results.get("_summary", {}).get("is_fully_consistent", False),
        "divergent_params": consistency_results.get("_summary", {}).get("divergent_params", []),
        "tolerance": 1e-6
    }
    
    return propagation_results