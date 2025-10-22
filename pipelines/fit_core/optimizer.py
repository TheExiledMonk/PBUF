"""
Parameter optimization engine for PBUF cosmology fitting.

This module provides the ParameterOptimizer class and CMB-specific optimization routines
for selective parameter optimization in both ΛCDM and PBUF models.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from scipy import optimize
import time
import json
import sys
import platform
import hashlib
from datetime import datetime, timezone
from . import ParameterDict
from . import parameter
from . import likelihoods
from . import datasets
from . import integrity


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    model: str
    optimized_params: Dict[str, float]
    starting_params: Dict[str, float]
    final_chi2: float
    chi2_improvement: float
    convergence_status: str
    n_function_evaluations: int
    optimization_time: float
    bounds_reached: List[str]
    optimizer_info: Dict[str, str]
    covariance_scaling: float
    metadata: Dict[str, Any]


class ParameterOptimizer:
    """
    Parameter optimization engine with bounds validation and convergence diagnostics.
    
    Provides selective parameter optimization for both ΛCDM and PBUF models with
    automatic bounds enforcement and comprehensive result validation.
    
    Requirements: 1.1, 1.2, 1.3, 1.6
    """
    
    def __init__(self, covariance_scaling: float = 1.0):
        """
        Initialize parameter optimizer.
        
        Args:
            covariance_scaling: Scaling factor for covariance matrices (default: 1.0)
        """
        self.covariance_scaling = covariance_scaling
        self._function_evaluations = 0
        
    def optimize_parameters(
        self, 
        model: str,
        datasets_list: List[str], 
        optimize_params: List[str],
        starting_values: Optional[Dict[str, float]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        warm_start: bool = False,
        warm_start_max_age_hours: float = 24.0
    ) -> OptimizationResult:
        """
        Optimize specified parameters for given model and datasets.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            datasets_list: List of dataset names to include in optimization
            optimize_params: List of parameter names to optimize
            starting_values: Optional starting parameter values (uses defaults if None)
            optimizer_config: Optional optimizer configuration
            dry_run: If True, perform optimization without persisting results
            warm_start: If True, use recent optimization results as starting point
            warm_start_max_age_hours: Maximum age of optimization results for warm start
            
        Returns:
            OptimizationResult with optimized parameters and diagnostics
            
        Raises:
            ValueError: If optimization request is invalid
            
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
        """
        # Validate optimization request
        self.validate_optimization_request(model, optimize_params)
        
        # Validate datasets
        if not datasets_list:
            raise ValueError("At least one dataset must be specified for optimization")
        
        # Get starting parameter values with warm-start support
        starting_params = self._get_starting_parameters(
            model, starting_values, warm_start, warm_start_max_age_hours
        )
        
        # Set default optimizer configuration
        if optimizer_config is None:
            optimizer_config = {
                "method": "L-BFGS-B",
                "options": {"maxiter": 1000, "ftol": 1e-9}
            }
        
        # Record starting time
        start_time = time.time()
        self._function_evaluations = 0
        
        # Get optimization bounds for specified parameters
        bounds_dict = self.get_optimization_bounds_dict(model, optimize_params)
        
        # Build objective function
        objective_func = self._build_optimization_objective(
            model, datasets_list, optimize_params, starting_params
        )
        
        # Extract initial values for optimization parameters only
        initial_values = np.array([starting_params[param] for param in optimize_params])
        bounds_list = [bounds_dict[param] for param in optimize_params]
        
        # Compute initial χ²
        initial_chi2 = objective_func(initial_values)
        
        # Execute optimization
        try:
            result = optimize.minimize(
                objective_func,
                initial_values,
                method=optimizer_config["method"],
                bounds=bounds_list,
                options=optimizer_config.get("options", {})
            )
            
            convergence_status = "success" if result.success else "failed"
            final_chi2 = result.fun
            optimized_values = result.x
            
        except Exception as e:
            # Handle optimization failures gracefully
            convergence_status = f"error: {str(e)}"
            final_chi2 = initial_chi2
            optimized_values = initial_values
        
        # Record optimization time
        optimization_time = time.time() - start_time
        
        # Build optimized parameter dictionary
        optimized_params = starting_params.copy()
        for i, param_name in enumerate(optimize_params):
            optimized_params[param_name] = float(optimized_values[i])
        
        # Check which bounds were reached
        bounds_reached = []
        tolerance = 1e-6
        for i, param_name in enumerate(optimize_params):
            min_bound, max_bound = bounds_dict[param_name]
            value = optimized_values[i]
            if abs(value - min_bound) < tolerance:
                bounds_reached.append(f"{param_name}_min")
            elif abs(value - max_bound) < tolerance:
                bounds_reached.append(f"{param_name}_max")
        
        # Compute χ² improvement
        chi2_improvement = initial_chi2 - final_chi2
        
        # Build comprehensive optimizer info with provenance tracking
        optimizer_info = self._build_optimizer_provenance(optimizer_config)
        
        # Build comprehensive metadata with provenance tracking
        metadata = self._build_optimization_metadata(
            datasets_list, initial_chi2, optimizer_config, 
            starting_params, optimize_params, dry_run, warm_start
        )
        
        return OptimizationResult(
            model=model,
            optimized_params={param: optimized_params[param] for param in optimize_params},
            starting_params={param: starting_params[param] for param in optimize_params},
            final_chi2=final_chi2,
            chi2_improvement=chi2_improvement,
            convergence_status=convergence_status,
            n_function_evaluations=self._function_evaluations,
            optimization_time=optimization_time,
            bounds_reached=bounds_reached,
            optimizer_info=optimizer_info,
            covariance_scaling=self.covariance_scaling,
            metadata=metadata
        )
    
    def get_optimization_bounds(self, model: str, param: str) -> Tuple[float, float]:
        """
        Get optimization bounds for a specific parameter.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            param: Parameter name
            
        Returns:
            Tuple of (min_value, max_value) for optimization bounds
            
        Requirements: 1.1, 1.2, 1.3, 1.6
        """
        return parameter.get_optimization_bounds(model, param)
    
    def get_optimization_bounds_dict(self, model: str, optimize_params: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Get optimization bounds for multiple parameters as a dictionary.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            optimize_params: List of parameter names to get bounds for
            
        Returns:
            Dictionary mapping parameter names to (min, max) bounds tuples
            
        Requirements: 1.1, 1.2, 1.3, 1.6
        """
        return parameter.get_optimization_bounds_dict(model, optimize_params)
    
    def validate_optimization_request(self, model: str, optimize_params: List[str]) -> bool:
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
        return parameter.validate_optimization_request(model, optimize_params)
    
    def _build_optimization_objective(
        self,
        model: str,
        datasets_list: List[str],
        optimize_params: List[str],
        base_params: ParameterDict
    ) -> Callable[[np.ndarray], float]:
        """
        Build objective function for parameter optimization.
        
        Args:
            model: Model type for parameter construction
            datasets_list: List of datasets to include in χ²
            optimize_params: List of parameters being optimized
            base_params: Base parameter dictionary (fixed parameters)
            
        Returns:
            Objective function that computes total χ² for parameter array
        """
        # Create likelihood dispatcher mapping
        likelihood_dispatch = {
            "cmb": likelihoods.likelihood_cmb,
            "bao": likelihoods.likelihood_bao,
            "bao_ani": likelihoods.likelihood_bao_ani,
            "sn": likelihoods.likelihood_sn
        }
        
        # Pre-load and cache datasets
        data_cache = {}
        for dataset_name in datasets_list:
            data_cache[dataset_name] = datasets.load_dataset(dataset_name)
            
            # Apply covariance scaling if specified
            if self.covariance_scaling != 1.0 and "covariance" in data_cache[dataset_name]:
                data_cache[dataset_name]["covariance"] *= self.covariance_scaling
        
        def objective(param_values: np.ndarray) -> float:
            """
            Objective function that computes total χ² for given parameter values.
            
            Args:
                param_values: Array of parameter values being optimized
                
            Returns:
                Total χ² across all datasets
            """
            self._function_evaluations += 1
            
            try:
                # Build full parameter dictionary
                params = base_params.copy()
                for i, param_name in enumerate(optimize_params):
                    params[param_name] = float(param_values[i])
                
                # Rebuild parameters with derived quantities
                full_params = parameter.build_params(model, params)
                
                total_chi2 = 0.0
                
                # Sum χ² contributions from all requested datasets
                for dataset_name in datasets_list:
                    if dataset_name not in likelihood_dispatch:
                        raise ValueError(f"Unknown dataset: {dataset_name}")
                    
                    # Get likelihood function and data
                    likelihood_func = likelihood_dispatch[dataset_name]
                    data = data_cache[dataset_name]
                    
                    # Compute χ² for this dataset
                    chi2, _ = likelihood_func(full_params, data)
                    total_chi2 += chi2
                
                return total_chi2
                
            except Exception as e:
                # Return large χ² for invalid parameter combinations
                return 1e10
        
        return objective
    
    def _get_starting_parameters(
        self,
        model: str,
        starting_values: Optional[Dict[str, float]],
        warm_start: bool,
        warm_start_max_age_hours: float
    ) -> ParameterDict:
        """
        Get starting parameter values with warm-start support.
        
        Args:
            model: Model type ("lcdm" or "pbuf")
            starting_values: Optional explicit starting values
            warm_start: Whether to use warm-start from recent optimizations
            warm_start_max_age_hours: Maximum age for warm-start data
            
        Returns:
            Dictionary of starting parameter values
            
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
        """
        # Start with model defaults
        starting_params = parameter.get_defaults(model)
        
        # Apply warm-start if requested
        if warm_start:
            try:
                from .parameter_store import OptimizedParameterStore
                store = OptimizedParameterStore()
                warm_start_params = store.get_warm_start_params(model, warm_start_max_age_hours)
                
                if warm_start_params:
                    # Update starting parameters with warm-start values
                    starting_params.update(warm_start_params)
                    print(f"[WARM_START] Using recent optimization results for {model} model")
                    print(f"[WARM_START] Loaded {len(warm_start_params)} optimized parameters")
                else:
                    print(f"[WARM_START] No recent optimization results found for {model} model (max age: {warm_start_max_age_hours}h)")
                    
            except Exception as e:
                print(f"[WARM_START] Warning: Failed to load warm-start parameters: {str(e)}")
                print(f"[WARM_START] Falling back to default parameters")
        
        # Apply explicit starting values (highest priority)
        if starting_values is not None:
            starting_params.update(starting_values)
            print(f"[PARAMETERS] Applied {len(starting_values)} explicit starting values")
        
        return starting_params
    
    def _validate_warm_start_compatibility(
        self,
        model: str,
        warm_start_params: Dict[str, float],
        optimize_params: List[str]
    ) -> Dict[str, Any]:
        """
        Validate compatibility of warm-start parameters with current optimization request.
        
        Args:
            model: Model type for validation
            warm_start_params: Warm-start parameter values
            optimize_params: Parameters being optimized in current run
            
        Returns:
            Dictionary with compatibility analysis results
            
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
        """
        compatibility = {
            "is_compatible": True,
            "warnings": [],
            "parameter_overlap": [],
            "missing_parameters": [],
            "extra_parameters": []
        }
        
        # Check parameter overlap
        warm_start_param_names = set(warm_start_params.keys())
        optimize_param_names = set(optimize_params)
        
        compatibility["parameter_overlap"] = list(warm_start_param_names & optimize_param_names)
        compatibility["missing_parameters"] = list(optimize_param_names - warm_start_param_names)
        compatibility["extra_parameters"] = list(warm_start_param_names - optimize_param_names)
        
        # Generate warnings for potential issues
        if compatibility["missing_parameters"]:
            compatibility["warnings"].append(
                f"Warm-start missing parameters: {compatibility['missing_parameters']}"
            )
        
        if compatibility["extra_parameters"]:
            compatibility["warnings"].append(
                f"Warm-start has extra parameters: {compatibility['extra_parameters']}"
            )
        
        # Validate parameter values are within bounds
        try:
            for param_name, value in warm_start_params.items():
                if param_name in optimize_params:
                    min_bound, max_bound = parameter.get_optimization_bounds(model, param_name)
                    if not (min_bound <= value <= max_bound):
                        compatibility["warnings"].append(
                            f"Warm-start parameter {param_name}={value} outside bounds [{min_bound}, {max_bound}]"
                        )
                        compatibility["is_compatible"] = False
        except Exception as e:
            compatibility["warnings"].append(f"Bounds validation failed: {str(e)}")
        
        return compatibility
    
    def _build_optimizer_provenance(self, optimizer_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Build comprehensive optimizer provenance information for reproducibility.
        
        Args:
            optimizer_config: Optimizer configuration dictionary
            
        Returns:
            Dictionary with detailed optimizer provenance information
            
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        try:
            import scipy
            scipy_version = scipy.__version__
        except (ImportError, AttributeError):
            scipy_version = "unknown"
        
        try:
            numpy_version = np.__version__
        except AttributeError:
            numpy_version = "unknown"
        
        # Get system information
        system_info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor() or "unknown"
        }
        
        # Create configuration checksum for reproducibility
        config_str = json.dumps(optimizer_config, sort_keys=True)
        config_checksum = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return {
            "method": optimizer_config["method"],
            "library": "scipy",
            "version": scipy_version,  # For backward compatibility
            "scipy_version": scipy_version,
            "numpy_version": numpy_version,
            "python_version": system_info["python_version"],
            "platform": system_info["platform"],
            "architecture": system_info["architecture"],
            "processor": system_info["processor"],
            "config_checksum": config_checksum,
            "covariance_scaling": str(self.covariance_scaling),
            "provenance_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def _build_optimization_metadata(
        self,
        datasets_list: List[str],
        initial_chi2: float,
        optimizer_config: Dict[str, Any],
        starting_params: ParameterDict,
        optimize_params: List[str],
        dry_run: bool = False,
        warm_start: bool = False
    ) -> Dict[str, Any]:
        """
        Build comprehensive optimization metadata for provenance tracking.
        
        Args:
            datasets_list: List of datasets used in optimization
            initial_chi2: Initial χ² value before optimization
            optimizer_config: Optimizer configuration
            starting_params: Starting parameter values
            optimize_params: List of parameters being optimized
            
        Returns:
            Dictionary with comprehensive optimization metadata
            
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        # Create parameter checksums for reproducibility
        param_str = json.dumps({k: starting_params[k] for k in optimize_params}, sort_keys=True)
        param_checksum = hashlib.md5(param_str.encode()).hexdigest()[:8]
        
        # Create dataset fingerprint
        dataset_str = json.dumps(sorted(datasets_list))
        dataset_checksum = hashlib.md5(dataset_str.encode()).hexdigest()[:8]
        
        return {
            "datasets": datasets_list,
            "dataset_checksum": dataset_checksum,
            "initial_chi2": initial_chi2,
            "optimization_config": optimizer_config,
            "starting_param_checksum": param_checksum,
            "optimized_param_list": optimize_params,
            "optimization_timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": hashlib.md5(f"{time.time()}_{id(self)}".encode()).hexdigest()[:12],
            "covariance_scaling_applied": self.covariance_scaling,
            "dry_run_mode": dry_run,
            "warm_start_enabled": warm_start,
            "execution_mode": "dry_run" if dry_run else "production",
            "reproducibility_info": {
                "parameter_count": len(optimize_params),
                "dataset_count": len(datasets_list),
                "optimization_type": "bounded_minimization",
                "persistence_enabled": not dry_run
            }
        }
    
    def log_optimization_provenance(self, result: OptimizationResult) -> None:
        """
        Log comprehensive optimization provenance information.
        
        Args:
            result: OptimizationResult to log provenance for
            
        Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
        """
        print(f"\n[PROVENANCE] Optimization Session Report")
        print(f"[PROVENANCE] Session ID: {result.metadata.get('session_id', 'unknown')}")
        print(f"[PROVENANCE] Timestamp: {result.metadata.get('optimization_timestamp', 'unknown')}")
        print(f"[PROVENANCE] Execution mode: {result.metadata.get('execution_mode', 'unknown')}")
        
        # Log dry-run and warm-start status
        if result.metadata.get('dry_run_mode', False):
            print(f"[DRY_RUN] Mode enabled - results will not be persisted")
        
        if result.metadata.get('warm_start_enabled', False):
            print(f"[WARM_START] Mode enabled - using recent optimization results")
        
        # Log optimizer information
        opt_info = result.optimizer_info
        print(f"[PROVENANCE] Optimizer: {opt_info.get('method', 'unknown')} "
              f"({opt_info.get('library', 'unknown')} v{opt_info.get('scipy_version', 'unknown')})")
        print(f"[PROVENANCE] Environment: Python {opt_info.get('python_version', 'unknown')} "
              f"on {opt_info.get('platform', 'unknown')}")
        print(f"[PROVENANCE] NumPy: v{opt_info.get('numpy_version', 'unknown')}")
        
        # Log configuration checksums
        print(f"[PROVENANCE] Config checksum: {opt_info.get('config_checksum', 'unknown')}")
        print(f"[PROVENANCE] Parameter checksum: {result.metadata.get('starting_param_checksum', 'unknown')}")
        print(f"[PROVENANCE] Dataset checksum: {result.metadata.get('dataset_checksum', 'unknown')}")
        
        # Log convergence diagnostics
        print(f"[CONVERGENCE] Status: {result.convergence_status}")
        print(f"[CONVERGENCE] Function evaluations: {result.n_function_evaluations}")
        print(f"[CONVERGENCE] Optimization time: {result.optimization_time:.3f}s")
        print(f"[CONVERGENCE] χ² improvement: {result.chi2_improvement:.6f}")
        
        # Log parameter changes with precision
        print(f"[PARAMETERS] Optimized {len(result.optimized_params)} parameters:")
        for param, final_val in result.optimized_params.items():
            start_val = result.starting_params[param]
            change = final_val - start_val
            rel_change = abs(change / start_val) if start_val != 0 else float('inf')
            print(f"[PARAMETERS]   {param}: {start_val:.8f} → {final_val:.8f} "
                  f"(Δ={change:+.8f}, rel={rel_change:.2e})")
        
        # Log bounds information
        if result.bounds_reached:
            print(f"[BOUNDS] Reached bounds: {', '.join(result.bounds_reached)}")
        else:
            print(f"[BOUNDS] No bounds reached during optimization")
        
        # Log reproducibility information
        repro_info = result.metadata.get('reproducibility_info', {})
        print(f"[REPRODUCIBILITY] Parameter count: {repro_info.get('parameter_count', 'unknown')}")
        print(f"[REPRODUCIBILITY] Dataset count: {repro_info.get('dataset_count', 'unknown')}")
        print(f"[REPRODUCIBILITY] Covariance scaling: {result.covariance_scaling}")


def optimize_cmb_parameters(
    model: str,
    optimize_params: List[str],
    starting_params: Optional[ParameterDict] = None,
    covariance_scaling: float = 1.0,
    dry_run: bool = False,
    warm_start: bool = False,
    warm_start_max_age_hours: float = 24.0
) -> OptimizationResult:
    """
    Perform CMB-specific parameter optimization with χ² objective.
    
    This function provides a specialized interface for CMB parameter optimization
    with convergence diagnostics, provenance logging, dataset integrity validation,
    dry-run mode, and warm-start capabilities.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimize_params: List of parameter names to optimize
        starting_params: Optional starting parameter values
        covariance_scaling: Scaling factor for CMB covariance matrix
        dry_run: If True, perform optimization without persisting results
        warm_start: If True, use recent optimization results as starting point
        warm_start_max_age_hours: Maximum age of optimization results for warm start
        
    Returns:
        OptimizationResult with CMB-specific optimization results
        
    Raises:
        ValueError: If CMB dataset validation fails or parameters are invalid
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
    """
    # Validate dataset integrity before optimization
    try:
        cmb_data = datasets.load_dataset("cmb")
        # Run basic integrity checks on CMB dataset
        integrity_results = integrity.run_integrity_suite(starting_params, ["cmb"])
        if integrity_results["overall_status"] != "PASS":
            raise ValueError(f"Integrity check failed: {integrity_results['failures']}")
    except Exception as e:
        raise ValueError(f"CMB dataset integrity validation failed: {str(e)}")
    
    # Create optimizer with CMB-specific configuration
    optimizer = ParameterOptimizer(covariance_scaling=covariance_scaling)
    
    # Set CMB-optimized configuration
    optimizer_config = {
        "method": "L-BFGS-B",
        "options": {
            "maxiter": 2000,  # Higher iteration limit for CMB precision
            "ftol": 1e-10,    # Tighter convergence tolerance
            "gtol": 1e-8      # Gradient tolerance
        }
    }
    
    # Execute optimization with dry-run and warm-start support
    result = optimizer.optimize_parameters(
        model=model,
        datasets_list=["cmb"],
        optimize_params=optimize_params,
        starting_values=starting_params,
        optimizer_config=optimizer_config,
        dry_run=dry_run,
        warm_start=warm_start,
        warm_start_max_age_hours=warm_start_max_age_hours
    )
    
    # Add CMB-specific metadata
    result.metadata.update({
        "optimization_type": "cmb_specific",
        "covariance_scaling": covariance_scaling,
        "dry_run": dry_run,
        "dataset_integrity_validated": True
    })
    
    # Log comprehensive provenance and convergence diagnostics
    optimizer.log_optimization_provenance(result)
    _log_convergence_diagnostics(result)
    
    # Validate optimization result
    _validate_optimization_result(result, model)
    
    return result


def _log_convergence_diagnostics(result: OptimizationResult) -> None:
    """
    Log convergence diagnostics and provenance information.
    
    Args:
        result: OptimizationResult to log diagnostics for
        
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
    """
    print(f"[OPTIMIZATION] model={result.model} status={result.convergence_status}")
    print(f"[CONVERGENCE] χ²_initial={result.metadata['initial_chi2']:.6f} χ²_final={result.final_chi2:.6f} improvement={result.chi2_improvement:.6f}")
    print(f"[PERFORMANCE] evaluations={result.n_function_evaluations} time={result.optimization_time:.3f}s")
    
    if result.bounds_reached:
        print(f"[BOUNDS] reached={','.join(result.bounds_reached)}")
    
    # Log optimizer provenance
    optimizer_info = result.optimizer_info
    print(f"[PROVENANCE] method={optimizer_info['method']} library={optimizer_info['library']} version={optimizer_info['version']}")
    
    # Log parameter changes
    for param in result.optimized_params:
        start_val = result.starting_params[param]
        final_val = result.optimized_params[param]
        change = final_val - start_val
        print(f"[PARAM] {param}: {start_val:.6f} → {final_val:.6f} (Δ={change:+.6f})")


def _validate_optimization_result(result: OptimizationResult, model: str) -> None:
    """
    Validate optimization result for physical consistency and bounds compliance.
    
    Args:
        result: OptimizationResult to validate
        model: Model type for validation rules
        
    Raises:
        ValueError: If optimization result is invalid or unphysical
        
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
    """
    # Check convergence status
    if result.convergence_status.startswith("error"):
        raise ValueError(f"Optimization failed: {result.convergence_status}")
    
    # Validate optimized parameters are within bounds
    for param_name, value in result.optimized_params.items():
        try:
            min_bound, max_bound = parameter.get_optimization_bounds(model, param_name)
            if not (min_bound <= value <= max_bound):
                raise ValueError(
                    f"Optimized parameter {param_name}={value} outside bounds "
                    f"[{min_bound}, {max_bound}]"
                )
        except ValueError:
            # Parameter might not have bounds defined - skip validation
            continue
    
    # Check for reasonable χ² improvement
    if result.chi2_improvement < -1e-6:  # Allow small numerical errors
        print(f"Warning: Optimization resulted in χ² increase of {-result.chi2_improvement:.6f}")
    
    # Validate physical consistency by building full parameter set
    try:
        full_params = parameter.get_defaults(model)
        full_params.update(result.optimized_params)
        parameter.validate_params(full_params, model)
    except Exception as e:
        raise ValueError(f"Optimized parameters failed physical validation: {str(e)}")
    
    print(f"[VALIDATION] Optimization result validated successfully")