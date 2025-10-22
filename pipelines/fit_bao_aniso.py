#!/usr/bin/env python3
"""
Enhanced BAO anisotropic fitting wrapper script for PBUF cosmology pipeline.

This script provides an enhanced wrapper around the unified optimization engine
for anisotropic BAO fitting with optimized parameter integration and comprehensive
validation capabilities.
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional
from fit_core import engine
from fit_core.parameter import ParameterDict
from fit_core import integrity
from fit_core.parameter_store import OptimizedParameterStore

# Import OPTIMIZABLE_PARAMETERS for test compatibility
try:
    from fit_core.parameter import OPTIMIZABLE_PARAMETERS
except ImportError:
    OPTIMIZABLE_PARAMETERS = {
        "lcdm": ["H0", "Om0", "Obh2", "ns"],
        "pbuf": ["H0", "Om0", "Obh2", "ns", "alpha", "Rmax", "eps0", "n_eps", "k_sat"]
    }


def validate_params(params: Dict[str, Any], model: str) -> None:
    """
    Validate parameter dictionary for the given model.
    
    Args:
        params: Parameter dictionary to validate
        model: Model type ("lcdm" or "pbuf")
        
    Raises:
        ValueError: If parameters are invalid
    """
    from fit_core.parameter import validate_params as core_validate_params
    core_validate_params(params, model)


def get_default_params(model: str) -> Dict[str, Any]:
    """
    Get default parameters for the given model.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        
    Returns:
        Dictionary of default parameters
    """
    from fit_core.parameter import get_defaults
    return get_defaults(model)


def main():
    """
    Main entry point for enhanced anisotropic BAO fitting.
    
    Requirements: 1.1, 3.1, 4.1
    """
    try:
        args = parse_arguments()
        
        # Build parameter overrides from command line
        overrides = {}
        if args.H0 is not None:
            overrides["H0"] = args.H0
        if args.Om0 is not None:
            overrides["Om0"] = args.Om0
        if args.Obh2 is not None:
            overrides["Obh2"] = args.Obh2
        if args.ns is not None:
            overrides["ns"] = args.ns
        
        # Add PBUF-specific parameters if model is pbuf
        if args.model == "pbuf":
            if args.alpha is not None:
                overrides["alpha"] = args.alpha
            if args.Rmax is not None:
                overrides["Rmax"] = args.Rmax
            if args.eps0 is not None:
                overrides["eps0"] = args.eps0
            if args.n_eps is not None:
                overrides["n_eps"] = args.n_eps
            if args.k_sat is not None:
                overrides["k_sat"] = args.k_sat
        
        # Apply BAO anisotropic safety checks before fitting
        try:
            from fit_core.bao_aniso_validation import validate_no_auto_add_datasets
            validate_no_auto_add_datasets(["bao_ani"])
            print("✅ BAO anisotropic safety checks passed")
        except ImportError:
            print("⚠️  BAO anisotropic validation module not available")
        except Exception as e:
            print(f"❌ BAO anisotropic safety check failed: {e}")
            return 1
        
        # Run enhanced anisotropic BAO fitting
        results = run_bao_aniso_fit(
            model=args.model,
            overrides=overrides if overrides else None,
            verify_integrity=args.verify_integrity,
            integrity_tolerance=args.integrity_tolerance
        )
        
        # Output results with enhanced formatting
        if args.output_format == "json":
            formatted_results = format_json_results(results)
            print(json.dumps(formatted_results, indent=2, default=str))
        else:
            print_human_readable_results(results)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for enhanced anisotropic BAO fitting.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Enhanced anisotropic BAO fitting using unified PBUF cosmology pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        "--model", 
        choices=["lcdm", "pbuf"], 
        default="pbuf",
        help="Cosmological model to fit"
    )
    
    # Common cosmological parameters
    parser.add_argument("--H0", type=float, help="Hubble constant (km/s/Mpc)")
    parser.add_argument("--Om0", type=float, help="Matter density fraction")
    parser.add_argument("--Obh2", type=float, help="Physical baryon density")
    parser.add_argument("--ns", type=float, help="Scalar spectral index")
    
    # PBUF-specific parameters
    parser.add_argument("--alpha", type=float, help="Elasticity amplitude")
    parser.add_argument("--Rmax", type=float, help="Saturation length scale")
    parser.add_argument("--eps0", type=float, help="Elasticity bias term")
    parser.add_argument("--n_eps", type=float, help="Evolution exponent")
    parser.add_argument("--k_sat", type=float, help="Saturation coefficient")
    
    # Options
    parser.add_argument(
        "--verify-integrity", 
        action="store_true",
        help="Run integrity checks before fitting"
    )
    parser.add_argument(
        "--integrity-tolerance",
        type=float,
        default=1e-4,
        help="Tolerance for physics consistency checks (default: 1e-4)"
    )
    parser.add_argument(
        "--output-format",
        choices=["human", "json"],
        default="human",
        help="Output format for results"
    )
    
    return parser.parse_args()


def run_bao_aniso_fit(
    model: str,
    overrides: Optional[ParameterDict] = None,
    verify_integrity: bool = False,
    integrity_tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Execute enhanced anisotropic BAO fitting using unified engine with automatic optimized parameter usage.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        overrides: Optional parameter overrides
        verify_integrity: Whether to run integrity checks
        integrity_tolerance: Tolerance for physics consistency checks
        
    Returns:
        Complete results dictionary with parameter source information
        
    Requirements: 1.1, 3.1, 4.1
    """
    # Initialize parameter store with enhanced error handling and validation
    try:
        param_store = OptimizedParameterStore()
        
        # Verify storage integrity before proceeding
        integrity_check = param_store.verify_storage_integrity()
        if integrity_check["overall_status"] not in ["healthy", "recovered"]:
            print("Warning: Parameter store integrity issues detected")
            for action in integrity_check.get("recovery_actions", []):
                print(f"  Recovery action: {action}")
    
    except Exception as e:
        print(f"Warning: Failed to initialize parameter store: {e}")
        print("Falling back to hardcoded defaults")
        # Fallback to hardcoded defaults if parameter store fails
        from fit_core.parameter import get_defaults
        base_params = get_defaults(model)
        parameter_source_info = {
            "source": "hardcoded_fallback",
            "cmb_optimized": False,
            "fallback_reason": str(e),
            "overrides_applied": len(overrides) if overrides else 0,
            "override_params": list(overrides.keys()) if overrides else []
        }
        
        if overrides:
            base_params.update(overrides)
            print(f"Applied {len(overrides)} parameter override(s) to fallback defaults")
        
        return _execute_fit_with_params(
            model, base_params, parameter_source_info, 
            verify_integrity, integrity_tolerance
        )
    
    # Enhanced parameter retrieval with multiple optimization sources
    parameter_source_info = _get_optimized_parameters_with_metadata(param_store, model)
    base_params = parameter_source_info["final_params"]
    
    # Enhanced parameter override handling with validation
    if overrides:
        override_info = _apply_parameter_overrides(base_params, overrides, model)
        parameter_source_info.update(override_info)
        print(f"Applied {len(overrides)} parameter override(s)")
        
        # Log which parameters were overridden and their sources
        for param_name in overrides.keys():
            original_source = parameter_source_info.get("param_sources", {}).get(param_name, "unknown")
            print(f"  {param_name}: {original_source} -> user_override")
    
    # Enhanced logging of parameter sources
    _log_parameter_source_details(parameter_source_info, model)
    
    return _execute_fit_with_params(
        model, base_params, parameter_source_info, 
        verify_integrity, integrity_tolerance
    )


def _get_optimized_parameters_with_metadata(param_store: OptimizedParameterStore, model: str) -> Dict[str, Any]:
    """
    Get optimized parameters with comprehensive source metadata tracking.
    
    Args:
        param_store: Initialized parameter store
        model: Model type ("lcdm" or "pbuf")
        
    Returns:
        Dictionary with parameters and detailed source metadata
        
    Requirements: 1.1, 3.1, 3.2
    """
    # Check multiple optimization sources in priority order
    optimization_sources = ["cmb", "bao", "sn", "joint"]
    
    # Get base defaults first
    try:
        base_params = param_store.get_model_defaults(model)
        param_sources = {param: "stored_defaults" for param in base_params.keys()}
    except Exception as e:
        print(f"Warning: Failed to get stored defaults, using hardcoded: {e}")
        from fit_core.parameter import get_defaults
        base_params = get_defaults(model)
        param_sources = {param: "hardcoded_defaults" for param in base_params.keys()}
    
    # Track optimization metadata
    optimization_metadata = {
        "available_optimizations": [],
        "used_optimization": None,
        "optimization_age_hours": None,
        "convergence_status": None
    }
    
    # Check for optimizations in priority order
    best_optimization = None
    for dataset in optimization_sources:
        if param_store.is_optimized(model, dataset):
            optimization_metadata["available_optimizations"].append(dataset)
            
            # Get optimization history for this dataset
            history = param_store.get_optimization_history(model)
            dataset_optimizations = [h for h in history if h.dataset == dataset]
            
            if dataset_optimizations and dataset_optimizations[0].convergence_status == "success":
                if best_optimization is None or dataset == "cmb":  # Prefer CMB optimization
                    best_optimization = dataset_optimizations[0]
                    optimization_metadata["used_optimization"] = dataset
    
    # Apply best available optimization
    if best_optimization:
        # Calculate optimization age
        try:
            from datetime import datetime, timezone
            record_time = datetime.fromisoformat(best_optimization.timestamp.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            age_hours = (current_time - record_time).total_seconds() / 3600
            optimization_metadata["optimization_age_hours"] = age_hours
        except Exception:
            optimization_metadata["optimization_age_hours"] = None
        
        optimization_metadata["convergence_status"] = best_optimization.convergence_status
        
        # Update parameters with optimized values
        for param_name, param_value in best_optimization.final_values.items():
            if param_name in base_params:
                base_params[param_name] = param_value
                param_sources[param_name] = f"{best_optimization.dataset}_optimized"
        
        print(f"Using {best_optimization.dataset.upper()}-optimized parameters for {model.upper()} model")
        print(f"Optimization age: {optimization_metadata['optimization_age_hours']:.1f} hours")
        print(f"Optimized parameters: {', '.join(best_optimization.optimized_params)}")
    else:
        print(f"No valid optimizations found for {model.upper()} model, using defaults")
        if optimization_metadata["available_optimizations"]:
            print(f"Available optimizations (but not used): {', '.join(optimization_metadata['available_optimizations'])}")
    
    # Check for warm-start parameters as additional fallback
    warm_start_params = param_store.get_warm_start_params(model, max_age_hours=48.0)
    if warm_start_params and not best_optimization:
        print(f"Using warm-start parameters from recent optimization")
        for param_name, param_value in warm_start_params.items():
            if param_name in base_params:
                base_params[param_name] = param_value
                param_sources[param_name] = "warm_start"
    
    return {
        "final_params": base_params,
        "param_sources": param_sources,
        "source": optimization_metadata["used_optimization"] + "_optimized" if optimization_metadata["used_optimization"] else "defaults",
        "cmb_optimized": optimization_metadata["used_optimization"] == "cmb",
        "optimization_metadata": optimization_metadata,
        "overrides_applied": 0,
        "override_params": []
    }


def _apply_parameter_overrides(
    base_params: ParameterDict, 
    overrides: ParameterDict, 
    model: str
) -> Dict[str, Any]:
    """
    Apply parameter overrides with validation and source tracking.
    
    Args:
        base_params: Base parameter dictionary to modify
        overrides: Parameter overrides to apply
        model: Model type for validation
        
    Returns:
        Dictionary with override metadata
        
    Requirements: 1.1, 3.1, 3.2
    """
    from fit_core.parameter import validate_params, OPTIMIZABLE_PARAMETERS
    
    # Validate override parameters
    invalid_params = []
    for param_name, param_value in overrides.items():
        if param_name not in base_params:
            invalid_params.append(f"{param_name} (not in model)")
            continue
        
        # Type validation
        if not isinstance(param_value, (int, float)):
            invalid_params.append(f"{param_name} (invalid type: {type(param_value)})")
            continue
        
        # Range validation for known parameters
        if param_name == "H0" and not (20.0 <= param_value <= 120.0):
            invalid_params.append(f"{param_name} (out of range: {param_value})")
        elif param_name == "Om0" and not (0.01 <= param_value <= 0.99):
            invalid_params.append(f"{param_name} (out of range: {param_value})")
        elif param_name == "Obh2" and not (0.005 <= param_value <= 0.05):
            invalid_params.append(f"{param_name} (out of range: {param_value})")
        elif param_name == "ns" and not (0.8 <= param_value <= 1.2):
            invalid_params.append(f"{param_name} (out of range: {param_value})")
    
    if invalid_params:
        raise ValueError(f"Invalid parameter overrides: {', '.join(invalid_params)}")
    
    # Apply overrides
    original_values = {}
    for param_name, param_value in overrides.items():
        original_values[param_name] = base_params[param_name]
        base_params[param_name] = param_value
    
    # Validate final parameter set
    try:
        validate_params(base_params, model)
    except Exception as e:
        # Restore original values if validation fails
        for param_name, original_value in original_values.items():
            base_params[param_name] = original_value
        raise ValueError(f"Parameter overrides result in invalid parameter set: {e}")
    
    # Check if overrides affect optimizable parameters
    optimizable_overrides = []
    if model in OPTIMIZABLE_PARAMETERS:
        for param_name in overrides.keys():
            if param_name in OPTIMIZABLE_PARAMETERS[model]:
                optimizable_overrides.append(param_name)
    
    if optimizable_overrides:
        print(f"Warning: Overriding optimizable parameters may affect fit quality: {', '.join(optimizable_overrides)}")
    
    return {
        "overrides_applied": len(overrides),
        "override_params": list(overrides.keys()),
        "optimizable_overrides": optimizable_overrides,
        "original_values": original_values
    }


def _log_parameter_source_details(parameter_source_info: Dict[str, Any], model: str) -> None:
    """
    Log detailed parameter source information for transparency.
    
    Args:
        parameter_source_info: Parameter source metadata
        model: Model type
        
    Requirements: 1.1, 3.1, 4.1
    """
    print(f"\nParameter Source Details for {model.upper()} model:")
    print("=" * 50)
    
    # Overall source summary
    source = parameter_source_info.get("source", "unknown")
    print(f"Primary source: {source}")
    
    # Optimization details
    opt_metadata = parameter_source_info.get("optimization_metadata", {})
    available_opts = opt_metadata.get("available_optimizations", [])
    used_opt = opt_metadata.get("used_optimization")
    
    if available_opts:
        print(f"Available optimizations: {', '.join(available_opts)}")
        if used_opt:
            print(f"Selected optimization: {used_opt}")
            age = opt_metadata.get("optimization_age_hours")
            if age is not None:
                print(f"Optimization age: {age:.1f} hours")
            
            convergence = opt_metadata.get("convergence_status")
            if convergence:
                print(f"Convergence status: {convergence}")
    else:
        print("No optimizations available")
    
    # Parameter-by-parameter sources
    param_sources = parameter_source_info.get("param_sources", {})
    if param_sources:
        print(f"\nParameter sources:")
        source_groups = {}
        for param, source in param_sources.items():
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(param)
        
        for source, params in source_groups.items():
            print(f"  {source}: {', '.join(sorted(params))}")
    
    # Override information
    overrides_applied = parameter_source_info.get("overrides_applied", 0)
    if overrides_applied > 0:
        override_params = parameter_source_info.get("override_params", [])
        print(f"\nUser overrides: {', '.join(override_params)}")
        
        optimizable_overrides = parameter_source_info.get("optimizable_overrides", [])
        if optimizable_overrides:
            print(f"Optimizable parameters overridden: {', '.join(optimizable_overrides)}")
    
    print("=" * 50)


def _execute_fit_with_params(
    model: str,
    params: ParameterDict,
    parameter_source_info: Dict[str, Any],
    verify_integrity: bool,
    integrity_tolerance: float
) -> Dict[str, Any]:
    """
    Execute the actual fitting with prepared parameters and metadata.
    
    Args:
        model: Model type
        params: Final parameter dictionary
        parameter_source_info: Parameter source metadata
        verify_integrity: Whether to run integrity checks
        integrity_tolerance: Tolerance for physics consistency checks
        
    Returns:
        Complete results dictionary
        
    Requirements: 1.1, 3.1, 4.1
    """
    # Run integrity checks if requested
    if verify_integrity:
        print("Running integrity checks...")
        
        # Configure tolerances
        tolerances = {
            "h_ratios": integrity_tolerance,
            "recombination": integrity_tolerance,
            "sound_horizon": integrity_tolerance
        }
        
        integrity_results = integrity.run_integrity_suite(
            params=None,  # Will use defaults
            datasets=["bao_ani"],
            tolerances=tolerances
        )
        
        # Print comprehensive integrity report
        print_integrity_report(integrity_results)
        
        if integrity_results["overall_status"] != "PASS":
            print("Warning: Some integrity checks failed")
            return {"error": "Integrity checks failed", "integrity_results": integrity_results}
    
    # Execute anisotropic BAO fitting using unified engine with optimized parameters
    results = engine.run_fit(
        model=model,
        datasets_list=["bao_ani"],
        mode="individual",
        overrides=params  # Use final prepared parameters
    )
    
    # Add enhanced parameter source information to results
    results["parameter_source"] = parameter_source_info
    
    # Add enhanced validation metadata for anisotropic BAO
    results = _add_enhanced_validation_metadata(results, verify_integrity, integrity_tolerance)
    
    return results


def print_integrity_report(integrity_results: Dict[str, Any]) -> None:
    """
    Print comprehensive integrity validation report.
    
    Args:
        integrity_results: Results from integrity.run_integrity_suite()
    """
    print("\n" + "=" * 60)
    print("INTEGRITY VALIDATION REPORT")
    print("=" * 60)
    
    # Overall status
    status = integrity_results["overall_status"]
    status_symbol = "✓" if status == "PASS" else "✗"
    print(f"Overall Status: {status_symbol} {status}")
    
    # Summary statistics
    summary = integrity_results.get("summary", {})
    print(f"Tests Run: {summary.get('total_tests', 0)}")
    print(f"Passed: {summary.get('passed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Warnings: {summary.get('warnings', 0)}")
    
    # Tolerances used
    tolerances = integrity_results.get("tolerances_used", {})
    if tolerances:
        print(f"\nTolerances Used:")
        for test_name, tolerance in tolerances.items():
            print(f"  {test_name:20s}: {tolerance:.2e}")
    
    # Detailed test results
    print(f"\nDetailed Results:")
    for test_name in integrity_results.get("tests_run", []):
        test_result = integrity_results.get(test_name, {})
        status = test_result.get("status", "UNKNOWN")
        description = test_result.get("description", "No description")
        symbol = "✓" if status == "PASS" else "✗"
        
        print(f"  {symbol} {test_name:20s}: {status}")
        print(f"    {description}")
        
        # Show specific values for some tests
        if test_name == "recombination":
            computed = test_result.get("computed_z_recomb")
            reference = test_result.get("reference_z_recomb")
            if computed and reference:
                print(f"    Computed z*: {computed:.2f}, Reference: {reference:.2f}")
        
        elif test_name == "sound_horizon":
            computed = test_result.get("computed_r_s_drag")
            reference = test_result.get("reference_r_s_drag")
            if computed and reference:
                print(f"    Computed r_s: {computed:.2f} Mpc, Reference: {reference:.2f} Mpc")
    
    # Failed tests details
    failures = integrity_results.get("failures", [])
    if failures:
        print(f"\nFailed Tests: {', '.join(failures)}")
    
    print("=" * 60)


def print_human_readable_results(results: Dict[str, Any]) -> None:
    """
    Print results in human-readable format with enhanced anisotropic BAO predictions,
    parameter source information, and comprehensive validation metadata.
    
    Args:
        results: Results dictionary from engine.run_fit()
        
    Requirements: 4.1, 5.1
    """
    print("=" * 80)
    print("ENHANCED ANISOTROPIC BAO FITTING RESULTS")
    print("=" * 80)
    
    # Print model and parameters
    params = results.get("params", {})
    parameter_source = results.get("parameter_source", {})
    
    print(f"Model: {params.get('model_class', 'unknown').upper()}")
    
    # Enhanced parameter source information display
    _print_parameter_source_summary(parameter_source)
    
    # Enhanced parameter display with source annotations
    _print_enhanced_parameters(params, parameter_source)
    
    # Enhanced fit statistics with additional metrics
    _print_enhanced_fit_statistics(results)
    
    # Detailed anisotropic BAO predictions and analysis
    _print_detailed_bao_anisotropic_results(results)
    
    # Validation metadata and optimization status
    _print_validation_and_optimization_status(results, parameter_source)
    
    print("=" * 80)


def _print_parameter_source_summary(parameter_source: Dict[str, Any]) -> None:
    """
    Print enhanced parameter source summary with optimization details.
    
    Args:
        parameter_source: Parameter source metadata
        
    Requirements: 4.1, 5.1
    """
    print(f"\nParameter Source Summary:")
    print("-" * 40)
    
    source = parameter_source.get("source", "unknown")
    cmb_optimized = parameter_source.get("cmb_optimized", False)
    overrides_applied = parameter_source.get("overrides_applied", 0)
    
    # Display primary parameter source with enhanced details
    if source == "hardcoded_fallback":
        fallback_reason = parameter_source.get("fallback_reason", "unknown")
        print(f"Primary Source: ⚠️  Hardcoded fallback")
        print(f"Fallback Reason: {fallback_reason}")
        print(f"Reliability: Low (no optimization available)")
    elif source.endswith("_optimized"):
        opt_type = source.replace("_optimized", "").upper()
        print(f"Primary Source: ✅ {opt_type}-optimized parameters")
        
        # Show detailed optimization metadata
        opt_metadata = parameter_source.get("optimization_metadata", {})
        age_hours = opt_metadata.get("optimization_age_hours")
        if age_hours is not None:
            freshness = "Fresh" if age_hours < 24 else "Stale" if age_hours < 168 else "Old"
            print(f"Optimization Age: {age_hours:.1f} hours ({freshness})")
        
        convergence = opt_metadata.get("convergence_status")
        if convergence:
            status_icon = "✅" if convergence == "success" else "⚠️"
            print(f"Convergence: {status_icon} {convergence}")
        
        available_opts = opt_metadata.get("available_optimizations", [])
        if len(available_opts) > 1:
            other_opts = [opt.upper() for opt in available_opts if opt != opt_type.lower()]
            print(f"Alternative Sources: {', '.join(other_opts)}")
        
        print(f"Reliability: High (recent optimization)")
    else:
        print(f"Primary Source: 📋 Default parameters")
        print(f"Reliability: Medium (no recent optimization)")
    
    # Show override information with impact assessment
    if overrides_applied > 0:
        override_params = parameter_source.get("override_params", [])
        print(f"\nUser Overrides: {overrides_applied} parameter(s)")
        print(f"Overridden Parameters: {', '.join(override_params)}")
        
        optimizable_overrides = parameter_source.get("optimizable_overrides", [])
        if optimizable_overrides:
            print(f"⚠️  Impact: Overriding optimizable parameters may affect fit quality")
            print(f"Affected Optimizable: {', '.join(optimizable_overrides)}")
    
    # Show parameter-by-parameter sources if mixed
    param_sources = parameter_source.get("param_sources", {})
    if param_sources and len(set(param_sources.values())) > 1:
        print(f"\nParameter Source Breakdown:")
        source_groups = {}
        for param, param_source in param_sources.items():
            if param_source not in source_groups:
                source_groups[param_source] = []
            source_groups[param_source].append(param)
        
        for param_source, param_list in source_groups.items():
            icon = "✅" if "optimized" in param_source else "📋" if "default" in param_source else "⚠️"
            print(f"  {icon} {param_source}: {', '.join(sorted(param_list))}")


def _print_enhanced_parameters(params: Dict[str, Any], parameter_source: Dict[str, Any]) -> None:
    """
    Print parameters with enhanced formatting and source annotations.
    
    Args:
        params: Parameter dictionary
        parameter_source: Parameter source metadata
        
    Requirements: 4.1, 5.1
    """
    print(f"\nModel Parameters:")
    print("-" * 40)
    
    param_sources = parameter_source.get("param_sources", {})
    override_params = set(parameter_source.get("override_params", []))
    
    # Core cosmological parameters
    core_params = ["H0", "Om0", "Obh2", "ns"]
    print("Core Cosmological Parameters:")
    for param in core_params:
        if param in params:
            value = params[param]
            source_info = _get_parameter_source_annotation(param, param_sources, override_params)
            print(f"  {param:8s} = {value:10.6f} {source_info}")
    
    # PBUF-specific parameters if present
    pbuf_params = ["alpha", "Rmax", "eps0", "n_eps", "k_sat"]
    pbuf_present = any(param in params for param in pbuf_params)
    if pbuf_present:
        print("\nPBUF Model Parameters:")
        for param in pbuf_params:
            if param in params:
                value = params[param]
                source_info = _get_parameter_source_annotation(param, param_sources, override_params)
                print(f"  {param:8s} = {value:10.6f} {source_info}")
    
    # Derived parameters if available
    derived_params = ["z_recomb", "r_s_drag"]
    derived_present = any(param in params for param in derived_params)
    if derived_present:
        print("\nDerived Parameters:")
        for param in derived_params:
            if param in params:
                value = params[param]
                print(f"  {param:8s} = {value:10.6f} (computed)")


def _get_parameter_source_annotation(param: str, param_sources: Dict[str, str], override_params: set) -> str:
    """
    Get source annotation for a parameter.
    
    Args:
        param: Parameter name
        param_sources: Parameter source mapping
        override_params: Set of overridden parameters
        
    Returns:
        Source annotation string
    """
    if param in override_params:
        return "[USER]"
    
    source = param_sources.get(param, "unknown")
    if "optimized" in source:
        return "[OPT]"
    elif "default" in source:
        return "[DEF]"
    elif "warm_start" in source:
        return "[WARM]"
    else:
        return "[?]"


def _print_enhanced_fit_statistics(results: Dict[str, Any]) -> None:
    """
    Print enhanced fit statistics with additional metrics and interpretations.
    
    Args:
        results: Results dictionary
        
    Requirements: 4.1, 5.1
    """
    print(f"\nFit Statistics:")
    print("-" * 40)
    
    metrics = results.get("metrics", {})
    chi2_breakdown = results.get("chi2_breakdown", {})
    
    # Primary statistics
    chi2 = metrics.get('total_chi2')
    aic = metrics.get('aic')
    bic = metrics.get('bic')
    dof = metrics.get('dof')
    p_value = metrics.get('p_value')
    
    print("Primary Metrics:")
    print(f"  χ²       = {chi2:.3f}" if isinstance(chi2, (int, float)) else "  χ²       = N/A")
    print(f"  DOF      = {dof}" if dof is not None else "  DOF      = N/A")
    
    # Reduced chi-squared with interpretation
    if isinstance(chi2, (int, float)) and dof is not None and dof > 0:
        reduced_chi2 = chi2 / dof
        interpretation = _interpret_reduced_chi2(reduced_chi2)
        print(f"  χ²/DOF   = {reduced_chi2:.3f} ({interpretation})")
    
    print(f"  p-value  = {p_value:.6f}" if isinstance(p_value, (int, float)) else "  p-value  = N/A")
    
    # Information criteria
    print("\nModel Selection Criteria:")
    print(f"  AIC      = {aic:.3f}" if isinstance(aic, (int, float)) else "  AIC      = N/A")
    print(f"  BIC      = {bic:.3f}" if isinstance(bic, (int, float)) else "  BIC      = N/A")
    
    # Chi-squared breakdown by dataset
    if chi2_breakdown:
        print("\nChi-squared Breakdown:")
        for dataset, dataset_chi2 in chi2_breakdown.items():
            if isinstance(dataset_chi2, (int, float)):
                contribution = (dataset_chi2 / chi2 * 100) if isinstance(chi2, (int, float)) and chi2 > 0 else 0
                print(f"  {dataset:12s}: {dataset_chi2:8.3f} ({contribution:5.1f}%)")


def _interpret_reduced_chi2(reduced_chi2: float) -> str:
    """
    Interpret reduced chi-squared value.
    
    Args:
        reduced_chi2: Reduced chi-squared value
        
    Returns:
        Interpretation string
    """
    if reduced_chi2 < 0.5:
        return "Overfitted"
    elif reduced_chi2 < 1.2:
        return "Good fit"
    elif reduced_chi2 < 2.0:
        return "Acceptable"
    elif reduced_chi2 < 5.0:
        return "Poor fit"
    else:
        return "Very poor fit"


def _print_detailed_bao_anisotropic_results(results: Dict[str, Any]) -> None:
    """
    Print detailed anisotropic BAO predictions and analysis.
    
    Args:
        results: Results dictionary
        
    Requirements: 4.1, 5.1
    """
    bao_ani_results = results.get("results", {}).get("bao_ani", {})
    if not bao_ani_results:
        print(f"\nAnisotropic BAO Results: No data available")
        return
    
    print(f"\nAnisotropic BAO Analysis:")
    print("-" * 40)
    
    predictions = bao_ani_results.get("predictions", {})
    observations = bao_ani_results.get("observations", {})
    
    # Display chi-squared contribution prominently
    bao_ani_chi2 = bao_ani_results.get("chi2")
    if bao_ani_chi2 is not None:
        print(f"Dataset χ² = {bao_ani_chi2:.3f}")
    
    # Enhanced transverse BAO measurements (D_M/r_s)
    if "DM_over_rs" in predictions:
        print(f"\nTransverse BAO Measurements (D_M/r_s):")
        dm_predictions = predictions["DM_over_rs"]
        dm_observations = observations.get("DM_over_rs", {})
        
        if isinstance(dm_predictions, dict) and isinstance(dm_observations, dict):
            print(f"  {'z':>6s} {'Observed':>10s} {'Predicted':>10s} {'Residual':>10s}")
            print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
            for z in sorted(dm_predictions.keys()):
                pred = dm_predictions[z]
                obs = dm_observations.get(z, 0.0)
                residual = obs - pred if obs != 0.0 else 0.0
                print(f"  {z:6.2f} {obs:10.3f} {pred:10.3f} {residual:+10.3f}")
        else:
            # Handle array or other formats safely
            pred_str = _format_array_for_display(dm_predictions)
            print(f"  Predicted: {pred_str}")
            if _has_data(dm_observations):
                obs_str = _format_array_for_display(dm_observations)
                print(f"  Observed:  {obs_str}")
    
    # Enhanced radial BAO measurements (H*r_s)
    if "H_times_rs" in predictions:
        print(f"\nRadial BAO Measurements (H*r_s):")
        h_predictions = predictions["H_times_rs"]
        h_observations = observations.get("H_times_rs", {})
        
        if isinstance(h_predictions, dict) and isinstance(h_observations, dict):
            print(f"  {'z':>6s} {'Observed':>10s} {'Predicted':>10s} {'Residual':>10s}")
            print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
            for z in sorted(h_predictions.keys()):
                pred = h_predictions[z]
                obs = h_observations.get(z, 0.0)
                residual = obs - pred if obs != 0.0 else 0.0
                print(f"  {z:6.2f} {obs:10.3f} {pred:10.3f} {residual:+10.3f}")
        else:
            # Handle array or other formats safely
            pred_str = _format_array_for_display(h_predictions)
            print(f"  Predicted: {pred_str}")
            if _has_data(h_observations):
                obs_str = _format_array_for_display(h_observations)
                print(f"  Observed:  {obs_str}")
    
    # Enhanced anisotropic parameters display
    if "alpha_parallel" in predictions and "alpha_perpendicular" in predictions:
        print(f"\nAnisotropic BAO Parameters:")
        alpha_par = predictions["alpha_parallel"]
        alpha_perp = predictions["alpha_perpendicular"]
        
        if isinstance(alpha_par, dict) and isinstance(alpha_perp, dict):
            print(f"  {'z':>6s} {'α∥':>8s} {'α⊥':>8s} {'Anisotropy':>10s}")
            print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
            for z in sorted(alpha_par.keys()):
                if z in alpha_perp:
                    par = alpha_par[z]
                    perp = alpha_perp[z]
                    anisotropy = par / perp if perp != 0 else 0.0
                    print(f"  {z:6.2f} {par:8.3f} {perp:8.3f} {anisotropy:10.3f}")
        else:
            print(f"  α∥ (parallel)      = {alpha_par}")
            print(f"  α⊥ (perpendicular) = {alpha_perp}")
    
    # Residual analysis if available
    residuals = bao_ani_results.get("residuals")
    if residuals is not None:
        if hasattr(residuals, '__len__') and len(residuals) > 0:
            import numpy as np
            residuals_array = np.array(residuals)
            print(f"\nResidual Analysis:")
            print(f"  RMS residual = {np.sqrt(np.mean(residuals_array**2)):.3f}")
            print(f"  Max residual = {np.max(np.abs(residuals_array)):.3f}")
            print(f"  Mean residual = {np.mean(residuals_array):+.3f}")


def _print_validation_and_optimization_status(results: Dict[str, Any], parameter_source: Dict[str, Any]) -> None:
    """
    Print validation metadata and optimization status information.
    
    Args:
        results: Results dictionary
        parameter_source: Parameter source metadata
        
    Requirements: 4.1, 5.1
    """
    print(f"\nValidation & Optimization Status:")
    print("-" * 40)
    
    # Dataset validation metadata
    bao_ani_results = results.get("results", {}).get("bao_ani", {})
    validation_metadata = bao_ani_results.get("validation_metadata", {})
    
    if validation_metadata:
        print("Dataset Validation:")
        
        covariance_condition = validation_metadata.get("covariance_condition_number")
        if covariance_condition is not None:
            condition_status = "Good" if covariance_condition < 1e12 else "Poor"
            print(f"  Covariance condition: {covariance_condition:.2e} ({condition_status})")
        
        n_redshift_bins = validation_metadata.get("n_redshift_bins")
        if n_redshift_bins is not None:
            print(f"  Redshift bins: {n_redshift_bins}")
            print(f"  Covariance matrix: {2*n_redshift_bins}×{2*n_redshift_bins}")
        
        data_quality = validation_metadata.get("data_quality", "unknown")
        print(f"  Data quality: {data_quality}")
    
    # Optimization status from parameter source
    opt_metadata = parameter_source.get("optimization_metadata", {})
    if opt_metadata:
        print("\nOptimization Status:")
        
        available_opts = opt_metadata.get("available_optimizations", [])
        used_opt = opt_metadata.get("used_optimization")
        
        if available_opts:
            print(f"  Available optimizations: {', '.join([opt.upper() for opt in available_opts])}")
            if used_opt:
                print(f"  Selected optimization: {used_opt.upper()}")
            else:
                print(f"  Selected optimization: None (using defaults)")
        else:
            print(f"  Available optimizations: None")
        
        convergence = opt_metadata.get("convergence_status")
        if convergence:
            status_icon = "✅" if convergence == "success" else "⚠️"
            print(f"  Convergence status: {status_icon} {convergence}")
        
        age_hours = opt_metadata.get("optimization_age_hours")
        if age_hours is not None:
            freshness = "Fresh" if age_hours < 24 else "Stale" if age_hours < 168 else "Old"
            print(f"  Optimization freshness: {freshness} ({age_hours:.1f}h ago)")
    
    # Diagnostics from engine results
    diagnostics = results.get("diagnostics", {})
    if diagnostics:
        print("\nFit Diagnostics:")
        
        convergence_status = diagnostics.get("convergence_status", "unknown")
        print(f"  Fit convergence: {convergence_status}")
        
        total_datasets = diagnostics.get("total_datasets", 0)
        total_data_points = diagnostics.get("total_data_points", 0)
        print(f"  Datasets used: {total_datasets}")
        print(f"  Data points: {total_data_points}")
    
    # Parameter override warnings
    override_params = parameter_source.get("override_params", [])
    optimizable_overrides = parameter_source.get("optimizable_overrides", [])
    if optimizable_overrides:
        print(f"\n⚠️  Warnings:")
        print(f"  Optimizable parameters overridden: {', '.join(optimizable_overrides)}")
        print(f"  This may reduce fit quality compared to optimized values")


def format_json_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format results for JSON output with enhanced structure for programmatic access.
    
    Args:
        results: Raw results dictionary from engine.run_fit()
        
    Returns:
        Enhanced results dictionary optimized for JSON serialization
        
    Requirements: 4.1, 5.1
    """
    import numpy as np
    from datetime import datetime, timezone
    
    # Create enhanced JSON structure
    formatted_results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "anisotropic_bao_fit",
            "version": "enhanced_v1.0",
            "model": results.get("model", "unknown")
        },
        "parameters": {
            "values": {},
            "sources": {},
            "optimization_status": {}
        },
        "fit_results": {
            "statistics": {},
            "chi2_breakdown": {},
            "datasets": {}
        },
        "anisotropic_bao": {
            "predictions": {},
            "observations": {},
            "residuals": {},
            "validation": {}
        },
        "diagnostics": {
            "parameter_source": {},
            "validation_metadata": {},
            "warnings": []
        }
    }
    
    # Enhanced parameter information
    params = results.get("params", {})
    parameter_source = results.get("parameter_source", {})
    
    # Separate parameters by type
    core_params = ["H0", "Om0", "Obh2", "ns"]
    pbuf_params = ["alpha", "Rmax", "eps0", "n_eps", "k_sat"]
    derived_params = ["z_recomb", "r_s_drag", "model_class"]
    
    formatted_results["parameters"]["values"] = {
        "core": {param: params.get(param) for param in core_params if param in params},
        "pbuf": {param: params.get(param) for param in pbuf_params if param in params},
        "derived": {param: params.get(param) for param in derived_params if param in params}
    }
    
    # Parameter source information
    param_sources = parameter_source.get("param_sources", {})
    formatted_results["parameters"]["sources"] = {
        "primary_source": parameter_source.get("source", "unknown"),
        "cmb_optimized": parameter_source.get("cmb_optimized", False),
        "parameter_sources": param_sources,
        "overrides": {
            "count": parameter_source.get("overrides_applied", 0),
            "parameters": parameter_source.get("override_params", []),
            "optimizable_overrides": parameter_source.get("optimizable_overrides", [])
        }
    }
    
    # Optimization status
    opt_metadata = parameter_source.get("optimization_metadata", {})
    formatted_results["parameters"]["optimization_status"] = {
        "available_optimizations": opt_metadata.get("available_optimizations", []),
        "used_optimization": opt_metadata.get("used_optimization"),
        "optimization_age_hours": opt_metadata.get("optimization_age_hours"),
        "convergence_status": opt_metadata.get("convergence_status")
    }
    
    # Enhanced fit statistics
    metrics = results.get("metrics", {})
    formatted_results["fit_results"]["statistics"] = {
        "chi2": metrics.get("total_chi2"),
        "dof": metrics.get("dof"),
        "reduced_chi2": metrics.get("total_chi2") / metrics.get("dof") if metrics.get("dof", 0) > 0 else None,
        "p_value": metrics.get("p_value"),
        "aic": metrics.get("aic"),
        "bic": metrics.get("bic")
    }
    
    # Chi-squared breakdown
    formatted_results["fit_results"]["chi2_breakdown"] = results.get("chi2_breakdown", {})
    
    # Dataset information
    formatted_results["fit_results"]["datasets"] = {
        "used": results.get("datasets", []),
        "count": len(results.get("datasets", []))
    }
    
    # Enhanced anisotropic BAO results
    bao_ani_results = results.get("results", {}).get("bao_ani", {})
    if bao_ani_results:
        predictions = bao_ani_results.get("predictions", {})
        observations = bao_ani_results.get("observations", {})
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            else:
                return obj
        
        formatted_results["anisotropic_bao"]["predictions"] = convert_numpy(predictions)
        formatted_results["anisotropic_bao"]["observations"] = convert_numpy(observations)
        
        # Residuals if available
        residuals = bao_ani_results.get("residuals")
        if residuals is not None:
            formatted_results["anisotropic_bao"]["residuals"] = {
                "values": convert_numpy(residuals),
                "statistics": _compute_residual_statistics(residuals)
            }
        
        # Validation metadata
        validation_metadata = bao_ani_results.get("validation_metadata", {})
        formatted_results["anisotropic_bao"]["validation"] = validation_metadata
        
        # Dataset-specific chi2
        formatted_results["anisotropic_bao"]["chi2"] = bao_ani_results.get("chi2")
    
    # Enhanced diagnostics
    formatted_results["diagnostics"]["parameter_source"] = parameter_source
    formatted_results["diagnostics"]["validation_metadata"] = results.get("validation_metadata", {})
    
    # Collect warnings
    warnings = []
    if parameter_source.get("optimizable_overrides"):
        warnings.append({
            "type": "parameter_override",
            "message": "Optimizable parameters overridden",
            "affected_parameters": parameter_source.get("optimizable_overrides", [])
        })
    
    if parameter_source.get("source") == "hardcoded_fallback":
        warnings.append({
            "type": "fallback_parameters",
            "message": "Using hardcoded fallback parameters",
            "reason": parameter_source.get("fallback_reason", "unknown")
        })
    
    formatted_results["diagnostics"]["warnings"] = warnings
    
    return formatted_results


def _compute_residual_statistics(residuals) -> Dict[str, float]:
    """
    Compute statistical summary of residuals.
    
    Args:
        residuals: Residual array or list
        
    Returns:
        Dictionary with residual statistics
    """
    import numpy as np
    
    if residuals is None or len(residuals) == 0:
        return {}
    
    residuals_array = np.array(residuals)
    
    return {
        "rms": float(np.sqrt(np.mean(residuals_array**2))),
        "mean": float(np.mean(residuals_array)),
        "std": float(np.std(residuals_array)),
        "max_abs": float(np.max(np.abs(residuals_array))),
        "min": float(np.min(residuals_array)),
        "max": float(np.max(residuals_array))
    }


def _has_data(data) -> bool:
    """
    Check if data structure has content.
    
    Args:
        data: Data to check
        
    Returns:
        True if data has content, False otherwise
    """
    import numpy as np
    
    if data is None:
        return False
    
    try:
        if isinstance(data, np.ndarray):
            return data.size > 0
        elif isinstance(data, (list, tuple)):
            return len(data) > 0
        elif isinstance(data, dict):
            return len(data) > 0
        else:
            return bool(data)
    except Exception:
        return False


def _format_array_for_display(data) -> str:
    """
    Format array or other data types for safe display.
    
    Args:
        data: Data to format (array, list, dict, etc.)
        
    Returns:
        String representation suitable for display
    """
    import numpy as np
    
    if data is None:
        return "N/A"
    
    try:
        if isinstance(data, np.ndarray):
            if data.size <= 10:  # Small arrays - show all values
                return f"[{', '.join(f'{x:.3f}' for x in data.flatten())}]"
            else:  # Large arrays - show summary
                return f"Array({data.shape}) [{data.min():.3f} ... {data.max():.3f}]"
        elif isinstance(data, (list, tuple)):
            if len(data) <= 10:
                return f"[{', '.join(f'{x:.3f}' if isinstance(x, (int, float)) else str(x) for x in data)}]"
            else:
                return f"List({len(data)}) [{min(data):.3f} ... {max(data):.3f}]"
        elif isinstance(data, dict):
            if len(data) <= 5:
                items = [f"{k}:{v:.3f}" if isinstance(v, (int, float)) else f"{k}:{v}" for k, v in data.items()]
                return f"{{{', '.join(items)}}}"
            else:
                return f"Dict({len(data)} items)"
        else:
            return str(data)
    except Exception:
        return str(data)


def _add_enhanced_validation_metadata(
    results: Dict[str, Any], 
    verify_integrity: bool, 
    integrity_tolerance: float
) -> Dict[str, Any]:
    """
    Add enhanced validation metadata to results for anisotropic BAO analysis.
    
    Args:
        results: Results dictionary from engine
        verify_integrity: Whether integrity checks were run
        integrity_tolerance: Tolerance used for checks
        
    Returns:
        Enhanced results dictionary with validation metadata
        
    Requirements: 4.1, 5.1
    """
    import numpy as np
    
    # Add top-level validation metadata
    validation_metadata = {
        "integrity_checks_run": verify_integrity,
        "integrity_tolerance": integrity_tolerance,
        "analysis_timestamp": results.get("metadata", {}).get("timestamp"),
        "validation_version": "enhanced_v1.0"
    }
    
    # Enhance anisotropic BAO results with validation metadata
    bao_ani_results = results.get("results", {}).get("bao_ani", {})
    if bao_ani_results:
        # Analyze covariance matrix if available
        covariance = bao_ani_results.get("covariance")
        if covariance is not None:
            try:
                cov_array = np.array(covariance)
                condition_number = np.linalg.cond(cov_array)
                eigenvals = np.linalg.eigvals(cov_array)
                
                bao_validation = {
                    "covariance_condition_number": float(condition_number),
                    "covariance_shape": list(cov_array.shape),
                    "covariance_positive_definite": bool(np.all(eigenvals > 0)),
                    "covariance_min_eigenvalue": float(np.min(eigenvals)),
                    "covariance_max_eigenvalue": float(np.max(eigenvals))
                }
                
                # Infer number of redshift bins
                if len(cov_array.shape) == 2 and cov_array.shape[0] == cov_array.shape[1]:
                    n_total = cov_array.shape[0]
                    if n_total % 2 == 0:  # Should be 2N for N redshift bins
                        n_redshift_bins = n_total // 2
                        bao_validation["n_redshift_bins"] = n_redshift_bins
                        bao_validation["expected_structure"] = f"{n_total}x{n_total} (2N for N={n_redshift_bins} redshift bins)"
                
                # Data quality assessment
                if condition_number < 1e6:
                    data_quality = "excellent"
                elif condition_number < 1e9:
                    data_quality = "good"
                elif condition_number < 1e12:
                    data_quality = "acceptable"
                else:
                    data_quality = "poor"
                
                bao_validation["data_quality"] = data_quality
                
            except Exception as e:
                bao_validation = {
                    "covariance_analysis_error": str(e),
                    "data_quality": "unknown"
                }
        else:
            bao_validation = {
                "covariance_available": False,
                "data_quality": "unknown"
            }
        
        # Analyze predictions and observations
        predictions = bao_ani_results.get("predictions", {})
        observations = bao_ani_results.get("observations", {})
        
        if predictions and observations:
            # Check data consistency
            pred_keys = set(predictions.keys())
            obs_keys = set(observations.keys())
            
            bao_validation.update({
                "prediction_types": list(pred_keys),
                "observation_types": list(obs_keys),
                "data_consistency": pred_keys == obs_keys,
                "missing_predictions": list(obs_keys - pred_keys),
                "extra_predictions": list(pred_keys - obs_keys)
            })
            
            # Analyze redshift coverage if available
            if "DM_over_rs" in predictions and isinstance(predictions["DM_over_rs"], dict):
                redshifts = list(predictions["DM_over_rs"].keys())
                if redshifts:
                    try:
                        z_values = [float(z) for z in redshifts]
                        bao_validation.update({
                            "redshift_range": [min(z_values), max(z_values)],
                            "n_redshift_points": len(z_values),
                            "redshift_spacing": "uniform" if len(set(np.diff(sorted(z_values)))) == 1 else "non-uniform"
                        })
                    except (ValueError, TypeError):
                        pass
        
        # Add validation metadata to BAO results
        if "validation_metadata" not in bao_ani_results:
            bao_ani_results["validation_metadata"] = {}
        bao_ani_results["validation_metadata"].update(bao_validation)
    
    # Add top-level validation metadata
    results["validation_metadata"] = validation_metadata
    
    return results


if __name__ == "__main__":
    sys.exit(main())