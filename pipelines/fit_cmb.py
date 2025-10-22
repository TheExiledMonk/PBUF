#!/usr/bin/env python3
"""
CMB fitting wrapper script for PBUF cosmology pipeline.

This script provides a thin wrapper around the unified optimization engine
for CMB-only fitting, maintaining backward compatibility with legacy interfaces.
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional, List
from fit_core import engine
from fit_core.parameter import ParameterDict
from fit_core import integrity
from fit_core import config
from fit_core.optimizer import ParameterOptimizer, optimize_cmb_parameters
from fit_core.parameter_store import OptimizedParameterStore


def main():
    """
    Main entry point for CMB fitting.
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    try:
        args = parse_arguments()
        
        # Handle configuration file creation
        if args.create_config:
            config_manager = config.ConfigurationManager()
            config_manager.create_example_config(args.create_config, args.config_format)
            print(f"Created example configuration file: {args.create_config}")
            return 0
        
        # Load configuration
        config_manager = None
        if args.config:
            config_manager = config.load_configuration(args.config)
        else:
            # Try to find configuration file automatically
            auto_config = config.find_config_file()
            if auto_config:
                print(f"Found configuration file: {auto_config}")
                config_manager = config.load_configuration(auto_config)
        
        # Get configuration settings
        if config_manager:
            config_overrides = config_manager.get_parameter_overrides()
            optimizer_config = config_manager.get_optimizer_config()
            output_config = config_manager.get_output_config()
            integrity_config = config_manager.get_integrity_config()
            optimization_config = config_manager.get_optimization_config()
        else:
            config_overrides = {}
            optimizer_config = {}
            output_config = {}
            integrity_config = {}
            optimization_config = {}
        
        # Merge optimization settings (CLI takes precedence over config file)
        optimization_config = config.merge_optimization_config(optimization_config, args)
        
        # Build parameter overrides (command line takes precedence over config file)
        overrides = config_overrides.copy()
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
        
        # Determine settings (command line overrides config file)
        verify_integrity = args.verify_integrity or integrity_config.get('enabled', False)
        integrity_tolerance = args.integrity_tolerance or integrity_config.get('tolerances', {}).get('h_ratios', 1e-4)
        output_format = args.output_format or output_config.get('format', 'human')
        optimizer_method = args.optimizer or optimizer_config.get('method', 'minimize')
        
        # Check if optimization is requested
        optimize_params = optimization_config.get('optimize_parameters', [])
        if optimize_params:
            # Run CMB optimization
            results = run_cmb_optimization(
                model=args.model,
                optimize_params=optimize_params,
                overrides=overrides if overrides else None,
                verify_integrity=verify_integrity,
                integrity_tolerance=integrity_tolerance,
                optimization_config=optimization_config
            )
        else:
            # Run standard CMB fitting
            results = run_cmb_fit(
                model=args.model,
                overrides=overrides if overrides else None,
                verify_integrity=verify_integrity,
                integrity_tolerance=integrity_tolerance,
                optimizer_config=optimizer_config if optimizer_method != 'minimize' else None,
                optimization_config=None
            )
        
        # Output results
        if output_format == "json":
            output_text = json.dumps(results, indent=2, default=str)
        elif output_format == "csv":
            output_text = format_csv_results(results)
        else:
            output_text = format_human_readable_results(results)
        
        # Print or save results
        if args.save_results or output_config.get('save_results'):
            output_file = args.save_results or output_config.get('results_file', 'cmb_results.json')
            with open(output_file, 'w') as f:
                f.write(output_text)
            print(f"Results saved to: {output_file}")
        else:
            print(output_text)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for CMB fitting.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="CMB fitting using unified PBUF cosmology pipeline",
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
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file (JSON, YAML, or INI format)"
    )
    parser.add_argument(
        "--create-config",
        type=str,
        help="Create example configuration file and exit"
    )
    parser.add_argument(
        "--config-format",
        choices=["json", "yaml", "ini"],
        default="json",
        help="Format for created configuration file"
    )
    
    # Options
    parser.add_argument(
        "--verify-integrity", 
        action="store_true",
        help="Run integrity checks before fitting"
    )
    parser.add_argument(
        "--integrity-tolerance",
        type=float,
        help="Tolerance for physics consistency checks (default: 1e-4)"
    )
    parser.add_argument(
        "--output-format",
        choices=["human", "json", "csv"],
        help="Output format for results"
    )
    parser.add_argument(
        "--optimizer",
        choices=["minimize", "differential_evolution"],
        help="Optimization algorithm to use"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save results to file"
    )
    
    # Add optimization arguments
    config.add_optimization_arguments(parser)
    
    return parser.parse_args()


def run_cmb_optimization(
    model: str,
    optimize_params: List[str],
    overrides: Optional[ParameterDict] = None,
    verify_integrity: bool = False,
    integrity_tolerance: float = 1e-4,
    optimization_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute CMB parameter optimization using ParameterOptimizer.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimize_params: List of parameter names to optimize
        overrides: Optional parameter overrides
        verify_integrity: Whether to run integrity checks
        integrity_tolerance: Tolerance for physics consistency checks
        optimization_config: Optional optimization configuration
        
    Returns:
        Complete results dictionary with optimization metadata
        
    Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 7.1, 7.2, 7.3, 7.4, 7.5
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
            datasets=["cmb"],
            tolerances=tolerances
        )
        
        # Print comprehensive integrity report
        print_integrity_report(integrity_results)
        
        if integrity_results["overall_status"] != "PASS":
            print("Warning: Some integrity checks failed")
            return {"error": "Integrity checks failed", "integrity_results": integrity_results}
    
    # Get optimization configuration parameters
    covariance_scaling = optimization_config.get('covariance_scaling', 1.0)
    dry_run = optimization_config.get('dry_run', False)
    warm_start = optimization_config.get('warm_start', False)
    
    # Initialize parameter store
    param_store = OptimizedParameterStore()
    
    # Get starting parameters (with warm start support)
    starting_params = None
    if warm_start:
        warm_start_params = param_store.get_warm_start_params(model)
        if warm_start_params:
            print(f"Using warm start parameters from recent optimization")
            starting_params = warm_start_params
    
    # Apply any parameter overrides
    if overrides:
        if starting_params is None:
            starting_params = param_store.get_model_defaults(model)
        starting_params.update(overrides)
    
    # Execute CMB optimization
    try:
        optimization_result = optimize_cmb_parameters(
            model=model,
            optimize_params=optimize_params,
            starting_params=starting_params,
            covariance_scaling=covariance_scaling,
            dry_run=dry_run
        )
        
        # Update parameter store with optimization results (unless dry run)
        if not dry_run and optimization_result.convergence_status == "success":
            optimization_metadata = {
                "cmb_optimized": optimization_result.metadata["timestamp"],
                "source_dataset": "cmb",
                "optimized_params": optimize_params,
                "chi2_improvement": optimization_result.chi2_improvement,
                "convergence_status": optimization_result.convergence_status,
                "optimizer_info": optimization_result.optimizer_info,
                "covariance_scaling": covariance_scaling
            }
            
            param_store.update_model_defaults(
                model=model,
                optimized_params=optimization_result.optimized_params,
                optimization_metadata=optimization_metadata,
                dry_run=False
            )
            
            print(f"Updated {model} model defaults with optimized parameters")
        
        # Build results dictionary compatible with existing format
        results = build_optimization_results(model, optimization_result, param_store)
        
        return results
        
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return {"error": f"Optimization failed: {str(e)}"}


def run_cmb_fit(
    model: str,
    overrides: Optional[ParameterDict] = None,
    verify_integrity: bool = False,
    integrity_tolerance: float = 1e-4,
    optimizer_config: Optional[Dict[str, Any]] = None,
    optimization_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute CMB fitting using unified engine.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        overrides: Optional parameter overrides
        verify_integrity: Whether to run integrity checks
        integrity_tolerance: Tolerance for physics consistency checks
        optimizer_config: Optional optimizer configuration
        optimization_config: Optional optimization configuration
        
    Returns:
        Complete results dictionary
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
            datasets=["cmb"],
            tolerances=tolerances
        )
        
        # Print comprehensive integrity report
        print_integrity_report(integrity_results)
        
        if integrity_results["overall_status"] != "PASS":
            print("Warning: Some integrity checks failed")
            return {"error": "Integrity checks failed", "integrity_results": integrity_results}
    
    # Check degrees of freedom and add BAO if needed
    from fit_core.statistics import compute_dof
    from fit_core.parameter import build_params
    
    params = build_params(model, overrides)
    n_params = len([k for k, v in params.items() if isinstance(v, (int, float))])
    
    # Check if CMB alone has sufficient degrees of freedom
    try:
        dof = compute_dof(["cmb"], n_params)
        datasets_to_use = ["cmb"]
    except ValueError:
        # CMB alone has insufficient DOF, add BAO for more data points
        datasets_to_use = ["cmb", "bao"]
        print("Note: Adding BAO data to CMB for sufficient degrees of freedom")
    
    # Execute CMB fitting using unified engine
    results = engine.run_fit(
        model=model,
        datasets_list=datasets_to_use,
        mode="joint" if len(datasets_to_use) > 1 else "individual",
        overrides=overrides,
        optimizer_config=optimizer_config
    )
    
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


def format_csv_results(results: Dict[str, Any]) -> str:
    """
    Format results as CSV.
    
    Args:
        results: Results dictionary from engine.run_fit()
        
    Returns:
        CSV formatted string
    """
    lines = []
    
    # Header
    lines.append("parameter,value")
    
    # Parameters
    params = results.get("params", {})
    for param, value in params.items():
        if isinstance(value, (int, float)):
            lines.append(f"{param},{value}")
    
    # Metrics
    metrics = results.get("metrics", {})
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append(f"{metric},{value}")
    
    return "\n".join(lines)


def build_optimization_results(
    model: str, 
    optimization_result, 
    param_store: OptimizedParameterStore
) -> Dict[str, Any]:
    """
    Build results dictionary from optimization result in format compatible with existing engine.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        optimization_result: OptimizationResult from optimizer
        param_store: Parameter store for getting full parameter set
        
    Returns:
        Results dictionary compatible with existing format
        
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    # Get full parameter set (optimized + fixed)
    full_params = param_store.get_model_defaults(model)
    
    # Update with optimized parameter values
    full_params.update(optimization_result.optimized_params)
    
    # Mark which parameters were optimized
    optimization_metadata = {
        "optimization_performed": True,
        "optimized_parameters": list(optimization_result.optimized_params.keys()),
        "fixed_parameters": [p for p in full_params.keys() 
                           if p not in optimization_result.optimized_params],
        "optimization_status": optimization_result.convergence_status,
        "chi2_improvement": optimization_result.chi2_improvement,
        "starting_values": optimization_result.starting_params,
        "final_values": optimization_result.optimized_params,
        "optimizer_info": optimization_result.optimizer_info,
        "covariance_scaling": optimization_result.covariance_scaling
    }
    
    # Build results in standard format
    results = {
        "params": full_params.copy(),
        "metrics": {
            "total_chi2": optimization_result.final_chi2,
            "chi2_improvement": optimization_result.chi2_improvement,
            "optimization_time": optimization_result.optimization_time,
            "function_evaluations": optimization_result.n_function_evaluations
        },
        "results": {
            "cmb": {
                "chi2": optimization_result.final_chi2,
                "optimization_metadata": optimization_metadata
            }
        },
        "optimization_summary": optimization_metadata
    }
    
    # Add model class for compatibility
    results["params"]["model_class"] = model.upper()
    
    return results


def format_human_readable_results(results: Dict[str, Any]) -> str:
    """
    Format results in human-readable format.
    
    Args:
        results: Results dictionary from engine.run_fit()
        
    Returns:
        Human-readable formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CMB FITTING RESULTS")
    lines.append("=" * 60)
    
    # Print model and parameters
    params = results.get("params", {})
    optimization_summary = results.get("optimization_summary", {})
    
    # Get model from multiple possible locations
    model = (params.get('model_class') or 
             results.get('model') or 
             'unknown').upper()
    
    lines.append(f"Model: {model}")
    
    # Check if optimization was performed
    optimization_performed = optimization_summary.get("optimization_performed", False)
    
    if optimization_performed:
        optimized_params = set(optimization_summary.get("optimized_parameters", []))
        fixed_params = set(optimization_summary.get("fixed_parameters", []))
        starting_values = optimization_summary.get("starting_values", {})
        
        lines.append(f"\nOptimization Status: {optimization_summary.get('optimization_status', 'unknown')}")
        lines.append(f"χ² Improvement: {optimization_summary.get('chi2_improvement', 0):.6f}")
        
        lines.append("\nOptimized Parameters:")
        # Core parameters
        core_params = ["H0", "Om0", "Obh2", "ns"]
        for param in core_params:
            if param in params and param in optimized_params:
                start_val = starting_values.get(param, "N/A")
                final_val = params[param]
                if isinstance(start_val, (int, float)):
                    change = final_val - start_val
                    lines.append(f"  {param:8s} = {final_val:.6f} (was {start_val:.6f}, Δ={change:+.6f})")
                else:
                    lines.append(f"  {param:8s} = {final_val:.6f} (optimized)")
        
        # PBUF parameters if present and optimized
        pbuf_params = ["alpha", "Rmax", "eps0", "n_eps", "k_sat"]
        pbuf_optimized = any(param in params and param in optimized_params for param in pbuf_params)
        if pbuf_optimized:
            lines.append("\nOptimized PBUF Parameters:")
            for param in pbuf_params:
                if param in params and param in optimized_params:
                    start_val = starting_values.get(param, "N/A")
                    final_val = params[param]
                    if isinstance(start_val, (int, float)):
                        change = final_val - start_val
                        lines.append(f"  {param:8s} = {final_val:.6f} (was {start_val:.6f}, Δ={change:+.6f})")
                    else:
                        lines.append(f"  {param:8s} = {final_val:.6f} (optimized)")
        
        # Show fixed parameters if any
        fixed_core = [p for p in core_params if p in params and p in fixed_params]
        fixed_pbuf = [p for p in pbuf_params if p in params and p in fixed_params]
        
        if fixed_core or fixed_pbuf:
            lines.append("\nFixed Parameters:")
            for param in fixed_core + fixed_pbuf:
                if param in params:
                    lines.append(f"  {param:8s} = {params[param]:.6f} (fixed)")
    else:
        lines.append("\nParameters (no optimization):")
        
        # Core parameters
        core_params = ["H0", "Om0", "Obh2", "ns"]
        for param in core_params:
            if param in params:
                lines.append(f"  {param:8s} = {params[param]:.6f}")
        
        # PBUF parameters if present
        pbuf_params = ["alpha", "Rmax", "eps0", "n_eps", "k_sat"]
        pbuf_present = any(param in params for param in pbuf_params)
        if pbuf_present:
            lines.append("\nPBUF Parameters:")
            for param in pbuf_params:
                if param in params:
                    lines.append(f"  {param:8s} = {params[param]:.6f}")
    
    # Fit statistics
    metrics = results.get("metrics", {})
    lines.append(f"\nFit Statistics:")
    
    # Format numeric values safely
    chi2 = metrics.get('total_chi2')
    aic = metrics.get('aic')
    bic = metrics.get('bic')
    dof = metrics.get('dof')
    p_value = metrics.get('p_value')
    
    lines.append(f"  χ²       = {chi2:.3f}" if isinstance(chi2, (int, float)) else "  χ²       = N/A")
    lines.append(f"  AIC      = {aic:.3f}" if isinstance(aic, (int, float)) else "  AIC      = N/A")
    lines.append(f"  BIC      = {bic:.3f}" if isinstance(bic, (int, float)) else "  BIC      = N/A")
    lines.append(f"  DOF      = {dof}" if dof is not None else "  DOF      = N/A")
    lines.append(f"  p-value  = {p_value:.6f}" if isinstance(p_value, (int, float)) else "  p-value  = N/A")
    
    # CMB-specific results
    cmb_results = results.get("results", {}).get("cmb", {})
    if cmb_results:
        predictions = cmb_results.get("predictions", {})
        lines.append(f"\nCMB Predictions:")
        if "R" in predictions:
            R = predictions['R']
            lines.append(f"  R        = {R:.3f}" if isinstance(R, (int, float)) else "  R        = N/A")
        if "l_A" in predictions:
            l_A = predictions['l_A']
            lines.append(f"  ℓ_A      = {l_A:.3f}" if isinstance(l_A, (int, float)) else "  ℓ_A      = N/A")
        if "theta_star" in predictions:
            theta_star = predictions['theta_star']
            lines.append(f"  θ*       = {theta_star:.6f}" if isinstance(theta_star, (int, float)) else "  θ*       = N/A")
    
    lines.append("=" * 60)
    return "\n".join(lines)


def print_human_readable_results(results: Dict[str, Any]) -> None:
    """
    Print results in human-readable format.
    
    Args:
        results: Results dictionary from engine.run_fit()
    """
    print(format_human_readable_results(results))


if __name__ == "__main__":
    sys.exit(main())