#!/usr/bin/env python3
"""
CMB fitting wrapper script for PBUF cosmology pipeline.

This script provides a thin wrapper around the unified optimization engine
for CMB-only fitting, maintaining backward compatibility with legacy interfaces.
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional
from fit_core import engine
from fit_core.parameter import ParameterDict
from fit_core import integrity
from fit_core import config


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
        else:
            config_overrides = {}
            optimizer_config = {}
            output_config = {}
            integrity_config = {}
        
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
        
        # Run CMB fitting
        results = run_cmb_fit(
            model=args.model,
            overrides=overrides if overrides else None,
            verify_integrity=verify_integrity,
            integrity_tolerance=integrity_tolerance,
            optimizer_config=optimizer_config if optimizer_method != 'minimize' else None
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
    
    return parser.parse_args()


def run_cmb_fit(
    model: str,
    overrides: Optional[ParameterDict] = None,
    verify_integrity: bool = False,
    integrity_tolerance: float = 1e-4,
    optimizer_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute CMB fitting using unified engine.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        overrides: Optional parameter overrides
        verify_integrity: Whether to run integrity checks
        integrity_tolerance: Tolerance for physics consistency checks
        optimizer_config: Optional optimizer configuration
        
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
    
    # Execute CMB fitting using unified engine
    results = engine.run_fit(
        model=model,
        datasets_list=["cmb"],
        mode="individual",
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
    lines.append(f"Model: {params.get('model_class', 'unknown')}")
    lines.append("\nOptimized Parameters:")
    
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
    lines.append(f"  χ²       = {metrics.get('total_chi2', 'N/A'):.3f}")
    lines.append(f"  AIC      = {metrics.get('aic', 'N/A'):.3f}")
    lines.append(f"  BIC      = {metrics.get('bic', 'N/A'):.3f}")
    lines.append(f"  DOF      = {metrics.get('dof', 'N/A')}")
    lines.append(f"  p-value  = {metrics.get('p_value', 'N/A'):.6f}")
    
    # CMB-specific results
    cmb_results = results.get("results", {}).get("cmb", {})
    if cmb_results:
        predictions = cmb_results.get("predictions", {})
        lines.append(f"\nCMB Predictions:")
        if "R" in predictions:
            lines.append(f"  R        = {predictions['R']:.3f}")
        if "l_A" in predictions:
            lines.append(f"  ℓ_A      = {predictions['l_A']:.3f}")
        if "theta_star" in predictions:
            lines.append(f"  θ*       = {predictions['theta_star']:.6f}")
    
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