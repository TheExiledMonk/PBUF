#!/usr/bin/env python3
"""
Supernova fitting wrapper script for PBUF cosmology pipeline.

This script provides a thin wrapper around the unified optimization engine
for supernova-only fitting.
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional
from fit_core import engine
from fit_core.parameter import ParameterDict
from fit_core import integrity


def main():
    """
    Main entry point for supernova fitting.
    
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
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
        
        # Run supernova fitting
        results = run_sn_fit(
            model=args.model,
            overrides=overrides if overrides else None,
            verify_integrity=args.verify_integrity,
            integrity_tolerance=args.integrity_tolerance
        )
        
        # Output results
        if args.output_format == "json":
            print(json.dumps(results, indent=2, default=str))
        else:
            print_human_readable_results(results)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for supernova fitting.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Supernova fitting using unified PBUF cosmology pipeline",
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


def run_sn_fit(
    model: str,
    overrides: Optional[ParameterDict] = None,
    verify_integrity: bool = False,
    integrity_tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Execute supernova fitting using unified engine.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        overrides: Optional parameter overrides
        verify_integrity: Whether to run integrity checks
        integrity_tolerance: Tolerance for physics consistency checks
        
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
            datasets=["sn"],
            tolerances=tolerances
        )
        
        # Print comprehensive integrity report
        print_integrity_report(integrity_results)
        
        if integrity_results["overall_status"] != "PASS":
            print("Warning: Some integrity checks failed")
            return {"error": "Integrity checks failed", "integrity_results": integrity_results}
    
    # Execute supernova fitting using unified engine
    results = engine.run_fit(
        model=model,
        datasets_list=["sn"],
        mode="individual",
        overrides=overrides
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


def print_human_readable_results(results: Dict[str, Any]) -> None:
    """
    Print results in human-readable format.
    
    Args:
        results: Results dictionary from engine.run_fit()
    """
    print("=" * 60)
    print("SUPERNOVA FITTING RESULTS")
    print("=" * 60)
    
    # Print model and parameters
    params = results.get("params", {})
    print(f"Model: {params.get('model_class', 'unknown')}")
    print("\nOptimized Parameters:")
    
    # Core parameters
    core_params = ["H0", "Om0", "Obh2", "ns"]
    for param in core_params:
        if param in params:
            print(f"  {param:8s} = {params[param]:.6f}")
    
    # PBUF parameters if present
    pbuf_params = ["alpha", "Rmax", "eps0", "n_eps", "k_sat"]
    pbuf_present = any(param in params for param in pbuf_params)
    if pbuf_present:
        print("\nPBUF Parameters:")
        for param in pbuf_params:
            if param in params:
                print(f"  {param:8s} = {params[param]:.6f}")
    
    # Fit statistics
    metrics = results.get("metrics", {})
    print(f"\nFit Statistics:")
    print(f"  χ²       = {metrics.get('total_chi2', 'N/A'):.3f}")
    print(f"  AIC      = {metrics.get('aic', 'N/A'):.3f}")
    print(f"  BIC      = {metrics.get('bic', 'N/A'):.3f}")
    print(f"  DOF      = {metrics.get('dof', 'N/A')}")
    print(f"  p-value  = {metrics.get('p_value', 'N/A'):.6f}")
    
    # Supernova-specific results
    sn_results = results.get("results", {}).get("sn", {})
    if sn_results:
        predictions = sn_results.get("predictions", {})
        print(f"\nSupernova Predictions:")
        if "distance_modulus" in predictions:
            mu_values = predictions["distance_modulus"]
            if isinstance(mu_values, dict):
                print(f"  Distance moduli (sample):")
                # Show first few values as example
                for i, (z, mu) in enumerate(list(mu_values.items())[:5]):
                    print(f"    z={z:.3f}: μ={mu:.3f}")
                if len(mu_values) > 5:
                    print(f"    ... and {len(mu_values)-5} more")
            else:
                print(f"    μ values: {mu_values}")
        
        if "magnitude_offset" in predictions:
            print(f"  Magnitude offset: {predictions['magnitude_offset']:.6f}")
    
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())