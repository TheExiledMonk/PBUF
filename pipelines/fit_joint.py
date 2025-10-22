#!/usr/bin/env python3
"""
Joint fitting wrapper script for PBUF cosmology pipeline.

This script provides a thin wrapper around the unified optimization engine
for multi-dataset joint fitting.
"""

import argparse
import sys
import json
from typing import Dict, Any, Optional, List
from fit_core import engine
from fit_core.parameter import ParameterDict
from fit_core import integrity
from fit_core import config


def main():
    """
    Main entry point for joint fitting.
    
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
        
        # Run joint fitting
        results = run_joint_fit(
            model=args.model,
            datasets=args.datasets,
            overrides=overrides if overrides else None,
            verify_integrity=args.verify_integrity,
            integrity_tolerance=args.integrity_tolerance,
            optimizer=args.optimizer
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
    Parse command-line arguments for joint fitting.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Joint multi-dataset fitting using unified PBUF cosmology pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model selection
    parser.add_argument(
        "--model", 
        choices=["lcdm", "pbuf"], 
        default="pbuf",
        help="Cosmological model to fit"
    )
    
    # Dataset selection
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cmb", "bao", "bao_ani", "sn"],
        default=["cmb", "bao", "sn"],
        help="Datasets to include in joint fit"
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


def run_joint_fit(
    model: str,
    datasets: List[str],
    overrides: Optional[ParameterDict] = None,
    verify_integrity: bool = False,
    integrity_tolerance: float = 1e-4,
    optimizer: str = "minimize"
) -> Dict[str, Any]:
    """
    Execute joint fitting using unified engine.
    
    Args:
        model: Model type ("lcdm" or "pbuf")
        datasets: List of datasets to include in joint fit
        overrides: Optional parameter overrides
        verify_integrity: Whether to run integrity checks
        integrity_tolerance: Tolerance for physics consistency checks
        optimizer: Optimization algorithm to use
        
    Returns:
        Complete results dictionary
    """
    # Validate dataset list
    valid_datasets = ["cmb", "bao", "bao_ani", "sn"]
    for dataset in datasets:
        if dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {dataset}. Must be one of {valid_datasets}")
    
    if len(datasets) < 2:
        raise ValueError("Joint fitting requires at least 2 datasets")
    
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
            datasets=datasets,
            tolerances=tolerances
        )
        
        # Print comprehensive integrity report
        print_integrity_report(integrity_results)
        
        if integrity_results["overall_status"] != "PASS":
            print("Warning: Some integrity checks failed")
            return {"error": "Integrity checks failed", "integrity_results": integrity_results}
    
    # Configure optimizer
    optimizer_config = None
    if optimizer == "differential_evolution":
        optimizer_config = {
            "method": "differential_evolution",
            "options": {"maxiter": 1000, "tol": 1e-9}
        }
    
    # Execute joint fitting using unified engine
    results = engine.run_fit(
        model=model,
        datasets_list=datasets,
        mode="joint",
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


def print_human_readable_results(results: Dict[str, Any]) -> None:
    """
    Print results in human-readable format.
    
    Args:
        results: Results dictionary from engine.run_fit()
    """
    print("=" * 60)
    print("JOINT FITTING RESULTS")
    print("=" * 60)
    
    # Print model and parameters
    params = results.get("params", {})
    print(f"Model: {params.get('model_class', 'unknown')}")
    
    # Show which datasets were included
    dataset_results = results.get("results", {})
    included_datasets = list(dataset_results.keys())
    print(f"Datasets: {', '.join(included_datasets)}")
    
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
    
    # Overall fit statistics
    metrics = results.get("metrics", {})
    print(f"\nOverall Fit Statistics:")
    print(f"  Total χ² = {metrics.get('total_chi2', 'N/A'):.3f}")
    print(f"  AIC      = {metrics.get('aic', 'N/A'):.3f}")
    print(f"  BIC      = {metrics.get('bic', 'N/A'):.3f}")
    print(f"  DOF      = {metrics.get('dof', 'N/A')}")
    print(f"  p-value  = {metrics.get('p_value', 'N/A'):.6f}")
    
    # Per-dataset breakdown
    print(f"\nPer-Dataset Breakdown:")
    for dataset_name in included_datasets:
        dataset_result = dataset_results.get(dataset_name, {})
        chi2 = dataset_result.get("chi2", "N/A")
        print(f"  {dataset_name:8s}: χ² = {chi2:.3f}" if chi2 != "N/A" else f"  {dataset_name:8s}: χ² = N/A")
    
    # Key predictions summary
    print(f"\nKey Predictions Summary:")
    
    # CMB predictions
    if "cmb" in dataset_results:
        cmb_pred = dataset_results["cmb"].get("predictions", {})
        if "R" in cmb_pred:
            print(f"  CMB R        = {cmb_pred['R']:.3f}")
        if "l_A" in cmb_pred:
            print(f"  CMB ℓ_A      = {cmb_pred['l_A']:.3f}")
    
    # BAO predictions
    if "bao" in dataset_results:
        bao_pred = dataset_results["bao"].get("predictions", {})
        if "D_V_over_rs" in bao_pred:
            print(f"  BAO D_V/r_s  = (multiple redshifts)")
    
    if "bao_ani" in dataset_results:
        bao_ani_pred = dataset_results["bao_ani"].get("predictions", {})
        if "D_M_over_rs" in bao_ani_pred:
            print(f"  BAO D_M/r_s  = (multiple redshifts)")
    
    # SN predictions
    if "sn" in dataset_results:
        sn_pred = dataset_results["sn"].get("predictions", {})
        if "distance_modulus" in sn_pred:
            print(f"  SN μ         = (multiple redshifts)")
    
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())