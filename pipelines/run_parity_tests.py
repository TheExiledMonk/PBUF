#!/usr/bin/env python3
"""
Command-line interface for running parity tests between legacy and unified systems.

This script provides a convenient interface for executing comprehensive numerical
equivalence validation across all cosmological blocks and models.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import argparse
import sys
import json
import os
from typing import List, Optional
from pathlib import Path

# Add the pipelines directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fit_core.parity_testing import (
    ParityTester, 
    ParityConfig, 
    run_comprehensive_parity_suite
)


def main():
    """Main entry point for parity testing CLI."""
    args = parse_arguments()
    
    try:
        # Create configuration
        config = ParityConfig(
            tolerance=args.tolerance,
            relative_tolerance=args.relative_tolerance,
            legacy_scripts_path=args.legacy_path,
            output_dir=args.output_dir,
            verbose=args.verbose,
            save_intermediate=args.save_intermediate
        )
        
        if args.comprehensive:
            # Run comprehensive test suite
            print("Running comprehensive parity test suite...")
            reports = run_comprehensive_parity_suite(
                config=config,
                models=args.models,
                dataset_combinations=parse_dataset_combinations(args.dataset_combinations)
            )
            
            # Generate summary report
            generate_summary_report(reports, config.output_dir)
            
        else:
            # Run individual test
            tester = ParityTester(config)
            
            test_name = args.test_name or f"{args.model}_{'_'.join(args.datasets)}"
            
            # Parse parameter overrides
            overrides = {}
            if args.overrides:
                try:
                    overrides = json.loads(args.overrides)
                except json.JSONDecodeError as e:
                    print(f"Error parsing parameter overrides: {e}", file=sys.stderr)
                    return 1
            
            # Run single parity test
            report = tester.run_parity_test(
                test_name=test_name,
                model=args.model,
                datasets=args.datasets,
                parameters=overrides if overrides else None
            )
            
            # Save and display report
            report_path = tester.save_report(report)
            
            if args.show_report:
                print("\n" + tester.generate_parity_report(report))
            
            # Exit with appropriate code
            return 0 if report.overall_pass else 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parity tests between legacy and unified cosmology systems",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Test configuration
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive test suite across all models and dataset combinations"
    )
    
    parser.add_argument(
        "--test-name",
        type=str,
        help="Name for the test (auto-generated if not provided)"
    )
    
    # System configuration
    parser.add_argument(
        "--model",
        choices=["lcdm", "pbuf"],
        default="pbuf",
        help="Cosmological model to test"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["cmb", "bao", "bao_ani", "sn"],
        default=["cmb"],
        help="Datasets to include in the test"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lcdm", "pbuf"],
        default=["lcdm", "pbuf"],
        help="Models to test in comprehensive mode"
    )
    
    parser.add_argument(
        "--dataset-combinations",
        type=str,
        help="JSON string specifying dataset combinations for comprehensive testing"
    )
    
    # Parameter overrides
    parser.add_argument(
        "--overrides",
        type=str,
        help="JSON string with parameter overrides (e.g., '{\"H0\": 70.0, \"Om0\": 0.3}')"
    )
    
    # Tolerance settings
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numerical comparisons"
    )
    
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1e-6,
        help="Relative tolerance for numerical comparisons"
    )
    
    # Paths and output
    parser.add_argument(
        "--legacy-path",
        type=str,
        help="Path to legacy scripts directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="parity_results",
        help="Directory for output files"
    )
    
    # Display options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--show-report",
        action="store_true",
        help="Display the parity report after completion"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=True,
        help="Save intermediate results for debugging"
    )
    
    return parser.parse_args()


def parse_dataset_combinations(combinations_str: Optional[str]) -> Optional[List[List[str]]]:
    """Parse dataset combinations from JSON string."""
    if not combinations_str:
        return None
    
    try:
        return json.loads(combinations_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing dataset combinations: {e}", file=sys.stderr)
        return None


def generate_summary_report(reports: List, output_dir: str) -> None:
    """Generate a summary report for comprehensive testing."""
    summary_path = os.path.join(output_dir, "parity_summary.txt")
    
    total_tests = len(reports)
    passed_tests = sum(1 for report in reports if report.overall_pass)
    failed_tests = total_tests - passed_tests
    
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE PARITY TEST SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Passed:      {passed_tests}\n")
        f.write(f"Failed:      {failed_tests}\n")
        f.write(f"Success Rate: {passed_tests/total_tests*100:.1f}%\n")
        f.write("\n")
        
        if failed_tests > 0:
            f.write("FAILED TESTS:\n")
            f.write("-" * 40 + "\n")
            for report in reports:
                if not report.overall_pass:
                    f.write(f"  {report.test_name}\n")
                    failed_comparisons = [c for c in report.comparisons if not c.passes_tolerance]
                    for comp in failed_comparisons[:3]:  # Show first 3 failures
                        f.write(f"    - {comp.metric_name}: diff={comp.absolute_diff:.2e}\n")
                    if len(failed_comparisons) > 3:
                        f.write(f"    - ... and {len(failed_comparisons)-3} more\n")
            f.write("\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 40 + "\n")
        for report in reports:
            status = "PASS" if report.overall_pass else "FAIL"
            f.write(f"  {report.test_name:25s} {status:4s} "
                   f"({len([c for c in report.comparisons if c.passes_tolerance])}"
                   f"/{len(report.comparisons)} comparisons passed)\n")
    
    print(f"Summary report saved to: {summary_path}")


if __name__ == "__main__":
    sys.exit(main())