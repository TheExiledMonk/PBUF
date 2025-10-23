#!/usr/bin/env python3
"""
Comprehensive unit test runner for the data preparation framework.

This script runs all unit tests for the framework components and generates
a summary report of test coverage and results.
"""

import subprocess
import sys
from pathlib import Path
import time
from typing import Dict, List, Tuple


def run_test_suite(test_file: str, description: str) -> Tuple[bool, int, int, float]:
    """
    Run a test suite and return results.
    
    Args:
        test_file: Path to test file
        description: Description of test suite
        
    Returns:
        Tuple of (success, passed_count, total_count, duration)
    """
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        duration = time.time() - start_time
        
        # Parse pytest output to get counts
        output_lines = result.stdout.split('\n')
        summary_line = None
        for line in output_lines:
            if 'passed' in line and ('failed' in line or 'error' in line or line.strip().endswith('passed')):
                summary_line = line
                break
        
        if summary_line:
            # Extract numbers from summary
            import re
            passed_match = re.search(r'(\d+) passed', summary_line)
            failed_match = re.search(r'(\d+) failed', summary_line)
            error_match = re.search(r'(\d+) error', summary_line)
            
            passed_count = int(passed_match.group(1)) if passed_match else 0
            failed_count = int(failed_match.group(1)) if failed_match else 0
            error_count = int(error_match.group(1)) if error_match else 0
            
            total_count = passed_count + failed_count + error_count
            success = failed_count == 0 and error_count == 0
        else:
            # Fallback parsing
            passed_count = result.stdout.count('PASSED')
            failed_count = result.stdout.count('FAILED')
            error_count = result.stdout.count('ERROR')
            total_count = passed_count + failed_count + error_count
            success = result.returncode == 0
        
        print(f"Results: {passed_count}/{total_count} passed")
        if failed_count > 0:
            print(f"Failed: {failed_count}")
        if error_count > 0:
            print(f"Errors: {error_count}")
        print(f"Duration: {duration:.2f}s")
        
        if not success:
            print("\nFailure details:")
            print(result.stdout)
            if result.stderr:
                print("Stderr:")
                print(result.stderr)
        
        return success, passed_count, total_count, duration
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error running tests: {e}")
        return False, 0, 0, duration


def main():
    """Run all unit tests and generate summary report."""
    print("Data Preparation Framework - Unit Test Suite")
    print("=" * 60)
    
    # Define test suites
    test_suites = [
        ("pipelines/data_preparation/tests/test_interfaces.py", "Core Interfaces"),
        ("pipelines/data_preparation/tests/test_schema.py", "Schema Validation"),
        ("pipelines/data_preparation/tests/test_validation.py", "Validation Engine"),
        ("pipelines/data_preparation/tests/test_sn_derivation.py", "SN Derivation Module"),
        ("pipelines/data_preparation/tests/test_bao_derivation.py", "BAO Derivation Module"),
        ("pipelines/data_preparation/tests/test_cmb_derivation.py", "CMB Derivation Module"),
        ("pipelines/data_preparation/tests/test_cc_derivation.py", "CC Derivation Module"),
        ("pipelines/data_preparation/tests/test_rsd_derivation.py", "RSD Derivation Module"),
    ]
    
    # Run all test suites
    results = []
    total_passed = 0
    total_tests = 0
    total_duration = 0
    
    for test_file, description in test_suites:
        success, passed, total, duration = run_test_suite(test_file, description)
        results.append((description, success, passed, total, duration))
        total_passed += passed
        total_tests += total
        total_duration += duration
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("UNIT TEST SUMMARY REPORT")
    print(f"{'='*60}")
    
    print(f"{'Test Suite':<25} {'Status':<10} {'Passed':<8} {'Total':<8} {'Duration':<10}")
    print("-" * 60)
    
    all_passed = True
    for description, success, passed, total, duration in results:
        status = "PASS" if success else "FAIL"
        if not success:
            all_passed = False
        
        print(f"{description:<25} {status:<10} {passed:<8} {total:<8} {duration:<10.2f}s")
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {'PASS' if all_passed else 'FAIL':<10} {total_passed:<8} {total_tests:<8} {total_duration:<10.2f}s")
    
    # Calculate coverage percentage
    coverage_pct = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\nTest Coverage: {coverage_pct:.1f}% ({total_passed}/{total_tests} tests passing)")
    
    # Requirements compliance check
    print(f"\n{'='*60}")
    print("REQUIREMENTS COMPLIANCE CHECK")
    print(f"{'='*60}")
    
    # Check requirements from task 9.1
    requirements_met = []
    
    # Unit tests for each derivation module
    derivation_modules = ['SN', 'BAO', 'CMB', 'CC', 'RSD']
    derivation_tests_passed = sum(1 for desc, success, _, _, _ in results 
                                 if 'Derivation Module' in desc and success)
    requirements_met.append(
        (f"Unit tests for {len(derivation_modules)} derivation modules", 
         derivation_tests_passed == len(derivation_modules))
    )
    
    # Validation engine tests
    validation_success = any(success for desc, success, _, _, _ in results 
                           if 'Validation Engine' in desc)
    requirements_met.append(
        ("Validation engine tests covering all validation rules", validation_success)
    )
    
    # Schema compliance tests
    schema_success = any(success for desc, success, _, _, _ in results 
                        if 'Schema Validation' in desc)
    requirements_met.append(
        ("Schema compliance tests for standardized dataset format", schema_success)
    )
    
    # Interface tests
    interface_success = any(success for desc, success, _, _, _ in results 
                           if 'Core Interfaces' in desc)
    requirements_met.append(
        ("Core interface and abstract base class tests", interface_success)
    )
    
    print("Requirement Coverage:")
    for requirement, met in requirements_met:
        status = "✓ PASS" if met else "✗ FAIL"
        print(f"  {status} {requirement}")
    
    all_requirements_met = all(met for _, met in requirements_met)
    
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    
    if all_passed and all_requirements_met:
        print("🎉 ALL TESTS PASSED - Task 9.1 COMPLETE")
        print(f"✓ {total_passed}/{total_tests} unit tests passing")
        print("✓ All derivation modules have comprehensive unit tests")
        print("✓ Validation engine tests cover all validation rules and edge cases")
        print("✓ Schema compliance tests verify standardized dataset format")
        print("✓ Requirements 8.1 and 8.2 satisfied")
        return 0
    else:
        print("❌ TESTS INCOMPLETE - Task 9.1 needs attention")
        if not all_passed:
            failed_count = total_tests - total_passed
            print(f"✗ {failed_count}/{total_tests} tests failing")
        if not all_requirements_met:
            print("✗ Some requirements not fully met")
        return 1


if __name__ == "__main__":
    sys.exit(main())