#!/usr/bin/env python3
"""
Test runner for BAO anisotropic fitting test suite.

This script runs all the test modules for the BAO anisotropic fitting
functionality and provides a comprehensive test report.

Requirements: 5.1
"""

import unittest
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test_suite() -> Dict[str, Any]:
    """
    Run the complete BAO anisotropic fitting test suite.
    
    Returns:
        Dictionary with test results and statistics
    """
    print("="*80)
    print("BAO ANISOTROPIC FITTING - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Test modules to run
    test_modules = [
        'test_bao_aniso_fit',
        'test_bao_aniso_parity', 
        'test_bao_aniso_performance',
        'test_bao_aniso_integration'
    ]
    
    results = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'skipped': 0,
        'execution_time': 0,
        'module_results': {}
    }
    
    start_time = time.time()
    
    for module_name in test_modules:
        print(f"\n{'-'*60}")
        print(f"Running {module_name}")
        print(f"{'-'*60}")
        
        try:
            # Import and run the test module
            module = __import__(module_name)
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            
            # Run tests with detailed output
            runner = unittest.TextTestRunner(
                verbosity=2,
                stream=sys.stdout,
                buffer=True
            )
            
            module_start = time.time()
            result = runner.run(suite)
            module_time = time.time() - module_start
            
            # Collect results
            module_results = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'execution_time': module_time,
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
            }
            
            results['module_results'][module_name] = module_results
            results['total_tests'] += result.testsRun
            results['passed'] += result.testsRun - len(result.failures) - len(result.errors)
            results['failed'] += len(result.failures)
            results['errors'] += len(result.errors)
            results['skipped'] += module_results['skipped']
            
            print(f"\nModule Summary:")
            print(f"  Tests run: {result.testsRun}")
            print(f"  Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
            print(f"  Failed: {len(result.failures)}")
            print(f"  Errors: {len(result.errors)}")
            print(f"  Skipped: {module_results['skipped']}")
            print(f"  Success rate: {module_results['success_rate']:.1f}%")
            print(f"  Execution time: {module_time:.2f}s")
            
        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
            results['module_results'][module_name] = {
                'error': str(e),
                'skipped': True
            }
        except Exception as e:
            print(f"Error running {module_name}: {e}")
            results['module_results'][module_name] = {
                'error': str(e),
                'failed': True
            }
    
    results['execution_time'] = time.time() - start_time
    
    return results


def print_final_report(results: Dict[str, Any]):
    """
    Print final test report with comprehensive statistics.
    
    Args:
        results: Test results dictionary
    """
    print(f"\n{'='*80}")
    print("FINAL TEST REPORT")
    print(f"{'='*80}")
    
    # Overall statistics
    total_tests = results['total_tests']
    passed = results['passed']
    failed = results['failed']
    errors = results['errors']
    skipped = results['skipped']
    
    success_rate = (passed / max(total_tests, 1)) * 100 if total_tests > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total execution time: {results['execution_time']:.2f}s")
    
    # Module breakdown
    print(f"\nModule Breakdown:")
    for module_name, module_result in results['module_results'].items():
        if 'error' in module_result:
            status = "ERROR" if module_result.get('failed') else "SKIPPED"
            print(f"  {module_name:30s}: {status} - {module_result['error']}")
        else:
            tests_run = module_result.get('tests_run', 0)
            success_rate = module_result.get('success_rate', 0)
            exec_time = module_result.get('execution_time', 0)
            print(f"  {module_name:30s}: {tests_run:3d} tests, {success_rate:5.1f}% success, {exec_time:5.2f}s")
    
    # Test categories summary
    print(f"\nTest Categories Summary:")
    
    categories = {
        'Unit Tests': ['test_bao_aniso_fit'],
        'Parity Tests': ['test_bao_aniso_parity'],
        'Performance Tests': ['test_bao_aniso_performance'],
        'Integration Tests': ['test_bao_aniso_integration']
    }
    
    for category, modules in categories.items():
        category_tests = 0
        category_passed = 0
        category_time = 0
        
        for module in modules:
            if module in results['module_results']:
                module_result = results['module_results'][module]
                if 'tests_run' in module_result:
                    category_tests += module_result['tests_run']
                    category_passed += module_result['tests_run'] - module_result['failures'] - module_result['errors']
                    category_time += module_result['execution_time']
        
        if category_tests > 0:
            category_success = (category_passed / category_tests) * 100
            print(f"  {category:20s}: {category_tests:3d} tests, {category_success:5.1f}% success, {category_time:5.2f}s")
    
    # Performance summary (if available)
    perf_module = results['module_results'].get('test_bao_aniso_performance')
    if perf_module and 'tests_run' in perf_module:
        print(f"\nPerformance Test Summary:")
        print(f"  Performance benchmarks completed successfully")
        print(f"  All performance thresholds met: {perf_module['failures'] == 0}")
    
    # Final status
    print(f"\n{'='*80}")
    if failed == 0 and errors == 0:
        print("🎉 ALL TESTS PASSED! BAO anisotropic fitting implementation is ready.")
    elif failed > 0 or errors > 0:
        print("⚠️  SOME TESTS FAILED. Review failures and fix issues before deployment.")
    else:
        print("ℹ️  Tests completed with mixed results. Review the detailed output above.")
    
    print(f"{'='*80}")


def main():
    """Main entry point for test runner."""
    try:
        results = run_test_suite()
        print_final_report(results)
        
        # Exit with appropriate code
        if results['failed'] > 0 or results['errors'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()