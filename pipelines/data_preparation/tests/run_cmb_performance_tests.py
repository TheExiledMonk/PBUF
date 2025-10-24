#!/usr/bin/env python3
"""
Test runner for CMB performance and stress tests.

This script runs comprehensive performance and stress tests for the CMB raw parameter
integration feature, providing detailed reporting and monitoring.

Usage:
    python run_cmb_performance_tests.py [--quick] [--stress-only] [--report-file output.json]
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import psutil

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def run_performance_tests(quick_mode: bool = False, stress_only: bool = False, 
                         report_file: str = None) -> Dict[str, Any]:
    """
    Run CMB performance and stress tests with comprehensive reporting.
    
    Args:
        quick_mode: If True, run only essential performance tests
        stress_only: If True, run only stress tests
        report_file: Path to save detailed test report
        
    Returns:
        Dictionary containing test results and performance metrics
    """
    print("=" * 80)
    print("CMB Raw Parameter Integration - Performance & Stress Tests")
    print("=" * 80)
    
    # Initialize test report
    test_report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': get_system_info(),
        'test_configuration': {
            'quick_mode': quick_mode,
            'stress_only': stress_only
        },
        'test_results': {},
        'performance_metrics': {},
        'summary': {}
    }
    
    # Define test modules and their categories
    test_modules = {
        'performance': [
            'test_cmb_performance_stress::TestProcessingTimePerformance',
            'test_cmb_performance_stress::TestMemoryUsageMonitoring',
            'test_cmb_performance_stress::TestPerformanceRegression'
        ],
        'stress': [
            'test_cmb_performance_stress::TestNumericalStabilityExtreme',
            'test_cmb_performance_stress::TestStressConditions'
        ]
    }
    
    # Select tests to run based on mode
    if stress_only:
        tests_to_run = test_modules['stress']
        print("Running STRESS TESTS ONLY")
    elif quick_mode:
        tests_to_run = [
            'test_cmb_performance_stress::TestProcessingTimePerformance::test_single_parameter_set_processing_time',
            'test_cmb_performance_stress::TestMemoryUsageMonitoring::test_distance_prior_memory_usage',
            'test_cmb_performance_stress::TestNumericalStabilityExtreme::test_extreme_valid_parameter_ranges'
        ]
        print("Running QUICK PERFORMANCE TESTS")
    else:
        tests_to_run = test_modules['performance'] + test_modules['stress']
        print("Running FULL PERFORMANCE & STRESS TEST SUITE")
    
    print(f"Total test modules: {len(tests_to_run)}")
    print()
    
    # Run tests
    overall_start_time = time.time()
    
    for i, test_module in enumerate(tests_to_run, 1):
        print(f"[{i}/{len(tests_to_run)}] Running {test_module}")
        print("-" * 60)
        
        # Run individual test module
        result = run_test_module(test_module)
        test_report['test_results'][test_module] = result
        
        # Print summary
        if result['success']:
            print(f"✓ PASSED in {result['duration']:.2f}s")
            if result['performance_metrics']:
                print("  Performance metrics:")
                for metric, value in result['performance_metrics'].items():
                    print(f"    {metric}: {value}")
        else:
            print(f"✗ FAILED in {result['duration']:.2f}s")
            print(f"  Error: {result['error']}")
        
        print()
    
    overall_duration = time.time() - overall_start_time
    
    # Generate summary
    test_report['summary'] = generate_test_summary(test_report['test_results'], overall_duration)
    
    # Print final summary
    print_test_summary(test_report['summary'])
    
    # Save detailed report if requested
    if report_file:
        save_test_report(test_report, report_file)
        print(f"\nDetailed report saved to: {report_file}")
    
    return test_report


def run_test_module(test_module: str) -> Dict[str, Any]:
    """
    Run a single test module and capture results.
    
    Args:
        test_module: Test module specification (e.g., 'module::Class::method')
        
    Returns:
        Dictionary with test results and performance metrics
    """
    start_time = time.time()
    start_memory = get_memory_usage_mb()
    
    try:
        # Run pytest with specific test
        test_path = f'pipelines/data_preparation/tests/{test_module}'
        cmd = [
            sys.executable, '-m', 'pytest',
            test_path,
            '-v',
            '--tb=short',
            '--no-header',
            '--quiet'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=300  # 5 minute timeout per test module
        )
        
        duration = time.time() - start_time
        end_memory = get_memory_usage_mb()
        
        # Parse performance metrics from output if available
        performance_metrics = parse_performance_metrics(result.stdout)
        
        return {
            'success': result.returncode == 0,
            'duration': duration,
            'memory_usage_mb': end_memory - start_memory,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'performance_metrics': performance_metrics,
            'error': result.stderr if result.returncode != 0 else None
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'duration': time.time() - start_time,
            'memory_usage_mb': 0,
            'stdout': '',
            'stderr': 'Test timed out after 5 minutes',
            'performance_metrics': {},
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'success': False,
            'duration': time.time() - start_time,
            'memory_usage_mb': 0,
            'stdout': '',
            'stderr': str(e),
            'performance_metrics': {},
            'error': str(e)
        }


def parse_performance_metrics(test_output: str) -> Dict[str, Any]:
    """
    Parse performance metrics from test output.
    
    Args:
        test_output: Captured test output
        
    Returns:
        Dictionary of parsed performance metrics
    """
    metrics = {}
    
    # Look for common performance indicators in output
    lines = test_output.split('\n')
    for line in lines:
        line = line.strip()
        
        # Parse timing information
        if 'processing:' in line.lower() and 's' in line:
            try:
                # Extract timing value
                parts = line.split(':')
                if len(parts) >= 2:
                    time_part = parts[1].strip()
                    if 's' in time_part:
                        time_value = float(time_part.replace('s', '').strip())
                        metric_name = parts[0].strip().lower().replace(' ', '_')
                        metrics[f'{metric_name}_seconds'] = time_value
            except (ValueError, IndexError):
                continue
        
        # Parse memory information
        if 'memory' in line.lower() and 'mb' in line.lower():
            try:
                # Extract memory value
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        memory_part = parts[1].strip().lower()
                        if 'mb' in memory_part:
                            memory_value = float(memory_part.replace('mb', '').strip())
                            metric_name = parts[0].strip().lower().replace(' ', '_')
                            metrics[f'{metric_name}_mb'] = memory_value
            except (ValueError, IndexError):
                continue
    
    return metrics


def get_system_info() -> Dict[str, Any]:
    """Get system information for test reporting."""
    return {
        'platform': sys.platform,
        'python_version': sys.version,
        'cpu_count': os.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3)
    }


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def generate_test_summary(test_results: Dict[str, Any], total_duration: float) -> Dict[str, Any]:
    """Generate summary statistics from test results."""
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result['success'])
    failed_tests = total_tests - passed_tests
    
    # Calculate performance statistics
    durations = [result['duration'] for result in test_results.values()]
    memory_usage = [result['memory_usage_mb'] for result in test_results.values()]
    
    return {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
        'total_duration': total_duration,
        'average_test_duration': sum(durations) / len(durations) if durations else 0,
        'max_test_duration': max(durations) if durations else 0,
        'total_memory_usage_mb': sum(memory_usage),
        'max_memory_usage_mb': max(memory_usage) if memory_usage else 0,
        'failed_test_names': [name for name, result in test_results.items() if not result['success']]
    }


def print_test_summary(summary: Dict[str, Any]):
    """Print formatted test summary."""
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"Total Tests:     {summary['total_tests']}")
    print(f"Passed:          {summary['passed_tests']}")
    print(f"Failed:          {summary['failed_tests']}")
    print(f"Success Rate:    {summary['success_rate']:.1%}")
    print()
    
    print(f"Total Duration:  {summary['total_duration']:.2f}s")
    print(f"Average/Test:    {summary['average_test_duration']:.2f}s")
    print(f"Longest Test:    {summary['max_test_duration']:.2f}s")
    print()
    
    print(f"Total Memory:    {summary['total_memory_usage_mb']:.1f}MB")
    print(f"Peak Memory:     {summary['max_memory_usage_mb']:.1f}MB")
    
    if summary['failed_tests'] > 0:
        print()
        print("FAILED TESTS:")
        for test_name in summary['failed_test_names']:
            print(f"  - {test_name}")
    
    print("=" * 80)


def save_test_report(report: Dict[str, Any], file_path: str):
    """Save detailed test report to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description='Run CMB performance and stress tests',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run only essential performance tests (faster)'
    )
    
    parser.add_argument(
        '--stress-only',
        action='store_true',
        help='Run only stress tests'
    )
    
    parser.add_argument(
        '--report-file',
        type=str,
        help='Save detailed test report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Run tests
    try:
        report = run_performance_tests(
            quick_mode=args.quick,
            stress_only=args.stress_only,
            report_file=args.report_file
        )
        
        # Exit with appropriate code
        if report['summary']['failed_tests'] > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()