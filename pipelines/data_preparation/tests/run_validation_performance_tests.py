#!/usr/bin/env python3
"""
Comprehensive validation and performance test runner for the data preparation framework.

This script runs all validation and performance tests including:
- Round-trip deterministic behavior tests
- Cross-validation tests with legacy loaders
- Performance benchmarks for Phase A datasets

Requirements: 8.3, 9.1 - Task 9.3 implementation
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime


def run_test_category(test_pattern: str, description: str, timeout: int = 300) -> Tuple[bool, Dict[str, Any]]:
    """
    Run a category of tests and return results.
    
    Args:
        test_pattern: Pytest pattern to match tests
        description: Description of test category
        timeout: Maximum time to allow for tests (seconds)
        
    Returns:
        Tuple of (success, results_dict)
    """
    print(f"\n{'='*70}")
    print(f"Running {description}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # Run pytest with specific pattern
        cmd = [
            sys.executable, '-m', 'pytest', 
            'pipelines/data_preparation/tests/test_validation_performance.py',
            '-k', test_pattern,
            '-v', '--tb=short', '--no-header'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        duration = time.time() - start_time
        
        # Parse pytest output
        output_lines = result.stdout.split('\n')
        
        # Count results
        passed_count = result.stdout.count('PASSED')
        failed_count = result.stdout.count('FAILED')
        error_count = result.stdout.count('ERROR')
        skipped_count = result.stdout.count('SKIPPED')
        
        total_count = passed_count + failed_count + error_count
        success = result.returncode == 0 and failed_count == 0 and error_count == 0
        
        # Extract test names and results
        test_results = []
        for line in output_lines:
            if '::test_' in line and ('PASSED' in line or 'FAILED' in line or 'ERROR' in line or 'SKIPPED' in line):
                test_results.append(line.strip())
        
        results = {
            'success': success,
            'passed': passed_count,
            'failed': failed_count,
            'errors': error_count,
            'skipped': skipped_count,
            'total': total_count,
            'duration': duration,
            'test_results': test_results,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        # Print summary
        print(f"Results: {passed_count} passed, {failed_count} failed, {error_count} errors, {skipped_count} skipped")
        print(f"Duration: {duration:.2f}s")
        
        if not success:
            print(f"\n❌ FAILURES/ERRORS:")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            # Show failed test details
            failure_lines = [line for line in output_lines if 'FAILED' in line or 'ERROR' in line]
            for line in failure_lines:
                print(f"  {line}")
        
        return success, results
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Error running tests: {e}")
        return False, {
            'success': False,
            'error': str(e),
            'duration': duration,
            'passed': 0,
            'failed': 0,
            'errors': 1,
            'total': 0
        }


def generate_performance_report(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive performance report."""
    
    # Calculate overall statistics
    total_tests = sum(r.get('total', 0) for r in results.values())
    total_passed = sum(r.get('passed', 0) for r in results.values())
    total_failed = sum(r.get('failed', 0) for r in results.values())
    total_errors = sum(r.get('errors', 0) for r in results.values())
    total_skipped = sum(r.get('skipped', 0) for r in results.values())
    total_duration = sum(r.get('duration', 0) for r in results.values())
    
    overall_success = all(r.get('success', False) for r in results.values())
    
    # Extract performance metrics from test outputs
    performance_metrics = {}
    
    for category, result in results.items():
        if 'performance' in category.lower() and result.get('stdout'):
            stdout = result['stdout']
            
            # Extract Phase A pipeline performance if available
            if 'Phase A Pipeline Performance Results' in stdout:
                lines = stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'Total processing time:' in line:
                        # Extract time from line like "Total processing time: 45.23s (0.75 minutes)"
                        import re
                        time_match = re.search(r'(\d+\.?\d*)s', line)
                        if time_match:
                            performance_metrics['phase_a_total_time'] = float(time_match.group(1))
                    elif 'Performance target:' in line:
                        time_match = re.search(r'(\d+\.?\d*)s', line)
                        if time_match:
                            performance_metrics['phase_a_target_time'] = float(time_match.group(1))
            
            # Extract individual dataset performance
            if 'Individual Performance Summary' in stdout:
                lines = stdout.split('\n')
                for line in lines:
                    if 'Total processing time:' in line and 'Individual' not in line:
                        import re
                        time_match = re.search(r'(\d+\.?\d*)s', line)
                        if time_match:
                            performance_metrics['individual_total_time'] = float(time_match.group(1))
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_success': overall_success,
        'summary': {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'errors': total_errors,
            'skipped': total_skipped,
            'duration_seconds': total_duration,
            'duration_minutes': total_duration / 60,
            'pass_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0
        },
        'categories': results,
        'performance_metrics': performance_metrics,
        'requirements_compliance': {
            'round_trip_deterministic': results.get('round_trip', {}).get('success', False),
            'cross_validation_legacy': results.get('cross_validation', {}).get('success', False),
            'performance_benchmarks': results.get('performance', {}).get('success', False),
            'phase_a_10_min_target': performance_metrics.get('phase_a_total_time', float('inf')) <= 600
        }
    }
    
    return report


def main():
    """Run all validation and performance tests and generate comprehensive report."""
    
    print("🧪 Data Preparation Framework - Validation & Performance Test Suite")
    print("=" * 80)
    print("Task 9.3: Implement validation and performance tests")
    print("Requirements: 8.3, 9.1")
    print("=" * 80)
    
    # Define test categories
    test_categories = [
        {
            'pattern': 'TestRoundTripDeterministic',
            'description': 'Round-trip Deterministic Behavior Tests',
            'timeout': 120,
            'key': 'round_trip'
        },
        {
            'pattern': 'TestCrossValidationWithLegacy',
            'description': 'Cross-validation with Legacy Loaders',
            'timeout': 180,
            'key': 'cross_validation'
        },
        {
            'pattern': 'TestPerformanceBenchmarks',
            'description': 'Performance Benchmarks (Phase A ≤ 10 min)',
            'timeout': 720,  # 12 minutes to allow for 10-minute test plus overhead
            'key': 'performance'
        }
    ]
    
    # Run all test categories
    all_results = {}
    overall_start_time = time.time()
    
    for category in test_categories:
        success, results = run_test_category(
            category['pattern'], 
            category['description'], 
            category['timeout']
        )
        all_results[category['key']] = results
        
        if not success:
            print(f"\n⚠️  {category['description']} had failures - continuing with remaining tests")
    
    overall_duration = time.time() - overall_start_time
    
    # Generate comprehensive report
    print(f"\n{'='*80}")
    print("VALIDATION & PERFORMANCE TEST REPORT")
    print(f"{'='*80}")
    
    report = generate_performance_report(all_results)
    
    # Print summary
    summary = report['summary']
    print(f"Overall Duration: {overall_duration:.2f}s ({overall_duration/60:.2f} minutes)")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Pass Rate: {summary['pass_rate']:.1f}%")
    
    # Print category results
    print(f"\n{'Category Results:':<40} {'Status':<10} {'Tests':<8} {'Duration':<10}")
    print("-" * 70)
    
    for category_name, category_data in all_results.items():
        status = "✅ PASS" if category_data.get('success', False) else "❌ FAIL"
        tests = f"{category_data.get('passed', 0)}/{category_data.get('total', 0)}"
        duration = f"{category_data.get('duration', 0):.1f}s"
        
        display_name = {
            'round_trip': 'Round-trip Deterministic Tests',
            'cross_validation': 'Cross-validation Tests',
            'performance': 'Performance Benchmarks'
        }.get(category_name, category_name)
        
        print(f"{display_name:<40} {status:<10} {tests:<8} {duration:<10}")
    
    # Print requirements compliance
    print(f"\n{'='*80}")
    print("REQUIREMENTS COMPLIANCE")
    print(f"{'='*80}")
    
    compliance = report['requirements_compliance']
    
    requirements = [
        ('Round-trip deterministic behavior', compliance['round_trip_deterministic']),
        ('Cross-validation with legacy loaders', compliance['cross_validation_legacy']),
        ('Performance benchmarks', compliance['performance_benchmarks']),
        ('Phase A ≤ 10 min target', compliance['phase_a_10_min_target'])
    ]
    
    for req_name, req_met in requirements:
        status = "✅ PASS" if req_met else "❌ FAIL"
        print(f"  {status} {req_name}")
    
    # Print performance metrics
    if report['performance_metrics']:
        print(f"\n{'='*80}")
        print("PERFORMANCE METRICS")
        print(f"{'='*80}")
        
        metrics = report['performance_metrics']
        
        if 'phase_a_total_time' in metrics:
            phase_a_time = metrics['phase_a_total_time']
            phase_a_target = metrics.get('phase_a_target_time', 600)
            print(f"Phase A Pipeline: {phase_a_time:.2f}s / {phase_a_target}s ({phase_a_time/60:.2f} min)")
            
        if 'individual_total_time' in metrics:
            individual_time = metrics['individual_total_time']
            print(f"Individual Datasets: {individual_time:.2f}s ({individual_time/60:.2f} min)")
    
    # Save detailed report
    report_file = Path("pipelines/data_preparation/tests/validation_performance_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Final result
    print(f"\n{'='*80}")
    print("FINAL RESULT")
    print(f"{'='*80}")
    
    all_requirements_met = all(compliance.values())
    overall_success = report['overall_success']
    
    if overall_success and all_requirements_met:
        print("🎉 ALL VALIDATION & PERFORMANCE TESTS PASSED")
        print("✅ Task 9.3 COMPLETE - Comprehensive testing suite implemented")
        print("✅ Round-trip deterministic behavior verified")
        print("✅ Cross-validation with legacy loaders confirmed")
        print("✅ Performance benchmarks met (Phase A ≤ 10 min)")
        print("✅ Requirements 8.3 and 9.1 satisfied")
        return 0
    else:
        print("❌ VALIDATION & PERFORMANCE TESTS INCOMPLETE")
        
        if not overall_success:
            failed_tests = summary['failed'] + summary['errors']
            print(f"✗ {failed_tests} test(s) failed or had errors")
        
        if not all_requirements_met:
            failed_reqs = [name for name, met in requirements if not met]
            print(f"✗ Requirements not met: {', '.join(failed_reqs)}")
        
        print("⚠️  Task 9.3 needs attention before completion")
        return 1


if __name__ == "__main__":
    sys.exit(main())