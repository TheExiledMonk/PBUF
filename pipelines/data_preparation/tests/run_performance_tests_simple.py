#!/usr/bin/env python3
"""
Simple test runner for CMB performance and stress tests.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_performance_tests():
    """Run CMB performance tests with simple reporting."""
    
    print("=" * 80)
    print("CMB Raw Parameter Integration - Performance & Stress Tests")
    print("=" * 80)
    
    # Define test cases to run
    test_cases = [
        "TestProcessingTimePerformance::test_single_parameter_set_processing_time",
        "TestProcessingTimePerformance::test_jacobian_computation_performance", 
        "TestProcessingTimePerformance::test_batch_processing_performance",
        "TestMemoryUsageMonitoring::test_distance_prior_memory_usage",
        "TestMemoryUsageMonitoring::test_covariance_propagation_memory_scaling",
        "TestNumericalStabilityExtreme::test_extreme_valid_parameter_ranges",
        "TestNumericalStabilityExtreme::test_numerical_precision_limits",
        "TestStressConditions::test_concurrent_processing_simulation",
        "TestStressConditions::test_repeated_processing_stability"
    ]
    
    passed = 0
    failed = 0
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Running {test_case}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Run the test
        cmd = [
            sys.executable, '-m', 'pytest',
            f'pipelines/data_preparation/tests/test_cmb_performance_stress.py::{test_case}',
            '-v', '--tb=short'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            duration = time.time() - start_time
            total_time += duration
            
            if result.returncode == 0:
                print(f"✓ PASSED in {duration:.2f}s")
                passed += 1
            else:
                print(f"✗ FAILED in {duration:.2f}s")
                print(f"Error output: {result.stderr}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            total_time += duration
            print(f"✗ TIMEOUT after {duration:.2f}s")
            failed += 1
        except Exception as e:
            duration = time.time() - start_time
            total_time += duration
            print(f"✗ ERROR: {e}")
            failed += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests:     {len(test_cases)}")
    print(f"Passed:          {passed}")
    print(f"Failed:          {failed}")
    print(f"Success Rate:    {passed/len(test_cases):.1%}")
    print(f"Total Duration:  {total_time:.2f}s")
    print(f"Average/Test:    {total_time/len(test_cases):.2f}s")
    print("=" * 80)
    
    return failed == 0

if __name__ == '__main__':
    success = run_performance_tests()
    sys.exit(0 if success else 1)