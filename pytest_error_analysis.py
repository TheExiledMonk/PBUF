#!/usr/bin/env python3
"""
Pytest Error Analysis Script
Groups pytest failures by error type and provides a summary
"""

import subprocess
import re
from collections import defaultdict

def run_pytest_and_analyze():
    """Run pytest and analyze the error patterns"""
    
    # Run pytest with minimal output to get error summary
    result = subprocess.run(['pytest', '--tb=no', '-v'], 
                          capture_output=True, text=True)
    
    output = result.stdout + result.stderr
    
    # Extract failed tests
    failed_tests = []
    for line in output.split('\n'):
        if 'FAILED' in line:
            failed_tests.append(line.strip())
    
    # Group errors by type
    error_groups = defaultdict(list)
    
    for test in failed_tests:
        # Extract error type from test name and common patterns
        if 'ValueError: Input file does not exist' in test:
            error_groups['File Not Found Errors'].append(test)
        elif 'ValueError: not enough values to unpack' in test:
            error_groups['Unpacking Errors'].append(test)
        elif 'AssertionError: Regex pattern did not match' in test:
            error_groups['Regex Pattern Errors'].append(test)
        elif 'ProcessingError: Processing failed for dataset' in test:
            error_groups['Dataset Processing Errors'].append(test)
        elif 'FileNotFoundError: Dataset file not found' in test:
            error_groups['Dataset File Missing'].append(test)
        elif 'RuntimeError: Dataset verification failed' in test:
            error_groups['Dataset Verification Errors'].append(test)
        elif 'AssertionError: Parameter test failed' in test:
            error_groups['Parameter Configuration Errors'].append(test)
        elif 'Missing required columns' in test:
            error_groups['Column Validation Errors'].append(test)
        elif 'test_bao_derivation' in test:
            error_groups['BAO Derivation Module Errors'].append(test)
        elif 'test_cc_derivation' in test:
            error_groups['CC Derivation Module Errors'].append(test)
        elif 'test_cmb_derivation' in test:
            error_groups['CMB Derivation Module Errors'].append(test)
        elif 'test_sn_derivation' in test:
            error_groups['SN Derivation Module Errors'].append(test)
        elif 'test_rsd_derivation' in test:
            error_groups['RSD Derivation Module Errors'].append(test)
        elif 'test_validation_performance' in test:
            error_groups['Performance/Validation Errors'].append(test)
        elif 'test_output_manager' in test:
            error_groups['Output Manager Errors'].append(test)
        else:
            error_groups['Other Errors'].append(test)
    
    return error_groups, len(failed_tests)

def print_analysis(error_groups, total_failures):
    """Print the error analysis"""
    
    print("=" * 80)
    print("PYTEST ERROR ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Failed Tests: {total_failures}")
    print(f"Error Categories: {len(error_groups)}")
    print()
    
    # Sort by number of errors (descending)
    sorted_groups = sorted(error_groups.items(), 
                          key=lambda x: len(x[1]), reverse=True)
    
    for category, tests in sorted_groups:
        print(f"📋 {category} ({len(tests)} failures)")
        print("-" * 60)
        
        # Show first few examples
        for i, test in enumerate(tests[:3]):
            # Clean up the test name for readability
            clean_test = test.replace('FAILED ', '').split(' - ')[0]
            print(f"  {i+1}. {clean_test}")
        
        if len(tests) > 3:
            print(f"  ... and {len(tests) - 3} more similar failures")
        print()
    
    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    # Provide specific recommendations based on error patterns
    if 'File Not Found Errors' in error_groups:
        print("🔧 File Not Found Issues:")
        print("   - Check test data setup and file paths")
        print("   - Ensure test fixtures create required files")
        print()
    
    if 'Dataset Processing Errors' in error_groups:
        print("🔧 Dataset Processing Issues:")
        print("   - Review dataset validation logic")
        print("   - Check mock data generation in tests")
        print("   - Verify derivation module implementations")
        print()
    
    if 'Column Validation Errors' in error_groups:
        print("🔧 Column Validation Issues:")
        print("   - Review expected column names in test data")
        print("   - Check data format consistency")
        print("   - Verify column mapping logic")
        print()
    
    if any('Derivation Module' in cat for cat in error_groups.keys()):
        print("🔧 Derivation Module Issues:")
        print("   - Multiple derivation modules failing similarly")
        print("   - Likely common interface or base class issue")
        print("   - Review shared validation/processing logic")
        print()

if __name__ == "__main__":
    error_groups, total_failures = run_pytest_and_analyze()
    print_analysis(error_groups, total_failures)