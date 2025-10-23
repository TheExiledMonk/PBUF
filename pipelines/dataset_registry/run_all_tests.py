#!/usr/bin/env python3
"""
Comprehensive test runner for dataset registry

Runs all unit tests, integration tests, and performance benchmarks
for the dataset registry system.
"""

import sys
import time
import subprocess
from pathlib import Path


def run_test_suite(test_file, description):
    """Run a test suite and report results"""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        
        if result.returncode == 0:
            print(f"✅ {description} PASSED")
            return True, duration
        else:
            print(f"❌ {description} FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False, duration
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"💥 {description} ERROR: {e}")
        return False, duration


def main():
    """Run all dataset registry tests"""
    print("Dataset Registry Comprehensive Test Suite")
    print("=" * 60)
    
    test_suites = [
        ("pipelines/dataset_registry/test_manifest_schema.py", "Manifest Schema Unit Tests"),
        ("pipelines/dataset_registry/test_download_manager.py", "Download Manager Unit Tests"),
        ("pipelines/dataset_registry/test_verification_engine.py", "Verification Engine Unit Tests"),
        ("pipelines/dataset_registry/test_registry_manager.py", "Registry Manager Unit Tests"),
        ("pipelines/dataset_registry/test_integration.py", "Integration Tests"),
    ]
    
    results = []
    total_start_time = time.time()
    
    for test_file, description in test_suites:
        success, duration = run_test_suite(test_file, description)
        results.append((description, success, duration))
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for description, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{description:<40} {status} ({duration:.2f}s)")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal Duration: {total_duration:.2f} seconds")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 {failed} test suite(s) failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())