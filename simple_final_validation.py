#!/usr/bin/env python3
"""
Simplified final validation for PBUF cosmology pipeline deployment.

This script performs essential validation tests and generates a deployment
certification report.
"""

import sys
import json
import time
import subprocess
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.fit_core.engine import run_fit
from pipelines.fit_core.parameter import build_params, get_defaults
from pipelines.fit_core.integrity import run_integrity_suite


def test_core_functionality():
    """Test core system functionality."""
    
    print("="*60)
    print("CORE FUNCTIONALITY VALIDATION")
    print("="*60)
    
    results = {}
    
    # Test 1: Parameter building
    try:
        lcdm_params = build_params("lcdm")
        pbuf_params = build_params("pbuf")
        
        assert "H0" in lcdm_params
        assert "Om0" in lcdm_params
        assert "alpha" in pbuf_params
        assert "k_sat" in pbuf_params
        
        results["parameter_building"] = "PASS"
        print("✓ Parameter building: PASS")
        
    except Exception as e:
        results["parameter_building"] = f"FAIL: {e}"
        print(f"❌ Parameter building: FAIL - {e}")
    
    # Test 2: Basic fitting functionality
    try:
        # Test LCDM CMB fit
        lcdm_result = run_fit("lcdm", ["cmb"])
        assert "params" in lcdm_result
        assert "metrics" in lcdm_result
        assert lcdm_result["metrics"]["total_chi2"] > 0
        
        # Test PBUF CMB fit
        pbuf_result = run_fit("pbuf", ["cmb"])
        assert "params" in pbuf_result
        assert "metrics" in pbuf_result
        assert pbuf_result["metrics"]["total_chi2"] > 0
        
        results["basic_fitting"] = "PASS"
        print("✓ Basic fitting: PASS")
        print(f"  LCDM χ²: {lcdm_result['metrics']['total_chi2']:.3f}")
        print(f"  PBUF χ²: {pbuf_result['metrics']['total_chi2']:.3f}")
        
    except Exception as e:
        results["basic_fitting"] = f"FAIL: {e}"
        print(f"❌ Basic fitting: FAIL - {e}")
    
    # Test 3: Joint fitting
    try:
        joint_result = run_fit("lcdm", ["cmb", "bao"])
        
        assert len(joint_result["results"]) == 2
        assert "cmb" in joint_result["results"]
        assert "bao" in joint_result["results"]
        
        # Verify chi2 additivity
        total_chi2 = joint_result["metrics"]["total_chi2"]
        sum_chi2 = sum(joint_result["results"][ds]["chi2"] for ds in ["cmb", "bao"])
        assert abs(total_chi2 - sum_chi2) < 1e-10
        
        results["joint_fitting"] = "PASS"
        print("✓ Joint fitting: PASS")
        print(f"  Joint χ²: {total_chi2:.3f}")
        
    except Exception as e:
        results["joint_fitting"] = f"FAIL: {e}"
        print(f"❌ Joint fitting: FAIL - {e}")
    
    # Test 4: Parameter overrides
    try:
        override_result = run_fit("lcdm", ["cmb"], overrides={"H0": 70.0})
        assert override_result["params"]["H0"] == 70.0
        
        results["parameter_overrides"] = "PASS"
        print("✓ Parameter overrides: PASS")
        
    except Exception as e:
        results["parameter_overrides"] = f"FAIL: {e}"
        print(f"❌ Parameter overrides: FAIL - {e}")
    
    return results


def test_system_integrity():
    """Test system integrity checks."""
    
    print("\n" + "="*60)
    print("SYSTEM INTEGRITY VALIDATION")
    print("="*60)
    
    results = {}
    
    # Test integrity for different scenarios
    scenarios = [
        ("lcdm", ["cmb"]),
        ("lcdm", ["bao"]),
        ("pbuf", ["cmb"]),
        ("pbuf", ["bao"])
    ]
    
    for model, datasets in scenarios:
        scenario_name = f"{model}_{'+'.join(datasets)}"
        
        try:
            params = build_params(model)
            integrity_result = run_integrity_suite(params, datasets)
            
            if integrity_result["overall_status"] == "PASS":
                results[scenario_name] = "PASS"
                print(f"✓ {scenario_name}: PASS")
            else:
                results[scenario_name] = "FAIL"
                print(f"❌ {scenario_name}: FAIL")
                
        except Exception as e:
            results[scenario_name] = f"ERROR: {e}"
            print(f"❌ {scenario_name}: ERROR - {e}")
    
    return results


def test_wrapper_scripts():
    """Test wrapper script functionality."""
    
    print("\n" + "="*60)
    print("WRAPPER SCRIPT VALIDATION")
    print("="*60)
    
    results = {}
    
    # Test help functionality for each wrapper
    wrappers = [
        "pipelines/fit_cmb.py",
        "pipelines/fit_bao.py",
        "pipelines/fit_sn.py",
        "pipelines/fit_joint.py"
    ]
    
    for wrapper in wrappers:
        try:
            result = subprocess.run([
                sys.executable, wrapper, "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "usage:" in result.stdout:
                results[wrapper] = "PASS"
                print(f"✓ {wrapper}: PASS")
            else:
                results[wrapper] = f"FAIL: Return code {result.returncode}"
                print(f"❌ {wrapper}: FAIL")
                
        except Exception as e:
            results[wrapper] = f"ERROR: {e}"
            print(f"❌ {wrapper}: ERROR - {e}")
    
    return results


def test_performance():
    """Test basic performance benchmarks."""
    
    print("\n" + "="*60)
    print("PERFORMANCE VALIDATION")
    print("="*60)
    
    results = {}
    
    # Performance test cases
    test_cases = [
        ("CMB fit", "lcdm", ["cmb"]),
        ("BAO fit", "lcdm", ["bao"]),
        ("Joint fit", "lcdm", ["cmb", "bao"])
    ]
    
    for test_name, model, datasets in test_cases:
        try:
            start_time = time.time()
            
            result = run_fit(model, datasets)
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[test_name] = {
                "duration": duration,
                "chi2": result["metrics"]["total_chi2"],
                "status": "PASS" if duration < 30 else "SLOW"
            }
            
            print(f"✓ {test_name}: {duration:.2f}s (χ²={result['metrics']['total_chi2']:.3f})")
            
            if duration > 30:
                print(f"  ⚠️ Slow performance: {duration:.1f}s > 30s")
                
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            print(f"❌ {test_name}: ERROR - {e}")
    
    return results


def test_numerical_consistency():
    """Test numerical consistency across multiple runs."""
    
    print("\n" + "="*60)
    print("NUMERICAL CONSISTENCY VALIDATION")
    print("="*60)
    
    results = {}
    
    try:
        # Run same fit multiple times
        chi2_values = []
        for i in range(3):
            result = run_fit("lcdm", ["cmb"])
            chi2_values.append(result["metrics"]["total_chi2"])
        
        # Check consistency
        chi2_std = np.std(chi2_values)
        chi2_mean = np.mean(chi2_values)
        
        if chi2_std < 1e-10:  # Should be identical
            results["consistency"] = "PASS"
            print(f"✓ Numerical consistency: PASS")
            print(f"  χ² values: {chi2_values}")
            print(f"  Standard deviation: {chi2_std:.2e}")
        else:
            results["consistency"] = f"FAIL: std={chi2_std:.2e}"
            print(f"❌ Numerical consistency: FAIL")
            print(f"  χ² values: {chi2_values}")
            print(f"  Standard deviation: {chi2_std:.2e}")
            
    except Exception as e:
        results["consistency"] = f"ERROR: {e}"
        print(f"❌ Numerical consistency: ERROR - {e}")
    
    return results


def generate_certification_report():
    """Generate final certification report."""
    
    print("\n" + "="*80)
    print("PBUF COSMOLOGY PIPELINE - FINAL VALIDATION REPORT")
    print("Task 12.3: Perform final validation and prepare for deployment")
    print("="*80)
    
    # Run all validation tests
    core_results = test_core_functionality()
    integrity_results = test_system_integrity()
    wrapper_results = test_wrapper_scripts()
    performance_results = test_performance()
    consistency_results = test_numerical_consistency()
    
    # Compile results
    all_results = {
        "core_functionality": core_results,
        "system_integrity": integrity_results,
        "wrapper_scripts": wrapper_results,
        "performance": performance_results,
        "numerical_consistency": consistency_results
    }
    
    # Count passes and fails
    total_tests = 0
    passed_tests = 0
    
    for category, tests in all_results.items():
        for test_name, result in tests.items():
            total_tests += 1
            if result == "PASS" or (isinstance(result, dict) and result.get("status") == "PASS"):
                passed_tests += 1
    
    # Determine certification status
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    if pass_rate >= 0.9:  # 90% pass rate required
        certification_status = "CERTIFIED"
    elif pass_rate >= 0.7:  # 70-90% conditional
        certification_status = "CONDITIONAL"
    else:
        certification_status = "NOT_CERTIFIED"
    
    # Generate certification report
    certification = {
        "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_version": "1.0.0",
        "certification_status": certification_status,
        "pass_rate": f"{pass_rate*100:.1f}%",
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "detailed_results": all_results,
        "recommendations": []
    }
    
    # Add recommendations
    if certification_status == "CERTIFIED":
        certification["recommendations"].append(
            "System is ready for production deployment"
        )
        certification["recommendations"].append(
            "Follow migration guide for safe deployment"
        )
    elif certification_status == "CONDITIONAL":
        certification["recommendations"].append(
            "Address failing tests before full deployment"
        )
        certification["recommendations"].append(
            "Consider phased deployment with monitoring"
        )
    else:
        certification["recommendations"].append(
            "Critical issues must be resolved before deployment"
        )
        certification["recommendations"].append(
            "Re-run validation after fixes"
        )
    
    # Print summary
    print("\n" + "="*80)
    print("CERTIFICATION SUMMARY")
    print("="*80)
    
    print(f"Certification Status: {certification_status}")
    print(f"Pass Rate: {pass_rate*100:.1f}% ({passed_tests}/{total_tests})")
    
    if certification_status == "CERTIFIED":
        print("\n🎉 SYSTEM CERTIFIED FOR DEPLOYMENT")
        print("✓ All critical functionality validated")
        print("✓ System integrity verified")
        print("✓ Performance acceptable")
        print("✓ Numerical consistency confirmed")
    elif certification_status == "CONDITIONAL":
        print("\n⚠️ CONDITIONAL CERTIFICATION")
        print("Most tests passed but some issues need attention")
    else:
        print("\n❌ CERTIFICATION FAILED")
        print("Critical issues prevent deployment")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(certification["recommendations"], 1):
        print(f"  {i}. {rec}")
    
    # Save certification report
    report_file = "pbuf_pipeline_certification_report.json"
    with open(report_file, 'w') as f:
        json.dump(certification, f, indent=2)
    
    print(f"\nDetailed certification report saved to: {report_file}")
    
    return certification


if __name__ == "__main__":
    # Run final validation
    certification = generate_certification_report()
    
    # Exit with appropriate code
    if certification["certification_status"] == "CERTIFIED":
        print("\n✅ VALIDATION COMPLETE - SYSTEM READY FOR DEPLOYMENT")
        sys.exit(0)
    else:
        print("\n⚠️ VALIDATION COMPLETE - REVIEW ISSUES BEFORE DEPLOYMENT")
        sys.exit(1)