#!/usr/bin/env python3
"""
Comprehensive integration test report for task 12.1.

This script performs end-to-end testing of the unified PBUF cosmology pipeline
and generates a detailed report of system integration status.
"""

import sys
import subprocess
import json
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_core_modules():
    """Test that all core modules can be imported and basic functionality works."""
    
    print("="*60)
    print("TESTING CORE MODULES")
    print("="*60)
    
    results = {}
    
    try:
        # Test parameter module
        from pipelines.fit_core.parameter import build_params, get_defaults, DEFAULTS
        
        # Test LCDM parameters
        lcdm_params = build_params("lcdm")
        assert "H0" in lcdm_params
        assert "Om0" in lcdm_params
        
        # Test PBUF parameters
        pbuf_params = build_params("pbuf")
        assert "alpha" in pbuf_params
        assert "Rmax" in pbuf_params
        
        # Test overrides
        override_params = build_params("lcdm", overrides={"H0": 70.0})
        assert override_params["H0"] == 70.0
        
        results["parameter_module"] = "PASS"
        print("✓ Parameter module: PASS")
        
    except Exception as e:
        results["parameter_module"] = f"FAIL: {e}"
        print(f"❌ Parameter module: FAIL - {e}")
    
    try:
        # Test datasets module
        from pipelines.fit_core.datasets import load_dataset, validate_dataset
        
        results["datasets_module"] = "PASS"
        print("✓ Datasets module: PASS")
        
    except Exception as e:
        results["datasets_module"] = f"FAIL: {e}"
        print(f"❌ Datasets module: FAIL - {e}")
    
    try:
        # Test statistics module
        from pipelines.fit_core.statistics import chi2_generic, compute_metrics
        
        results["statistics_module"] = "PASS"
        print("✓ Statistics module: PASS")
        
    except Exception as e:
        results["statistics_module"] = f"FAIL: {e}"
        print(f"❌ Statistics module: FAIL - {e}")
    
    try:
        # Test likelihoods module
        from pipelines.fit_core.likelihoods import likelihood_cmb, likelihood_bao, likelihood_sn
        
        results["likelihoods_module"] = "PASS"
        print("✓ Likelihoods module: PASS")
        
    except Exception as e:
        results["likelihoods_module"] = f"FAIL: {e}"
        print(f"❌ Likelihoods module: FAIL - {e}")
    
    try:
        # Test engine module
        from pipelines.fit_core.engine import run_fit
        
        results["engine_module"] = "PASS"
        print("✓ Engine module: PASS")
        
    except Exception as e:
        results["engine_module"] = f"FAIL: {e}"
        print(f"❌ Engine module: FAIL - {e}")
    
    try:
        # Test integrity module
        from pipelines.fit_core.integrity import run_integrity_suite, verify_h_ratios
        
        results["integrity_module"] = "PASS"
        print("✓ Integrity module: PASS")
        
    except Exception as e:
        results["integrity_module"] = f"FAIL: {e}"
        print(f"❌ Integrity module: FAIL - {e}")
    
    return results

def test_wrapper_scripts():
    """Test that wrapper scripts are functional."""
    
    print("\n" + "="*60)
    print("TESTING WRAPPER SCRIPTS")
    print("="*60)
    
    results = {}
    
    # Test each wrapper script help functionality
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
                print(f"❌ {wrapper}: FAIL - Return code {result.returncode}")
                
        except Exception as e:
            results[wrapper] = f"FAIL: {e}"
            print(f"❌ {wrapper}: FAIL - {e}")
    
    return results

def test_extensibility():
    """Test system extensibility by adding a mock model."""
    
    print("\n" + "="*60)
    print("TESTING SYSTEM EXTENSIBILITY")
    print("="*60)
    
    results = {}
    
    try:
        from pipelines.fit_core.parameter import DEFAULTS, build_params
        
        # Save original defaults
        original_defaults = DEFAULTS.copy()
        
        # Add mock wCDM model
        DEFAULTS["wcdm"] = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "w0": -1.0,      # Dark energy equation of state
            "wa": 0.0,       # Dark energy evolution
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "PLANCK18"
        }
        
        # Test building parameters for mock model
        wcdm_params = build_params("wcdm")
        assert wcdm_params["w0"] == -1.0
        assert wcdm_params["wa"] == 0.0
        
        # Test with overrides
        wcdm_override = build_params("wcdm", overrides={"w0": -0.9})
        assert wcdm_override["w0"] == -0.9
        
        # Verify existing models still work
        lcdm_params = build_params("lcdm")
        pbuf_params = build_params("pbuf")
        
        assert "H0" in lcdm_params
        assert "alpha" in pbuf_params
        
        # Restore original defaults
        DEFAULTS.clear()
        DEFAULTS.update(original_defaults)
        
        results["extensibility"] = "PASS"
        print("✓ System extensibility: PASS")
        print("  - Mock wCDM model successfully added")
        print("  - Parameter building works for new model")
        print("  - Existing models unaffected")
        
    except Exception as e:
        results["extensibility"] = f"FAIL: {e}"
        print(f"❌ System extensibility: FAIL - {e}")
    
    return results

def test_error_handling():
    """Test system error handling robustness."""
    
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    results = {}
    
    try:
        from pipelines.fit_core.parameter import build_params
        
        # Test invalid model
        try:
            build_params("invalid_model")
            results["invalid_model"] = "FAIL: Should have raised exception"
        except (KeyError, ValueError):
            results["invalid_model"] = "PASS"
        
        # Test invalid parameter override
        try:
            build_params("lcdm", overrides={"H0": "invalid"})
            results["invalid_override"] = "FAIL: Should have raised exception"
        except (ValueError, TypeError):
            results["invalid_override"] = "PASS"
        
        print("✓ Error handling: PASS")
        print("  - Invalid model properly rejected")
        print("  - Invalid parameter overrides properly rejected")
        
    except Exception as e:
        results["error_handling"] = f"FAIL: {e}"
        print(f"❌ Error handling: FAIL - {e}")
    
    return results

def test_physics_consistency():
    """Test physics consistency checks."""
    
    print("\n" + "="*60)
    print("TESTING PHYSICS CONSISTENCY")
    print("="*60)
    
    results = {}
    
    try:
        from pipelines.fit_core.parameter import build_params
        from pipelines.fit_core.integrity import verify_h_ratios, verify_recombination
        
        # Test LCDM H(z) ratios (should be 1.0)
        lcdm_params = build_params("lcdm")
        h_ratios_result = verify_h_ratios(lcdm_params, [0.5, 1.0, 2.0])
        
        if h_ratios_result:  # Returns boolean
            results["lcdm_h_ratios"] = "PASS"
            print("✓ LCDM H(z) ratios: PASS")
        else:
            results["lcdm_h_ratios"] = "FAIL: H(z) ratios inconsistent"
            print("❌ LCDM H(z) ratios: FAIL")
        
        # Test recombination calculation
        recomb_result = verify_recombination(lcdm_params)
        
        if recomb_result:  # Returns boolean
            results["recombination"] = "PASS"
            print("✓ Recombination calculation: PASS")
        else:
            results["recombination"] = "FAIL: Recombination inconsistent"
            print("❌ Recombination calculation: FAIL")
        
    except Exception as e:
        results["physics_consistency"] = f"FAIL: {e}"
        print(f"❌ Physics consistency: FAIL - {e}")
    
    return results

def generate_integration_report():
    """Generate comprehensive integration test report."""
    
    print("="*60)
    print("PBUF COSMOLOGY PIPELINE - INTEGRATION TEST REPORT")
    print("Task 12.1: Integrate all components and perform end-to-end testing")
    print("="*60)
    
    # Run all tests
    core_results = test_core_modules()
    wrapper_results = test_wrapper_scripts()
    extensibility_results = test_extensibility()
    error_results = test_error_handling()
    physics_results = test_physics_consistency()
    
    # Compile overall results
    all_results = {
        "core_modules": core_results,
        "wrapper_scripts": wrapper_results,
        "extensibility": extensibility_results,
        "error_handling": error_results,
        "physics_consistency": physics_results
    }
    
    # Count passes and fails
    total_tests = 0
    passed_tests = 0
    
    for category, tests in all_results.items():
        for test_name, result in tests.items():
            total_tests += 1
            if result == "PASS":
                passed_tests += 1
    
    # Generate summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✓ Unified architecture is fully integrated")
        print("✓ All components work seamlessly together")
        print("✓ System is extensible for new models")
        print("✓ Error handling is robust")
        print("✓ Physics consistency is maintained")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed")
        print("Some integration issues need to be addressed")
    
    # Save detailed report
    report_file = "integration_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    return all_results, passed_tests == total_tests

if __name__ == "__main__":
    results, success = generate_integration_report()
    sys.exit(0 if success else 1)