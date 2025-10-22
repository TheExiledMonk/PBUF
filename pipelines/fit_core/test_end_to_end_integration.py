#!/usr/bin/env python3
"""
End-to-end integration tests for the unified PBUF cosmology pipeline.

This module tests complete system integration across all fitters and validates
seamless operation of the unified architecture with all datasets.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path

# Import all core components
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.fit_core.engine import run_fit
from pipelines.fit_core.parameter import build_params, get_defaults, DEFAULTS
from pipelines.fit_core.datasets import load_dataset, validate_dataset
from pipelines.fit_core.statistics import compute_metrics, chi2_generic
from pipelines.fit_core.integrity import run_integrity_suite
from pipelines.fit_core.logging_utils import log_run, format_results_table


class TestEndToEndIntegration:
    """Complete system integration tests."""
    
    def test_all_individual_fitters_integration(self):
        """Test that all individual fitters work seamlessly with unified architecture."""
        
        # Test parameters for both models
        test_cases = [
            ("lcdm", ["cmb"]),
            ("lcdm", ["bao"]),
            ("lcdm", ["sn"]),
            ("pbuf", ["cmb"]),
            ("pbuf", ["bao"]),
            ("pbuf", ["sn"])
        ]
        
        results = {}
        
        for model, datasets in test_cases:
            print(f"\nTesting {model} model with {datasets} dataset(s)...")
            
            # Run fit using unified engine
            result = run_fit(
                model=model,
                datasets_list=datasets,
                mode="individual",
                overrides=None
            )
            
            # Verify result structure
            assert "params" in result
            assert "results" in result
            assert "metrics" in result
            assert "diagnostics" in result
            
            # Verify model-specific parameters are present
            if model == "lcdm":
                assert "H0" in result["params"]
                assert "Om0" in result["params"]
                assert "Obh2" in result["params"]
            elif model == "pbuf":
                assert "alpha" in result["params"]
                assert "Rmax" in result["params"]
                assert "eps0" in result["params"]
            
            # Verify dataset-specific results
            for dataset in datasets:
                assert dataset in result["results"]
                assert "chi2" in result["results"][dataset]
                assert "predictions" in result["results"][dataset]
            
            # Verify metrics computation
            assert "total_chi2" in result["metrics"]
            assert "aic" in result["metrics"]
            assert "bic" in result["metrics"]
            assert "dof" in result["metrics"]
            
            # Store for cross-validation
            key = f"{model}_{'+'.join(datasets)}"
            results[key] = result
            
            print(f"✓ {key}: χ²={result['metrics']['total_chi2']:.3f}, "
                  f"AIC={result['metrics']['aic']:.3f}")
        
        return results
    
    def test_joint_fitting_integration(self):
        """Test joint fitting with multiple datasets."""
        
        # Test joint fitting scenarios
        joint_cases = [
            ("lcdm", ["cmb", "bao"]),
            ("lcdm", ["cmb", "sn"]),
            ("lcdm", ["bao", "sn"]),
            ("lcdm", ["cmb", "bao", "sn"]),
            ("pbuf", ["cmb", "bao"]),
            ("pbuf", ["cmb", "sn"]),
            ("pbuf", ["bao", "sn"]),
            ("pbuf", ["cmb", "bao", "sn"])
        ]
        
        joint_results = {}
        
        for model, datasets in joint_cases:
            print(f"\nTesting joint fit: {model} with {datasets}...")
            
            result = run_fit(
                model=model,
                datasets_list=datasets,
                mode="joint",
                overrides=None
            )
            
            # Verify joint result structure
            assert "params" in result
            assert "results" in result
            assert "metrics" in result
            
            # Verify all requested datasets are present
            for dataset in datasets:
                assert dataset in result["results"]
            
            # Verify total chi2 is sum of individual contributions
            total_chi2_computed = sum(
                result["results"][ds]["chi2"] for ds in datasets
            )
            assert abs(result["metrics"]["total_chi2"] - total_chi2_computed) < 1e-10
            
            # Store results
            key = f"{model}_joint_{'_'.join(datasets)}"
            joint_results[key] = result
            
            print(f"✓ {key}: χ²={result['metrics']['total_chi2']:.3f}")
        
        return joint_results
    
    def test_parameter_consistency_across_blocks(self):
        """Test that parameter handling is consistent across all blocks."""
        
        # Test parameter consistency
        for model in ["lcdm", "pbuf"]:
            # Get default parameters
            defaults = get_defaults(model)
            
            # Build parameters multiple times - should be identical
            params1 = build_params(model)
            params2 = build_params(model)
            params3 = build_params(model, overrides={})
            
            # Verify identical parameter dictionaries
            assert params1 == params2 == params3
            
            # Test with overrides
            if model == "lcdm":
                overrides = {"H0": 70.0, "Om0": 0.3}
            else:
                overrides = {"alpha": 1e-3, "eps0": 0.8}
            
            params_override = build_params(model, overrides=overrides)
            
            # Verify overrides were applied
            for key, value in overrides.items():
                assert params_override[key] == value
            
            # Verify non-overridden parameters remain at defaults
            for key, value in defaults.items():
                if key not in overrides:
                    assert params_override[key] == value
            
            print(f"✓ Parameter consistency verified for {model}")
    
    def test_statistical_consistency(self):
        """Test that statistical computations are consistent across all usage patterns."""
        
        # Run same fit multiple times with different interfaces
        model = "lcdm"
        datasets = ["cmb", "bao"]
        
        # Method 1: Direct engine call
        result1 = run_fit(model=model, datasets_list=datasets, mode="joint")
        
        # Method 2: Individual fits summed
        result_cmb = run_fit(model=model, datasets_list=["cmb"], mode="individual")
        result_bao = run_fit(model=model, datasets_list=["bao"], mode="individual")
        
        # Verify chi2 additivity (should be close but not exact due to optimization)
        individual_sum = (result_cmb["metrics"]["total_chi2"] + 
                         result_bao["metrics"]["total_chi2"])
        joint_chi2 = result1["metrics"]["total_chi2"]
        
        print(f"Individual sum χ²: {individual_sum:.6f}")
        print(f"Joint fit χ²: {joint_chi2:.6f}")
        print(f"Difference: {abs(joint_chi2 - individual_sum):.6f}")
        
        # Should be reasonably close (within optimization tolerance)
        assert abs(joint_chi2 - individual_sum) < 1.0  # Allow for optimization differences
        
        # Verify degrees of freedom computation
        expected_dof = sum(
            len(load_dataset(ds)["observations"]) for ds in datasets
        ) - len(result1["params"])
        
        assert result1["metrics"]["dof"] == expected_dof
        
        print("✓ Statistical consistency verified")
    
    def test_integrity_validation_integration(self):
        """Test that integrity validation works across all scenarios."""
        
        test_cases = [
            ("lcdm", ["cmb"]),
            ("lcdm", ["bao"]),
            ("lcdm", ["cmb", "bao", "sn"]),
            ("pbuf", ["cmb"]),
            ("pbuf", ["bao"]),
            ("pbuf", ["cmb", "bao", "sn"])
        ]
        
        for model, datasets in test_cases:
            # Build parameters
            params = build_params(model)
            
            # Run integrity suite
            integrity_results = run_integrity_suite(params, datasets)
            
            # Verify integrity results structure
            assert "h_ratios" in integrity_results
            assert "recombination" in integrity_results
            assert "covariance" in integrity_results
            assert "overall_status" in integrity_results
            
            # For LCDM, H ratios should be exactly 1.0
            if model == "lcdm":
                h_ratios = integrity_results["h_ratios"]["ratios"]
                for ratio in h_ratios:
                    assert abs(ratio - 1.0) < 1e-10
            
            print(f"✓ Integrity validation passed for {model} with {datasets}")
    
    def test_extensibility_mock_model(self):
        """Test system extensibility by adding a mock cosmological model."""
        
        # Add mock model to DEFAULTS
        original_defaults = DEFAULTS.copy()
        
        try:
            # Define mock "wcdm" (w-CDM) model parameters
            DEFAULTS["wcdm"] = {
                "H0": 67.4,
                "Om0": 0.315,
                "Obh2": 0.02237,
                "ns": 0.9649,
                "w0": -1.0,      # Dark energy equation of state
                "wa": 0.0,       # Dark energy evolution parameter
                "Neff": 3.046,
                "Tcmb": 2.7255,
                "recomb_method": "PLANCK18"
            }
            
            # Test parameter building for mock model
            mock_params = build_params("wcdm")
            
            # Verify mock parameters
            assert mock_params["w0"] == -1.0
            assert mock_params["wa"] == 0.0
            assert mock_params["H0"] == 67.4
            
            # Test with overrides
            mock_params_override = build_params("wcdm", overrides={"w0": -0.9, "wa": 0.1})
            assert mock_params_override["w0"] == -0.9
            assert mock_params_override["wa"] == 0.1
            
            print("✓ Mock wCDM model successfully integrated")
            
            # Test that existing functionality still works
            lcdm_params = build_params("lcdm")
            pbuf_params = build_params("pbuf")
            
            assert "H0" in lcdm_params
            assert "alpha" in pbuf_params
            
            print("✓ Existing models unaffected by extension")
            
        finally:
            # Restore original defaults
            DEFAULTS.clear()
            DEFAULTS.update(original_defaults)
    
    def test_error_handling_robustness(self):
        """Test system robustness to various error conditions."""
        
        # Test invalid model
        with pytest.raises((KeyError, ValueError)):
            build_params("invalid_model")
        
        # Test invalid dataset
        with pytest.raises((KeyError, FileNotFoundError, ValueError)):
            run_fit(model="lcdm", datasets_list=["invalid_dataset"])
        
        # Test invalid parameter overrides
        with pytest.raises((ValueError, TypeError)):
            build_params("lcdm", overrides={"H0": "invalid"})
        
        # Test empty datasets list
        with pytest.raises((ValueError, IndexError)):
            run_fit(model="lcdm", datasets_list=[])
        
        print("✓ Error handling robustness verified")
    
    def test_performance_benchmarks(self):
        """Basic performance benchmarks for system integration."""
        
        import time
        
        # Benchmark individual fits
        start_time = time.time()
        
        for _ in range(3):  # Run multiple times for averaging
            result = run_fit(model="lcdm", datasets_list=["cmb"], mode="individual")
        
        individual_time = (time.time() - start_time) / 3
        
        # Benchmark joint fit
        start_time = time.time()
        
        for _ in range(3):
            result = run_fit(model="lcdm", datasets_list=["cmb", "bao", "sn"], mode="joint")
        
        joint_time = (time.time() - start_time) / 3
        
        print(f"Average individual fit time: {individual_time:.3f}s")
        print(f"Average joint fit time: {joint_time:.3f}s")
        
        # Basic performance assertions (should complete in reasonable time)
        assert individual_time < 10.0  # Should complete in under 10 seconds
        assert joint_time < 30.0       # Joint fit should complete in under 30 seconds
        
        print("✓ Performance benchmarks passed")


def run_comprehensive_integration_tests():
    """Run all integration tests and generate summary report."""
    
    print("="*60)
    print("COMPREHENSIVE END-TO-END INTEGRATION TESTING")
    print("="*60)
    
    test_suite = TestEndToEndIntegration()
    
    try:
        # Run all integration tests
        print("\n1. Testing individual fitters integration...")
        individual_results = test_suite.test_all_individual_fitters_integration()
        
        print("\n2. Testing joint fitting integration...")
        joint_results = test_suite.test_joint_fitting_integration()
        
        print("\n3. Testing parameter consistency...")
        test_suite.test_parameter_consistency_across_blocks()
        
        print("\n4. Testing statistical consistency...")
        test_suite.test_statistical_consistency()
        
        print("\n5. Testing integrity validation...")
        test_suite.test_integrity_validation_integration()
        
        print("\n6. Testing system extensibility...")
        test_suite.test_extensibility_mock_model()
        
        print("\n7. Testing error handling robustness...")
        test_suite.test_error_handling_robustness()
        
        print("\n8. Running performance benchmarks...")
        test_suite.test_performance_benchmarks()
        
        print("\n" + "="*60)
        print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY")
        print("="*60)
        
        # Generate summary report
        summary = {
            "status": "PASSED",
            "individual_fitters_tested": len(individual_results),
            "joint_scenarios_tested": len(joint_results),
            "extensibility_verified": True,
            "error_handling_robust": True,
            "performance_acceptable": True
        }
        
        return summary
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    # Run integration tests when executed directly
    summary = run_comprehensive_integration_tests()
    print(f"\nIntegration test summary: {summary}")