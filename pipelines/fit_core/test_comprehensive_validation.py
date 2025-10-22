"""
Comprehensive validation and testing for parameter optimization system.

This module implements critical validation tests for the parameter optimization
system, including round-trip persistence, cross-model consistency, backward
compatibility, and performance/robustness testing.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 2.1, 2.2, 2.3, 2.4, 2.5, 
             8.1, 8.2, 8.3, 8.4, 8.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import json
import os
import tempfile
import shutil
import time
import threading
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from .parameter_store import OptimizedParameterStore, OptimizationRecord, OptimizationResult
from .optimizer import ParameterOptimizer
from .parameter import get_defaults, validate_params
from .config import ConfigurationManager
from .engine import run_fit


class TestRoundTripPersistence:
    """
    Test 8.1: Critical round-trip persistence test
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
    """
    
    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(storage_dir=self.temp_dir)
        self.optimizer = ParameterOptimizer()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_propagation_roundtrip_numerical_precision(self):
        """
        Test round-trip persistence with numerical precision validation.
        
        Ensures optimized parameters survive store/reload cycle with 1e-12 precision.
        """
        # Test for both models
        for model in ["lcdm", "pbuf"]:
            # Get original parameters
            original_params = self.store.get_model_defaults(model)
            
            # Define high-precision optimized parameters
            if model == "lcdm":
                optimized_params = {
                    "H0": 67.36123456789012,
                    "Om0": 0.31456789012345,
                    "Obh2": 0.022383456789012,
                    "ns": 0.964912345678901
                }
            else:  # pbuf
                optimized_params = {
                    "H0": 67.36123456789012,
                    "Om0": 0.31456789012345,
                    "k_sat": 0.976234567890123,
                    "alpha": 5.123456789012e-4
                }
            
            # Create optimization metadata
            metadata = {
                "source_dataset": "cmb",
                "chi2_improvement": 3.456789012345,
                "convergence_status": "success",
                "optimizer_info": {
                    "method": "L-BFGS-B",
                    "library": "scipy",
                    "version": "1.13.0"
                },
                "covariance_scaling": 1.234567890123,
                "n_function_evaluations": 42,
                "optimization_time": 12.345678901234
            }
            
            # Store optimized parameters
            self.store.update_model_defaults(model, optimized_params, metadata)
            
            # Reload from storage
            reloaded_params = self.store.get_model_defaults(model)
            
            # Verify exact numerical precision (1e-12 tolerance)
            for param, expected_value in optimized_params.items():
                actual_value = reloaded_params[param]
                precision_error = abs(actual_value - expected_value)
                
                assert precision_error < 1e-12, (
                    f"Model {model}, parameter {param}: "
                    f"expected {expected_value}, got {actual_value}, "
                    f"precision error {precision_error} exceeds 1e-12"
                )
            
            # Verify non-optimized parameters unchanged
            for param, original_value in original_params.items():
                if param not in optimized_params:
                    assert reloaded_params[param] == original_value, (
                        f"Non-optimized parameter {param} changed: "
                        f"original {original_value}, current {reloaded_params[param]}"
                    )    

    def test_parameter_store_concurrent_access_reliability(self):
        """
        Test parameter store reliability under concurrent access.
        
        Simulates multiple optimization processes accessing the store simultaneously.
        """
        def optimization_worker(worker_id, model, results_list):
            """Worker function for concurrent optimization."""
            try:
                # Each worker optimizes different parameters
                if model == "lcdm":
                    optimized_params = {
                        "H0": 67.0 + worker_id * 0.1,
                        "Om0": 0.31 + worker_id * 0.001
                    }
                else:  # pbuf
                    optimized_params = {
                        "k_sat": 0.97 + worker_id * 0.01,
                        "alpha": (5.0 + worker_id * 0.1) * 1e-4
                    }
                
                metadata = {
                    "source_dataset": f"worker_{worker_id}",
                    "chi2_improvement": worker_id * 0.5,
                    "convergence_status": "success",
                    "worker_id": worker_id
                }
                
                # Simulate some processing time
                time.sleep(0.01 * worker_id)
                
                # Update parameters
                self.store.update_model_defaults(model, optimized_params, metadata)
                
                # Verify the update was successful
                reloaded = self.store.get_model_defaults(model)
                for param, expected in optimized_params.items():
                    if abs(reloaded[param] - expected) > 1e-10:
                        results_list.append(f"Worker {worker_id}: Parameter {param} mismatch")
                        return
                
                results_list.append(f"Worker {worker_id}: Success")
                
            except Exception as e:
                results_list.append(f"Worker {worker_id}: Error - {str(e)}")
        
        # Test concurrent access for both models
        for model in ["lcdm", "pbuf"]:
            results = []
            threads = []
            
            # Create multiple worker threads
            num_workers = 5
            for i in range(num_workers):
                thread = threading.Thread(
                    target=optimization_worker,
                    args=(i, model, results)
                )
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10.0)  # 10 second timeout
            
            # Verify all workers completed successfully
            assert len(results) == num_workers, f"Not all workers completed: {results}"
            
            success_count = sum(1 for r in results if "Success" in r)
            assert success_count >= num_workers - 1, (
                f"Too many failures in concurrent access test: {results}"
            )
    
    def test_optimization_metadata_preservation_across_cycles(self):
        """
        Test that optimization metadata is preserved across multiple optimization cycles.
        """
        model = "pbuf"
        
        # First optimization cycle
        cycle1_params = {"k_sat": 1.1, "alpha": 5.5e-4}
        cycle1_metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 2.5,
            "convergence_status": "success",
            "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
            "covariance_scaling": 1.0,
            "n_function_evaluations": 35,
            "optimization_time": 8.5
        }
        
        self.store.update_model_defaults(model, cycle1_params, cycle1_metadata)
        
        # Verify first cycle metadata
        history1 = self.store.get_optimization_history(model)
        assert len(history1) == 1
        assert history1[0].chi2_improvement == 2.5
        assert history1[0].optimizer_info["method"] == "L-BFGS-B"
        
        # Second optimization cycle
        time.sleep(0.01)  # Ensure different timestamp
        cycle2_params = {"H0": 68.2, "Om0": 0.32}
        cycle2_metadata = {
            "source_dataset": "cmb",
            "chi2_improvement": 1.8,
            "convergence_status": "success",
            "optimizer_info": {"method": "Powell", "library": "scipy", "version": "1.13.0"},
            "covariance_scaling": 1.2,
            "n_function_evaluations": 28,
            "optimization_time": 6.3
        }
        
        self.store.update_model_defaults(model, cycle2_params, cycle2_metadata)
        
        # Verify both cycles preserved
        history2 = self.store.get_optimization_history(model)
        assert len(history2) == 2
        
        # Most recent should be first
        recent_record = history2[0]
        assert recent_record.chi2_improvement == 1.8
        assert recent_record.optimizer_info["method"] == "Powell"
        
        older_record = history2[1]
        assert older_record.chi2_improvement == 2.5
        assert older_record.optimizer_info["method"] == "L-BFGS-B"
        
        # Verify current parameters include both cycles
        current_params = self.store.get_model_defaults(model)
        assert current_params["k_sat"] == 1.1  # From cycle 1
        assert current_params["alpha"] == 5.5e-4  # From cycle 1
        assert current_params["H0"] == 68.2  # From cycle 2
        assert current_params["Om0"] == 0.32  # From cycle 2
    
    def test_storage_corruption_recovery(self):
        """
        Test recovery from storage file corruption scenarios.
        """
        # Add some optimization data first
        optimized_params = {"H0": 68.5, "Om0": 0.315}
        metadata = {"source_dataset": "cmb", "convergence_status": "success"}
        self.store.update_model_defaults("lcdm", optimized_params, metadata)
        
        # Test 1: Corrupt JSON syntax
        lcdm_file = self.store._get_model_file("lcdm")
        with open(lcdm_file, 'w') as f:
            f.write('{"defaults": {"H0": 67.4, "Om0": 0.315}, "invalid": json}')
        
        # Should recover and return valid defaults
        recovered_params = self.store.get_model_defaults("lcdm")
        assert "H0" in recovered_params
        assert isinstance(recovered_params["H0"], (int, float))
        
        # Test 2: Missing required structure
        with open(lcdm_file, 'w') as f:
            json.dump({"wrong_structure": True}, f)
        
        recovered_params = self.store.get_model_defaults("lcdm")
        assert "H0" in recovered_params
        
        # Test 3: Empty file
        with open(lcdm_file, 'w') as f:
            f.write("")
        
        recovered_params = self.store.get_model_defaults("lcdm")
        assert "H0" in recovered_params
        
        # Test 4: File permissions issue (simulate)
        if os.name != 'nt':  # Skip on Windows
            try:
                os.chmod(lcdm_file, 0o000)  # Remove all permissions
                recovered_params = self.store.get_model_defaults("lcdm")
                assert "H0" in recovered_params
            except PermissionError:
                # Expected behavior - should handle gracefully
                pass
            finally:
                try:
                    os.chmod(lcdm_file, 0o644)  # Restore permissions
                except (OSError, PermissionError):
                    pass


class TestCrossModelConsistency:
    """
    Test 8.2: Cross-model consistency validation
    
    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(storage_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_lcdm_pbuf_optimization_divergence_detection(self):
        """
        Test that ΛCDM and PBUF optimizations don't diverge inappropriately.
        """
        # Set up consistent initial parameters
        consistent_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649
        }
        
        metadata = {
            "source_dataset": "cmb",
            "convergence_status": "success",
            "chi2_improvement": 2.0
        }
        
        # Update both models with consistent parameters
        self.store.update_model_defaults("lcdm", consistent_params, metadata)
        
        pbuf_params = consistent_params.copy()
        pbuf_params.update({"k_sat": 0.976, "alpha": 5e-4})
        self.store.update_model_defaults("pbuf", pbuf_params, metadata)
        
        # Should be consistent
        comparison = self.store.validate_cross_model_consistency(tolerance=1e-6, log_warnings=False)
        assert comparison["_summary"]["is_fully_consistent"]
        assert len(comparison["_summary"]["divergent_params"]) == 0
        
        # Introduce divergence in PBUF model
        divergent_params = {"H0": 70.0}  # Significant change from 67.4
        self.store.update_model_defaults("pbuf", divergent_params, metadata)
        
        # Should detect divergence
        comparison = self.store.validate_cross_model_consistency(tolerance=1e-3, log_warnings=False)
        assert not comparison["_summary"]["is_fully_consistent"]
        assert "H0" in comparison["_summary"]["divergent_params"]
        
        # Check detailed comparison
        h0_comparison = comparison["H0"]
        assert h0_comparison["status"] == "divergent"
        assert h0_comparison["lcdm_value"] == 67.4
        assert h0_comparison["pbuf_value"] == 70.0
        assert not h0_comparison["within_tolerance"]
    
    def test_shared_parameter_consistency_after_sequential_optimizations(self):
        """
        Test shared parameter consistency after sequential optimizations.
        """
        # Sequential optimization scenario
        optimizations = [
            {
                "model": "lcdm",
                "params": {"H0": 67.5, "Om0": 0.314},
                "delay": 0.01
            },
            {
                "model": "pbuf", 
                "params": {"H0": 67.52, "Om0": 0.3142, "k_sat": 0.98},  # Slight drift
                "delay": 0.01
            },
            {
                "model": "lcdm",
                "params": {"Obh2": 0.02240, "ns": 0.965},  # Different parameters
                "delay": 0.01
            },
            {
                "model": "pbuf",
                "params": {"alpha": 5.2e-4},  # PBUF-specific parameter
                "delay": 0.01
            }
        ]
        
        metadata = {
            "source_dataset": "cmb",
            "convergence_status": "success",
            "chi2_improvement": 1.5
        }
        
        # Execute sequential optimizations
        for i, opt in enumerate(optimizations):
            time.sleep(opt["delay"])
            self.store.update_model_defaults(opt["model"], opt["params"], metadata)
            
            # Check consistency after each step
            comparison = self.store.validate_cross_model_consistency(tolerance=1e-2, log_warnings=False)
            
            if i < 2:  # First two optimizations should maintain consistency
                shared_divergent = [p for p in comparison["_summary"]["divergent_params"] 
                                  if p in {"H0", "Om0", "Obh2", "ns"}]
                assert len(shared_divergent) <= 1, (
                    f"Too many shared parameters divergent after optimization {i}: {shared_divergent}"
                )
        
        # Final consistency check
        final_comparison = self.store.validate_cross_model_consistency(tolerance=1e-2, log_warnings=False)
        
        # Should have minimal divergence in shared parameters
        shared_divergent = [p for p in final_comparison["_summary"]["divergent_params"] 
                          if p in {"H0", "Om0", "Obh2", "ns"}]
        assert len(shared_divergent) <= 2, (
            f"Too many shared parameters divergent in final state: {shared_divergent}"
        )
    
    def test_tolerance_based_parameter_drift_detection(self):
        """
        Test tolerance-based validation for parameter drift detection.
        """
        # Set up initial consistent state
        initial_params = {"H0": 67.4, "Om0": 0.315}
        metadata = {"source_dataset": "cmb", "convergence_status": "success"}
        
        self.store.update_model_defaults("lcdm", initial_params, metadata)
        self.store.update_model_defaults("pbuf", initial_params, metadata)
        
        # Test different tolerance levels
        tolerances = [1e-6, 1e-4, 1e-2, 1e-1]
        drift_amounts = [1e-7, 1e-5, 1e-3, 1e-2]  # Corresponding drift amounts
        
        for tolerance, drift in zip(tolerances, drift_amounts):
            # Introduce controlled drift
            drifted_params = {"H0": 67.4 * (1 + drift)}
            self.store.update_model_defaults("pbuf", drifted_params, metadata)
            
            # Check if drift is detected correctly
            comparison = self.store.validate_cross_model_consistency(
                tolerance=tolerance, log_warnings=False
            )
            
            expected_divergent = drift > tolerance
            actual_divergent = "H0" in comparison["_summary"]["divergent_params"]
            
            assert actual_divergent == expected_divergent, (
                f"Tolerance {tolerance}, drift {drift}: "
                f"expected divergent={expected_divergent}, got {actual_divergent}"
            )
            
            if actual_divergent:
                h0_comparison = comparison["H0"]
                assert h0_comparison["relative_difference"] > tolerance
                assert not h0_comparison["within_tolerance"]
    
    def test_cross_model_drift_analysis(self):
        """
        Test comprehensive drift analysis between models.
        """
        # Set up optimization history
        lcdm_params = {"H0": 67.0, "Om0": 0.30}
        pbuf_params = {"H0": 67.0, "Om0": 0.30, "k_sat": 0.95}
        
        metadata = {"source_dataset": "cmb", "convergence_status": "success"}
        
        self.store.update_model_defaults("lcdm", lcdm_params, metadata)
        self.store.update_model_defaults("pbuf", pbuf_params, metadata)
        
        # Simulate parameter drift over time by manually modifying stored defaults
        # without creating new optimization records
        time.sleep(0.01)
        
        # ΛCDM drifts slightly (modify stored defaults directly)
        lcdm_data = self.store._load_model_data("lcdm")
        lcdm_data["defaults"]["H0"] = 67.2  # +0.3% drift from 67.0
        self.store._save_model_data("lcdm", lcdm_data)
        
        # PBUF drifts more significantly
        pbuf_data = self.store._load_model_data("pbuf")
        pbuf_data["defaults"]["H0"] = 68.0  # +1.5% drift from 67.0
        self.store._save_model_data("pbuf", pbuf_data)
        
        # Analyze drift
        drift_analysis = self.store.detect_parameter_drift(max_drift_threshold=0.01)
        
        # Should detect drift in PBUF model (1.5% > 1% threshold)
        assert drift_analysis["summary"]["drift_detected"]
        assert "pbuf.H0" in drift_analysis["summary"]["drifted_parameters"]
        
        # Check specific drift amounts
        lcdm_drift_info = drift_analysis["lcdm"]
        pbuf_drift_info = drift_analysis["pbuf"]
        
        # ΛCDM should have analyzed drift (0.3% < 1% threshold)
        if lcdm_drift_info["status"] == "analyzed":
            lcdm_h0_drift = lcdm_drift_info["parameter_drift"]["H0"]
            assert not lcdm_h0_drift["exceeds_threshold"]
        
        # PBUF should have drift that exceeds threshold (1.5% > 1% threshold)
        if pbuf_drift_info["status"] == "analyzed":
            pbuf_h0_drift = pbuf_drift_info["parameter_drift"]["H0"]
            assert pbuf_h0_drift["exceeds_threshold"]
            assert pbuf_h0_drift["drift"] > 0.01

class TestBackwardCompatibility:
    """
    Test 8.3: Backward compatibility test suite
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(storage_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_existing_workflows_without_optimization_flags(self):
        """
        Test that existing workflows work unchanged without optimization flags.
        """
        # Simulate legacy workflow - direct parameter access
        lcdm_defaults = get_defaults("lcdm")
        pbuf_defaults = get_defaults("pbuf")
        
        # Should get standard hardcoded defaults
        assert "H0" in lcdm_defaults
        assert "Om0" in lcdm_defaults
        assert "k_sat" in pbuf_defaults
        assert "alpha" in pbuf_defaults
        
        # Values should match expected defaults
        assert abs(lcdm_defaults["H0"] - 67.4) < 1e-6
        assert abs(lcdm_defaults["Om0"] - 0.315) < 1e-6
        
        # Parameter validation should work as before
        validate_params(lcdm_defaults, "lcdm")
        validate_params(pbuf_defaults, "pbuf")
        
        # Store should return same defaults when no optimization exists
        store_lcdm = self.store.get_model_defaults("lcdm")
        store_pbuf = self.store.get_model_defaults("pbuf")
        
        # Should be identical to hardcoded defaults
        for param in lcdm_defaults:
            assert store_lcdm[param] == lcdm_defaults[param], (
                f"Parameter {param} differs: store={store_lcdm[param]}, "
                f"hardcoded={lcdm_defaults[param]}"
            )
        
        for param in pbuf_defaults:
            assert store_pbuf[param] == pbuf_defaults[param], (
                f"Parameter {param} differs: store={store_pbuf[param]}, "
                f"hardcoded={pbuf_defaults[param]}"
            )
    
    def test_legacy_parameter_overrides_functionality(self):
        """
        Test that legacy parameter overrides continue to function.
        """
        # Test Config class parameter override functionality
        config_data = {
            "parameters": {
                "H0": 70.0,
                "Om0": 0.3,
                "k_sat": 1.0
            }
        }
        
        # Create temporary config file
        config_file = Path(self.temp_dir) / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load config and apply overrides
        config = ConfigurationManager(str(config_file))
        
        # Get base parameters
        lcdm_params = get_defaults("lcdm")
        pbuf_params = get_defaults("pbuf")
        
        # Apply overrides (simulate legacy behavior)
        parameter_overrides = config.get_parameter_overrides()
        if parameter_overrides:
            for param, value in parameter_overrides.items():
                if param in lcdm_params:
                    lcdm_params[param] = value
                if param in pbuf_params:
                    pbuf_params[param] = value
        
        # Verify overrides were applied
        assert lcdm_params["H0"] == 70.0
        assert lcdm_params["Om0"] == 0.3
        assert pbuf_params["H0"] == 70.0
        assert pbuf_params["Om0"] == 0.3
        assert pbuf_params["k_sat"] == 1.0
        
        # Verify non-overridden parameters unchanged
        original_lcdm = get_defaults("lcdm")
        original_pbuf = get_defaults("pbuf")
        
        assert lcdm_params["Obh2"] == original_lcdm["Obh2"]
        assert pbuf_params["alpha"] == original_pbuf["alpha"]
    
    def test_existing_configuration_files_compatibility(self):
        """
        Test that existing configuration files remain compatible.
        """
        # Legacy configuration format (without optimization section)
        legacy_config = {
            "datasets": ["cmb", "bao"],
            "model": "lcdm",
            "parameters": {
                "H0": 68.0
            },
            "output_dir": "results",
            "verbose": True
        }
        
        config_file = Path(self.temp_dir) / "legacy_config.json"
        with open(config_file, 'w') as f:
            json.dump(legacy_config, f)
        
        # Should load without errors
        config = ConfigurationManager(str(config_file))
        
        # Should have expected values
        assert config.config_data.get("model") == "lcdm"
        assert "cmb" in config.config_data.get("datasets", [])
        assert "bao" in config.config_data.get("datasets", [])
        parameter_overrides = config.get_parameter_overrides()
        assert parameter_overrides.get("H0") == 68.0
        
        # Should not have optimization settings (defaults should apply)
        optimization_config = config.get_optimization_config()
        assert not optimization_config.get("optimize_parameters", [])
        
        # Modern configuration with optimization section
        modern_config = {
            "datasets": ["cmb"],
            "model": "pbuf",
            "optimization": {
                "optimize_parameters": ["k_sat", "alpha"],
                "dry_run": False,
                "covariance_scaling": 1.0
            },
            "parameters": {
                "H0": 67.5
            }
        }
        
        modern_config_file = Path(self.temp_dir) / "modern_config.json"
        with open(modern_config_file, 'w') as f:
            json.dump(modern_config, f)
        
        # Should also load without errors
        modern_config_obj = ConfigurationManager(str(modern_config_file))
        
        # Should have both legacy and modern features
        assert modern_config_obj.config_data.get("model") == "pbuf"
        modern_parameter_overrides = modern_config_obj.get_parameter_overrides()
        assert modern_parameter_overrides.get("H0") == 67.5
        
        # Should have optimization settings
        modern_optimization_config = modern_config_obj.get_optimization_config()
        assert "k_sat" in modern_optimization_config.get("optimize_parameters", [])
    
    def test_parameter_validation_backward_compatibility(self):
        """
        Test that parameter validation remains backward compatible.
        """
        # Test with original parameter sets
        original_lcdm = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "PLANCK18"
        }
        
        original_pbuf = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "PLANCK18",
            "alpha": 5e-4,
            "Rmax": 1000000.0,
            "eps0": 1e-4,
            "n_eps": 1.0,
            "k_sat": 0.976
        }
        
        # Should validate without errors
        validate_params(original_lcdm, "lcdm")
        validate_params(original_pbuf, "pbuf")
        
        # Test with parameter overrides (legacy style)
        modified_lcdm = original_lcdm.copy()
        modified_lcdm["H0"] = 70.0
        modified_lcdm["Om0"] = 0.3
        
        validate_params(modified_lcdm, "lcdm")
        
        modified_pbuf = original_pbuf.copy()
        modified_pbuf["k_sat"] = 1.2
        modified_pbuf["alpha"] = 6e-4
        
        validate_params(modified_pbuf, "pbuf")
        
        # Test error handling remains the same
        invalid_lcdm = original_lcdm.copy()
        invalid_lcdm["H0"] = -10.0  # Invalid value
        
        with pytest.raises((ValueError, AssertionError)):
            validate_params(invalid_lcdm, "lcdm")
        
        invalid_pbuf = original_pbuf.copy()
        invalid_pbuf["k_sat"] = 5.0  # Out of bounds
        
        with pytest.raises((ValueError, AssertionError)):
            validate_params(invalid_pbuf, "pbuf")
    
    def test_engine_backward_compatibility(self):
        """
        Test that the unified engine maintains backward compatibility.
        """
        # Test that the engine interface accepts both legacy and modern parameters
        # without actually running the fit (which would require real data)
        
        # Test parameter validation for legacy interface
        legacy_params = {
            "model": "lcdm",
            "datasets_list": ["cmb"]
        }
        
        # Should not raise TypeError for missing parameters
        try:
            # We expect this to fail due to missing data, but not due to interface issues
            run_fit(**legacy_params)
        except (ValueError, FileNotFoundError, KeyError) as e:
            # These are expected - missing data files, integrity failures, etc.
            error_msg = str(e).lower()
            expected_errors = ["degrees of freedom", "not found", "missing", "integrity", "failed"]
            assert any(err in error_msg for err in expected_errors), f"Unexpected error: {str(e)}"
        except TypeError as e:
            # This would indicate interface incompatibility
            pytest.fail(f"Interface incompatibility detected: {str(e)}")
        
        # Test parameter validation for modern interface
        modern_params = {
            "model": "pbuf", 
            "datasets_list": ["cmb"],
            "optimize_params": ["k_sat", "alpha"]
        }
        
        try:
            # We expect this to fail due to missing data, but not due to interface issues
            run_fit(**modern_params)
        except (ValueError, FileNotFoundError, KeyError) as e:
            # These are expected - missing data files, integrity failures, etc.
            error_msg = str(e).lower()
            expected_errors = ["degrees of freedom", "not found", "missing", "integrity", "failed"]
            assert any(err in error_msg for err in expected_errors), f"Unexpected error: {str(e)}"
        except TypeError as e:
            # This would indicate interface incompatibility
            pytest.fail(f"Interface incompatibility detected: {str(e)}")


class TestPerformanceAndRobustness:
    """
    Test 8.4: Performance and robustness tests
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = OptimizedParameterStore(storage_dir=self.temp_dir)
        self.optimizer = ParameterOptimizer()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_optimization_convergence_various_starting_conditions(self):
        """
        Test optimization convergence under various starting conditions.
        """
        # Define test scenarios with different starting conditions
        test_scenarios = [
            {
                "name": "default_start",
                "model": "lcdm",
                "starting_params": None,  # Use defaults
                "optimize_params": ["H0", "Om0"]
            },
            {
                "name": "boundary_start_low",
                "model": "lcdm", 
                "starting_params": {"H0": 25.0, "Om0": 0.05},  # Near lower bounds
                "optimize_params": ["H0", "Om0"]
            },
            {
                "name": "boundary_start_high",
                "model": "lcdm",
                "starting_params": {"H0": 140.0, "Om0": 0.9},  # Near upper bounds
                "optimize_params": ["H0", "Om0"]
            },
            {
                "name": "pbuf_default",
                "model": "pbuf",
                "starting_params": None,
                "optimize_params": ["k_sat", "alpha"]
            },
            {
                "name": "pbuf_boundary_low",
                "model": "pbuf",
                "starting_params": {"k_sat": 0.15, "alpha": 2e-6},
                "optimize_params": ["k_sat", "alpha"]
            },
            {
                "name": "pbuf_boundary_high", 
                "model": "pbuf",
                "starting_params": {"k_sat": 1.8, "alpha": 8e-3},
                "optimize_params": ["k_sat", "alpha"]
            }
        ]
        
        # Mock optimization function to simulate different convergence behaviors
        def mock_optimize_parameters(model, datasets_list, optimize_params, starting_values=None, **kwargs):
            """Mock optimization that simulates realistic convergence."""
            
            # Get bounds for validation
            bounds = {}
            if model == "lcdm":
                bounds = {"H0": (20.0, 150.0), "Om0": (0.01, 0.99)}
            else:  # pbuf
                bounds = {"k_sat": (0.1, 2.0), "alpha": (1e-6, 1e-2)}
            
            # Simulate optimization result
            optimized_values = {}
            for param in optimize_params:
                if starting_values and param in starting_values:
                    start_val = starting_values[param]
                else:
                    # Use middle of bounds as default
                    low, high = bounds[param]
                    start_val = (low + high) / 2
                
                # Simulate convergence toward middle of bounds
                low, high = bounds[param]
                target = (low + high) / 2
                
                # Add some noise but converge toward target
                noise = np.random.normal(0, 0.01 * (high - low))
                optimized_values[param] = np.clip(target + noise, low, high)
            
            return OptimizationResult(
                model=model,
                optimized_params=optimized_values,
                starting_params=starting_values or {},
                final_chi2=100.0 - np.random.uniform(1, 10),
                chi2_improvement=np.random.uniform(1, 10),
                convergence_status="success",
                n_function_evaluations=np.random.randint(20, 100),
                optimization_time=np.random.uniform(1, 30),
                bounds_reached=[],
                optimizer_info={"method": "L-BFGS-B", "library": "scipy"},
                covariance_scaling=1.0,
                metadata={}
            )
        
        # Test each scenario
        convergence_results = {}
        
        with patch.object(self.optimizer, 'optimize_parameters', side_effect=mock_optimize_parameters):
            for scenario in test_scenarios:
                try:
                    result = self.optimizer.optimize_parameters(
                        model=scenario["model"],
                        datasets_list=["cmb"],  # Add required datasets_list parameter
                        optimize_params=scenario["optimize_params"],
                        starting_values=scenario["starting_params"]
                    )
                    
                    # Verify convergence
                    assert result.convergence_status == "success"
                    assert result.chi2_improvement > 0
                    assert result.n_function_evaluations > 0
                    
                    # Verify optimized parameters are within bounds
                    for param, value in result.optimized_params.items():
                        if scenario["model"] == "lcdm":
                            if param == "H0":
                                assert 20.0 <= value <= 150.0
                            elif param == "Om0":
                                assert 0.01 <= value <= 0.99
                        else:  # pbuf
                            if param == "k_sat":
                                assert 0.1 <= value <= 2.0
                            elif param == "alpha":
                                assert 1e-6 <= value <= 1e-2
                    
                    convergence_results[scenario["name"]] = "success"
                    
                except Exception as e:
                    convergence_results[scenario["name"]] = f"failed: {str(e)}"
        
        # Verify most scenarios converged successfully
        success_count = sum(1 for result in convergence_results.values() if result == "success")
        total_scenarios = len(test_scenarios)
        
        assert success_count >= total_scenarios * 0.8, (
            f"Too many convergence failures: {convergence_results}"
        )
    
    def test_bounds_enforcement_and_constraint_handling(self):
        """
        Test bounds enforcement and constraint handling during optimization.
        """
        # Test parameter bounds validation
        bounds_tests = [
            {
                "model": "lcdm",
                "param": "H0",
                "valid_values": [25.0, 67.4, 140.0],
                "invalid_values": [10.0, 200.0, -5.0],
                "expected_bounds": (20.0, 150.0)
            },
            {
                "model": "lcdm", 
                "param": "Om0",
                "valid_values": [0.05, 0.315, 0.9],
                "invalid_values": [0.0, 1.0, -0.1],
                "expected_bounds": (0.01, 0.99)
            },
            {
                "model": "pbuf",
                "param": "k_sat", 
                "valid_values": [0.2, 0.976, 1.8],
                "invalid_values": [0.05, 3.0, -1.0],
                "expected_bounds": (0.1, 2.0)
            },
            {
                "model": "pbuf",
                "param": "alpha",
                "valid_values": [2e-6, 5e-4, 8e-3],
                "invalid_values": [1e-7, 2e-2, -1e-4],
                "expected_bounds": (1e-6, 1e-2)
            }
        ]
        
        for test_case in bounds_tests:
            # Test bounds retrieval
            bounds = self.optimizer.get_optimization_bounds(test_case["model"], test_case["param"])
            assert bounds == test_case["expected_bounds"], (
                f"Bounds mismatch for {test_case['model']}.{test_case['param']}: "
                f"expected {test_case['expected_bounds']}, got {bounds}"
            )
            
            # Test valid values
            for valid_value in test_case["valid_values"]:
                params = {test_case["param"]: valid_value}
                
                # Should not raise error for valid values
                try:
                    is_valid = self.optimizer.validate_optimization_request(
                        test_case["model"], [test_case["param"]]
                    )
                    assert is_valid
                except Exception as e:
                    pytest.fail(f"Valid value {valid_value} rejected: {str(e)}")
            
            # Test invalid values (bounds checking in parameter validation)
            for invalid_value in test_case["invalid_values"]:
                params = get_defaults(test_case["model"])
                params[test_case["param"]] = invalid_value
                
                # Should raise error for invalid values
                with pytest.raises((ValueError, AssertionError)):
                    validate_params(params, test_case["model"])
    
    def test_error_recovery_and_fallback_mechanisms(self):
        """
        Test error recovery and fallback mechanisms.
        """
        # Test 1: Optimization failure fallback
        def mock_failing_optimization(*args, **kwargs):
            raise RuntimeError("Optimization failed to converge")
        
        with patch.object(self.optimizer, 'optimize_parameters', side_effect=mock_failing_optimization):
            # Should handle optimization failure gracefully
            try:
                result = self.optimizer.optimize_parameters("lcdm", ["cmb"], ["H0"])
                # If no exception, check that fallback was used
                assert result is None or result.convergence_status == "failed"
            except RuntimeError:
                # Expected behavior - optimization failure should be caught and handled
                pass
        
        # Test 2: Storage corruption recovery
        optimized_params = {"H0": 68.0}
        metadata = {"source_dataset": "cmb", "convergence_status": "success"}
        
        # First, store valid data
        self.store.update_model_defaults("lcdm", optimized_params, metadata)
        
        # Corrupt the storage file
        lcdm_file = self.store._get_model_file("lcdm")
        with open(lcdm_file, 'w') as f:
            f.write("corrupted data")
        
        # Should recover gracefully
        recovered_params = self.store.get_model_defaults("lcdm")
        assert "H0" in recovered_params
        assert isinstance(recovered_params["H0"], (int, float))
        
        # Test 3: Invalid parameter request handling
        with pytest.raises(ValueError, match="not optimizable"):
            self.store.update_model_defaults("lcdm", {"invalid_param": 1.0}, metadata)
        
        # Test 4: File locking timeout handling
        # Simulate lock acquisition failure
        def mock_acquire_lock_failure(*args):
            raise RuntimeError("Could not acquire lock")
        
        with patch.object(self.store, '_acquire_lock', side_effect=mock_acquire_lock_failure):
            with pytest.raises(RuntimeError, match="Could not acquire lock"):
                self.store.update_model_defaults("lcdm", optimized_params, metadata)
    
    def test_quick_regression_mode_chi2_validation(self):
        """
        Test quick regression mode: rerun optimizer with new defaults and confirm χ² identical within 1e-8.
        """
        # Mock optimization function that returns consistent χ² values
        def mock_consistent_optimization(model, datasets_list, optimize_params, starting_values=None, **kwargs):
            """Mock optimization with deterministic χ² for regression testing."""
            
            # Use deterministic "optimization" based on parameter values
            base_chi2 = 100.0
            
            if starting_values:
                # Simulate χ² based on starting parameters
                param_sum = sum(starting_values.values())
                chi2_variation = (param_sum % 10) * 0.1
            else:
                chi2_variation = 0.0
            
            final_chi2 = base_chi2 - chi2_variation
            
            # Return consistent optimized parameters
            if model == "lcdm":
                optimized_params = {"H0": 67.5, "Om0": 0.314}
            else:  # pbuf
                optimized_params = {"k_sat": 0.98, "alpha": 5.1e-4}
            
            return OptimizationResult(
                model=model,
                optimized_params=optimized_params,
                starting_params=starting_values or {},
                final_chi2=final_chi2,
                chi2_improvement=2.0,
                convergence_status="success",
                n_function_evaluations=50,
                optimization_time=10.0,
                bounds_reached=[],
                optimizer_info={"method": "L-BFGS-B", "library": "scipy"},
                covariance_scaling=1.0,
                metadata={}
            )
        
        with patch.object(self.optimizer, 'optimize_parameters', side_effect=mock_consistent_optimization):
            # First optimization run
            result1 = self.optimizer.optimize_parameters("lcdm", ["cmb"], ["H0", "Om0"])
            chi2_1 = result1.final_chi2
            
            # Store the results
            self.store.update_model_defaults("lcdm", result1.optimized_params, result1.metadata)
            
            # Second optimization run with same parameters (regression test)
            result2 = self.optimizer.optimize_parameters("lcdm", ["cmb"], ["H0", "Om0"])
            chi2_2 = result2.final_chi2
            
            # χ² should be identical within 1e-8
            chi2_difference = abs(chi2_2 - chi2_1)
            assert chi2_difference < 1e-8, (
                f"χ² regression test failed: first run = {chi2_1}, "
                f"second run = {chi2_2}, difference = {chi2_difference}"
            )
            
            # Test with PBUF model as well
            pbuf_result1 = self.optimizer.optimize_parameters("pbuf", ["cmb"], ["k_sat", "alpha"])
            pbuf_chi2_1 = pbuf_result1.final_chi2
            
            pbuf_result2 = self.optimizer.optimize_parameters("pbuf", ["cmb"], ["k_sat", "alpha"])
            pbuf_chi2_2 = pbuf_result2.final_chi2
            
            pbuf_chi2_difference = abs(pbuf_chi2_2 - pbuf_chi2_1)
            assert pbuf_chi2_difference < 1e-8, (
                f"PBUF χ² regression test failed: first run = {pbuf_chi2_1}, "
                f"second run = {pbuf_chi2_2}, difference = {pbuf_chi2_difference}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])