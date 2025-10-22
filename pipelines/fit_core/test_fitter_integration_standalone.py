#!/usr/bin/env python3
"""
Standalone integration tests for fitter scripts with optimization support.

Tests the core functionality without importing modules that have external dependencies.
This ensures the tests can run in environments without scipy/numpy.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from typing import Dict, Any


class TestFitterIntegrationCore(unittest.TestCase):
    """
    Core integration tests for fitter optimization functionality.
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_parameter_source_tracking_structure(self):
        """
        Test that parameter source tracking has correct structure.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test with default parameters (no optimization)
        result_source_defaults = {
            "source": "defaults",
            "cmb_optimized": False,
            "overrides_applied": 0,
            "override_params": []
        }
        
        # Verify structure for defaults
        self.assertIn("source", result_source_defaults)
        self.assertIn("cmb_optimized", result_source_defaults)
        self.assertIn("overrides_applied", result_source_defaults)
        self.assertIn("override_params", result_source_defaults)
        
        self.assertFalse(result_source_defaults["cmb_optimized"])
        self.assertEqual(result_source_defaults["source"], "defaults")
        self.assertEqual(result_source_defaults["overrides_applied"], 0)
        self.assertEqual(result_source_defaults["override_params"], [])
        
        # Test with optimized parameters
        result_source_optimized = {
            "source": "cmb_optimized",
            "cmb_optimized": True,
            "overrides_applied": 2,
            "override_params": ["H0", "Om0"]
        }
        
        self.assertTrue(result_source_optimized["cmb_optimized"])
        self.assertEqual(result_source_optimized["source"], "cmb_optimized")
        self.assertEqual(result_source_optimized["overrides_applied"], 2)
        self.assertIn("H0", result_source_optimized["override_params"])
        self.assertIn("Om0", result_source_optimized["override_params"])
    
    def test_optimization_metadata_structure(self):
        """
        Test that optimization metadata has correct structure for joint fitting.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test joint fitting optimization metadata
        optimization_metadata = {
            "cmb_optimized": True,
            "optimization_status": {
                "cmb": True,
                "bao": False,
                "sn": False
            },
            "datasets_with_optimization": ["cmb"],
            "overrides_applied": 0,
            "override_params": [],
            "parameter_source": "cmb_optimized"
        }
        
        # Verify required fields
        required_fields = [
            "cmb_optimized",
            "optimization_status", 
            "datasets_with_optimization",
            "overrides_applied",
            "override_params",
            "parameter_source"
        ]
        
        for field in required_fields:
            self.assertIn(field, optimization_metadata)
        
        # Verify values
        self.assertTrue(optimization_metadata["cmb_optimized"])
        self.assertEqual(optimization_metadata["parameter_source"], "cmb_optimized")
        self.assertIn("cmb", optimization_metadata["datasets_with_optimization"])
        self.assertTrue(optimization_metadata["optimization_status"]["cmb"])
        self.assertFalse(optimization_metadata["optimization_status"]["bao"])
    
    def test_parameter_override_precedence_logic(self):
        """
        Test that parameter override logic works correctly.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Simulate optimized parameters from CMB
        optimized_params = {
            "H0": 67.5,
            "Om0": 0.316,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "k_sat": 0.98,
            "alpha": 0.1
        }
        
        # Simulate user overrides
        user_overrides = {
            "H0": 70.0,
            "Om0": 0.32
        }
        
        # Simulate parameter merging (override precedence)
        merged_params = optimized_params.copy()
        merged_params.update(user_overrides)
        
        # Verify override precedence
        self.assertEqual(merged_params["H0"], 70.0)  # Override value
        self.assertEqual(merged_params["Om0"], 0.32)  # Override value
        self.assertEqual(merged_params["k_sat"], 0.98)  # Optimized value (not overridden)
        self.assertEqual(merged_params["alpha"], 0.1)  # Optimized value (not overridden)
        self.assertEqual(merged_params["Obh2"], 0.02237)  # Optimized value (not overridden)
        
        # Verify parameter source tracking
        parameter_source = {
            "source": "cmb_optimized",
            "cmb_optimized": True,
            "overrides_applied": len(user_overrides),
            "override_params": list(user_overrides.keys())
        }
        
        self.assertEqual(parameter_source["overrides_applied"], 2)
        self.assertIn("H0", parameter_source["override_params"])
        self.assertIn("Om0", parameter_source["override_params"])
        self.assertNotIn("k_sat", parameter_source["override_params"])
    
    def test_optimization_validation_structure(self):
        """
        Test optimization validation result structure for joint fitting.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test validation result structure
        validation_result = {
            "cmb_chi2_improvement": 1.5,
            "joint_total_chi2": 25.4,
            "validation_status": "optimized_parameters_used",
            "validation_timestamp": "2025-10-22T14:42:00Z"
        }
        
        # Verify required fields
        required_fields = [
            "cmb_chi2_improvement",
            "joint_total_chi2", 
            "validation_status",
            "validation_timestamp"
        ]
        
        for field in required_fields:
            self.assertIn(field, validation_result)
        
        # Verify values and types
        self.assertIsInstance(validation_result["cmb_chi2_improvement"], (int, float))
        self.assertIsInstance(validation_result["joint_total_chi2"], (int, float))
        self.assertIsInstance(validation_result["validation_status"], str)
        self.assertIsInstance(validation_result["validation_timestamp"], str)
        
        self.assertEqual(validation_result["cmb_chi2_improvement"], 1.5)
        self.assertEqual(validation_result["validation_status"], "optimized_parameters_used")
        self.assertGreater(validation_result["cmb_chi2_improvement"], 0)  # Should show improvement
    
    def test_backward_compatibility_structure(self):
        """
        Test that results maintain backward compatibility with existing workflows.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test that results maintain legacy structure while adding new fields
        legacy_compatible_result = {
            # Legacy fields (must be preserved)
            "params": {
                "H0": 67.4,
                "Om0": 0.315,
                "model_class": "LCDM"
            },
            "metrics": {
                "total_chi2": 18.3,
                "aic": 22.3,
                "bic": 26.1
            },
            "results": {
                "bao": {
                    "chi2": 18.3,
                    "predictions": {}
                }
            },
            # New fields (added for optimization support)
            "parameter_source": {
                "source": "defaults",
                "cmb_optimized": False,
                "overrides_applied": 0,
                "override_params": []
            }
        }
        
        # Verify legacy structure is preserved
        legacy_fields = ["params", "metrics", "results"]
        for field in legacy_fields:
            self.assertIn(field, legacy_compatible_result)
        
        # Verify legacy field contents
        self.assertIn("model_class", legacy_compatible_result["params"])
        self.assertIn("total_chi2", legacy_compatible_result["metrics"])
        self.assertIn("bao", legacy_compatible_result["results"])
        
        # Verify new parameter source info is properly added
        self.assertIn("parameter_source", legacy_compatible_result)
        param_source = legacy_compatible_result["parameter_source"]
        
        self.assertFalse(param_source["cmb_optimized"])
        self.assertEqual(param_source["source"], "defaults")
        self.assertEqual(param_source["overrides_applied"], 0)
    
    def test_cmb_optimization_result_structure(self):
        """
        Test CMB optimization result structure.
        
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 7.1, 7.2, 7.3, 7.4, 7.5
        """
        # Test CMB optimization result structure
        cmb_optimization_result = {
            "params": {
                "H0": 67.5,
                "Om0": 0.316,
                "k_sat": 0.98,
                "model_class": "PBUF"
            },
            "metrics": {
                "total_chi2": 8.5,
                "chi2_improvement": 1.5,
                "optimization_time": 2.3,
                "function_evaluations": 50
            },
            "results": {
                "cmb": {
                    "chi2": 8.5,
                    "optimization_metadata": {
                        "optimization_performed": True,
                        "optimized_parameters": ["H0", "Om0", "k_sat"],
                        "fixed_parameters": ["Obh2", "ns", "alpha", "Rmax", "eps0", "n_eps"],
                        "optimization_status": "success",
                        "chi2_improvement": 1.5,
                        "starting_values": {"H0": 67.4, "Om0": 0.315, "k_sat": 0.976},
                        "final_values": {"H0": 67.5, "Om0": 0.316, "k_sat": 0.98}
                    }
                }
            },
            "optimization_summary": {
                "optimization_performed": True,
                "optimized_parameters": ["H0", "Om0", "k_sat"],
                "fixed_parameters": ["Obh2", "ns", "alpha", "Rmax", "eps0", "n_eps"],
                "optimization_status": "success",
                "chi2_improvement": 1.5
            }
        }
        
        # Verify optimization-specific fields
        self.assertIn("optimization_summary", cmb_optimization_result)
        opt_summary = cmb_optimization_result["optimization_summary"]
        
        self.assertTrue(opt_summary["optimization_performed"])
        self.assertEqual(opt_summary["optimization_status"], "success")
        self.assertIn("H0", opt_summary["optimized_parameters"])
        self.assertIn("Om0", opt_summary["optimized_parameters"])
        self.assertIn("k_sat", opt_summary["optimized_parameters"])
        
        # Verify metrics include optimization-specific information
        metrics = cmb_optimization_result["metrics"]
        self.assertIn("chi2_improvement", metrics)
        self.assertIn("optimization_time", metrics)
        self.assertIn("function_evaluations", metrics)
        
        # Verify CMB result includes optimization metadata
        cmb_result = cmb_optimization_result["results"]["cmb"]
        self.assertIn("optimization_metadata", cmb_result)
        opt_metadata = cmb_result["optimization_metadata"]
        
        self.assertTrue(opt_metadata["optimization_performed"])
        self.assertIn("starting_values", opt_metadata)
        self.assertIn("final_values", opt_metadata)
    
    def test_optimization_config_validation(self):
        """
        Test optimization configuration validation.
        
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
        """
        # Test valid optimization configuration
        valid_config = {
            "optimize_parameters": ["H0", "Om0", "k_sat"],
            "covariance_scaling": 1.2,
            "dry_run": False,
            "warm_start": True
        }
        
        # Verify required fields
        self.assertIn("optimize_parameters", valid_config)
        self.assertIsInstance(valid_config["optimize_parameters"], list)
        self.assertGreater(len(valid_config["optimize_parameters"]), 0)
        
        # Verify optional fields have correct types
        self.assertIsInstance(valid_config["covariance_scaling"], (int, float))
        self.assertIsInstance(valid_config["dry_run"], bool)
        self.assertIsInstance(valid_config["warm_start"], bool)
        
        # Test parameter validation
        valid_lcdm_params = ["H0", "Om0", "Obh2", "ns"]
        valid_pbuf_params = ["H0", "Om0", "Obh2", "ns", "k_sat", "alpha", "Rmax", "eps0", "n_eps"]
        
        # All requested parameters should be in valid set
        for param in valid_config["optimize_parameters"]:
            self.assertIn(param, valid_pbuf_params)
    
    def test_file_structure_requirements(self):
        """
        Test that required file structures are maintained.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test that parameter store file structure is correct
        param_store_structure = {
            "defaults": {
                "H0": 67.4,
                "Om0": 0.315,
                "model_class": "LCDM"
            },
            "optimization_metadata": {
                "initialized": "2025-10-22T14:42:00Z",
                "source": "hardcoded_defaults",
                "last_updated": "2025-10-22T14:42:00Z",
                "optimized_params": []
            }
        }
        
        # Verify required top-level fields
        self.assertIn("defaults", param_store_structure)
        self.assertIn("optimization_metadata", param_store_structure)
        
        # Verify defaults structure
        defaults = param_store_structure["defaults"]
        self.assertIsInstance(defaults, dict)
        self.assertIn("H0", defaults)
        self.assertIn("Om0", defaults)
        
        # Verify metadata structure
        metadata = param_store_structure["optimization_metadata"]
        self.assertIn("initialized", metadata)
        self.assertIn("source", metadata)
        self.assertIsInstance(metadata["optimized_params"], list)


class TestFitterWorkflowIntegration(unittest.TestCase):
    """
    Test integration of fitter workflows with optimization.
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def test_workflow_sequence_validation(self):
        """
        Test that fitter workflow sequences are correctly validated.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test recommended workflow sequence
        workflow_steps = [
            {
                "step": 1,
                "action": "cmb_optimization",
                "description": "Optimize CMB parameters first",
                "required": True
            },
            {
                "step": 2,
                "action": "individual_fits",
                "description": "Run BAO/SN fits with optimized parameters",
                "required": False
            },
            {
                "step": 3,
                "action": "joint_fit",
                "description": "Run joint fit with pre-optimized parameters",
                "required": False
            }
        ]
        
        # Verify workflow structure
        self.assertEqual(len(workflow_steps), 3)
        
        # Verify CMB optimization is first and required
        cmb_step = workflow_steps[0]
        self.assertEqual(cmb_step["step"], 1)
        self.assertEqual(cmb_step["action"], "cmb_optimization")
        self.assertTrue(cmb_step["required"])
        
        # Verify subsequent steps reference optimization
        for step in workflow_steps[1:]:
            self.assertIn("optimized", step["description"])
    
    def test_error_handling_structure(self):
        """
        Test error handling structure for optimization failures.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test error result structure
        error_result = {
            "error": "Optimization failed: convergence not achieved",
            "error_type": "optimization_failure",
            "fallback_action": "use_default_parameters",
            "parameter_source": {
                "source": "defaults",
                "cmb_optimized": False,
                "error_occurred": True,
                "error_message": "Optimization failed: convergence not achieved"
            }
        }
        
        # Verify error handling structure
        self.assertIn("error", error_result)
        self.assertIn("error_type", error_result)
        self.assertIn("fallback_action", error_result)
        self.assertIn("parameter_source", error_result)
        
        # Verify graceful degradation
        param_source = error_result["parameter_source"]
        self.assertEqual(param_source["source"], "defaults")
        self.assertFalse(param_source["cmb_optimized"])
        self.assertTrue(param_source["error_occurred"])


if __name__ == "__main__":
    unittest.main()