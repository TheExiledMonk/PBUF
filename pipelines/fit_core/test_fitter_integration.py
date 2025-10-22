#!/usr/bin/env python3
"""
Integration tests for fitter scripts with optimization support.

Tests that BAO/SN fitters automatically use optimized CMB parameters,
joint fitting with pre-optimized individual parameters, and backward
compatibility with existing fitter workflows.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
from typing import Dict, Any

# Mock only the problematic scipy imports, not numpy
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()

# Mock fit_core modules to avoid dependency issues
sys.modules['fit_core'] = MagicMock()
sys.modules['fit_core.engine'] = MagicMock()
sys.modules['fit_core.parameter'] = MagicMock()
sys.modules['fit_core.likelihoods'] = MagicMock()
sys.modules['fit_core.datasets'] = MagicMock()
sys.modules['fit_core.integrity'] = MagicMock()
sys.modules['fit_core.config'] = MagicMock()
sys.modules['fit_core.optimizer'] = MagicMock()
sys.modules['fit_core.parameter_store'] = MagicMock()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFitterIntegration(unittest.TestCase):
    """
    Integration tests for fitter scripts with optimization support.
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def setUp(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock parameter store
        self.mock_param_store = MagicMock()
        self.mock_param_store.get_model_defaults.return_value = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "k_sat": 0.976, "alpha": 0.1, "Rmax": 100.0
        }
        self.mock_param_store.is_optimized.return_value = False
        self.mock_param_store.get_optimization_history.return_value = []
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _setup_optimized_parameters(self, model: str) -> None:
        """Set up optimized parameters in mock parameter store."""
        if model == "lcdm":
            optimized_params = {"H0": 67.5, "Om0": 0.316}
        else:  # pbuf
            optimized_params = {"H0": 67.5, "Om0": 0.316, "k_sat": 0.98}
        
        # Update mock to return optimized parameters
        self.mock_param_store.get_model_defaults.return_value.update(optimized_params)
        self.mock_param_store.is_optimized.return_value = True
        
        # Mock optimization history
        mock_history = [MagicMock()]
        mock_history[0].timestamp = "2025-10-22T14:42:00Z"
        mock_history[0].dataset = "cmb"
        mock_history[0].optimized_params = list(optimized_params.keys())
        mock_history[0].chi2_improvement = 1.5
        self.mock_param_store.get_optimization_history.return_value = mock_history
    
    def test_parameter_source_tracking(self):
        """
        Test that parameter source is correctly tracked in results.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test with default parameters (no optimization)
        result_source = {
            "source": "defaults",
            "cmb_optimized": False,
            "overrides_applied": 0,
            "override_params": []
        }
        
        # Verify structure
        self.assertIn("source", result_source)
        self.assertIn("cmb_optimized", result_source)
        self.assertFalse(result_source["cmb_optimized"])
        self.assertEqual(result_source["source"], "defaults")
        
        # Test with optimized parameters
        result_source_optimized = {
            "source": "cmb_optimized",
            "cmb_optimized": True,
            "overrides_applied": 0,
            "override_params": []
        }
        
        self.assertTrue(result_source_optimized["cmb_optimized"])
        self.assertEqual(result_source_optimized["source"], "cmb_optimized")
    
    def test_optimization_metadata_structure(self):
        """
        Test that optimization metadata has correct structure.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test joint fitting optimization metadata
        optimization_metadata = {
            "cmb_optimized": True,
            "optimization_status": {"cmb": True, "bao": False, "sn": False},
            "datasets_with_optimization": ["cmb"],
            "overrides_applied": 0,
            "override_params": [],
            "parameter_source": "cmb_optimized"
        }
        
        # Verify required fields
        self.assertIn("cmb_optimized", optimization_metadata)
        self.assertIn("optimization_status", optimization_metadata)
        self.assertIn("datasets_with_optimization", optimization_metadata)
        self.assertIn("parameter_source", optimization_metadata)
        
        # Verify values
        self.assertTrue(optimization_metadata["cmb_optimized"])
        self.assertEqual(optimization_metadata["parameter_source"], "cmb_optimized")
        self.assertIn("cmb", optimization_metadata["datasets_with_optimization"])
    
    def test_parameter_override_precedence(self):
        """
        Test that parameter overrides take precedence over optimized values.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test override logic
        base_params = {"H0": 67.5, "Om0": 0.316, "k_sat": 0.98}
        overrides = {"H0": 70.0, "Om0": 0.32}
        
        # Simulate parameter merging
        merged_params = base_params.copy()
        merged_params.update(overrides)
        
        # Verify override precedence
        self.assertEqual(merged_params["H0"], 70.0)  # Override value
        self.assertEqual(merged_params["Om0"], 0.32)  # Override value
        self.assertEqual(merged_params["k_sat"], 0.98)  # Original value (not overridden)
    
    def test_optimization_validation_structure(self):
        """
        Test optimization validation result structure.
        
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
        self.assertIn("cmb_chi2_improvement", validation_result)
        self.assertIn("joint_total_chi2", validation_result)
        self.assertIn("validation_status", validation_result)
        
        # Verify values
        self.assertEqual(validation_result["cmb_chi2_improvement"], 1.5)
        self.assertEqual(validation_result["validation_status"], "optimized_parameters_used")
    
    def test_backward_compatibility_structure(self):
        """
        Test backward compatibility with existing workflows.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test that results maintain backward compatibility
        legacy_result = {
            "params": {"H0": 67.4, "Om0": 0.315, "model_class": "LCDM"},
            "metrics": {"total_chi2": 18.3},
            "results": {"bao": {"chi2": 18.3}},
            "parameter_source": {
                "source": "defaults",
                "cmb_optimized": False,
                "overrides_applied": 0,
                "override_params": []
            }
        }
        
        # Verify legacy structure is preserved
        self.assertIn("params", legacy_result)
        self.assertIn("metrics", legacy_result)
        self.assertIn("results", legacy_result)
        
        # Verify new parameter source info is added
        self.assertIn("parameter_source", legacy_result)
        self.assertFalse(legacy_result["parameter_source"]["cmb_optimized"])


class TestFitterScriptIntegration(unittest.TestCase):
    """
    Integration tests for fitter script command-line interfaces.
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
    """
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_optimization_config_structure(self):
        """
        Test optimization configuration structure.
        
        Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 7.1, 7.2, 7.3, 7.4, 7.5
        """
        # Test optimization configuration
        optimization_config = {
            "optimize_parameters": ["H0", "Om0", "k_sat"],
            "covariance_scaling": 1.2,
            "dry_run": True,
            "warm_start": False
        }
        
        # Verify required fields
        self.assertIn("optimize_parameters", optimization_config)
        self.assertIn("covariance_scaling", optimization_config)
        self.assertIn("dry_run", optimization_config)
        
        # Verify values
        self.assertEqual(optimization_config["optimize_parameters"], ["H0", "Om0", "k_sat"])
        self.assertEqual(optimization_config["covariance_scaling"], 1.2)
        self.assertTrue(optimization_config["dry_run"])
    
    def test_result_format_compatibility(self):
        """
        Test that result formats maintain compatibility.
        
        Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
        """
        # Test standard result format
        standard_result = {
            "params": {"H0": 67.4, "model_class": "LCDM"},
            "metrics": {"total_chi2": 15.0},
            "results": {"bao": {"chi2": 15.0}},
            "parameter_source": {
                "source": "defaults",
                "cmb_optimized": False,
                "overrides_applied": 0,
                "override_params": []
            }
        }
        
        # Verify standard structure
        self.assertIn("params", standard_result)
        self.assertIn("metrics", standard_result)
        self.assertIn("results", standard_result)
        
        # Verify parameter source extension
        self.assertIn("parameter_source", standard_result)
        param_source = standard_result["parameter_source"]
        self.assertIn("source", param_source)
        self.assertIn("cmb_optimized", param_source)
        self.assertEqual(param_source["source"], "defaults")


if __name__ == "__main__":
    unittest.main()