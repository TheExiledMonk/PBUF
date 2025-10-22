#!/usr/bin/env python3
"""
Integration tests for BAO anisotropic fitting with fit_core infrastructure.

This module provides comprehensive integration tests that verify the BAO anisotropic
fitting works correctly with all fit_core components including engine, parameter_store,
datasets, likelihoods, and integrity validation.

Requirements: 5.1
"""

import unittest
import tempfile
import shutil
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
from typing import Dict, Any, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock scipy to avoid dependency issues
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()


class TestEngineIntegration(unittest.TestCase):
    """
    Integration tests with the fit_core engine.
    
    Tests that BAO anisotropic fitting integrates correctly with the
    unified optimization engine.
    """
    
    def setUp(self):
        """Set up engine integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.test_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "model_class": "lcdm"
        }
        
        self.expected_engine_results = {
            "params": self.test_params,
            "metrics": {
                "total_chi2": 15.234,
                "aic": 23.234,
                "bic": 31.456,
                "dof": 8,
                "p_value": 0.054
            },
            "results": {
                "bao_ani": {
                    "chi2": 15.234,
                    "predictions": {
                        "DM_over_rs": np.array([8.467, 13.156, 16.789]),
                        "H_times_rs": np.array([147.8, 98.2, 81.3])
                    },
                    "residuals": np.array([0.1, -0.2, 0.05])
                }
            },
            "chi2_breakdown": {
                "bao_ani": 15.234
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_engine_call_with_correct_parameters(self, mock_store_class, mock_run_fit):
        """Test that engine is called with correct parameters for BAO anisotropic fitting."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        mock_run_fit.return_value = self.expected_engine_results
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": self.test_params,
                "source": "defaults",
                "cmb_optimized": False
            }
            
            try:
                from fit_bao_aniso import run_bao_aniso_fit
                
                result = run_bao_aniso_fit("lcdm")
                
                # Verify engine was called with correct parameters
                mock_run_fit.assert_called_once_with(
                    model="lcdm",
                    datasets_list=["bao_ani"],
                    mode="individual",
                    overrides=self.test_params
                )
                
                # Verify result structure includes engine results
                self.assertIn("params", result)
                self.assertIn("metrics", result)
                self.assertEqual(result["params"], self.test_params)
            
            except ImportError:
                self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_engine_individual_mode_specification(self, mock_store_class, mock_run_fit):
        """Test that engine is called in individual mode for BAO anisotropic fitting."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        mock_run_fit.return_value = self.expected_engine_results
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": self.test_params,
                "source": "defaults",
                "cmb_optimized": False
            }
            
            try:
                from fit_bao_aniso import run_bao_aniso_fit
                
                run_bao_aniso_fit("lcdm")
                
                # Verify mode parameter
                call_args = mock_run_fit.call_args
                self.assertEqual(call_args[1]["mode"], "individual")
            
            except ImportError:
                self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_engine_dataset_specification(self, mock_store_class, mock_run_fit):
        """Test that bao_ani dataset is correctly specified to engine."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        mock_run_fit.return_value = self.expected_engine_results
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": self.test_params,
                "source": "defaults",
                "cmb_optimized": False
            }
            
            try:
                from fit_bao_aniso import run_bao_aniso_fit
                
                run_bao_aniso_fit("lcdm")
                
                # Verify datasets_list parameter
                call_args = mock_run_fit.call_args
                self.assertEqual(call_args[1]["datasets_list"], ["bao_ani"])
            
            except ImportError:
                self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_engine_error_handling(self, mock_store_class, mock_run_fit):
        """Test error handling when engine fails."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        # Make engine raise an exception
        mock_run_fit.side_effect = Exception("Engine optimization failed")
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": self.test_params,
                "source": "defaults",
                "cmb_optimized": False
            }
            
            try:
                from fit_bao_aniso import run_bao_aniso_fit
                
                with self.assertRaises(Exception) as context:
                    run_bao_aniso_fit("lcdm")
                
                self.assertIn("Engine optimization failed", str(context.exception))
            
            except ImportError:
                self.skipTest("Could not import run_bao_aniso_fit")


class TestParameterStoreIntegration(unittest.TestCase):
    """
    Integration tests with OptimizedParameterStore.
    
    Tests that BAO anisotropic fitting correctly integrates with the
    parameter optimization and storage system.
    """
    
    def setUp(self):
        """Set up parameter store integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_parameter_store_initialization(self, mock_store_class):
        """Test that parameter store is properly initialized."""
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": {"H0": 67.4, "Om0": 0.315},
                "source": "defaults",
                "cmb_optimized": False
            }
            
            with patch('fit_bao_aniso._execute_fit_with_params') as mock_execute:
                mock_execute.return_value = {"success": True}
                
                try:
                    from fit_bao_aniso import run_bao_aniso_fit
                    
                    run_bao_aniso_fit("lcdm")
                    
                    # Verify parameter store was initialized
                    mock_store_class.assert_called_once()
                    mock_store.verify_storage_integrity.assert_called_once()
                
                except ImportError:
                    self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_parameter_store_integrity_check(self, mock_store_class):
        """Test parameter store integrity verification."""
        mock_store = MagicMock()
        
        # Test different integrity statuses
        integrity_statuses = ["healthy", "recovered", "corrupted"]
        
        for status in integrity_statuses:
            with self.subTest(status=status):
                mock_store.verify_storage_integrity.return_value = {
                    "overall_status": status,
                    "recovery_actions": ["action1", "action2"] if status == "recovered" else []
                }
                mock_store_class.return_value = mock_store
                
                with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                    mock_get_params.return_value = {
                        "final_params": {"H0": 67.4},
                        "source": "defaults",
                        "cmb_optimized": False
                    }
                    
                    with patch('fit_bao_aniso._execute_fit_with_params') as mock_execute:
                        mock_execute.return_value = {"success": True}
                        
                        try:
                            from fit_bao_aniso import run_bao_aniso_fit
                            
                            result = run_bao_aniso_fit("lcdm")
                            
                            # Should succeed regardless of integrity status
                            self.assertTrue(result.get("success", False))
                        
                        except ImportError:
                            self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_parameter_store_fallback_on_failure(self, mock_store_class):
        """Test fallback behavior when parameter store initialization fails."""
        # Make parameter store initialization fail
        mock_store_class.side_effect = Exception("Storage initialization failed")
        
        with patch('fit_bao_aniso.get_default_params') as mock_get_defaults:
            mock_get_defaults.return_value = {"H0": 67.4, "Om0": 0.315}
            
            with patch('fit_bao_aniso._execute_fit_with_params') as mock_execute:
                mock_execute.return_value = {
                    "success": True,
                    "parameter_source": {
                        "source": "hardcoded_fallback",
                        "fallback_reason": "Storage initialization failed"
                    }
                }
                
                try:
                    from fit_bao_aniso import run_bao_aniso_fit
                    
                    result = run_bao_aniso_fit("lcdm")
                    
                    # Verify fallback was used
                    param_source = result.get("parameter_source", {})
                    self.assertEqual(param_source.get("source"), "hardcoded_fallback")
                    self.assertIn("Storage initialization failed", param_source.get("fallback_reason", ""))
                
                except ImportError:
                    self.skipTest("Could not import run_bao_aniso_fit")


class TestIntegrityValidationIntegration(unittest.TestCase):
    """
    Integration tests with integrity validation system.
    
    Tests that BAO anisotropic fitting correctly integrates with the
    physics consistency and integrity checking system.
    """
    
    def setUp(self):
        """Set up integrity validation test environment."""
        self.integrity_results_pass = {
            "overall_status": "PASS",
            "summary": {
                "total_tests": 5,
                "passed": 5,
                "failed": 0,
                "warnings": 0
            },
            "tests_run": ["recombination", "sound_horizon", "h_ratios", "bao_consistency", "covariance_validation"],
            "tolerances_used": {
                "h_ratios": 1e-4,
                "recombination": 1e-4,
                "sound_horizon": 1e-4
            },
            "recombination": {
                "status": "PASS",
                "description": "Recombination redshift within expected range",
                "computed_z_recomb": 1089.2,
                "reference_z_recomb": 1089.0
            },
            "sound_horizon": {
                "status": "PASS",
                "description": "Sound horizon at drag epoch consistent",
                "computed_r_s_drag": 147.8,
                "reference_r_s_drag": 147.9
            }
        }
        
        self.integrity_results_fail = {
            "overall_status": "FAIL",
            "summary": {
                "total_tests": 5,
                "passed": 3,
                "failed": 2,
                "warnings": 0
            },
            "tests_run": ["recombination", "sound_horizon", "h_ratios"],
            "failures": ["h_ratios", "bao_consistency"]
        }
    
    @patch('fit_bao_aniso.integrity.run_integrity_suite')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_integrity_validation_enabled(self, mock_store_class, mock_integrity):
        """Test integrity validation when enabled."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        mock_integrity.return_value = self.integrity_results_pass
        
        with patch('fit_bao_aniso.engine.run_fit') as mock_run_fit:
            mock_run_fit.return_value = {"success": True}
            
            with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                mock_get_params.return_value = {
                    "final_params": {"H0": 67.4, "Om0": 0.315},
                    "source": "defaults",
                    "cmb_optimized": False
                }
                
                try:
                    from fit_bao_aniso import run_bao_aniso_fit
                    
                    result = run_bao_aniso_fit("lcdm", verify_integrity=True, integrity_tolerance=1e-4)
                    
                    # Verify integrity suite was called
                    mock_integrity.assert_called_once()
                    call_args = mock_integrity.call_args
                    
                    # Check call parameters
                    self.assertEqual(call_args[1]["datasets"], ["bao_ani"])
                    self.assertIn("tolerances", call_args[1])
                    
                    # Verify tolerances were set correctly
                    tolerances = call_args[1]["tolerances"]
                    self.assertEqual(tolerances["h_ratios"], 1e-4)
                    self.assertEqual(tolerances["recombination"], 1e-4)
                    self.assertEqual(tolerances["sound_horizon"], 1e-4)
                
                except ImportError:
                    self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.integrity.run_integrity_suite')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_integrity_validation_disabled(self, mock_store_class, mock_integrity):
        """Test that integrity validation is skipped when disabled."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        with patch('fit_bao_aniso.engine.run_fit') as mock_run_fit:
            mock_run_fit.return_value = {"success": True}
            
            with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                mock_get_params.return_value = {
                    "final_params": {"H0": 67.4, "Om0": 0.315},
                    "source": "defaults",
                    "cmb_optimized": False
                }
                
                try:
                    from fit_bao_aniso import run_bao_aniso_fit
                    
                    result = run_bao_aniso_fit("lcdm", verify_integrity=False)
                    
                    # Verify integrity suite was NOT called
                    mock_integrity.assert_not_called()
                
                except ImportError:
                    self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.integrity.run_integrity_suite')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_integrity_validation_failure_handling(self, mock_store_class, mock_integrity):
        """Test handling of integrity validation failures."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        mock_integrity.return_value = self.integrity_results_fail
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": {"H0": 67.4, "Om0": 0.315},
                "source": "defaults",
                "cmb_optimized": False
            }
            
            try:
                from fit_bao_aniso import run_bao_aniso_fit
                
                result = run_bao_aniso_fit("lcdm", verify_integrity=True)
                
                # Should return error result when integrity fails
                self.assertIn("error", result)
                self.assertIn("integrity_results", result)
                self.assertEqual(result["integrity_results"]["overall_status"], "FAIL")
            
            except ImportError:
                self.skipTest("Could not import run_bao_aniso_fit")


class TestDatasetIntegration(unittest.TestCase):
    """
    Integration tests with dataset loading and handling.
    
    Tests that BAO anisotropic fitting correctly integrates with the
    dataset loading and validation system.
    """
    
    def test_bao_ani_dataset_specification(self):
        """Test that bao_ani dataset is correctly specified."""
        # This test verifies the dataset name is correctly passed to the engine
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store_class.return_value = mock_store
            
            with patch('fit_bao_aniso.engine.run_fit') as mock_run_fit:
                mock_run_fit.return_value = {"success": True}
                
                with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                    mock_get_params.return_value = {
                        "final_params": {"H0": 67.4},
                        "source": "defaults",
                        "cmb_optimized": False
                    }
                    
                    try:
                        from fit_bao_aniso import run_bao_aniso_fit
                        
                        run_bao_aniso_fit("lcdm")
                        
                        # Verify correct dataset was specified
                        call_args = mock_run_fit.call_args
                        datasets_list = call_args[1]["datasets_list"]
                        self.assertEqual(datasets_list, ["bao_ani"])
                        self.assertNotIn("bao", datasets_list)  # Should not include isotropic BAO
                    
                    except ImportError:
                        self.skipTest("Could not import run_bao_aniso_fit")
    
    def test_dataset_separation_from_isotropic_bao(self):
        """Test that anisotropic BAO is properly separated from isotropic BAO."""
        # This test ensures we don't accidentally include both bao and bao_ani
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store_class.return_value = mock_store
            
            with patch('fit_bao_aniso.engine.run_fit') as mock_run_fit:
                mock_run_fit.return_value = {"success": True}
                
                with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                    mock_get_params.return_value = {
                        "final_params": {"H0": 67.4},
                        "source": "defaults",
                        "cmb_optimized": False
                    }
                    
                    try:
                        from fit_bao_aniso import run_bao_aniso_fit
                        
                        run_bao_aniso_fit("lcdm")
                        
                        # Verify only bao_ani is included, not bao
                        call_args = mock_run_fit.call_args
                        datasets_list = call_args[1]["datasets_list"]
                        
                        self.assertIn("bao_ani", datasets_list)
                        self.assertNotIn("bao", datasets_list)
                        self.assertEqual(len(datasets_list), 1)
                    
                    except ImportError:
                        self.skipTest("Could not import run_bao_aniso_fit")


class TestEndToEndIntegration(unittest.TestCase):
    """
    End-to-end integration tests.
    
    Tests the complete integration flow from parameter loading through
    fitting to result formatting.
    """
    
    @patch('fit_bao_aniso.OptimizedParameterStore')
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.integrity.run_integrity_suite')
    def test_complete_integration_flow_lcdm(self, mock_integrity, mock_run_fit, mock_store_class):
        """Test complete integration flow for LCDM model."""
        # Setup comprehensive mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store.get_model_defaults.return_value = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649
        }
        mock_store.is_optimized.return_value = True
        
        # Mock optimization record
        mock_record = MagicMock()
        mock_record.dataset = "cmb"
        mock_record.convergence_status = "success"
        mock_record.final_values = {"H0": 67.8, "Om0": 0.31}
        mock_record.optimized_params = ["H0", "Om0"]
        mock_record.timestamp = "2024-01-01T12:00:00Z"
        
        mock_store.get_optimization_history.return_value = [mock_record]
        mock_store_class.return_value = mock_store
        
        # Mock engine results (should reflect the override H0=70.0)
        mock_run_fit.return_value = {
            "params": {"H0": 70.0, "Om0": 0.31, "Obh2": 0.02237, "ns": 0.9649, "model_class": "lcdm"},
            "metrics": {"total_chi2": 15.234, "aic": 23.234, "dof": 8},
            "results": {"bao_ani": {"chi2": 15.234}}
        }
        
        # Mock integrity results
        mock_integrity.return_value = {
            "overall_status": "PASS",
            "summary": {"total_tests": 5, "passed": 5, "failed": 0}
        }
        
        try:
            from fit_bao_aniso import run_bao_aniso_fit
            
            # Run complete integration test
            result = run_bao_aniso_fit(
                model="lcdm",
                overrides={"H0": 70.0},
                verify_integrity=True,
                integrity_tolerance=1e-4
            )
            
            # Verify all components were called
            mock_store_class.assert_called_once()
            mock_run_fit.assert_called_once()
            mock_integrity.assert_called_once()
            
            # Verify result structure
            self.assertIn("params", result)
            self.assertIn("metrics", result)
            self.assertIn("parameter_source", result)
            
            # Verify parameter source information
            param_source = result["parameter_source"]
            self.assertIn("source", param_source)
            self.assertIn("cmb_optimized", param_source)
            self.assertIn("overrides_applied", param_source)
            
            # Verify override was applied
            self.assertEqual(result["params"]["H0"], 70.0)
            self.assertEqual(param_source["overrides_applied"], 1)
        
        except ImportError:
            self.skipTest("Could not import run_bao_aniso_fit")
    
    @patch('fit_bao_aniso.OptimizedParameterStore')
    @patch('fit_bao_aniso.engine.run_fit')
    def test_complete_integration_flow_pbuf(self, mock_run_fit, mock_store_class):
        """Test complete integration flow for PBUF model."""
        # Setup mocks for PBUF model
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store.get_model_defaults.return_value = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "alpha": 0.1, "Rmax": 100.0, "eps0": 0.01, "n_eps": 2.0, "k_sat": 0.1
        }
        mock_store.is_optimized.return_value = False
        mock_store.get_optimization_history.return_value = []
        mock_store.get_warm_start_params.return_value = None
        mock_store_class.return_value = mock_store
        
        # Mock engine results for PBUF
        mock_run_fit.return_value = {
            "params": {
                "H0": 68.0, "Om0": 0.31, "Obh2": 0.02237, "ns": 0.9649,
                "alpha": 0.12, "Rmax": 100.0, "eps0": 0.01, "n_eps": 2.0, "k_sat": 0.1,
                "model_class": "pbuf"
            },
            "metrics": {"total_chi2": 12.891, "aic": 22.891, "dof": 8},
            "results": {"bao_ani": {"chi2": 12.891}}
        }
        
        try:
            from fit_bao_aniso import run_bao_aniso_fit
            
            # Run PBUF integration test
            result = run_bao_aniso_fit(
                model="pbuf",
                overrides={"H0": 68.0, "alpha": 0.12}
            )
            
            # Verify PBUF-specific parameters
            self.assertEqual(result["params"]["model_class"], "pbuf")
            self.assertIn("alpha", result["params"])
            self.assertIn("Rmax", result["params"])
            
            # Verify overrides were applied
            self.assertEqual(result["params"]["H0"], 68.0)
            self.assertEqual(result["params"]["alpha"], 0.12)
        
        except ImportError:
            self.skipTest("Could not import run_bao_aniso_fit")


if __name__ == '__main__':
    # Run integration tests with detailed output
    unittest.main(verbosity=2, buffer=True)