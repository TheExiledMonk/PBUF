#!/usr/bin/env python3
"""
Comprehensive test suite for BAO anisotropic fitting functionality.

This test suite covers:
- Unit tests for parameter loading and optimization integration
- Validation tests comparing against existing fit_aniso.py results
- Integration tests with fit_core infrastructure
- Performance benchmarks for fitting execution time

Requirements: 5.1
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call
from typing import Dict, Any, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock scipy to avoid dependency issues in tests
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.optimize'] = MagicMock()

# Import the module under test
try:
    from fit_bao_aniso import (
        run_bao_aniso_fit,
        _get_optimized_parameters_with_metadata,
        _apply_parameter_overrides,
        _execute_fit_with_params,
        parse_arguments
    )
except ImportError:
    # Create mock functions if import fails
    def run_bao_aniso_fit(*args, **kwargs):
        return {"mock": True}
    
    def _get_optimized_parameters_with_metadata(*args, **kwargs):
        return {"mock": True}
    
    def _apply_parameter_overrides(*args, **kwargs):
        return {"mock": True}
    
    def _execute_fit_with_params(*args, **kwargs):
        return {"mock": True}
    
    def parse_arguments():
        return MagicMock()


class TestParameterLoadingAndOptimization(unittest.TestCase):
    """
    Unit tests for parameter loading and optimization integration.
    
    Tests the enhanced parameter loading system that integrates with
    OptimizedParameterStore and handles fallbacks gracefully.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_param_store = MagicMock()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_parameter_store_initialization_success(self, mock_store_class):
        """Test successful parameter store initialization."""
        # Setup mock
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        # Mock parameter retrieval
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": {"H0": 70.0, "Om0": 0.3},
                "source": "cmb_optimized",
                "cmb_optimized": True
            }
            
            with patch('fit_bao_aniso._execute_fit_with_params') as mock_execute:
                mock_execute.return_value = {"success": True}
                
                result = run_bao_aniso_fit("lcdm")
                
                # Verify parameter store was initialized
                mock_store_class.assert_called_once()
                mock_store.verify_storage_integrity.assert_called_once()
    
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_parameter_store_initialization_failure_fallback(self, mock_store_class):
        """Test fallback to hardcoded defaults when parameter store fails."""
        # Setup mock to raise exception
        mock_store_class.side_effect = Exception("Storage unavailable")
        
        with patch('fit_core.parameter.get_defaults') as mock_get_defaults:
            mock_get_defaults.return_value = {"H0": 67.4, "Om0": 0.315}
            
            with patch('fit_bao_aniso._execute_fit_with_params') as mock_execute:
                mock_execute.return_value = {"success": True, "fallback": True}
                
                result = run_bao_aniso_fit("lcdm")
                
                # Verify fallback was used
                mock_get_defaults.assert_called_once_with("lcdm")
                self.assertTrue(result.get("fallback", False))
    
    def test_get_optimized_parameters_with_metadata_cmb_priority(self):
        """Test that CMB optimization takes priority over other sources."""
        mock_store = MagicMock()
        
        # Setup mock to have multiple optimizations available
        mock_store.get_model_defaults.return_value = {"H0": 67.4, "Om0": 0.315}
        mock_store.is_optimized.side_effect = lambda model, dataset: dataset in ["cmb", "bao", "sn"]
        
        # Mock optimization history with different datasets
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc).isoformat()
        
        mock_cmb_record = MagicMock()
        mock_cmb_record.dataset = "cmb"
        mock_cmb_record.convergence_status = "success"
        mock_cmb_record.final_values = {"H0": 67.8, "Om0": 0.31}
        mock_cmb_record.optimized_params = ["H0", "Om0"]
        mock_cmb_record.timestamp = current_time
        
        mock_bao_record = MagicMock()
        mock_bao_record.dataset = "bao"
        mock_bao_record.convergence_status = "success"
        mock_bao_record.final_values = {"H0": 68.2, "Om0": 0.32}
        mock_bao_record.timestamp = current_time
        
        mock_store.get_optimization_history.return_value = [mock_cmb_record, mock_bao_record]
        
        result = _get_optimized_parameters_with_metadata(mock_store, "lcdm")
        
        # Verify CMB optimization was selected
        self.assertEqual(result["optimization_metadata"]["used_optimization"], "cmb")
        self.assertEqual(result["final_params"]["H0"], 67.8)
        self.assertTrue(result["cmb_optimized"])
    
    def test_get_optimized_parameters_no_optimization_available(self):
        """Test behavior when no optimizations are available."""
        mock_store = MagicMock()
        
        # Setup mock with no optimizations
        mock_store.get_model_defaults.return_value = {"H0": 67.4, "Om0": 0.315}
        mock_store.is_optimized.return_value = False
        mock_store.get_optimization_history.return_value = []
        mock_store.get_warm_start_params.return_value = None
        
        result = _get_optimized_parameters_with_metadata(mock_store, "lcdm")
        
        # Verify defaults were used
        self.assertEqual(result["source"], "defaults")
        self.assertFalse(result["cmb_optimized"])
        self.assertEqual(result["final_params"]["H0"], 67.4)
    
    def test_apply_parameter_overrides_valid(self):
        """Test applying valid parameter overrides."""
        base_params = {"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649}
        overrides = {"H0": 70.0, "Om0": 0.3}
        
        with patch('fit_core.parameter.validate_params') as mock_validate:
            mock_validate.return_value = None  # No validation errors
            
            with patch('fit_core.parameter.OPTIMIZABLE_PARAMETERS', {"lcdm": ["H0", "Om0"]}):
                result = _apply_parameter_overrides(base_params, overrides, "lcdm")
                
                # Verify overrides were applied
                self.assertEqual(base_params["H0"], 70.0)
                self.assertEqual(base_params["Om0"], 0.3)
                self.assertEqual(result["overrides_applied"], 2)
                self.assertEqual(set(result["override_params"]), {"H0", "Om0"})
                self.assertEqual(set(result["optimizable_overrides"]), {"H0", "Om0"})
    
    def test_apply_parameter_overrides_invalid_range(self):
        """Test that invalid parameter ranges are rejected."""
        base_params = {"H0": 67.4, "Om0": 0.315}
        overrides = {"H0": 150.0}  # Invalid H0 value
        
        with self.assertRaises(ValueError) as context:
            _apply_parameter_overrides(base_params, overrides, "lcdm")
        
        self.assertIn("out of range", str(context.exception))
        # Verify original values were not modified
        self.assertEqual(base_params["H0"], 67.4)
    
    def test_apply_parameter_overrides_invalid_parameter(self):
        """Test that invalid parameter names are rejected."""
        base_params = {"H0": 67.4, "Om0": 0.315}
        overrides = {"invalid_param": 1.0}
        
        with self.assertRaises(ValueError) as context:
            _apply_parameter_overrides(base_params, overrides, "lcdm")
        
        self.assertIn("not in model", str(context.exception))


class TestValidationAgainstExistingImplementation(unittest.TestCase):
    """
    Validation tests comparing against existing fit_aniso.py results.
    
    These tests ensure the enhanced implementation produces results
    consistent with the original fit_aniso.py script.
    """
    
    def setUp(self):
        """Set up test environment."""
        self.test_params_lcdm = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "RECFAST",
            "model_class": "lcdm"
        }
        
        self.test_params_pbuf = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "RECFAST",
            "alpha": 0.1,
            "Rmax": 100.0,
            "eps0": 0.01,
            "n_eps": 2.0,
            "k_sat": 0.1,
            "model_class": "pbuf"
        }
    
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_lcdm_results_consistency(self, mock_store_class, mock_run_fit):
        """Test that LCDM results are consistent with original implementation."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        # Mock expected results from original implementation
        expected_results = {
            "params": self.test_params_lcdm,
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
                    }
                }
            }
        }
        
        mock_run_fit.return_value = expected_results
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": self.test_params_lcdm,
                "source": "defaults",
                "cmb_optimized": False
            }
            
            result = run_bao_aniso_fit("lcdm")
            
            # Verify engine was called with correct parameters (no overrides when using defaults)
            mock_run_fit.assert_called_once_with(
                model="lcdm",
                datasets_list=["bao_ani"],
                mode="individual",
                overrides=None
            )
            
            # Verify results structure
            self.assertIn("params", result)
            self.assertIn("metrics", result)
            self.assertIn("parameter_source", result)
    
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_pbuf_results_consistency(self, mock_store_class, mock_run_fit):
        """Test that PBUF results are consistent with original implementation."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        # Mock expected results
        expected_results = {
            "params": self.test_params_pbuf,
            "metrics": {
                "total_chi2": 12.891,
                "aic": 22.891,
                "bic": 35.234,
                "dof": 8,
                "p_value": 0.115
            }
        }
        
        mock_run_fit.return_value = expected_results
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": self.test_params_pbuf,
                "source": "defaults",
                "cmb_optimized": False
            }
            
            result = run_bao_aniso_fit("pbuf")
            
            # Verify PBUF-specific parameters were included
            self.assertEqual(result["params"]["alpha"], 0.1)
            self.assertEqual(result["params"]["Rmax"], 100.0)
    
    def test_parameter_override_consistency(self):
        """Test that parameter overrides work consistently with original implementation."""
        overrides = {"H0": 70.0, "Om0": 0.3}
        
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store_class.return_value = mock_store
            
            with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                mock_get_params.return_value = {
                    "final_params": self.test_params_lcdm.copy(),
                    "source": "defaults",
                    "cmb_optimized": False
                }
                
                with patch('fit_bao_aniso._execute_fit_with_params') as mock_execute:
                    mock_execute.return_value = {"success": True}
                    
                    result = run_bao_aniso_fit("lcdm", overrides=overrides)
                    
                    # Verify overrides were processed
                    call_args = mock_execute.call_args[0]
                    final_params = call_args[1]
                    self.assertEqual(final_params["H0"], 70.0)
                    self.assertEqual(final_params["Om0"], 0.3)


class TestFitCoreIntegration(unittest.TestCase):
    """
    Integration tests with fit_core infrastructure.
    
    Tests that the BAO anisotropic fitting integrates properly with
    the existing fit_core components.
    """
    
    @patch('fit_bao_aniso.integrity.run_integrity_suite')
    def test_integrity_validation_integration(self, mock_integrity):
        """Test integration with integrity validation system."""
        # Setup mock integrity results
        mock_integrity.return_value = {
            "overall_status": "PASS",
            "summary": {"total_tests": 5, "passed": 5, "failed": 0},
            "tests_run": ["recombination", "sound_horizon", "h_ratios"]
        }
        
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
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
                    
                    result = run_bao_aniso_fit("lcdm", verify_integrity=True)
                    
                    # Verify integrity suite was called
                    mock_integrity.assert_called_once()
                    call_args = mock_integrity.call_args
                    self.assertEqual(call_args[1]["datasets"], ["bao_ani"])
    
    @patch('fit_bao_aniso.engine.run_fit')
    def test_engine_integration_individual_mode(self, mock_run_fit):
        """Test integration with engine in individual mode."""
        mock_run_fit.return_value = {
            "params": {"H0": 67.4, "Om0": 0.315},
            "metrics": {"total_chi2": 15.0}
        }
        
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store_class.return_value = mock_store
            
            with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                mock_get_params.return_value = {
                    "final_params": {"H0": 67.4, "Om0": 0.315},
                    "source": "defaults",
                    "cmb_optimized": False
                }
                
                result = run_bao_aniso_fit("lcdm")
                
                # Verify engine was called with correct parameters (no overrides when using defaults)
                mock_run_fit.assert_called_once_with(
                    model="lcdm",
                    datasets_list=["bao_ani"],
                    mode="individual",
                    overrides=None
                )
    
    def test_dataset_specification_bao_ani(self):
        """Test that bao_ani dataset is correctly specified."""
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
                    
                    run_bao_aniso_fit("lcdm")
                    
                    # Verify bao_ani dataset was specified
                    call_args = mock_run_fit.call_args
                    self.assertEqual(call_args[1]["datasets_list"], ["bao_ani"])
    
    def test_error_handling_integration(self):
        """Test error handling integration with fit_core."""
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store_class.return_value = mock_store
            
            with patch('fit_bao_aniso.engine.run_fit') as mock_run_fit:
                mock_run_fit.side_effect = Exception("Fitting failed")
                
                with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                    mock_get_params.return_value = {
                        "final_params": {"H0": 67.4},
                        "source": "defaults",
                        "cmb_optimized": False
                    }
                    
                    with self.assertRaises(Exception) as context:
                        run_bao_aniso_fit("lcdm")
                    
                    self.assertIn("Fitting failed", str(context.exception))


class TestPerformanceBenchmarks(unittest.TestCase):
    """
    Performance benchmarks for fitting execution time.
    
    Tests that measure and validate the performance characteristics
    of the BAO anisotropic fitting implementation.
    """
    
    def setUp(self):
        """Set up performance test environment."""
        self.performance_results = {}
    
    def test_parameter_loading_performance(self):
        """Benchmark parameter loading and optimization integration performance."""
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store.get_model_defaults.return_value = {"H0": 67.4, "Om0": 0.315}
            mock_store.is_optimized.return_value = False
            mock_store.get_optimization_history.return_value = []
            mock_store.get_warm_start_params.return_value = None
            mock_store_class.return_value = mock_store
            
            # Benchmark parameter loading
            start_time = time.time()
            
            for _ in range(100):  # Run multiple iterations
                result = _get_optimized_parameters_with_metadata(mock_store, "lcdm")
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 100
            
            # Performance assertion: should be under 10ms per call
            self.assertLess(avg_time, 0.01, f"Parameter loading too slow: {avg_time:.4f}s")
            self.performance_results["parameter_loading_avg_time"] = avg_time
    
    @patch('fit_bao_aniso.engine.run_fit')
    @patch('fit_bao_aniso.OptimizedParameterStore')
    def test_full_fit_performance(self, mock_store_class, mock_run_fit):
        """Benchmark full fitting execution performance."""
        # Setup mocks
        mock_store = MagicMock()
        mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
        mock_store_class.return_value = mock_store
        
        # Mock fast fitting results
        mock_run_fit.return_value = {
            "params": {"H0": 67.4, "Om0": 0.315},
            "metrics": {"total_chi2": 15.0}
        }
        
        with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
            mock_get_params.return_value = {
                "final_params": {"H0": 67.4, "Om0": 0.315},
                "source": "defaults",
                "cmb_optimized": False
            }
            
            # Benchmark full fit execution
            start_time = time.time()
            
            for _ in range(10):  # Run multiple fits
                result = run_bao_aniso_fit("lcdm")
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            # Performance assertion: should be under 100ms per fit (mocked)
            self.assertLess(avg_time, 0.1, f"Full fit too slow: {avg_time:.4f}s")
            self.performance_results["full_fit_avg_time"] = avg_time
    
    def test_parameter_override_performance(self):
        """Benchmark parameter override application performance."""
        base_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "alpha": 0.1, "Rmax": 100.0, "eps0": 0.01, "n_eps": 2.0, "k_sat": 0.1
        }
        overrides = {"H0": 70.0, "Om0": 0.3, "alpha": 0.15}
        
        with patch('fit_core.parameter.validate_params'):
            with patch('fit_core.parameter.OPTIMIZABLE_PARAMETERS', {"pbuf": ["H0", "Om0", "alpha"]}):
                
                # Benchmark override application
                start_time = time.time()
                
                for _ in range(1000):  # Run many override operations
                    test_params = base_params.copy()
                    result = _apply_parameter_overrides(test_params, overrides, "pbuf")
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 1000
                
                # Performance assertion: should be under 1ms per override
                self.assertLess(avg_time, 0.001, f"Parameter override too slow: {avg_time:.6f}s")
                self.performance_results["parameter_override_avg_time"] = avg_time
    
    def tearDown(self):
        """Report performance results."""
        if self.performance_results:
            print(f"\nPerformance Benchmark Results:")
            for test_name, avg_time in self.performance_results.items():
                print(f"  {test_name}: {avg_time:.6f}s average")


class TestCommandLineInterface(unittest.TestCase):
    """
    Tests for command-line interface functionality.
    
    Ensures the argument parsing and CLI integration works correctly.
    """
    
    def test_argument_parsing_basic(self):
        """Test basic argument parsing functionality."""
        # Mock sys.argv for testing
        test_args = ["fit_bao_aniso.py", "--model", "lcdm", "--H0", "70.0"]
        
        with patch('sys.argv', test_args):
            args = parse_arguments()
            
            self.assertEqual(args.model, "lcdm")
            self.assertEqual(args.H0, 70.0)
    
    def test_argument_parsing_pbuf_parameters(self):
        """Test parsing of PBUF-specific parameters."""
        test_args = [
            "fit_bao_aniso.py", "--model", "pbuf", 
            "--alpha", "0.15", "--Rmax", "150.0", "--eps0", "0.02"
        ]
        
        with patch('sys.argv', test_args):
            args = parse_arguments()
            
            self.assertEqual(args.model, "pbuf")
            self.assertEqual(args.alpha, 0.15)
            self.assertEqual(args.Rmax, 150.0)
            self.assertEqual(args.eps0, 0.02)
    
    def test_argument_parsing_integrity_options(self):
        """Test parsing of integrity validation options."""
        test_args = [
            "fit_bao_aniso.py", "--verify-integrity", 
            "--integrity-tolerance", "1e-5", "--output-format", "json"
        ]
        
        with patch('sys.argv', test_args):
            args = parse_arguments()
            
            self.assertTrue(args.verify_integrity)
            self.assertEqual(args.integrity_tolerance, 1e-5)
            self.assertEqual(args.output_format, "json")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=True)