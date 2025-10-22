"""
Unit tests for logging_utils module.

Tests standardized log output format, physics consistency checks,
diagnostic computations, and error handling.
"""

import unittest
from unittest.mock import patch, MagicMock, call
import logging
import io
import sys
import numpy as np
from typing import Dict, Any

from . import logging_utils
from . import ParameterDict, ResultsDict, MetricsDict, PredictionsDict


class TestLogRun(unittest.TestCase):
    """Test log_run function for standardized run logging."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger_patcher = patch('pipelines.fit_core.logging_utils.logging.getLogger')
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger
        
        # Sample test data
        self.sample_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "alpha": 5e-4,
            "Rmax": 1e9
        }
        
        self.sample_metrics = {
            "total_chi2": 15.67,
            "aic": 23.67,
            "bic": 31.45,
            "dof": 12,
            "reduced_chi2": 1.306,
            "p_value": 0.205
        }
        
        self.sample_results = {
            "params": self.sample_params,
            "datasets": ["cmb", "bao"],
            "chi2_breakdown": {"cmb": 8.23, "bao": 7.44},
            "results": {
                "cmb": {
                    "predictions": {
                        "z_recomb": 1089.9,
                        "r_s": 144.3,
                        "l_A": 301.845,
                        "theta_star": 1.04092
                    }
                },
                "bao": {
                    "predictions": {
                        "z_drag": 1059.7,
                        "r_s_drag": 147.2
                    }
                }
            },
            "metrics": self.sample_metrics
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.logger_patcher.stop()
    
    def test_log_run_single_dataset(self):
        """Test logging for single dataset fit."""
        results = {
            "params": self.sample_params,
            "datasets": ["cmb"],
            "chi2_breakdown": {"cmb": 8.23},
            "results": {
                "cmb": {
                    "predictions": {
                        "z_recomb": 1089.9,
                        "r_s": 144.3
                    }
                }
            }
        }
        
        logging_utils.log_run("pbuf", "individual", results, self.sample_metrics)
        
        # Verify main run log was called
        self.mock_logger.info.assert_any_call(
            "[RUN] model=pbuf block=cmb χ²=15.67 AIC=23.67 params={H0:67.4, Om0:0.315, Obh2:0.02237, ns:0.9649, alpha:0.0005, Rmax:1e+09}"
        )
        
        # Verify metrics log was called
        metrics_calls = [call for call in self.mock_logger.info.call_args_list 
                        if "[METRICS]" in str(call)]
        self.assertTrue(len(metrics_calls) > 0)
        
        # Verify predictions log was called
        pred_calls = [call for call in self.mock_logger.info.call_args_list 
                     if "[PRED]" in str(call)]
        self.assertTrue(len(pred_calls) > 0)
    
    def test_log_run_multiple_datasets(self):
        """Test logging for joint dataset fit."""
        logging_utils.log_run("lcdm", "joint", self.sample_results, self.sample_metrics)
        
        # Verify joint dataset naming
        self.mock_logger.info.assert_any_call(
            "[RUN] model=lcdm block=cmb+bao χ²=15.67 AIC=23.67 params={H0:67.4, Om0:0.315, Obh2:0.02237, ns:0.9649, alpha:0.0005, Rmax:1e+09}"
        )
        
        # Verify χ² breakdown was logged
        chi2_calls = [call for call in self.mock_logger.info.call_args_list 
                     if "[CHI2]" in str(call)]
        self.assertTrue(len(chi2_calls) > 0)
    
    def test_log_run_empty_datasets(self):
        """Test logging with empty datasets list."""
        results = {
            "params": self.sample_params,
            "datasets": [],
            "chi2_breakdown": {},
            "results": {}
        }
        
        logging_utils.log_run("pbuf", "test", results, self.sample_metrics)
        
        # Should handle empty datasets gracefully
        run_calls = [call for call in self.mock_logger.info.call_args_list 
                    if "[RUN]" in str(call)]
        self.assertTrue(len(run_calls) > 0)
        
        # Check that "none" is used for empty datasets
        run_call_str = str(run_calls[0])
        self.assertIn("block=none", run_call_str)
    
    def test_log_run_missing_data(self):
        """Test logging with missing data fields."""
        minimal_results = {"params": {}}
        minimal_metrics = {}
        
        # Should not raise exception
        logging_utils.log_run("lcdm", "test", minimal_results, minimal_metrics)
        
        # Verify some logging occurred
        self.assertTrue(self.mock_logger.info.called)


class TestLogDiagnostics(unittest.TestCase):
    """Test log_diagnostics function for physics consistency checks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger_patcher = patch('pipelines.fit_core.logging_utils.logging.getLogger')
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger
        
        self.sample_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Tcmb": 2.7255,
            "Neff": 3.046,
            "z_recomb": 1089.80,
            "alpha": 5e-4  # PBUF model
        }
        
        self.sample_predictions = {
            "covariance_matrix": np.eye(3),
            "eigenvalues": [1.0, 2.0, 3.0]
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.logger_patcher.stop()
    
    def test_log_diagnostics_pbuf_model(self):
        """Test diagnostics logging for PBUF model."""
        logging_utils.log_diagnostics(self.sample_params, self.sample_predictions)
        
        # Verify CHECK log was called
        check_calls = [call for call in self.mock_logger.info.call_args_list 
                      if "[CHECK]" in str(call)]
        self.assertTrue(len(check_calls) > 0)
        
        # Verify H_ratio is present (should show PBUF modifications)
        check_str = str(check_calls[0])
        self.assertIn("H_ratio=", check_str)
        self.assertIn("recomb_check=", check_str)
        self.assertIn("covariance=", check_str)
        self.assertIn("constants=", check_str)
    
    def test_log_diagnostics_lcdm_model(self):
        """Test diagnostics logging for ΛCDM model."""
        lcdm_params = self.sample_params.copy()
        del lcdm_params["alpha"]  # Remove PBUF parameter
        
        logging_utils.log_diagnostics(lcdm_params, self.sample_predictions)
        
        # Verify CHECK log was called
        check_calls = [call for call in self.mock_logger.info.call_args_list 
                      if "[CHECK]" in str(call)]
        self.assertTrue(len(check_calls) > 0)
        
        # For ΛCDM, H_ratio should be [1.000,1.000,1.000]
        check_str = str(check_calls[0])
        self.assertIn("H_ratio=[1.000,1.000,1.000]", check_str)
    
    def test_log_diagnostics_missing_recombination(self):
        """Test diagnostics with missing recombination redshift."""
        params_no_recomb = self.sample_params.copy()
        del params_no_recomb["z_recomb"]
        
        logging_utils.log_diagnostics(params_no_recomb, self.sample_predictions)
        
        # Should handle missing z_recomb gracefully
        check_calls = [call for call in self.mock_logger.info.call_args_list 
                      if "[CHECK]" in str(call)]
        self.assertTrue(len(check_calls) > 0)
        
        check_str = str(check_calls[0])
        self.assertIn("recomb_check=MISSING", check_str)
    
    def test_log_diagnostics_empty_predictions(self):
        """Test diagnostics with empty predictions."""
        logging_utils.log_diagnostics(self.sample_params, {})
        
        # Should not raise exception
        self.assertTrue(self.mock_logger.info.called)


class TestFormatResultsTable(unittest.TestCase):
    """Test format_results_table function for human-readable output."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_results = {
            "model": "pbuf",
            "datasets": ["cmb", "bao"],
            "params": {
                "H0": 67.4,
                "Om0": 0.315,
                "Obh2": 0.02237,
                "ns": 0.9649,
                "alpha": 5e-4,
                "Rmax": 1e9
            },
            "metrics": {
                "total_chi2": 15.67,
                "aic": 23.67,
                "bic": 31.45,
                "dof": 12,
                "reduced_chi2": 1.306,
                "p_value": 0.205
            },
            "chi2_breakdown": {
                "cmb": 8.23,
                "bao": 7.44
            },
            "results": {
                "cmb": {
                    "predictions": {
                        "z_recomb": 1089.9,
                        "r_s": 144.3,
                        "l_A": 301.845,
                        "theta_star": 1.04092
                    }
                },
                "bao": {
                    "predictions": {
                        "z_drag": 1059.7,
                        "r_s_drag": 147.2
                    }
                }
            }
        }
    
    def test_format_results_table_complete(self):
        """Test complete results table formatting."""
        table = logging_utils.format_results_table(self.sample_results)
        
        # Check header section
        self.assertIn("PBUF Cosmology Fit Results - Model: PBUF", table)
        self.assertIn("Datasets: cmb, bao", table)
        
        # Check parameters section
        self.assertIn("Fitted Parameters:", table)
        self.assertIn("H0           =    67.400000", table)
        self.assertIn("Om0          =     0.315000", table)
        self.assertIn("alpha        =     0.000500", table)
        
        # Check metrics section
        self.assertIn("Fit Quality Metrics:", table)
        self.assertIn("Total χ²     =       15.670", table)
        self.assertIn("DOF          =           12", table)
        self.assertIn("AIC          =       23.670", table)
        
        # Check χ² breakdown section
        self.assertIn("χ² Breakdown by Dataset:", table)
        self.assertIn("cmb          =        8.230", table)
        self.assertIn("bao          =        7.440", table)
        
        # Check predictions section
        self.assertIn("Key Predictions:", table)
        self.assertIn("CMB:", table)
        self.assertIn("z_recomb   =   1089.900", table)
        self.assertIn("BAO:", table)
        self.assertIn("z_drag     =   1059.700", table)
    
    def test_format_results_table_minimal(self):
        """Test results table with minimal data."""
        minimal_results = {
            "model": "lcdm",
            "datasets": ["cmb"],
            "params": {"H0": 67.4},
            "metrics": {"total_chi2": 5.0}
        }
        
        table = logging_utils.format_results_table(minimal_results)
        
        # Should handle minimal data gracefully
        self.assertIn("Model: LCDM", table)
        self.assertIn("Datasets: cmb", table)
        self.assertIn("H0           =    67.400000", table)
        self.assertIn("Total χ²     =        5.000", table)
    
    def test_format_results_table_supernova_predictions(self):
        """Test results table with supernova predictions (array case)."""
        sn_results = {
            "model": "pbuf",
            "datasets": ["sn"],
            "params": {"H0": 67.4},
            "metrics": {"total_chi2": 10.0},
            "results": {
                "sn": {
                    "predictions": {
                        "mu_theory": [35.1, 36.2, 37.3, 38.4, 39.5]
                    }
                }
            }
        }
        
        table = logging_utils.format_results_table(sn_results)
        
        # Should handle array predictions
        self.assertIn("SN:", table)
        # Mean of [35.1, 36.2, 37.3, 38.4, 39.5] = 37.3
        self.assertIn("mu_mean    =     37.300", table)
    
    def test_format_results_table_empty_results(self):
        """Test results table with empty results."""
        empty_results = {}
        
        table = logging_utils.format_results_table(empty_results)
        
        # Should handle empty results gracefully
        self.assertIn("Model: UNKNOWN", table)
        self.assertIn("Datasets:", table)


class TestSetupLogging(unittest.TestCase):
    """Test setup_logging function for logging configuration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Store original handlers to restore later
        self.original_handlers = logging.getLogger().handlers[:]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in self.original_handlers:
            root_logger.addHandler(handler)
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        logging_utils.setup_logging()
        
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)
        
        # Should have at least one handler (console)
        self.assertTrue(len(root_logger.handlers) >= 1)
        
        # Check that console handler is present
        console_handlers = [h for h in root_logger.handlers 
                          if isinstance(h, logging.StreamHandler)]
        self.assertTrue(len(console_handlers) > 0)
    
    def test_setup_logging_debug_level(self):
        """Test logging setup with DEBUG level."""
        logging_utils.setup_logging(level="DEBUG")
        
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.DEBUG)
    
    def test_setup_logging_invalid_level(self):
        """Test logging setup with invalid level (should default to INFO)."""
        logging_utils.setup_logging(level="INVALID")
        
        root_logger = logging.getLogger()
        self.assertEqual(root_logger.level, logging.INFO)
    
    @patch('logging.FileHandler')
    def test_setup_logging_with_file(self, mock_file_handler):
        """Test logging setup with file output."""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler
        
        logging_utils.setup_logging(log_file="test.log")
        
        # Verify file handler was created
        mock_file_handler.assert_called_once_with("test.log")
        mock_handler.setLevel.assert_called()
        mock_handler.setFormatter.assert_called()


class TestPrivateHelperFunctions(unittest.TestCase):
    """Test private helper functions for internal logic."""
    
    def test_format_parameter_summary(self):
        """Test parameter summary formatting."""
        params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "alpha": 5e-4,
            "Rmax": 1e9,
            "extra_param": "ignored"
        }
        
        summary = logging_utils._format_parameter_summary(params)
        
        # Should include key parameters in order
        self.assertIn("H0:67.4", summary)
        self.assertIn("Om0:0.315", summary)
        self.assertIn("alpha:0.0005", summary)
        self.assertIn("Rmax:1e+09", summary)
        
        # Should be properly formatted as dict-like string
        self.assertTrue(summary.startswith("{"))
        self.assertTrue(summary.endswith("}"))
    
    def test_format_parameter_summary_partial(self):
        """Test parameter summary with only some parameters."""
        params = {"H0": 67.4, "Om0": 0.315}
        
        summary = logging_utils._format_parameter_summary(params)
        
        self.assertIn("H0:67.4", summary)
        self.assertIn("Om0:0.315", summary)
        # Should not include missing parameters
        self.assertNotIn("alpha", summary)
    
    def test_format_metrics_summary(self):
        """Test metrics summary formatting."""
        metrics = {
            "dof": 12,
            "reduced_chi2": 1.306,
            "p_value": 0.205,
            "bic": 31.45,
            "extra_metric": "ignored"
        }
        
        summary = logging_utils._format_metrics_summary(metrics)
        
        # Should include key metrics
        self.assertIn("dof=12", summary)
        self.assertIn("reduced_chi2=1.306", summary)
        self.assertIn("p=0.2050", summary)  # p_value formatted as p=
        self.assertIn("bic=31.450", summary)
    
    def test_verify_h_ratios_lcdm(self):
        """Test H(z) ratio verification for ΛCDM model."""
        lcdm_params = {"H0": 67.4, "Om0": 0.315}  # No alpha parameter
        
        ratios = logging_utils._verify_h_ratios(lcdm_params)
        
        # ΛCDM should return ratios of 1.0
        self.assertEqual(len(ratios), 3)  # Default test redshifts
        for ratio in ratios:
            self.assertAlmostEqual(ratio, 1.0, places=3)
    
    def test_verify_h_ratios_pbuf(self):
        """Test H(z) ratio verification for PBUF model."""
        pbuf_params = {"H0": 67.4, "Om0": 0.315, "alpha": 5e-4}
        
        ratios = logging_utils._verify_h_ratios(pbuf_params)
        
        # PBUF should return modified ratios
        self.assertEqual(len(ratios), 3)
        for i, ratio in enumerate(ratios):
            # Should show small deviations from 1.0
            self.assertNotEqual(ratio, 1.0)
            self.assertTrue(0.99 < ratio < 1.01)  # Small deviations
    
    def test_verify_h_ratios_custom_redshifts(self):
        """Test H(z) ratio verification with custom redshifts."""
        params = {"alpha": 5e-4}
        custom_z = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        ratios = logging_utils._verify_h_ratios(params, custom_z)
        
        self.assertEqual(len(ratios), len(custom_z))
    
    def test_verify_recombination_pass(self):
        """Test recombination verification with good value."""
        params = {"z_recomb": 1089.80}  # Exact Planck 2018 value
        
        status = logging_utils._verify_recombination(params)
        
        self.assertEqual(status, "PASS")
    
    def test_verify_recombination_warn(self):
        """Test recombination verification with warning-level deviation."""
        params = {"z_recomb": 1090.5}  # Small deviation
        
        status = logging_utils._verify_recombination(params)
        
        self.assertEqual(status, "WARN")
    
    def test_verify_recombination_fail(self):
        """Test recombination verification with large deviation."""
        params = {"z_recomb": 1100.0}  # Large deviation
        
        status = logging_utils._verify_recombination(params)
        
        self.assertEqual(status, "FAIL")
    
    def test_verify_recombination_missing(self):
        """Test recombination verification with missing value."""
        params = {}  # No z_recomb
        
        status = logging_utils._verify_recombination(params)
        
        self.assertEqual(status, "MISSING")
    
    def test_verify_physical_constants_pass(self):
        """Test physical constants verification with standard values."""
        params = {"Tcmb": 2.7255, "Neff": 3.046}
        
        status = logging_utils._verify_physical_constants(params)
        
        self.assertEqual(status, "PASS")
    
    def test_verify_physical_constants_warn(self):
        """Test physical constants verification with non-standard values."""
        params = {"Tcmb": 2.8, "Neff": 3.1}  # Deviations
        
        status = logging_utils._verify_physical_constants(params)
        
        self.assertEqual(status, "WARN")
    
    def test_verify_physical_constants_missing(self):
        """Test physical constants verification with missing values."""
        params = {}  # Use defaults
        
        status = logging_utils._verify_physical_constants(params)
        
        self.assertEqual(status, "PASS")  # Should use defaults
    
    def test_format_predictions_summary_cmb(self):
        """Test predictions summary formatting for CMB."""
        predictions = {
            "z_recomb": 1089.9,
            "r_s": 144.3,
            "l_A": 301.845,
            "theta_star": 1.04092
        }
        
        summary = logging_utils._format_predictions_summary("cmb", predictions)
        
        self.assertIn("cmb:", summary)
        self.assertIn("z_recomb=1089.900", summary)
        self.assertIn("r_s=144.300", summary)
        self.assertIn("l_A=301.845", summary)
        self.assertIn("theta_star=1.041", summary)
    
    def test_format_predictions_summary_bao(self):
        """Test predictions summary formatting for BAO."""
        predictions = {
            "z_drag": 1059.7,
            "r_s_drag": 147.2
        }
        
        summary = logging_utils._format_predictions_summary("bao", predictions)
        
        self.assertIn("bao:", summary)
        self.assertIn("z_drag=1059.700", summary)
        self.assertIn("r_s_drag=147.200", summary)
    
    def test_format_predictions_summary_sn_array(self):
        """Test predictions summary formatting for supernova with array."""
        predictions = {
            "mu_theory": [35.1, 36.2, 37.3, 38.4, 39.5]
        }
        
        summary = logging_utils._format_predictions_summary("sn", predictions)
        
        self.assertIn("sn:", summary)
        self.assertIn("mu_theory_mean=37.300", summary)
    
    def test_format_predictions_summary_unknown_dataset(self):
        """Test predictions summary formatting for unknown dataset."""
        predictions = {
            "param1": 1.0,
            "param2": 2.0,
            "param3": 3.0,
            "param4": 4.0,
            "param5": 5.0,
            "param6": 6.0  # Should be truncated
        }
        
        summary = logging_utils._format_predictions_summary("unknown", predictions)
        
        self.assertIn("unknown:", summary)
        # Should show first 5 parameters
        self.assertIn("param1=1.000", summary)
        self.assertIn("param5=5.000", summary)
        # Should not show param6 (truncated)
        self.assertNotIn("param6", summary)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases in diagnostic functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger_patcher = patch('pipelines.fit_core.logging_utils.logging.getLogger')
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.logger_patcher.stop()
    
    def test_log_run_with_none_values(self):
        """Test log_run with None values in results."""
        results = {
            "params": None,
            "datasets": None,
            "chi2_breakdown": None,
            "results": None
        }
        metrics = None
        
        # Should not raise exception
        logging_utils.log_run("pbuf", "test", results, metrics)
        
        # Verify some logging occurred
        self.assertTrue(self.mock_logger.info.called)
    
    def test_log_diagnostics_with_invalid_params(self):
        """Test log_diagnostics with invalid parameter types."""
        invalid_params = {
            "H0": "invalid",  # String instead of float
            "z_recomb": None,
            "Tcmb": float('inf'),
            "Neff": float('nan')
        }
        
        # Should handle invalid values gracefully
        logging_utils.log_diagnostics(invalid_params, {})
        
        # Verify some logging occurred
        self.assertTrue(self.mock_logger.info.called)
        
        # Check that invalid recombination is handled
        check_calls = [call for call in self.mock_logger.info.call_args_list 
                      if "[CHECK]" in str(call)]
        self.assertTrue(len(check_calls) > 0)
        
        check_str = str(check_calls[0])
        # Should show MISSING or INVALID for bad z_recomb
        self.assertTrue("recomb_check=MISSING" in check_str or "recomb_check=INVALID" in check_str)
    
    def test_format_results_table_with_invalid_data(self):
        """Test format_results_table with invalid data types."""
        invalid_results = {
            "model": None,
            "datasets": "not_a_list",
            "params": {"H0": float('nan')},
            "metrics": {"total_chi2": "invalid"}
        }
        
        # Should handle invalid data gracefully
        table = logging_utils.format_results_table(invalid_results)
        self.assertIsInstance(table, str)
        # Should contain some expected content even with invalid data
        self.assertIn("Model: UNKNOWN", table)  # None gets converted to "unknown"
        self.assertIn("Datasets: not_a_list", table)
    
    def test_verify_h_ratios_with_empty_params(self):
        """Test _verify_h_ratios with empty parameters."""
        ratios = logging_utils._verify_h_ratios({})
        
        # Should return ΛCDM ratios (1.0) for empty params
        self.assertEqual(len(ratios), 3)
        for ratio in ratios:
            self.assertEqual(ratio, 1.0)
    
    def test_format_predictions_summary_with_invalid_array(self):
        """Test _format_predictions_summary with invalid array data."""
        predictions = {
            "mu_theory": [1, 2, "invalid", 4, 5]  # Mixed types
        }
        
        # Should handle mixed types gracefully
        try:
            summary = logging_utils._format_predictions_summary("sn", predictions)
            self.assertIsInstance(summary, str)
        except Exception as e:
            self.fail(f"_format_predictions_summary raised exception: {e}")
    
    def test_format_predictions_summary_with_empty_array(self):
        """Test _format_predictions_summary with empty array."""
        predictions = {
            "mu_theory": []  # Empty array
        }
        
        summary = logging_utils._format_predictions_summary("sn", predictions)
        
        # Should handle empty array gracefully
        self.assertIn("sn:", summary)
        self.assertIn("array[0]", summary)


if __name__ == '__main__':
    unittest.main()