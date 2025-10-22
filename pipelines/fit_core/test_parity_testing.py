"""
Unit tests for the parity testing framework.

Tests the numerical comparison functions, tolerance checking, and report generation
capabilities of the parity testing system.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
import numpy as np

from .parity_testing import (
    ParityTester,
    ParityConfig,
    ComparisonResult,
    ParityReport,
    run_comprehensive_parity_suite
)


class TestParityTesting(unittest.TestCase):
    """Test cases for parity testing framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ParityConfig(
            tolerance=1e-6,
            relative_tolerance=1e-6,
            output_dir=self.temp_dir,
            verbose=False,
            save_intermediate=True
        )
        self.tester = ParityTester(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_initialization(self):
        """Test ParityConfig initialization and defaults."""
        # Test default config
        default_config = ParityConfig()
        self.assertEqual(default_config.tolerance, 1e-6)
        self.assertEqual(default_config.relative_tolerance, 1e-6)
        self.assertEqual(default_config.output_dir, "parity_results")
        self.assertTrue(default_config.verbose)
        self.assertTrue(default_config.save_intermediate)
        
        # Test custom config
        custom_config = ParityConfig(
            tolerance=1e-8,
            relative_tolerance=1e-8,
            output_dir="/tmp/test",
            verbose=False,
            save_intermediate=False
        )
        self.assertEqual(custom_config.tolerance, 1e-8)
        self.assertEqual(custom_config.relative_tolerance, 1e-8)
        self.assertEqual(custom_config.output_dir, "/tmp/test")
        self.assertFalse(custom_config.verbose)
        self.assertFalse(custom_config.save_intermediate)
    
    def test_tester_initialization(self):
        """Test ParityTester initialization."""
        # Test with custom config
        tester = ParityTester(self.config)
        self.assertEqual(tester.config, self.config)
        self.assertTrue(os.path.exists(self.temp_dir))
        
        # Test with default config
        default_tester = ParityTester()
        self.assertEqual(default_tester.config.tolerance, 1e-6)
    
    def test_compare_scalar_values_within_tolerance(self):
        """Test scalar comparison within tolerance."""
        # Test values within absolute tolerance
        result = self.tester._compare_scalar_values(
            metric_name="test_metric",
            legacy_value=1.0,
            unified_value=1.0 + 5e-7  # Within 1e-6 tolerance
        )
        
        self.assertEqual(result.metric_name, "test_metric")
        self.assertEqual(result.legacy_value, 1.0)
        self.assertEqual(result.unified_value, 1.0 + 5e-7)
        self.assertAlmostEqual(result.absolute_diff, 5e-7, places=10)
        self.assertTrue(result.passes_tolerance)
    
    def test_compare_scalar_values_outside_tolerance(self):
        """Test scalar comparison outside tolerance."""
        # Test values outside absolute tolerance
        result = self.tester._compare_scalar_values(
            metric_name="test_metric",
            legacy_value=1.0,
            unified_value=1.0 + 2e-6  # Outside 1e-6 tolerance
        )
        
        self.assertEqual(result.metric_name, "test_metric")
        self.assertEqual(result.legacy_value, 1.0)
        self.assertEqual(result.unified_value, 1.0 + 2e-6)
        self.assertAlmostEqual(result.absolute_diff, 2e-6, places=10)
        self.assertFalse(result.passes_tolerance)
    
    def test_compare_scalar_values_relative_tolerance(self):
        """Test scalar comparison with relative tolerance."""
        # Test large values where relative tolerance matters
        result = self.tester._compare_scalar_values(
            metric_name="test_metric",
            legacy_value=1000.0,
            unified_value=1000.0 + 5e-4  # 5e-7 relative difference
        )
        
        expected_rel_diff = 5e-4 / 1000.0  # 5e-7
        self.assertAlmostEqual(result.relative_diff, expected_rel_diff, places=10)
        self.assertTrue(result.passes_tolerance)  # Within 1e-6 relative tolerance
    
    def test_compare_scalar_values_zero_legacy(self):
        """Test scalar comparison when legacy value is zero."""
        result = self.tester._compare_scalar_values(
            metric_name="test_metric",
            legacy_value=0.0,
            unified_value=5e-7
        )
        
        self.assertEqual(result.legacy_value, 0.0)
        self.assertEqual(result.unified_value, 5e-7)
        self.assertEqual(result.absolute_diff, 5e-7)
        self.assertEqual(result.relative_diff, 5e-7)  # Uses absolute diff when legacy is zero
        self.assertTrue(result.passes_tolerance)
    
    def test_compare_metrics(self):
        """Test comparison of overall metrics."""
        legacy_results = {
            "metrics": {
                "total_chi2": 10.5,
                "aic": 14.5,
                "bic": 18.2,
                "dof": 5,
                "p_value": 0.062
            }
        }
        
        unified_results = {
            "metrics": {
                "total_chi2": 10.5 + 1e-7,  # Small difference
                "aic": 14.5 + 1e-7,
                "bic": 18.2 + 1e-7,
                "dof": 5,
                "p_value": 0.062 + 1e-8
            }
        }
        
        comparisons = self.tester._compare_metrics(legacy_results, unified_results)
        
        # Should have 5 comparisons
        self.assertEqual(len(comparisons), 5)
        
        # All should pass tolerance
        for comp in comparisons:
            self.assertTrue(comp.passes_tolerance)
        
        # Check metric names
        metric_names = [comp.metric_name for comp in comparisons]
        expected_names = ["metrics.total_chi2", "metrics.aic", "metrics.bic", 
                         "metrics.dof", "metrics.p_value"]
        for name in expected_names:
            self.assertIn(name, metric_names)
    
    def test_compare_parameters(self):
        """Test comparison of optimized parameters."""
        legacy_results = {
            "params": {
                "H0": 67.4,
                "Om0": 0.315,
                "Obh2": 0.02237,
                "alpha": 5e-4,
                "Rmax": 1e9
            }
        }
        
        unified_results = {
            "params": {
                "H0": 67.4 + 1e-7,
                "Om0": 0.315 + 1e-8,
                "Obh2": 0.02237 + 1e-9,
                "alpha": 5e-4 + 1e-10,
                "Rmax": 1e9 + 1e-3  # Larger difference but still within relative tolerance
            }
        }
        
        comparisons = self.tester._compare_parameters(legacy_results, unified_results)
        
        # Should have 5 comparisons
        self.assertEqual(len(comparisons), 5)
        
        # All should pass tolerance
        for comp in comparisons:
            self.assertTrue(comp.passes_tolerance)
    
    def test_compare_dataset_results(self):
        """Test comparison of per-dataset results."""
        legacy_results = {
            "results": {
                "cmb": {"chi2": 1.79},
                "bao": {"chi2": 3.21},
                "sn": {"chi2": 5.43}
            }
        }
        
        unified_results = {
            "results": {
                "cmb": {"chi2": 1.79 + 1e-7},
                "bao": {"chi2": 3.21 + 1e-7},
                "sn": {"chi2": 5.43 + 1e-7}
            }
        }
        
        comparisons = self.tester._compare_dataset_results(legacy_results, unified_results)
        
        # Should have 3 comparisons
        self.assertEqual(len(comparisons), 3)
        
        # All should pass tolerance
        for comp in comparisons:
            self.assertTrue(comp.passes_tolerance)
        
        # Check dataset names
        dataset_names = [comp.metric_name.split('.')[1] for comp in comparisons]
        for dataset in ["cmb", "bao", "sn"]:
            self.assertIn(dataset, dataset_names)
    
    def test_compare_predictions(self):
        """Test comparison of theoretical predictions."""
        legacy_results = {
            "results": {
                "cmb": {
                    "predictions": {
                        "R": 1.7502,
                        "l_A": 301.845,
                        "theta_star": 1.04092
                    }
                }
            }
        }
        
        unified_results = {
            "results": {
                "cmb": {
                    "predictions": {
                        "R": 1.7502 + 1e-7,
                        "l_A": 301.845 + 1e-6,
                        "theta_star": 1.04092 + 1e-8
                    }
                }
            }
        }
        
        comparisons = self.tester._compare_predictions(legacy_results, unified_results)
        
        # Should have 3 comparisons
        self.assertEqual(len(comparisons), 3)
        
        # All should pass tolerance
        for comp in comparisons:
            self.assertTrue(comp.passes_tolerance)
    
    @patch('pipelines.fit_core.parity_testing.engine.run_fit')
    def test_run_unified_system(self, mock_run_fit):
        """Test execution of unified system."""
        # Mock the engine response
        mock_results = {
            "params": {"H0": 67.4, "Om0": 0.315},
            "metrics": {"total_chi2": 10.5, "aic": 14.5},
            "results": {"cmb": {"chi2": 1.79}}
        }
        mock_run_fit.return_value = mock_results
        
        # Run unified system
        results = self.tester.run_unified_system("pbuf", ["cmb"])
        
        # Verify engine was called correctly (parity testing adds BAO for sufficient DOF)
        mock_run_fit.assert_called_once_with(
            model="pbuf",
            datasets_list=["cmb", "bao"],
            mode="joint",
            overrides=None
        )
        
        # Verify results include execution time
        self.assertIn("_execution_time", results)
        self.assertIsInstance(results["_execution_time"], float)
        
        # Verify original results are preserved
        self.assertEqual(results["params"], mock_results["params"])
        self.assertEqual(results["metrics"], mock_results["metrics"])
    
    def test_generate_mock_legacy_results(self):
        """Test generation of mock legacy results."""
        with patch.object(self.tester, 'run_unified_system') as mock_unified:
            mock_unified_results = {
                "params": {"H0": 67.4},
                "metrics": {"total_chi2": 10.5, "aic": 14.5},
                "_execution_time": 1.0
            }
            mock_unified.return_value = mock_unified_results
            
            # Generate mock legacy results
            mock_results = self.tester._generate_mock_legacy_results("pbuf", ["cmb"])
            
            # Verify structure is preserved
            self.assertIn("params", mock_results)
            self.assertIn("metrics", mock_results)
            self.assertIn("_execution_time", mock_results)
            
            # Verify small perturbations were added
            self.assertNotEqual(
                mock_results["metrics"]["total_chi2"],
                mock_unified_results["metrics"]["total_chi2"]
            )
            
            # Verify perturbations are small
            chi2_diff = abs(
                float(mock_results["metrics"]["total_chi2"]) - 
                mock_unified_results["metrics"]["total_chi2"]
            )
            self.assertLess(chi2_diff, 1e-6)
    
    @patch.object(ParityTester, 'run_legacy_system')
    @patch.object(ParityTester, 'run_unified_system')
    def test_run_parity_test(self, mock_unified, mock_legacy):
        """Test complete parity test execution."""
        # Mock system results
        legacy_results = {
            "params": {"H0": 67.4},
            "metrics": {"total_chi2": 10.5},
            "_execution_time": 0.5
        }
        
        unified_results = {
            "params": {"H0": 67.4 + 1e-7},
            "metrics": {"total_chi2": 10.5 + 1e-7},
            "_execution_time": 1.0
        }
        
        mock_legacy.return_value = legacy_results
        mock_unified.return_value = unified_results
        
        # Run parity test
        report = self.tester.run_parity_test(
            test_name="test_cmb",
            model="pbuf",
            datasets=["cmb"]
        )
        
        # Verify report structure
        self.assertIsInstance(report, ParityReport)
        self.assertEqual(report.test_name, "test_cmb")
        self.assertEqual(report.model, "pbuf")
        self.assertEqual(report.datasets, ["cmb"])
        self.assertTrue(report.overall_pass)  # Small differences should pass
        self.assertEqual(report.execution_time_legacy, 0.5)
        self.assertEqual(report.execution_time_unified, 1.0)
        
        # Verify comparisons were made
        self.assertGreater(len(report.comparisons), 0)
    
    def test_generate_parity_report(self):
        """Test parity report generation."""
        # Create a sample report
        comparisons = [
            ComparisonResult(
                metric_name="metrics.total_chi2",
                legacy_value=10.5,
                unified_value=10.5 + 1e-7,
                absolute_diff=1e-7,
                relative_diff=1e-8,
                passes_tolerance=True,
                tolerance_used=1e-6
            ),
            ComparisonResult(
                metric_name="params.H0",
                legacy_value=67.4,
                unified_value=67.4 + 2e-6,  # Fails tolerance
                absolute_diff=2e-6,
                relative_diff=3e-8,
                passes_tolerance=False,
                tolerance_used=1e-6
            )
        ]
        
        report = ParityReport(
            test_name="test_report",
            model="pbuf",
            datasets=["cmb"],
            parameters={},
            comparisons=comparisons,
            overall_pass=False,  # One comparison failed
            execution_time_legacy=0.5,
            execution_time_unified=1.0,
            timestamp="2023-01-01T12:00:00",
            config=self.config
        )
        
        # Generate report text
        report_text = self.tester.generate_parity_report(report)
        
        # Verify key elements are present
        self.assertIn("PARITY TEST REPORT: test_report", report_text)
        self.assertIn("Model: pbuf", report_text)
        self.assertIn("Overall Result: FAIL", report_text)
        self.assertIn("Legacy System:  0.500s", report_text)
        self.assertIn("Unified System: 1.000s", report_text)
        self.assertIn("FAILED COMPARISONS:", report_text)
        self.assertIn("params.H0", report_text)
    
    def test_save_report(self):
        """Test saving parity reports to files."""
        # Create a minimal report
        report = ParityReport(
            test_name="test_save",
            model="pbuf",
            datasets=["cmb"],
            parameters={},
            comparisons=[],
            overall_pass=True,
            execution_time_legacy=0.5,
            execution_time_unified=1.0,
            timestamp="2023-01-01T12:00:00",
            config=self.config
        )
        
        # Save report
        report_path = self.tester.save_report(report)
        
        # Verify files were created
        self.assertTrue(os.path.exists(report_path))
        self.assertTrue(report_path.endswith(".txt"))
        
        json_path = report_path.replace(".txt", ".json")
        self.assertTrue(os.path.exists(json_path))
        
        # Verify content
        with open(report_path, "r") as f:
            content = f.read()
            self.assertIn("test_save", content)
        
        with open(json_path, "r") as f:
            json_data = json.load(f)
            self.assertEqual(json_data["test_name"], "test_save")
    
    @patch('pipelines.fit_core.parity_testing.ParityTester.run_parity_test')
    def test_comprehensive_parity_suite(self, mock_run_test):
        """Test comprehensive parity test suite."""
        # Mock individual test results
        mock_report = ParityReport(
            test_name="mock_test",
            model="pbuf",
            datasets=["cmb"],
            parameters={},
            comparisons=[],
            overall_pass=True,
            execution_time_legacy=0.5,
            execution_time_unified=1.0,
            timestamp="2023-01-01T12:00:00",
            config=self.config
        )
        mock_run_test.return_value = mock_report
        
        # Run comprehensive suite
        reports = run_comprehensive_parity_suite(
            config=self.config,
            models=["pbuf"],
            dataset_combinations=[["cmb"], ["bao"]]
        )
        
        # Verify correct number of tests were run
        self.assertEqual(len(reports), 2)  # 1 model × 2 dataset combinations
        
        # Verify test names
        expected_calls = 2  # pbuf_cmb, pbuf_bao
        self.assertEqual(mock_run_test.call_count, expected_calls)


if __name__ == "__main__":
    unittest.main()