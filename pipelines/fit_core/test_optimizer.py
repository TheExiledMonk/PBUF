"""
Unit tests for parameter optimization engine.

Tests parameter bounds validation, optimization convergence with mock objectives,
and error handling for invalid parameter requests.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from . import optimizer
from . import parameter


class TestParameterOptimizer(unittest.TestCase):
    """Test cases for ParameterOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = optimizer.ParameterOptimizer()
        
    def test_get_optimization_bounds_lcdm(self):
        """Test parameter bounds retrieval for ΛCDM model."""
        # Test valid ΛCDM parameters
        bounds = self.optimizer.get_optimization_bounds("lcdm", "H0")
        self.assertEqual(bounds, (20.0, 150.0))
        
        bounds = self.optimizer.get_optimization_bounds("lcdm", "Om0")
        self.assertEqual(bounds, (0.01, 0.99))
        
        bounds = self.optimizer.get_optimization_bounds("lcdm", "Obh2")
        self.assertEqual(bounds, (0.005, 0.1))
        
        bounds = self.optimizer.get_optimization_bounds("lcdm", "ns")
        self.assertEqual(bounds, (0.5, 1.5))
        
    def test_get_optimization_bounds_pbuf(self):
        """Test parameter bounds retrieval for PBUF model."""
        # Test PBUF-specific parameters
        bounds = self.optimizer.get_optimization_bounds("pbuf", "alpha")
        self.assertEqual(bounds, (1e-6, 1e-2))
        
        bounds = self.optimizer.get_optimization_bounds("pbuf", "k_sat")
        self.assertEqual(bounds, (0.1, 2.0))
        
        bounds = self.optimizer.get_optimization_bounds("pbuf", "Rmax")
        self.assertEqual(bounds, (1e6, 1e12))
        
        # Test that ΛCDM parameters also work for PBUF
        bounds = self.optimizer.get_optimization_bounds("pbuf", "H0")
        self.assertEqual(bounds, (20.0, 150.0))
        
    def test_get_optimization_bounds_invalid_model(self):
        """Test error handling for invalid model types."""
        with self.assertRaises(ValueError) as context:
            self.optimizer.get_optimization_bounds("invalid_model", "H0")
        
        self.assertIn("Unknown model type", str(context.exception))
        
    def test_get_optimization_bounds_invalid_parameter(self):
        """Test error handling for invalid parameter names."""
        # Test parameter not optimizable for ΛCDM
        with self.assertRaises(ValueError) as context:
            self.optimizer.get_optimization_bounds("lcdm", "alpha")
        
        self.assertIn("not optimizable for lcdm model", str(context.exception))
        
        # Test completely invalid parameter
        with self.assertRaises(ValueError) as context:
            self.optimizer.get_optimization_bounds("pbuf", "invalid_param")
        
        self.assertIn("not optimizable for pbuf model", str(context.exception))
        
    def test_get_optimization_bounds_dict(self):
        """Test bounds retrieval for multiple parameters."""
        params = ["H0", "Om0", "Obh2"]
        bounds_dict = self.optimizer.get_optimization_bounds_dict("lcdm", params)
        
        expected = {
            "H0": (20.0, 150.0),
            "Om0": (0.01, 0.99),
            "Obh2": (0.005, 0.1)
        }
        
        self.assertEqual(bounds_dict, expected)
        
    def test_validate_optimization_request_valid(self):
        """Test validation of valid optimization requests."""
        # Valid ΛCDM parameters
        result = self.optimizer.validate_optimization_request("lcdm", ["H0", "Om0"])
        self.assertTrue(result)
        
        # Valid PBUF parameters
        result = self.optimizer.validate_optimization_request("pbuf", ["H0", "alpha", "k_sat"])
        self.assertTrue(result)
        
    def test_validate_optimization_request_invalid_model(self):
        """Test validation with invalid model type."""
        with self.assertRaises(ValueError) as context:
            self.optimizer.validate_optimization_request("invalid", ["H0"])
        
        self.assertIn("Unknown model type", str(context.exception))
        
    def test_validate_optimization_request_invalid_parameters(self):
        """Test validation with invalid parameter names."""
        # PBUF-specific parameter for ΛCDM model
        with self.assertRaises(ValueError) as context:
            self.optimizer.validate_optimization_request("lcdm", ["H0", "alpha"])
        
        self.assertIn("Invalid optimization parameters", str(context.exception))
        
        # Completely invalid parameter
        with self.assertRaises(ValueError) as context:
            self.optimizer.validate_optimization_request("pbuf", ["invalid_param"])
        
        self.assertIn("Invalid optimization parameters", str(context.exception))
        
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    def test_optimize_parameters_mock_objective(self, mock_likelihood, mock_load_dataset):
        """Test optimization convergence with mock objective function."""
        # Mock dataset loading
        mock_data = {
            "observations": {"R": 1.7502, "l_A": 301.63, "theta_star": 1.04119},
            "covariance": np.eye(3) * 0.01
        }
        mock_load_dataset.return_value = mock_data
        
        # Mock likelihood function with simple quadratic objective
        def mock_chi2(params, data):
            # Simple quadratic objective centered at H0=67.4
            H0 = params["H0"]
            chi2 = (H0 - 67.4)**2
            predictions = {"R": 1.75, "l_A": 301.6, "theta_star": 1.041}
            return chi2, predictions
        
        mock_likelihood.return_value = (1.0, {})
        mock_likelihood.side_effect = mock_chi2
        
        # Test optimization
        result = self.optimizer.optimize_parameters(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=["H0"],
            starting_values={"H0": 70.0}  # Start away from optimum
        )
        
        # Check that optimization found the minimum
        self.assertEqual(result.model, "lcdm")
        self.assertIn("H0", result.optimized_params)
        self.assertLess(result.final_chi2, 10.0)  # Should find a good minimum
        self.assertGreater(result.chi2_improvement, 0)  # Should improve χ²
        
    def test_optimize_parameters_empty_datasets(self):
        """Test error handling for empty dataset list."""
        with self.assertRaises(ValueError) as context:
            self.optimizer.optimize_parameters(
                model="lcdm",
                datasets_list=[],
                optimize_params=["H0"]
            )
        
        self.assertIn("At least one dataset must be specified", str(context.exception))
        
    def test_optimize_parameters_invalid_optimization_request(self):
        """Test error handling for invalid optimization parameters."""
        with self.assertRaises(ValueError):
            self.optimizer.optimize_parameters(
                model="lcdm",
                datasets_list=["cmb"],
                optimize_params=["alpha"]  # PBUF parameter for ΛCDM model
            )
            
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    def test_bounds_enforcement(self, mock_likelihood, mock_load_dataset):
        """Test that optimization respects parameter bounds."""
        # Mock dataset loading
        mock_data = {
            "observations": {"R": 1.7502, "l_A": 301.63, "theta_star": 1.04119},
            "covariance": np.eye(3) * 0.01
        }
        mock_load_dataset.return_value = mock_data
        
        # Mock likelihood that prefers values outside bounds
        def mock_chi2_outside_bounds(params, data):
            # Objective that would prefer H0 < 20 (outside lower bound)
            H0 = params["H0"]
            chi2 = (H0 - 10.0)**2  # Minimum at H0=10, but bound is H0 >= 20
            predictions = {"R": 1.75, "l_A": 301.6, "theta_star": 1.041}
            return chi2, predictions
        
        mock_likelihood.side_effect = mock_chi2_outside_bounds
        
        # Test optimization
        result = self.optimizer.optimize_parameters(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=["H0"],
            starting_values={"H0": 50.0}
        )
        
        # Check that result respects bounds
        self.assertGreaterEqual(result.optimized_params["H0"], 20.0)  # Lower bound
        self.assertLessEqual(result.optimized_params["H0"], 150.0)    # Upper bound
        
        # Check if bound was reached
        if abs(result.optimized_params["H0"] - 20.0) < 1e-6:
            self.assertIn("H0_min", result.bounds_reached)
            
    def test_covariance_scaling(self):
        """Test covariance scaling functionality."""
        optimizer_scaled = optimizer.ParameterOptimizer(covariance_scaling=2.0)
        self.assertEqual(optimizer_scaled.covariance_scaling, 2.0)
        
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    def test_optimization_result_structure(self, mock_likelihood, mock_load_dataset):
        """Test that optimization result has correct structure."""
        # Mock dataset and likelihood
        mock_data = {"observations": {}, "covariance": np.eye(3)}
        mock_load_dataset.return_value = mock_data
        mock_likelihood.return_value = (1.0, {})
        
        result = self.optimizer.optimize_parameters(
            model="lcdm",
            datasets_list=["cmb"],
            optimize_params=["H0"]
        )
        
        # Check result structure
        self.assertIsInstance(result, optimizer.OptimizationResult)
        self.assertEqual(result.model, "lcdm")
        self.assertIn("H0", result.optimized_params)
        self.assertIn("H0", result.starting_params)
        self.assertIsInstance(result.final_chi2, float)
        self.assertIsInstance(result.chi2_improvement, float)
        self.assertIsInstance(result.convergence_status, str)
        self.assertIsInstance(result.n_function_evaluations, int)
        self.assertIsInstance(result.optimization_time, float)
        self.assertIsInstance(result.bounds_reached, list)
        self.assertIsInstance(result.optimizer_info, dict)
        self.assertIsInstance(result.metadata, dict)


class TestCMBOptimization(unittest.TestCase):
    """Test cases for CMB-specific optimization routine."""
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    def test_optimize_cmb_parameters_basic(self, mock_likelihood, mock_integrity, mock_load_dataset):
        """Test basic CMB parameter optimization."""
        # Mock dataset loading
        mock_data = {
            "observations": {"R": 1.7502, "l_A": 301.63, "theta_star": 1.04119},
            "covariance": np.eye(3) * 0.01
        }
        mock_load_dataset.return_value = mock_data
        
        # Mock integrity check
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        
        # Mock likelihood
        mock_likelihood.return_value = (1.0, {"R": 1.75, "l_A": 301.6, "theta_star": 1.041})
        
        # Test CMB optimization
        result = optimizer.optimize_cmb_parameters(
            model="lcdm",
            optimize_params=["H0", "Om0"]
        )
        
        # Check result
        self.assertEqual(result.model, "lcdm")
        self.assertIn("H0", result.optimized_params)
        self.assertIn("Om0", result.optimized_params)
        self.assertEqual(result.metadata["optimization_type"], "cmb_specific")
        self.assertTrue(result.metadata["dataset_integrity_validated"])
        
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    def test_optimize_cmb_parameters_integrity_failure(self, mock_integrity, mock_load_dataset):
        """Test CMB optimization with integrity check failure."""
        # Mock dataset loading
        mock_load_dataset.return_value = {"observations": {}, "covariance": np.eye(3)}
        
        # Mock integrity check failure
        mock_integrity.return_value = {
            "overall_status": "FAIL", 
            "failures": ["covariance_matrix_invalid"]
        }
        
        # Test that integrity failure raises error
        with self.assertRaises(ValueError) as context:
            optimizer.optimize_cmb_parameters(
                model="lcdm",
                optimize_params=["H0"]
            )
        
        self.assertIn("CMB dataset integrity validation failed", str(context.exception))
        
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    def test_optimize_cmb_parameters_covariance_scaling(self, mock_likelihood, mock_integrity, mock_load_dataset):
        """Test CMB optimization with covariance scaling."""
        # Mock dataset and integrity
        mock_data = {"observations": {}, "covariance": np.eye(3)}
        mock_load_dataset.return_value = mock_data
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        mock_likelihood.return_value = (1.0, {})
        
        # Test with covariance scaling
        result = optimizer.optimize_cmb_parameters(
            model="lcdm",
            optimize_params=["H0"],
            covariance_scaling=1.5
        )
        
        self.assertEqual(result.covariance_scaling, 1.5)
        self.assertEqual(result.metadata["covariance_scaling"], 1.5)
        
    @patch('pipelines.fit_core.datasets.load_dataset')
    @patch('pipelines.fit_core.integrity.run_integrity_suite')
    @patch('pipelines.fit_core.likelihoods.likelihood_cmb')
    def test_optimize_cmb_parameters_dry_run(self, mock_likelihood, mock_integrity, mock_load_dataset):
        """Test CMB optimization in dry run mode."""
        # Mock dataset and integrity
        mock_data = {"observations": {}, "covariance": np.eye(3)}
        mock_load_dataset.return_value = mock_data
        mock_integrity.return_value = {"overall_status": "PASS", "failures": []}
        mock_likelihood.return_value = (1.0, {})
        
        # Test dry run mode
        result = optimizer.optimize_cmb_parameters(
            model="lcdm",
            optimize_params=["H0"],
            dry_run=True
        )
        
        self.assertTrue(result.metadata["dry_run"])
        
    def test_optimize_cmb_parameters_invalid_model(self):
        """Test CMB optimization with invalid model."""
        with self.assertRaises(ValueError):
            optimizer.optimize_cmb_parameters(
                model="invalid_model",
                optimize_params=["H0"]
            )
            
    def test_optimize_cmb_parameters_invalid_parameters(self):
        """Test CMB optimization with invalid parameters."""
        with self.assertRaises(ValueError):
            optimizer.optimize_cmb_parameters(
                model="lcdm",
                optimize_params=["invalid_param"]
            )


class TestOptimizationValidation(unittest.TestCase):
    """Test cases for optimization result validation."""
    
    def test_validate_optimization_result_success(self):
        """Test validation of successful optimization result."""
        # Create a valid optimization result
        result = optimizer.OptimizationResult(
            model="lcdm",
            optimized_params={"H0": 67.4, "Om0": 0.315},
            starting_params={"H0": 70.0, "Om0": 0.3},
            final_chi2=10.0,
            chi2_improvement=2.0,
            convergence_status="success",
            n_function_evaluations=100,
            optimization_time=1.5,
            bounds_reached=[],
            optimizer_info={"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
            covariance_scaling=1.0,
            metadata={}
        )
        
        # Should not raise any exception
        optimizer._validate_optimization_result(result, "lcdm")
        
    def test_validate_optimization_result_error_status(self):
        """Test validation with error convergence status."""
        result = optimizer.OptimizationResult(
            model="lcdm",
            optimized_params={"H0": 67.4},
            starting_params={"H0": 70.0},
            final_chi2=10.0,
            chi2_improvement=2.0,
            convergence_status="error: optimization failed",
            n_function_evaluations=100,
            optimization_time=1.5,
            bounds_reached=[],
            optimizer_info={},
            covariance_scaling=1.0,
            metadata={}
        )
        
        with self.assertRaises(ValueError) as context:
            optimizer._validate_optimization_result(result, "lcdm")
        
        self.assertIn("Optimization failed", str(context.exception))
        
    def test_validate_optimization_result_bounds_violation(self):
        """Test validation with parameter outside bounds."""
        result = optimizer.OptimizationResult(
            model="lcdm",
            optimized_params={"H0": 200.0},  # Outside upper bound of 150.0
            starting_params={"H0": 70.0},
            final_chi2=10.0,
            chi2_improvement=2.0,
            convergence_status="success",
            n_function_evaluations=100,
            optimization_time=1.5,
            bounds_reached=[],
            optimizer_info={},
            covariance_scaling=1.0,
            metadata={}
        )
        
        with self.assertRaises(ValueError) as context:
            optimizer._validate_optimization_result(result, "lcdm")
        
        self.assertIn("physical bounds", str(context.exception))


if __name__ == "__main__":
    unittest.main()