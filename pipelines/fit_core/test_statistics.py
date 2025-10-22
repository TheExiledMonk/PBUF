"""
Unit tests for statistics module.

Tests χ² computation against analytical cases, AIC/BIC calculations,
degrees of freedom computation, and model comparison utilities.

Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.fit_core import statistics


class TestChi2Generic(unittest.TestCase):
    """Test χ² computation against analytical cases with known results."""
    
    def test_chi2_simple_scalar_case(self):
        """Test χ² computation for simple scalar case with known result."""
        # Simple case: single observation with unit variance
        predictions = {"obs1": 1.0}
        observations = {"obs1": 0.0}
        covariance = np.array([[1.0]])
        
        expected_chi2 = 1.0  # (1-0)² / 1 = 1
        result = statistics.chi2_generic(predictions, observations, covariance)
        
        self.assertAlmostEqual(result, expected_chi2, places=10)
    
    def test_chi2_multiple_observations_diagonal_covariance(self):
        """Test χ² with multiple observations and diagonal covariance."""
        predictions = {"obs1": 2.0, "obs2": 3.0}
        observations = {"obs1": 1.0, "obs2": 1.0}
        covariance = np.array([[1.0, 0.0], [0.0, 4.0]])
        
        # χ² = (2-1)²/1 + (3-1)²/4 = 1 + 1 = 2
        expected_chi2 = 2.0
        result = statistics.chi2_generic(predictions, observations, covariance)
        
        self.assertAlmostEqual(result, expected_chi2, places=10)
    
    def test_chi2_correlated_observations(self):
        """Test χ² with correlated observations (non-diagonal covariance)."""
        predictions = {"obs1": 1.0, "obs2": 1.0}
        observations = {"obs1": 0.0, "obs2": 0.0}
        # Covariance with correlation coefficient 0.5
        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        # Analytical result: residuals = [1, 1], C^-1 = [[4/3, -2/3], [-2/3, 4/3]]
        # χ² = [1, 1] * [[4/3, -2/3], [-2/3, 4/3]] * [1, 1]ᵀ = 4/3 - 4/3 + 4/3 = 4/3
        expected_chi2 = 4.0/3.0
        result = statistics.chi2_generic(predictions, observations, covariance)
        
        self.assertAlmostEqual(result, expected_chi2, places=10)
    
    def test_chi2_array_observations(self):
        """Test χ² with array-valued observations."""
        predictions = {"obs_array": np.array([1.0, 2.0])}
        observations = {"obs_array": np.array([0.0, 0.0])}
        covariance = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        # χ² = (1-0)² + (2-0)² = 1 + 4 = 5
        expected_chi2 = 5.0
        result = statistics.chi2_generic(predictions, observations, covariance)
        
        self.assertAlmostEqual(result, expected_chi2, places=10)
    
    def test_chi2_mixed_scalar_array(self):
        """Test χ² with mixed scalar and array observations."""
        predictions = {"scalar": 1.0, "array": np.array([2.0, 3.0])}
        observations = {"scalar": 0.0, "array": np.array([1.0, 1.0])}
        covariance = np.eye(3)  # 3x3 identity matrix
        
        # Residuals: [1, 1, 2], χ² = 1² + 1² + 2² = 6
        expected_chi2 = 6.0
        result = statistics.chi2_generic(predictions, observations, covariance)
        
        self.assertAlmostEqual(result, expected_chi2, places=10)
    
    def test_chi2_zero_residuals(self):
        """Test χ² when predictions exactly match observations."""
        predictions = {"obs1": 1.0, "obs2": 2.0}
        observations = {"obs1": 1.0, "obs2": 2.0}
        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        expected_chi2 = 0.0
        result = statistics.chi2_generic(predictions, observations, covariance)
        
        self.assertAlmostEqual(result, expected_chi2, places=10)
    
    def test_chi2_singular_covariance_fallback(self):
        """Test χ² computation with singular covariance matrix (uses pseudo-inverse)."""
        predictions = {"obs1": 1.0, "obs2": 1.0}
        observations = {"obs1": 0.0, "obs2": 0.0}
        # Singular covariance matrix (rank 1)
        covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
        
        # Should not raise error and return finite result
        result = statistics.chi2_generic(predictions, observations, covariance)
        self.assertTrue(np.isfinite(result))
        self.assertGreaterEqual(result, 0.0)
    
    def test_chi2_input_validation_errors(self):
        """Test χ² input validation error cases."""
        valid_pred = {"obs1": 1.0}
        valid_obs = {"obs1": 0.0}
        valid_cov = np.array([[1.0]])
        
        # Empty predictions
        with self.assertRaises(ValueError):
            statistics.chi2_generic({}, valid_obs, valid_cov)
        
        # Empty observations
        with self.assertRaises(ValueError):
            statistics.chi2_generic(valid_pred, {}, valid_cov)
        
        # Mismatched keys
        with self.assertRaises(ValueError):
            statistics.chi2_generic({"obs1": 1.0}, {"obs2": 0.0}, valid_cov)
        
        # Wrong covariance size
        with self.assertRaises(ValueError):
            statistics.chi2_generic(valid_pred, valid_obs, np.array([[1.0, 0.0], [0.0, 1.0]]))
        
        # Non-square covariance
        with self.assertRaises(ValueError):
            statistics.chi2_generic(valid_pred, valid_obs, np.array([[1.0, 0.0]]))
        
        # NaN in covariance
        with self.assertRaises(ValueError):
            statistics.chi2_generic(valid_pred, valid_obs, np.array([[np.nan]]))
        
        # NaN in predictions
        with self.assertRaises(ValueError):
            statistics.chi2_generic({"obs1": np.nan}, valid_obs, valid_cov)
        
        # NaN in observations
        with self.assertRaises(ValueError):
            statistics.chi2_generic(valid_pred, {"obs1": np.nan}, valid_cov)


class TestComputeMetrics(unittest.TestCase):
    """Test AIC, BIC, and degrees of freedom calculations."""
    
    @patch('pipelines.fit_core.statistics._get_dataset_size')
    def test_compute_metrics_basic(self, mock_get_size):
        """Test basic metrics computation with known values."""
        mock_get_size.return_value = 10  # 10 data points per dataset
        
        chi2 = 5.0
        n_params = 2
        datasets = ["test_dataset"]
        
        result = statistics.compute_metrics(chi2, n_params, datasets)
        
        # Expected values
        expected_aic = 5.0 + 2 * 2  # χ² + 2k = 9
        expected_bic = 5.0 + 2 * np.log(10)  # χ² + k*ln(N) ≈ 9.61
        expected_dof = 10 - 2  # N - k = 8
        expected_chi2_reduced = 5.0 / 8  # χ²/dof = 0.625
        
        self.assertAlmostEqual(result["aic"], expected_aic, places=10)
        self.assertAlmostEqual(result["bic"], expected_bic, places=6)
        self.assertEqual(result["dof"], expected_dof)
        self.assertAlmostEqual(result["chi2_reduced"], expected_chi2_reduced, places=10)
        self.assertEqual(result["n_params"], n_params)
        self.assertEqual(result["n_data"], 10)
    
    @patch('pipelines.fit_core.statistics._get_dataset_size')
    def test_compute_metrics_multiple_datasets(self, mock_get_size):
        """Test metrics computation with multiple datasets."""
        # Different sizes for different datasets
        size_map = {"dataset1": 5, "dataset2": 15, "dataset3": 10}
        mock_get_size.side_effect = lambda name: size_map[name]
        
        chi2 = 20.0
        n_params = 3
        datasets = ["dataset1", "dataset2", "dataset3"]
        
        result = statistics.compute_metrics(chi2, n_params, datasets)
        
        total_data = 5 + 15 + 10  # 30
        expected_aic = 20.0 + 2 * 3  # 26
        expected_bic = 20.0 + 3 * np.log(30)  # ≈ 30.20
        expected_dof = 30 - 3  # 27
        
        self.assertAlmostEqual(result["aic"], expected_aic, places=10)
        self.assertAlmostEqual(result["bic"], expected_bic, places=6)
        self.assertEqual(result["dof"], expected_dof)
        self.assertEqual(result["n_data"], total_data)
    
    @patch('pipelines.fit_core.statistics.compute_p_value')
    @patch('pipelines.fit_core.statistics._get_dataset_size')
    def test_compute_metrics_with_p_value(self, mock_get_size, mock_p_value):
        """Test that p-value is correctly included in metrics."""
        mock_get_size.return_value = 10
        mock_p_value.return_value = 0.123
        
        chi2 = 8.0
        n_params = 2
        datasets = ["test"]
        
        result = statistics.compute_metrics(chi2, n_params, datasets)
        
        self.assertEqual(result["p_value"], 0.123)
        mock_p_value.assert_called_once_with(8.0, 8)  # chi2, dof
    
    def test_compute_metrics_zero_dof(self):
        """Test metrics computation when degrees of freedom is zero."""
        with patch('pipelines.fit_core.statistics._get_dataset_size', return_value=2):
            chi2 = 5.0
            n_params = 2
            datasets = ["test"]
            
            result = statistics.compute_metrics(chi2, n_params, datasets)
            
            self.assertEqual(result["dof"], 0)
            self.assertEqual(result["chi2_reduced"], np.inf)


class TestComputeDof(unittest.TestCase):
    """Test degrees of freedom computation."""
    
    @patch('pipelines.fit_core.statistics._get_dataset_size')
    def test_compute_dof_single_dataset(self, mock_get_size):
        """Test DOF computation for single dataset."""
        mock_get_size.return_value = 15
        
        result = statistics.compute_dof(["dataset1"], 3)
        
        self.assertEqual(result, 12)  # 15 - 3
        mock_get_size.assert_called_once_with("dataset1")
    
    @patch('pipelines.fit_core.statistics._get_dataset_size')
    def test_compute_dof_multiple_datasets(self, mock_get_size):
        """Test DOF computation for multiple datasets."""
        size_map = {"ds1": 10, "ds2": 20, "ds3": 5}
        mock_get_size.side_effect = lambda name: size_map[name]
        
        result = statistics.compute_dof(["ds1", "ds2", "ds3"], 7)
        
        self.assertEqual(result, 28)  # (10+20+5) - 7
    
    @patch('pipelines.fit_core.statistics._get_dataset_size')
    def test_compute_dof_negative_error(self, mock_get_size):
        """Test that negative DOF raises ValueError."""
        mock_get_size.return_value = 5
        
        with self.assertRaises(ValueError) as cm:
            statistics.compute_dof(["dataset"], 10)
        
        self.assertIn("cannot be negative", str(cm.exception))
        self.assertIn("5 data points - 10 parameters = -5", str(cm.exception))


class TestDeltaAic(unittest.TestCase):
    """Test model comparison utilities."""
    
    def test_delta_aic_basic(self):
        """Test basic ΔAIC computation."""
        aic1 = 100.0
        aic2 = 95.0
        
        result = statistics.delta_aic(aic1, aic2)
        
        self.assertEqual(result, 5.0)
    
    def test_delta_aic_negative(self):
        """Test ΔAIC when first model is better."""
        aic1 = 90.0
        aic2 = 95.0
        
        result = statistics.delta_aic(aic1, aic2)
        
        self.assertEqual(result, -5.0)
    
    def test_delta_aic_equal(self):
        """Test ΔAIC when models are equivalent."""
        aic1 = 100.0
        aic2 = 100.0
        
        result = statistics.delta_aic(aic1, aic2)
        
        self.assertEqual(result, 0.0)


class TestComputePValue(unittest.TestCase):
    """Test p-value computation from χ² distribution."""
    
    def test_p_value_with_scipy(self):
        """Test p-value computation when scipy is available."""
        # Mock scipy.stats to be available
        with patch('pipelines.fit_core.statistics.stats') as mock_stats:
            mock_stats.chi2.cdf.return_value = 0.8
            
            result = statistics.compute_p_value(10.0, 5)
            
            self.assertAlmostEqual(result, 0.2, places=10)  # 1 - 0.8
            mock_stats.chi2.cdf.assert_called_once_with(10.0, 5)
    
    def test_p_value_without_scipy_large_dof(self):
        """Test p-value approximation when scipy unavailable but DOF > 30."""
        with patch('pipelines.fit_core.statistics.stats', None):
            # For large DOF, should use normal approximation
            result = statistics.compute_p_value(50.0, 50)
            
            # Should return a finite value between 0 and 1
            self.assertTrue(0 <= result <= 1)
            self.assertTrue(np.isfinite(result))
    
    def test_p_value_without_scipy_small_dof(self):
        """Test p-value when scipy unavailable and DOF <= 30."""
        with patch('pipelines.fit_core.statistics.stats', None):
            result = statistics.compute_p_value(10.0, 5)
            
            # Should return NaN when scipy not available for small DOF
            self.assertTrue(np.isnan(result))
    
    def test_p_value_edge_cases(self):
        """Test p-value edge cases."""
        # Zero or negative DOF
        self.assertTrue(np.isnan(statistics.compute_p_value(10.0, 0)))
        self.assertTrue(np.isnan(statistics.compute_p_value(10.0, -1)))
        
        # Negative chi2
        self.assertTrue(np.isnan(statistics.compute_p_value(-5.0, 5)))


class TestGetDatasetSize(unittest.TestCase):
    """Test dataset size determination for DOF calculation."""
    
    def test_get_dataset_size_known_fixed(self):
        """Test size retrieval for datasets with fixed sizes."""
        # CMB has fixed size of 3
        result = statistics._get_dataset_size("cmb")
        self.assertEqual(result, 3)
    
    def test_get_dataset_size_fallback(self):
        """Test fallback sizes for known datasets."""
        # Test fallback values when DATASET_SIZES entry is None
        with patch.dict('pipelines.fit_core.statistics.DATASET_SIZES', {"bao": None}):
            result = statistics._get_dataset_size("bao")
            self.assertEqual(result, 10)  # Actual BAO dataset size
    
    def test_get_dataset_size_from_datasets_module(self):
        """Test size retrieval from datasets module using real datasets."""
        # Test with a real dataset that has variable size (bao)
        # This tests the path where datasets.get_dataset_info is called
        with patch.dict('pipelines.fit_core.statistics.DATASET_SIZES', {"bao": None}):
            result = statistics._get_dataset_size("bao")
            
        # Should get the actual BAO dataset size
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)
    
    def test_get_dataset_size_from_loaded_data(self):
        """Test size calculation from loaded dataset."""
        # Test the fallback mechanism by using a dataset that exists
        # but forcing it to use the fallback path
        with patch.dict('pipelines.fit_core.statistics.DATASET_SIZES', {"sn": None}):
            result = statistics._get_dataset_size("sn")
            
        # Should get a reasonable size for supernova dataset
        self.assertIsInstance(result, int)
        self.assertGreater(result, 100)  # SN datasets are typically large
    
    def test_get_dataset_size_unknown_dataset(self):
        """Test error for completely unknown dataset."""
        with self.assertRaises(ValueError) as cm:
            statistics._get_dataset_size("unknown_dataset")
        
        # The error comes from datasets module first, then from statistics if that fails
        error_msg = str(cm.exception)
        self.assertTrue("unknown_dataset" in error_msg and ("Unknown" in error_msg or "Unsupported" in error_msg))


class TestValidateChi2Distribution(unittest.TestCase):
    """Test χ² distribution validation and goodness of fit assessment."""
    
    @patch('pipelines.fit_core.statistics.compute_p_value')
    def test_validate_chi2_distribution_good_fit(self, mock_p_value):
        """Test validation for a good fit (reasonable χ²)."""
        mock_p_value.return_value = 0.3  # Good p-value
        
        result = statistics.validate_chi2_distribution(8.0, 10, alpha=0.05)
        
        self.assertEqual(result["chi2"], 8.0)
        self.assertEqual(result["dof"], 10)
        self.assertAlmostEqual(result["chi2_reduced"], 0.8, places=10)
        self.assertEqual(result["p_value"], 0.3)
        self.assertTrue(result["fit_acceptable"])  # p > alpha
        self.assertFalse(result["overfitting"])  # χ²_red not < 0.5
        self.assertFalse(result["underfitting"])  # χ²_red not > 2.0
        self.assertTrue(result["reasonable_fit"])  # 0.5 <= χ²_red <= 2.0
    
    @patch('pipelines.fit_core.statistics.compute_p_value')
    def test_validate_chi2_distribution_poor_fit(self, mock_p_value):
        """Test validation for a poor fit (high χ²)."""
        mock_p_value.return_value = 0.01  # Poor p-value
        
        result = statistics.validate_chi2_distribution(25.0, 10, alpha=0.05)
        
        self.assertEqual(result["chi2"], 25.0)
        self.assertAlmostEqual(result["chi2_reduced"], 2.5, places=10)
        self.assertFalse(result["fit_acceptable"])  # p < alpha
        self.assertFalse(result["overfitting"])
        self.assertTrue(result["underfitting"])  # χ²_red > 2.0
        self.assertFalse(result["reasonable_fit"])
    
    @patch('pipelines.fit_core.statistics.compute_p_value')
    def test_validate_chi2_distribution_overfitting(self, mock_p_value):
        """Test validation for suspiciously good fit (low χ²)."""
        mock_p_value.return_value = 0.99  # Very high p-value
        
        result = statistics.validate_chi2_distribution(2.0, 10, alpha=0.05)
        
        self.assertAlmostEqual(result["chi2_reduced"], 0.2, places=10)
        self.assertTrue(result["fit_acceptable"])
        self.assertTrue(result["overfitting"])  # χ²_red < 0.5
        self.assertFalse(result["underfitting"])
        self.assertFalse(result["reasonable_fit"])
    
    @patch('pipelines.fit_core.statistics.stats')
    @patch('pipelines.fit_core.statistics.compute_p_value')
    def test_validate_chi2_distribution_with_confidence_intervals(self, mock_p_value, mock_stats):
        """Test validation with confidence intervals when scipy available."""
        mock_p_value.return_value = 0.3
        mock_stats.chi2.ppf.side_effect = [3.0, 20.0]  # 95% CI bounds
        
        result = statistics.validate_chi2_distribution(10.0, 8)
        
        self.assertEqual(result["chi2_ci_lower"], 3.0)
        self.assertEqual(result["chi2_ci_upper"], 20.0)
        self.assertTrue(result["within_ci"])  # 3.0 <= 10.0 <= 20.0
        
        # Check ppf was called correctly for 95% CI
        expected_calls = [unittest.mock.call(0.025, 8), unittest.mock.call(0.975, 8)]
        mock_stats.chi2.ppf.assert_has_calls(expected_calls)
    
    def test_validate_chi2_distribution_zero_dof(self):
        """Test validation with zero degrees of freedom."""
        result = statistics.validate_chi2_distribution(5.0, 0)
        
        self.assertEqual(result["dof"], 0)
        self.assertEqual(result["chi2_reduced"], np.inf)
        self.assertIsNone(result["overfitting"])
        self.assertIsNone(result["underfitting"])
        self.assertIsNone(result["reasonable_fit"])


class TestComputeConfidenceIntervals(unittest.TestCase):
    """Test confidence interval computation for χ² values."""
    
    @patch('pipelines.fit_core.statistics.stats')
    def test_confidence_intervals_with_scipy(self, mock_stats):
        """Test confidence interval computation when scipy available."""
        # Mock ppf to return known values
        mock_stats.chi2.ppf.side_effect = [2.0, 15.0, 1.0, 18.0, 0.5, 20.0]
        
        result = statistics.compute_confidence_intervals(10.0, 8, [0.68, 0.95, 0.99])
        
        expected = {
            "0.68": {"lower": 2.0, "upper": 15.0},
            "0.95": {"lower": 1.0, "upper": 18.0},
            "0.99": {"lower": 0.5, "upper": 20.0}
        }
        
        self.assertEqual(result, expected)
    
    def test_confidence_intervals_without_scipy(self):
        """Test confidence intervals when scipy not available."""
        with patch('pipelines.fit_core.statistics.stats', None):
            result = statistics.compute_confidence_intervals(10.0, 8)
            
            # Should return NaN for all intervals
            for level in ["0.68", "0.95", "0.99"]:
                self.assertTrue(np.isnan(result[level]["lower"]))
                self.assertTrue(np.isnan(result[level]["upper"]))
    
    def test_confidence_intervals_zero_dof(self):
        """Test confidence intervals with zero degrees of freedom."""
        with patch('pipelines.fit_core.statistics.stats') as mock_stats:
            result = statistics.compute_confidence_intervals(10.0, 0)
            
            # Should return NaN for zero DOF
            for level in ["0.68", "0.95", "0.99"]:
                self.assertTrue(np.isnan(result[level]["lower"]))
                self.assertTrue(np.isnan(result[level]["upper"]))
            
            # stats.chi2.ppf should not be called
            mock_stats.chi2.ppf.assert_not_called()
    
    def test_confidence_intervals_default_levels(self):
        """Test confidence intervals with default confidence levels."""
        with patch('pipelines.fit_core.statistics.stats') as mock_stats:
            mock_stats.chi2.ppf.return_value = 5.0  # Dummy value
            
            result = statistics.compute_confidence_intervals(10.0, 8)
            
            # Should have default levels
            expected_levels = {"0.68", "0.95", "0.99"}
            self.assertEqual(set(result.keys()), expected_levels)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)