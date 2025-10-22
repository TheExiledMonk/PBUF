"""
Unit tests for dataset loading and validation functionality.

Tests cover:
- Dataset loading for all supported observational blocks
- Covariance matrix validation and error handling  
- Dataset metadata extraction and format consistency
- Requirements: 4.1, 4.2, 4.3, 4.4
"""

import unittest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from .datasets import (
    load_dataset, validate_dataset, get_dataset_info,
    _validate_covariance_matrix, _extract_dataset_metadata,
    _compute_condition_number, _compute_data_vector_length,
    _validate_data_consistency, SUPPORTED_DATASETS
)


class TestDatasetLoading(unittest.TestCase):
    """Test dataset loading for all supported observational blocks."""
    
    def test_load_cmb_dataset(self):
        """Test CMB dataset loading and structure."""
        data = load_dataset("cmb")
        
        # Check required top-level keys
        self.assertIn("observations", data)
        self.assertIn("covariance", data)
        self.assertIn("metadata", data)
        self.assertIn("dataset_type", data)
        self.assertEqual(data["dataset_type"], "cmb")
        
        # Check CMB-specific observations
        obs = data["observations"]
        self.assertIn("R", obs)
        self.assertIn("l_A", obs)
        self.assertIn("theta_star", obs)
        
        # Verify data types and reasonable values
        self.assertIsInstance(obs["R"], (int, float))
        self.assertIsInstance(obs["l_A"], (int, float))
        self.assertIsInstance(obs["theta_star"], (int, float))
        
        # Check reasonable physical values
        self.assertGreater(obs["R"], 1.0)  # Shift parameter > 1
        self.assertGreater(obs["l_A"], 200)  # Acoustic scale ~ 300
        self.assertGreater(obs["theta_star"], 1.0)  # Angular scale ~ 1.04
        
        # Check covariance matrix
        cov = data["covariance"]
        self.assertIsInstance(cov, np.ndarray)
        self.assertEqual(cov.shape, (3, 3))
        
        # Check metadata
        metadata = data["metadata"]
        self.assertIn("source", metadata)
        self.assertIn("n_data_points", metadata)
        self.assertEqual(metadata["n_data_points"], 3)
    
    def test_load_bao_dataset(self):
        """Test isotropic BAO dataset loading and structure."""
        data = load_dataset("bao")
        
        # Check structure
        self.assertEqual(data["dataset_type"], "bao")
        
        # Check BAO-specific observations
        obs = data["observations"]
        self.assertIn("redshift", obs)
        self.assertIn("DV_over_rs", obs)
        
        # Verify arrays have same length
        redshifts = np.asarray(obs["redshift"])
        dv_ratios = np.asarray(obs["DV_over_rs"])
        self.assertEqual(len(redshifts), len(dv_ratios))
        
        # Check reasonable values
        self.assertTrue(np.all(redshifts > 0))
        self.assertTrue(np.all(redshifts < 5))  # Reasonable redshift range
        self.assertTrue(np.all(dv_ratios > 0))
        
        # Check covariance dimensions
        cov = data["covariance"]
        expected_dim = len(redshifts)
        self.assertEqual(cov.shape, (expected_dim, expected_dim))
    
    def test_load_bao_anisotropic_dataset(self):
        """Test anisotropic BAO dataset loading and structure."""
        data = load_dataset("bao_ani")
        
        # Check structure
        self.assertEqual(data["dataset_type"], "bao_ani")
        
        # Check anisotropic BAO observations
        obs = data["observations"]
        self.assertIn("redshift", obs)
        self.assertIn("DM_over_rs", obs)
        self.assertIn("H_times_rs", obs)
        
        # Verify arrays have same length
        redshifts = np.asarray(obs["redshift"])
        dm_ratios = np.asarray(obs["DM_over_rs"])
        h_ratios = np.asarray(obs["H_times_rs"])
        
        self.assertEqual(len(redshifts), len(dm_ratios))
        self.assertEqual(len(redshifts), len(h_ratios))
        
        # Check reasonable values
        self.assertTrue(np.all(redshifts > 0))
        self.assertTrue(np.all(dm_ratios > 0))
        self.assertTrue(np.all(h_ratios > 0))
        
        # Check covariance dimensions (2N x 2N for N redshift bins)
        cov = data["covariance"]
        expected_dim = 2 * len(redshifts)
        self.assertEqual(cov.shape, (expected_dim, expected_dim))
    
    def test_load_supernova_dataset(self):
        """Test supernova dataset loading and structure."""
        data = load_dataset("sn")
        
        # Check structure
        self.assertEqual(data["dataset_type"], "sn")
        
        # Check supernova observations
        obs = data["observations"]
        self.assertIn("redshift", obs)
        self.assertIn("distance_modulus", obs)
        self.assertIn("sigma_mu", obs)
        
        # Verify arrays have same length
        redshifts = np.asarray(obs["redshift"])
        distance_moduli = np.asarray(obs["distance_modulus"])
        uncertainties = np.asarray(obs["sigma_mu"])
        
        self.assertEqual(len(redshifts), len(distance_moduli))
        self.assertEqual(len(redshifts), len(uncertainties))
        
        # Check reasonable values
        self.assertTrue(np.all(redshifts > 0))
        self.assertTrue(np.all(uncertainties > 0))
        
        # Check covariance dimensions
        cov = data["covariance"]
        expected_dim = len(redshifts)
        self.assertEqual(cov.shape, (expected_dim, expected_dim))
    
    def test_load_unsupported_dataset(self):
        """Test error handling for unsupported datasets."""
        with self.assertRaises(ValueError) as context:
            load_dataset("unknown_dataset")
        
        self.assertIn("Unsupported dataset", str(context.exception))
        self.assertIn("unknown_dataset", str(context.exception))
    
    def test_supported_datasets_configuration(self):
        """Test that all supported datasets are properly configured."""
        for dataset_name in SUPPORTED_DATASETS:
            config = SUPPORTED_DATASETS[dataset_name]
            
            # Check required configuration keys
            self.assertIn("description", config)
            self.assertIn("expected_observables", config)
            self.assertIn("covariance_shape", config)
            
            # Verify we can actually load the dataset
            data = load_dataset(dataset_name)
            self.assertEqual(data["dataset_type"], dataset_name)


class TestCovarianceValidation(unittest.TestCase):
    """Test covariance matrix validation and error handling."""
    
    def test_valid_covariance_matrix(self):
        """Test validation of valid covariance matrices."""
        # Simple 2x2 positive definite matrix
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.assertTrue(_validate_covariance_matrix(cov))
        
        # Identity matrix
        cov_identity = np.eye(3)
        self.assertTrue(_validate_covariance_matrix(cov_identity))
        
        # Diagonal matrix with positive values
        cov_diag = np.diag([1.0, 2.0, 0.5])
        self.assertTrue(_validate_covariance_matrix(cov_diag))
    
    def test_invalid_covariance_matrices(self):
        """Test rejection of invalid covariance matrices."""
        # Non-square matrix
        cov_nonsquare = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3]])
        self.assertFalse(_validate_covariance_matrix(cov_nonsquare))
        
        # 1D array
        cov_1d = np.array([1.0, 2.0, 3.0])
        self.assertFalse(_validate_covariance_matrix(cov_1d))
        
        # Non-symmetric matrix
        cov_nonsym = np.array([[1.0, 0.5], [0.3, 1.0]])
        self.assertFalse(_validate_covariance_matrix(cov_nonsym))
        
        # Negative eigenvalues (not positive definite)
        cov_negative = np.array([[1.0, 2.0], [2.0, 1.0]])
        self.assertFalse(_validate_covariance_matrix(cov_negative))
        
        # Zero eigenvalue (positive semi-definite, not positive definite)
        cov_zero = np.array([[1.0, 1.0], [1.0, 1.0]])
        self.assertFalse(_validate_covariance_matrix(cov_zero))
    
    def test_poorly_conditioned_matrix_warning(self):
        """Test warning for poorly conditioned matrices."""
        # Create poorly conditioned matrix
        cov_poor = np.array([[1.0, 0.0], [0.0, 1e-15]])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _validate_covariance_matrix(cov_poor)
            
            # Should still be valid but generate warning
            self.assertTrue(result)
            self.assertEqual(len(w), 1)
            self.assertIn("poorly conditioned", str(w[0].message))
    
    def test_covariance_eigenvalue_computation_failure(self):
        """Test handling of eigenvalue computation failures."""
        # Create matrix that might cause numerical issues
        cov_bad = np.array([[np.inf, 1.0], [1.0, np.inf]])
        self.assertFalse(_validate_covariance_matrix(cov_bad))
        
        # Matrix with NaN values
        cov_nan = np.array([[1.0, np.nan], [np.nan, 1.0]])
        self.assertFalse(_validate_covariance_matrix(cov_nan))


class TestDatasetValidation(unittest.TestCase):
    """Test dataset format validation and error handling."""
    
    def setUp(self):
        """Set up valid dataset for testing."""
        self.valid_cmb_data = {
            "observations": {
                "R": 1.7502,
                "l_A": 301.845,
                "theta_star": 1.04092
            },
            "covariance": np.eye(3) * 1e-6,
            "metadata": {
                "source": "Test",
                "n_data_points": 3,
                "observables": ["R", "l_A", "theta_star"]
            },
            "dataset_type": "cmb"
        }
    
    def test_valid_dataset_validation(self):
        """Test validation of properly formatted datasets."""
        # Test with valid CMB dataset
        result = validate_dataset(self.valid_cmb_data, "cmb")
        self.assertTrue(result)
        
        # Test with loaded datasets
        for dataset_name in SUPPORTED_DATASETS:
            data = load_dataset(dataset_name)
            result = validate_dataset(data, dataset_name)
            self.assertTrue(result)
    
    def test_missing_required_keys(self):
        """Test error handling for missing required keys."""
        # Missing observations
        data_no_obs = self.valid_cmb_data.copy()
        del data_no_obs["observations"]
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_no_obs, "cmb")
        self.assertIn("Missing required key: observations", str(context.exception))
        
        # Missing covariance
        data_no_cov = self.valid_cmb_data.copy()
        del data_no_cov["covariance"]
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_no_cov, "cmb")
        self.assertIn("Missing required key: covariance", str(context.exception))
        
        # Missing metadata
        data_no_meta = self.valid_cmb_data.copy()
        del data_no_meta["metadata"]
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_no_meta, "cmb")
        self.assertIn("Missing required key: metadata", str(context.exception))
    
    def test_invalid_dataset_type(self):
        """Test error handling for invalid dataset types."""
        data_bad_type = self.valid_cmb_data.copy()
        data_bad_type["dataset_type"] = "invalid_type"
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_bad_type, "cmb")
        self.assertIn("Unknown dataset type: invalid_type", str(context.exception))
    
    def test_invalid_observations_structure(self):
        """Test error handling for invalid observations structure."""
        # Observations not a dictionary
        data_bad_obs = self.valid_cmb_data.copy()
        data_bad_obs["observations"] = [1, 2, 3]
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_bad_obs, "cmb")
        self.assertIn("Observations must be a dictionary", str(context.exception))
    
    def test_missing_expected_observables(self):
        """Test error handling for missing expected observables."""
        data_missing_obs = self.valid_cmb_data.copy()
        del data_missing_obs["observations"]["R"]
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_missing_obs, "cmb")
        self.assertIn("Missing expected observable: R", str(context.exception))
    
    def test_invalid_covariance_type(self):
        """Test error handling for invalid covariance matrix type."""
        data_bad_cov = self.valid_cmb_data.copy()
        data_bad_cov["covariance"] = [[1, 0], [0, 1]]  # List instead of numpy array
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_bad_cov, "cmb")
        self.assertIn("Covariance must be a numpy array", str(context.exception))
    
    def test_invalid_covariance_properties(self):
        """Test error handling for invalid covariance matrix properties."""
        data_bad_cov = self.valid_cmb_data.copy()
        data_bad_cov["covariance"] = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive definite
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_bad_cov, "cmb")
        self.assertIn("Invalid covariance matrix properties", str(context.exception))
    
    def test_invalid_metadata_structure(self):
        """Test error handling for invalid metadata structure."""
        # Metadata not a dictionary
        data_bad_meta = self.valid_cmb_data.copy()
        data_bad_meta["metadata"] = "invalid_metadata"
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_bad_meta, "cmb")
        self.assertIn("Metadata must be a dictionary", str(context.exception))
        
        # Missing required metadata keys
        data_missing_meta = self.valid_cmb_data.copy()
        del data_missing_meta["metadata"]["source"]
        
        with self.assertRaises(ValueError) as context:
            validate_dataset(data_missing_meta, "cmb")
        self.assertIn("Missing required metadata: source", str(context.exception))


class TestDatasetMetadata(unittest.TestCase):
    """Test dataset metadata extraction and format consistency."""
    
    def test_get_dataset_info_valid_datasets(self):
        """Test metadata extraction for all valid datasets."""
        for dataset_name in SUPPORTED_DATASETS:
            info = get_dataset_info(dataset_name)
            
            # Check basic metadata structure
            self.assertIn("dataset_type", info)
            self.assertIn("covariance_shape", info)
            self.assertIn("covariance_condition_number", info)
            
            # Check dataset-specific metadata
            if dataset_name == "cmb":
                self.assertIn("recombination_redshift", info)
                self.assertEqual(info["recombination_redshift"], 1089.80)
                
            elif dataset_name in ["bao", "bao_ani"]:
                self.assertIn("redshift_range", info)
                self.assertIn("redshift_mean", info)
                
            elif dataset_name == "sn":
                self.assertIn("distance_modulus_range", info)
                self.assertIn("uncertainty_range", info)
    
    def test_get_dataset_info_invalid_dataset(self):
        """Test error handling for invalid dataset names."""
        with self.assertRaises(ValueError) as context:
            get_dataset_info("invalid_dataset")
        self.assertIn("Unsupported dataset: invalid_dataset", str(context.exception))
    
    def test_extract_dataset_metadata_cmb(self):
        """Test CMB-specific metadata extraction."""
        data = load_dataset("cmb")
        metadata = _extract_dataset_metadata(data, "cmb")
        
        self.assertEqual(metadata["dataset_type"], "cmb")
        self.assertIn("recombination_redshift", metadata)
        self.assertIn("observable_types", metadata)
        self.assertEqual(metadata["observable_types"], ["distance_priors"])
        self.assertEqual(metadata["data_vector_length"], 3)
        self.assertFalse(metadata["dimension_mismatch"])
    
    def test_extract_dataset_metadata_bao(self):
        """Test BAO-specific metadata extraction."""
        data = load_dataset("bao")
        metadata = _extract_dataset_metadata(data, "bao")
        
        self.assertEqual(metadata["dataset_type"], "bao")
        self.assertIn("redshift_range", metadata)
        self.assertIn("dv_ratio_range", metadata)
        self.assertIn("observable_types", metadata)
        self.assertEqual(metadata["observable_types"], ["isotropic_bao"])
        
        # Check redshift range is reasonable
        z_range = metadata["redshift_range"]
        self.assertGreater(z_range[1], z_range[0])
        self.assertGreater(z_range[0], 0)
    
    def test_extract_dataset_metadata_bao_anisotropic(self):
        """Test anisotropic BAO metadata extraction."""
        data = load_dataset("bao_ani")
        metadata = _extract_dataset_metadata(data, "bao_ani")
        
        self.assertEqual(metadata["dataset_type"], "bao_ani")
        self.assertIn("dm_ratio_range", metadata)
        self.assertIn("h_ratio_range", metadata)
        self.assertIn("observable_types", metadata)
        self.assertEqual(metadata["observable_types"], ["transverse_bao", "radial_bao"])
    
    def test_extract_dataset_metadata_supernova(self):
        """Test supernova-specific metadata extraction."""
        data = load_dataset("sn")
        metadata = _extract_dataset_metadata(data, "sn")
        
        self.assertEqual(metadata["dataset_type"], "sn")
        self.assertIn("distance_modulus_range", metadata)
        self.assertIn("uncertainty_range", metadata)
        self.assertIn("mean_uncertainty", metadata)
        self.assertIn("observable_types", metadata)
        self.assertEqual(metadata["observable_types"], ["distance_modulus"])
    
    def test_compute_condition_number(self):
        """Test condition number computation."""
        # Well-conditioned matrix
        matrix_good = np.eye(3)
        cond_good = _compute_condition_number(matrix_good)
        self.assertAlmostEqual(cond_good, 1.0, places=10)
        
        # Poorly conditioned matrix
        matrix_poor = np.array([[1.0, 0.0], [0.0, 1e-12]])
        cond_poor = _compute_condition_number(matrix_poor)
        self.assertGreater(cond_poor, 1e10)
        
        # Singular matrix (or nearly singular)
        matrix_singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        cond_singular = _compute_condition_number(matrix_singular)
        # Should be very large (either inf or very large finite number)
        self.assertGreater(cond_singular, 1e15)
    
    def test_compute_data_vector_length(self):
        """Test data vector length computation for different dataset types."""
        # CMB always has length 3
        cmb_obs = {"R": 1.0, "l_A": 300.0, "theta_star": 1.0}
        self.assertEqual(_compute_data_vector_length(cmb_obs, "cmb"), 3)
        
        # BAO length equals number of redshift bins
        bao_obs = {"redshift": [0.1, 0.2, 0.3], "DV_over_rs": [4, 5, 6]}
        self.assertEqual(_compute_data_vector_length(bao_obs, "bao"), 3)
        
        # Anisotropic BAO length is 2 * number of redshift bins
        bao_ani_obs = {"redshift": [0.1, 0.2], "DM_over_rs": [10, 12], "H_times_rs": [80, 90]}
        self.assertEqual(_compute_data_vector_length(bao_ani_obs, "bao_ani"), 4)
        
        # Supernova length equals number of supernovae
        sn_obs = {"redshift": [0.1, 0.2, 0.3, 0.4], "distance_modulus": [35, 36, 37, 38]}
        self.assertEqual(_compute_data_vector_length(sn_obs, "sn"), 4)


class TestDataConsistency(unittest.TestCase):
    """Test internal data consistency validation."""
    
    def test_consistent_array_lengths(self):
        """Test validation of consistent array lengths in observations."""
        # Valid consistent data
        consistent_data = {
            "observations": {
                "redshift": [0.1, 0.2, 0.3],
                "values": [1.0, 2.0, 3.0]
            },
            "covariance": np.eye(3),
            "metadata": {"source": "test", "n_data_points": 3, "observables": ["values"]},
            "dataset_type": "bao"
        }
        
        # Should not raise exception
        _validate_data_consistency(consistent_data)
    
    def test_inconsistent_array_lengths(self):
        """Test error handling for inconsistent array lengths."""
        inconsistent_data = {
            "observations": {
                "redshift": [0.1, 0.2, 0.3],
                "values": [1.0, 2.0]  # Different length
            },
            "covariance": np.eye(3),
            "metadata": {"source": "test", "n_data_points": 3, "observables": ["values"]},
            "dataset_type": "bao"
        }
        
        with self.assertRaises(ValueError) as context:
            _validate_data_consistency(inconsistent_data)
        self.assertIn("Inconsistent array lengths", str(context.exception))
    
    def test_covariance_dimension_mismatch(self):
        """Test error handling for covariance dimension mismatch."""
        mismatch_data = {
            "observations": {
                "redshift": [0.1, 0.2, 0.3],
                "DV_over_rs": [4.0, 5.0, 6.0]
            },
            "covariance": np.eye(2),  # Wrong dimension
            "metadata": {"source": "test", "n_data_points": 3, "observables": ["DV_over_rs"]},
            "dataset_type": "bao"
        }
        
        with self.assertRaises(ValueError) as context:
            _validate_data_consistency(mismatch_data)
        self.assertIn("Covariance matrix dimension", str(context.exception))
        self.assertIn("does not match expected data vector length", str(context.exception))


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error handling scenarios."""
    
    def test_dataset_loading_with_mock_failure(self):
        """Test error handling when dataset loading fails."""
        # Test get_dataset_info with loading failure
        with patch('pipelines.fit_core.datasets.load_dataset', side_effect=Exception("Mock loading error")):
            info = get_dataset_info("cmb")
            
            # Should return basic config info with error status
            self.assertIn("error", info)
            self.assertIn("status", info)
            self.assertEqual(info["status"], "unavailable")
            self.assertIn("Mock loading error", info["error"])
    
    def test_edge_cases_in_metadata_extraction(self):
        """Test edge cases in metadata extraction."""
        # Dataset with minimal observations
        minimal_data = {
            "observations": {},
            "covariance": np.eye(1),
            "metadata": {"source": "test", "n_data_points": 1, "observables": []},
            "dataset_type": "cmb"
        }
        
        metadata = _extract_dataset_metadata(minimal_data, "cmb")
        self.assertEqual(metadata["dataset_type"], "cmb")
        self.assertIn("covariance_condition_number", metadata)
    
    def test_numerical_edge_cases(self):
        """Test handling of numerical edge cases."""
        # Very small covariance values
        small_cov = np.eye(2) * 1e-20
        self.assertTrue(_validate_covariance_matrix(small_cov))
        
        # Very large covariance values
        large_cov = np.eye(2) * 1e20
        self.assertTrue(_validate_covariance_matrix(large_cov))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)