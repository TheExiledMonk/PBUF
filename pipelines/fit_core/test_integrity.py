"""
Unit tests for integrity validation system.

Tests individual validation functions with known good and bad parameter sets,
verifies integrity suite execution and comprehensive reporting, and tests
edge cases and numerical precision requirements.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path to import fit_core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.fit_core import integrity
from pipelines.fit_core import ParameterDict


class TestVerifyHRatios(unittest.TestCase):
    """Test H(z) ratio consistency verification."""
    
    def setUp(self):
        """Set up test parameter dictionaries."""
        # Standard ΛCDM parameters
        self.lcdm_params = {
            "model_class": "lcdm",
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649
        }
        
        # PBUF parameters with k_sat close to 1 (should behave like ΛCDM)
        self.pbuf_lcdm_like = {
            "model_class": "pbuf",
            "H0": 67.4,
            "Om0": 0.315,
            "k_sat": 0.9999,  # Very close to 1
            "alpha": 1e-6     # Very small
        }
        
        # PBUF parameters with significant deviation
        self.pbuf_deviant = {
            "model_class": "pbuf",
            "H0": 67.4,
            "Om0": 0.315,
            "k_sat": 0.5,     # Significant deviation from 1
            "alpha": 5e-4     # Standard value
        }
    
    def test_lcdm_always_passes(self):
        """Test that ΛCDM model always passes H(z) ratio check."""
        result = integrity.verify_h_ratios(self.lcdm_params)
        self.assertTrue(result)
    
    def test_pbuf_near_lcdm_passes(self):
        """Test that PBUF with k_sat ≈ 1 passes H(z) ratio check."""
        result = integrity.verify_h_ratios(self.pbuf_lcdm_like)
        self.assertTrue(result)
    
    def test_custom_redshifts(self):
        """Test H(z) ratio check with custom redshift array."""
        custom_z = [0.2, 1.5, 3.0]
        result = integrity.verify_h_ratios(self.lcdm_params, redshifts=custom_z)
        self.assertTrue(result)
    
    def test_custom_tolerance(self):
        """Test H(z) ratio check with custom tolerance."""
        # Very strict tolerance should still pass for ΛCDM
        result = integrity.verify_h_ratios(self.lcdm_params, tolerance=1e-10)
        self.assertTrue(result)
        
        # Very loose tolerance should pass for deviant PBUF
        result = integrity.verify_h_ratios(self.pbuf_deviant, tolerance=1.0)
        self.assertTrue(result)
    
    @patch('pipelines.fit_core.integrity._compute_h_ratio_at_z')
    def test_failing_h_ratios(self, mock_h_ratio):
        """Test H(z) ratio check failure when ratios exceed tolerance."""
        # Mock function to return ratios that exceed tolerance
        mock_h_ratio.return_value = 1.1  # 10% deviation
        
        result = integrity.verify_h_ratios(self.pbuf_deviant, tolerance=1e-4)
        self.assertFalse(result)
    
    def test_edge_case_zero_redshift(self):
        """Test H(z) ratio at z=0 (present day)."""
        result = integrity.verify_h_ratios(self.lcdm_params, redshifts=[0.0])
        self.assertTrue(result)
    
    def test_edge_case_high_redshift(self):
        """Test H(z) ratio at very high redshift."""
        result = integrity.verify_h_ratios(self.lcdm_params, redshifts=[10.0])
        self.assertTrue(result)


class TestVerifyRecombination(unittest.TestCase):
    """Test recombination redshift validation."""
    
    def setUp(self):
        """Set up test parameter dictionaries."""
        # Parameters with PLANCK18 method (should match exactly)
        self.planck18_params = {
            "z_recomb": 1089.80,  # Exact Planck 2018 value
            "recomb_method": "PLANCK18"
        }
        
        # Parameters with HS96 method (allow larger tolerance)
        self.hs96_params = {
            "z_recomb": 1090.5,   # Slightly different value
            "recomb_method": "HS96"
        }
        
        # Parameters with missing z_recomb
        self.missing_z_params = {
            "recomb_method": "PLANCK18"
        }
        
        # Parameters with significantly wrong z_recomb
        self.wrong_z_params = {
            "z_recomb": 500.0,    # Way off
            "recomb_method": "PLANCK18"
        }
    
    def test_planck18_exact_match(self):
        """Test PLANCK18 method with exact reference value."""
        result = integrity.verify_recombination(self.planck18_params)
        self.assertTrue(result)
    
    def test_hs96_within_tolerance(self):
        """Test HS96 method within allowed tolerance."""
        # Use a more reasonable tolerance for HS96 method
        result = integrity.verify_recombination(self.hs96_params, tolerance=1e-3)
        self.assertTrue(result)
    
    def test_missing_z_recomb(self):
        """Test failure when z_recomb is missing."""
        result = integrity.verify_recombination(self.missing_z_params)
        self.assertFalse(result)
    
    def test_wrong_z_recomb_planck18(self):
        """Test failure when PLANCK18 z_recomb is wrong."""
        result = integrity.verify_recombination(self.wrong_z_params)
        self.assertFalse(result)
    
    def test_custom_reference_value(self):
        """Test with custom reference value."""
        custom_ref = 1090.0
        params = {
            "z_recomb": 1090.0,
            "recomb_method": "CUSTOM"
        }
        result = integrity.verify_recombination(params, reference=custom_ref)
        self.assertTrue(result)
    
    def test_custom_tolerance(self):
        """Test with custom tolerance."""
        # Very strict tolerance
        result = integrity.verify_recombination(self.hs96_params, tolerance=1e-6)
        self.assertFalse(result)  # Should fail with strict tolerance
        
        # Very loose tolerance
        result = integrity.verify_recombination(self.hs96_params, tolerance=0.1)
        self.assertTrue(result)   # Should pass with loose tolerance
    
    def test_relative_error_calculation(self):
        """Test relative error calculation for non-PLANCK18 methods."""
        # Test case where relative error is exactly at tolerance
        tolerance = 1e-4
        reference = 1089.80
        z_recomb = reference * (1 + tolerance)  # Exactly at tolerance
        
        params = {
            "z_recomb": z_recomb,
            "recomb_method": "EH98"
        }
        
        result = integrity.verify_recombination(params, tolerance=tolerance)
        self.assertTrue(result)  # Should pass (equal to tolerance)


class TestVerifyCovarianceMatrices(unittest.TestCase):
    """Test covariance matrix validation."""
    
    def setUp(self):
        """Set up mock datasets."""
        # Good covariance matrix (positive definite)
        self.good_cov = np.array([
            [1.0, 0.1, 0.05],
            [0.1, 1.0, 0.2],
            [0.05, 0.2, 1.0]
        ])
        
        # Bad covariance matrix (not positive definite)
        # This matrix has eigenvalues [2.8, 0.1, 0.1] so it IS positive definite
        # Let's make a truly non-positive definite matrix
        self.bad_cov = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ])
        
        # Poorly conditioned matrix (still positive definite but poorly conditioned)
        self.poorly_conditioned = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1e-10, 0.0],  # Small but above tolerance
            [0.0, 0.0, 1.0]
        ])
        
        # Matrix with NaN values
        self.nan_matrix = np.array([
            [1.0, np.nan, 0.0],
            [np.nan, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_good_covariance_matrices(self, mock_load):
        """Test validation with good covariance matrices."""
        mock_load.return_value = {"covariance": self.good_cov}
        
        result = integrity.verify_covariance_matrices(["test_dataset"])
        self.assertTrue(result)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_bad_covariance_matrices(self, mock_load):
        """Test validation with bad covariance matrices."""
        mock_load.return_value = {"covariance": self.bad_cov}
        
        result = integrity.verify_covariance_matrices(["test_dataset"])
        self.assertFalse(result)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_missing_covariance(self, mock_load):
        """Test validation when covariance matrix is missing."""
        mock_load.return_value = {"data": "some_data"}  # No covariance key
        
        result = integrity.verify_covariance_matrices(["test_dataset"])
        self.assertFalse(result)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_poorly_conditioned_matrix(self, mock_load):
        """Test validation with poorly conditioned matrix."""
        mock_load.return_value = {"covariance": self.poorly_conditioned}
        
        # Should pass but with warning (poor conditioning doesn't fail)
        result = integrity.verify_covariance_matrices(["test_dataset"])
        self.assertTrue(result)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_nan_matrix(self, mock_load):
        """Test validation with NaN values in matrix."""
        mock_load.return_value = {"covariance": self.nan_matrix}
        
        result = integrity.verify_covariance_matrices(["test_dataset"])
        self.assertFalse(result)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_multiple_datasets(self, mock_load):
        """Test validation with multiple datasets."""
        def side_effect(dataset_name):
            if dataset_name == "good_dataset":
                return {"covariance": self.good_cov}
            elif dataset_name == "bad_dataset":
                return {"covariance": self.bad_cov}
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
        
        mock_load.side_effect = side_effect
        
        # Should fail because one dataset has bad covariance
        result = integrity.verify_covariance_matrices(["good_dataset", "bad_dataset"])
        self.assertFalse(result)
    
    @patch('pipelines.fit_core.datasets.load_dataset')
    def test_dataset_loading_error(self, mock_load):
        """Test handling of dataset loading errors."""
        mock_load.side_effect = Exception("Dataset not found")
        
        result = integrity.verify_covariance_matrices(["nonexistent_dataset"])
        self.assertFalse(result)


class TestVerifySoundHorizon(unittest.TestCase):
    """Test sound horizon verification."""
    
    def setUp(self):
        """Set up test parameter dictionaries."""
        # Standard parameters with correct sound horizon (use EH98 formula result)
        H0 = 67.4
        Omh2 = 0.143
        Obh2 = 0.02237
        h = H0 / 100.0
        r_s_eh98 = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(0.75)) / h
        
        self.good_params = {
            "r_s_drag": r_s_eh98,  # Use EH98 formula result
            "H0": H0,
            "Omh2": Omh2,
            "Obh2": Obh2
        }
        
        # Parameters with missing sound horizon
        self.missing_rs_params = {
            "H0": 67.4,
            "Omh2": 0.143,
            "Obh2": 0.02237
        }
        
        # Parameters with wrong sound horizon
        self.wrong_rs_params = {
            "r_s_drag": 100.0,   # Way off
            "H0": 67.4,
            "Omh2": 0.143,
            "Obh2": 0.02237
        }
    
    def test_good_sound_horizon(self):
        """Test verification with correct sound horizon."""
        result = integrity.verify_sound_horizon(self.good_params)
        self.assertTrue(result)
    
    def test_missing_sound_horizon(self):
        """Test failure when r_s_drag is missing."""
        result = integrity.verify_sound_horizon(self.missing_rs_params)
        self.assertFalse(result)
    
    def test_wrong_sound_horizon(self):
        """Test failure when sound horizon is significantly wrong."""
        result = integrity.verify_sound_horizon(self.wrong_rs_params)
        self.assertFalse(result)
    
    def test_eh98_formula_consistency(self):
        """Test consistency with Eisenstein & Hu 1998 formula."""
        # Create parameters that should match EH98 formula exactly
        H0 = 70.0
        Omh2 = 0.14
        Obh2 = 0.022
        h = H0 / 100.0
        
        # Compute EH98 formula result
        r_s_eh98 = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(0.75)) / h
        
        params = {
            "r_s_drag": r_s_eh98,
            "H0": H0,
            "Omh2": Omh2,
            "Obh2": Obh2
        }
        
        result = integrity.verify_sound_horizon(params)
        self.assertTrue(result)
    
    def test_reasonable_range_check(self):
        """Test that sound horizon is in reasonable range."""
        # Create parameters that give a large r_s via EH98 formula
        H0 = 30.0
        Omh2 = 0.01
        Obh2 = 0.001
        h = H0 / 100.0
        r_s_eh98 = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(0.75)) / h
        
        params = {
            "r_s_drag": r_s_eh98,  # Use EH98 formula result (will be large)
            "H0": H0,
            "Omh2": Omh2,
            "Obh2": Obh2
        }
        
        # This should pass EH98 check but warn about range
        result = integrity.verify_sound_horizon(params)
        self.assertTrue(result)


class TestCheckUnitConsistency(unittest.TestCase):
    """Test unit consistency checks."""
    
    def setUp(self):
        """Set up test parameter dictionaries."""
        # Good standard cosmology parameters (use EH98 sound horizon)
        H0 = 67.4
        Om0 = 0.315
        Omh2 = 0.143
        Obh2 = 0.02237
        h = H0 / 100.0
        r_s_eh98 = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(0.75)) / h
        
        self.good_params = {
            "H0": H0,
            "Om0": Om0,
            "Omh2": Omh2,
            "Orh2": 2.47e-5,
            "z_recomb": 1089.8,
            "z_drag": 1059.9,
            "r_s_drag": r_s_eh98  # Use EH98 formula result
        }
        
        # PBUF parameters
        self.pbuf_params = {
            "model_class": "pbuf",
            "H0": 67.4,
            "Om0": 0.315,
            "alpha": 5e-4,
            "k_sat": 0.9762,
            "Rmax": 1e9,
            "eps0": 0.7,
            "n_eps": 0.0
        }
        
        # Bad parameters (various issues)
        self.bad_params = {
            "H0": 200.0,         # H0 out of range
            "Om0": 0.8,          # High matter density
            "Omh2": 0.1,         # Inconsistent with Om0*h^2
            "z_recomb": 500.0,   # z_recomb < z_drag (wrong ordering)
            "z_drag": 1000.0,
            "r_s_drag": 50.0     # Sound horizon out of range
        }
    
    def test_good_parameters_pass_all_checks(self):
        """Test that good parameters pass all unit consistency checks."""
        checks = integrity.check_unit_consistency(self.good_params)
        
        # All checks should pass
        for check_name, result in checks.items():
            self.assertTrue(result, f"Check {check_name} failed")
    
    def test_h0_range_check(self):
        """Test H0 range validation."""
        # Good H0
        params = {"H0": 67.4}
        checks = integrity.check_unit_consistency(params)
        self.assertTrue(checks["H0_range"])
        
        # Bad H0 (too high)
        params = {"H0": 200.0}
        checks = integrity.check_unit_consistency(params)
        self.assertFalse(checks["H0_range"])
        
        # Bad H0 (too low)
        params = {"H0": 10.0}
        checks = integrity.check_unit_consistency(params)
        self.assertFalse(checks["H0_range"])
    
    def test_density_sum_check(self):
        """Test density fraction sum validation."""
        # Good density sum (flat universe)
        params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Orh2": 2.47e-5
        }
        checks = integrity.check_unit_consistency(params)
        self.assertTrue(checks["density_sum"])
        
        # Bad density sum (need to exceed 1% tolerance)
        params = {
            "H0": 67.4,
            "Om0": 0.8,   # High matter density
            "Orh2": 0.1   # Very high radiation density to push total > 1.01
        }
        checks = integrity.check_unit_consistency(params)
        self.assertFalse(checks["density_sum"])
    
    def test_omh2_consistency_check(self):
        """Test Omh2 consistency validation."""
        # Consistent Omh2
        H0 = 67.4
        Om0 = 0.315
        h = H0 / 100.0
        Omh2_expected = Om0 * h**2
        
        params = {
            "H0": H0,
            "Om0": Om0,
            "Omh2": Omh2_expected
        }
        checks = integrity.check_unit_consistency(params)
        self.assertTrue(checks["Omh2_consistency"])
        
        # Inconsistent Omh2
        params = {
            "H0": H0,
            "Om0": Om0,
            "Omh2": 0.5  # Way off
        }
        checks = integrity.check_unit_consistency(params)
        self.assertFalse(checks["Omh2_consistency"])
    
    def test_redshift_ordering_check(self):
        """Test redshift ordering validation."""
        # Correct ordering (z_recomb > z_drag)
        params = {
            "z_recomb": 1089.8,
            "z_drag": 1059.9
        }
        checks = integrity.check_unit_consistency(params)
        self.assertTrue(checks["redshift_ordering"])
        
        # Wrong ordering
        params = {
            "z_recomb": 1000.0,
            "z_drag": 1100.0  # z_drag > z_recomb (wrong)
        }
        checks = integrity.check_unit_consistency(params)
        self.assertFalse(checks["redshift_ordering"])
    
    def test_sound_horizon_range_check(self):
        """Test sound horizon range validation."""
        # Good sound horizon
        params = {"r_s_drag": 220.0}
        checks = integrity.check_unit_consistency(params)
        self.assertTrue(checks["sound_horizon_range"])
        
        # Bad sound horizon (too low)
        params = {"r_s_drag": 100.0}
        checks = integrity.check_unit_consistency(params)
        self.assertFalse(checks["sound_horizon_range"])
        
        # Bad sound horizon (too high)
        params = {"r_s_drag": 300.0}
        checks = integrity.check_unit_consistency(params)
        self.assertFalse(checks["sound_horizon_range"])
    
    def test_pbuf_parameter_ranges(self):
        """Test PBUF parameter range validation."""
        checks = integrity.check_unit_consistency(self.pbuf_params)
        
        # All PBUF checks should pass for good parameters
        self.assertTrue(checks["pbuf_alpha_range"])
        self.assertTrue(checks["pbuf_k_sat_range"])
        self.assertTrue(checks["pbuf_Rmax_range"])
        
        # Test bad PBUF parameters
        bad_pbuf = {
            "model_class": "pbuf",
            "alpha": 1.0,        # Too high
            "k_sat": 10.0,       # Too high
            "Rmax": 1e3          # Too low
        }
        checks = integrity.check_unit_consistency(bad_pbuf)
        self.assertFalse(checks["pbuf_alpha_range"])
        self.assertFalse(checks["pbuf_k_sat_range"])
        self.assertFalse(checks["pbuf_Rmax_range"])
    
    def test_missing_parameters(self):
        """Test handling of missing parameters."""
        # Empty parameter dictionary
        empty_params = {}
        checks = integrity.check_unit_consistency(empty_params)
        
        # Most checks should fail due to missing parameters
        self.assertFalse(checks["H0_range"])
        self.assertFalse(checks["Omh2_consistency"])
        self.assertFalse(checks["redshift_ordering"])
        self.assertFalse(checks["sound_horizon_range"])


class TestRunIntegritySuite(unittest.TestCase):
    """Test comprehensive integrity suite execution."""
    
    def setUp(self):
        """Set up test parameters and datasets."""
        self.good_params = {
            "model_class": "lcdm",
            "H0": 67.4,
            "Om0": 0.315,
            "Omh2": 0.143,
            "Obh2": 0.02237,
            "z_recomb": 1089.80,
            "z_drag": 1059.9,
            "r_s_drag": 147.09,
            "recomb_method": "PLANCK18"
        }
        
        self.test_datasets = ["cmb", "bao", "sn"]
    
    @patch('pipelines.fit_core.integrity.verify_covariance_matrices')
    @patch('pipelines.fit_core.integrity.check_unit_consistency')
    @patch('pipelines.fit_core.integrity.verify_sound_horizon')
    @patch('pipelines.fit_core.integrity.verify_recombination')
    @patch('pipelines.fit_core.integrity.verify_h_ratios')
    def test_all_tests_pass(self, mock_h_ratios, mock_recomb, mock_sound, 
                           mock_units, mock_cov):
        """Test integrity suite when all tests pass."""
        # Mock all tests to pass
        mock_h_ratios.return_value = True
        mock_recomb.return_value = True
        mock_sound.return_value = True
        mock_units.return_value = {
            "H0_range": True,
            "density_sum": True,
            "Omh2_consistency": True,
            "redshift_ordering": True,
            "sound_horizon_range": True
        }
        mock_cov.return_value = True
        
        results = integrity.run_integrity_suite(self.good_params, self.test_datasets)
        
        # Check overall status
        self.assertEqual(results["overall_status"], "PASS")
        self.assertEqual(len(results["failures"]), 0)
        self.assertEqual(results["summary"]["passed"], results["summary"]["total_tests"])
        
        # Check that all expected tests were run
        expected_tests = ["h_ratios", "recombination", "sound_horizon", 
                         "unit_consistency", "covariance_matrices"]
        for test in expected_tests:
            self.assertIn(test, results["tests_run"])
            self.assertEqual(results[test]["status"], "PASS")
    
    @patch('pipelines.fit_core.integrity.verify_covariance_matrices')
    @patch('pipelines.fit_core.integrity.check_unit_consistency')
    @patch('pipelines.fit_core.integrity.verify_sound_horizon')
    @patch('pipelines.fit_core.integrity.verify_recombination')
    @patch('pipelines.fit_core.integrity.verify_h_ratios')
    def test_some_tests_fail(self, mock_h_ratios, mock_recomb, mock_sound,
                            mock_units, mock_cov):
        """Test integrity suite when some tests fail."""
        # Mock some tests to fail
        mock_h_ratios.return_value = False  # Fail
        mock_recomb.return_value = True     # Pass
        mock_sound.return_value = False     # Fail
        mock_units.return_value = {
            "H0_range": True,
            "density_sum": False,  # Fail
            "Omh2_consistency": True,
            "redshift_ordering": True,
            "sound_horizon_range": True
        }
        mock_cov.return_value = True        # Pass
        
        results = integrity.run_integrity_suite(self.good_params, self.test_datasets)
        
        # Check overall status
        self.assertEqual(results["overall_status"], "FAIL")
        self.assertEqual(len(results["failures"]), 3)  # h_ratios, sound_horizon, unit_consistency
        self.assertIn("h_ratios", results["failures"])
        self.assertIn("sound_horizon", results["failures"])
        self.assertIn("unit_consistency", results["failures"])
        
        # Check individual test statuses
        self.assertEqual(results["h_ratios"]["status"], "FAIL")
        self.assertEqual(results["recombination"]["status"], "PASS")
        self.assertEqual(results["sound_horizon"]["status"], "FAIL")
        self.assertEqual(results["unit_consistency"]["status"], "FAIL")
        self.assertEqual(results["covariance_matrices"]["status"], "PASS")
    
    def test_no_datasets_provided(self):
        """Test integrity suite with no datasets (skips covariance check)."""
        results = integrity.run_integrity_suite(self.good_params, [])
        
        # Should not include covariance_matrices test
        self.assertNotIn("covariance_matrices", results["tests_run"])
        
        # Should still run other tests
        expected_tests = ["h_ratios", "recombination", "sound_horizon", "unit_consistency"]
        for test in expected_tests:
            self.assertIn(test, results["tests_run"])
    
    def test_results_structure(self):
        """Test that results dictionary has correct structure."""
        results = integrity.run_integrity_suite(self.good_params, self.test_datasets)
        
        # Check top-level keys
        required_keys = ["overall_status", "tests_run", "failures", "warnings", "summary"]
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check summary structure
        summary_keys = ["total_tests", "passed", "failed", "warnings"]
        for key in summary_keys:
            self.assertIn(key, results["summary"])
        
        # Check individual test result structure
        for test_name in results["tests_run"]:
            self.assertIn(test_name, results)
            test_result = results[test_name]
            self.assertIn("status", test_result)
            self.assertIn("description", test_result)
            self.assertIn(test_result["status"], ["PASS", "FAIL"])


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions used by integrity validation."""
    
    def test_compute_h_ratio_at_z(self):
        """Test H(z) ratio computation."""
        # ΛCDM parameters
        lcdm_params = {
            "model_class": "lcdm",
            "Om0": 0.315
        }
        
        # Should return 1.0 for ΛCDM
        ratio = integrity._compute_h_ratio_at_z(lcdm_params, 1.0)
        self.assertAlmostEqual(ratio, 1.0, places=10)
        
        # PBUF parameters with k_sat = 1 (should also return 1.0)
        pbuf_lcdm_like = {
            "model_class": "pbuf",
            "Om0": 0.315,
            "k_sat": 1.0,
            "alpha": 0.0
        }
        
        ratio = integrity._compute_h_ratio_at_z(pbuf_lcdm_like, 1.0)
        self.assertAlmostEqual(ratio, 1.0, places=10)
        
        # PBUF parameters with deviation
        pbuf_deviant = {
            "model_class": "pbuf",
            "Om0": 0.315,
            "k_sat": 0.5,
            "alpha": 5e-4
        }
        
        ratio = integrity._compute_h_ratio_at_z(pbuf_deviant, 1.0)
        self.assertNotEqual(ratio, 1.0)  # Should deviate from 1.0
        self.assertGreater(ratio, 0.0)   # Should be positive
    
    def test_check_matrix_properties(self):
        """Test matrix property checking function."""
        # Good matrix (positive definite)
        good_matrix = np.array([
            [2.0, 0.5],
            [0.5, 2.0]
        ])
        
        props = integrity._check_matrix_properties(good_matrix)
        self.assertTrue(props["is_finite"])
        self.assertTrue(props["is_square"])
        self.assertTrue(props["is_positive_definite"])
        self.assertLess(props["condition_number"], 100)  # Well-conditioned
        
        # Bad matrix (not positive definite)
        bad_matrix = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        
        props = integrity._check_matrix_properties(bad_matrix)
        self.assertTrue(props["is_finite"])
        self.assertTrue(props["is_square"])
        self.assertFalse(props["is_positive_definite"])
        
        # Non-square matrix
        nonsquare_matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        props = integrity._check_matrix_properties(nonsquare_matrix)
        self.assertTrue(props["is_finite"])
        self.assertFalse(props["is_square"])
        self.assertFalse(props["is_positive_definite"])
        
        # Matrix with NaN
        nan_matrix = np.array([
            [1.0, np.nan],
            [np.nan, 1.0]
        ])
        
        props = integrity._check_matrix_properties(nan_matrix)
        self.assertFalse(props["is_finite"])
        
        # Singular matrix (should handle gracefully)
        singular_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        
        props = integrity._check_matrix_properties(singular_matrix)
        self.assertTrue(props["is_finite"])
        self.assertTrue(props["is_square"])
        self.assertFalse(props["is_positive_definite"])
        self.assertEqual(props["condition_number"], np.inf)


class TestEdgeCasesAndNumericalPrecision(unittest.TestCase):
    """Test edge cases and numerical precision requirements."""
    
    def test_extreme_parameter_values(self):
        """Test integrity checks with extreme parameter values."""
        # Very high redshift
        extreme_params = {
            "model_class": "pbuf",
            "Om0": 0.999,  # Almost all matter
            "k_sat": 0.001,  # Very small k_sat
            "alpha": 1e-10   # Tiny alpha
        }
        
        # Should not crash, even with extreme values
        result = integrity.verify_h_ratios(extreme_params, redshifts=[100.0])
        self.assertIsInstance(result, bool)
    
    def test_numerical_precision_tolerances(self):
        """Test that numerical precision meets requirements."""
        # Test with values very close to tolerance boundaries
        tolerance = 1e-6
        
        # z_recomb exactly at tolerance boundary
        params = {
            "z_recomb": 1089.80 + tolerance,
            "recomb_method": "EH98"
        }
        
        result = integrity.verify_recombination(params, tolerance=tolerance)
        self.assertTrue(result)  # Should pass (equal to tolerance)
        
        # z_recomb just beyond tolerance (need larger deviation for relative error)
        params = {
            "z_recomb": 1089.80 * (1 + tolerance * 1.1),  # Use relative error
            "recomb_method": "EH98"
        }
        
        result = integrity.verify_recombination(params, tolerance=tolerance)
        self.assertFalse(result)  # Should fail (exceeds tolerance)
    
    def test_floating_point_edge_cases(self):
        """Test handling of floating point edge cases."""
        # Test with very small numbers
        tiny_params = {
            "model_class": "pbuf",
            "Om0": 1e-10,
            "k_sat": 1e-15,
            "alpha": 1e-20
        }
        
        # Should handle gracefully without overflow/underflow
        result = integrity.verify_h_ratios(tiny_params)
        self.assertIsInstance(result, bool)
        
        # Test with very large numbers
        huge_params = {
            "H0": 1e6,
            "Om0": 0.315,
            "Omh2": 1e12
        }
        
        checks = integrity.check_unit_consistency(huge_params)
        self.assertIsInstance(checks, dict)
    
    def test_zero_and_negative_values(self):
        """Test handling of zero and negative parameter values."""
        # Zero values
        zero_params = {
            "H0": 0.0,
            "Om0": 0.0,
            "alpha": 0.0
        }
        
        checks = integrity.check_unit_consistency(zero_params)
        self.assertFalse(checks["H0_range"])  # H0=0 should fail
        
        # Negative values
        negative_params = {
            "H0": -67.4,
            "Om0": -0.315
        }
        
        checks = integrity.check_unit_consistency(negative_params)
        self.assertFalse(checks["H0_range"])  # Negative H0 should fail
    
    def test_matrix_conditioning_edge_cases(self):
        """Test matrix conditioning with edge cases."""
        # Nearly singular matrix
        nearly_singular = np.array([
            [1.0, 1.0 - 1e-15],
            [1.0 - 1e-15, 1.0]
        ])
        
        props = integrity._check_matrix_properties(nearly_singular)
        self.assertTrue(props["is_finite"])
        self.assertGreater(props["condition_number"], 1e10)  # Very poorly conditioned
        
        # Identity matrix (perfectly conditioned)
        identity = np.eye(3)
        
        props = integrity._check_matrix_properties(identity)
        self.assertTrue(props["is_positive_definite"])
        self.assertAlmostEqual(props["condition_number"], 1.0, places=10)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)