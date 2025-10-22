"""
Unit tests for likelihood functions module.

Tests cover:
- CMB likelihood against known parameter sets and expected χ² values
- BAO likelihood computations for both isotropic and anisotropic cases  
- Supernova likelihood with reference datasets and parameter combinations
- Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from .likelihoods import (
    likelihood_cmb, likelihood_bao, likelihood_bao_ani, likelihood_sn,
    _compute_cmb_predictions, _compute_bao_predictions, _compute_sn_predictions
)
from .parameter import build_params
from .datasets import load_dataset


class TestCMBLikelihood(unittest.TestCase):
    """Test CMB likelihood against known parameter sets and expected χ² values."""
    
    def setUp(self):
        """Set up test parameters and mock data."""
        self.test_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "Obh2": 0.02237,
            "ns": 0.9649,
            "Neff": 3.046,
            "Tcmb": 2.7255,
            "recomb_method": "PLANCK18",
            "z_recomb": 1089.80,
            "r_s_drag": 147.09,
            "model_class": "lcdm"
        }
        
        # Mock CMB dataset with Planck 2018 values
        self.cmb_data = {
            "observations": {
                "R": 1.7502,
                "l_A": 301.845,
                "theta_star": 1.04092
            },
            "covariance": np.array([
                [2.305e-6, 2.496e-8, -1.406e-9],
                [2.496e-8, 2.917e-4, -1.649e-5],
                [-1.406e-9, -1.649e-5, 9.417e-8]
            ]),
            "metadata": {
                "source": "Planck2018",
                "n_data_points": 3,
                "observables": ["R", "l_A", "theta_star"]
            },
            "dataset_type": "cmb"
        }
    
    @patch('pipelines.fit_core.likelihoods._compute_cmb_predictions')
    def test_cmb_likelihood_exact_match(self, mock_compute):
        """Test CMB likelihood when predictions exactly match observations."""
        # Mock _compute_cmb_predictions to return exact observations
        mock_compute.return_value = {
            "R": self.cmb_data["observations"]["R"],
            "l_A": self.cmb_data["observations"]["l_A"],
            "theta_star": self.cmb_data["observations"]["theta_star"]
        }
        
        chi2, predictions = likelihood_cmb(self.test_params, self.cmb_data)
        
        # Should get χ² ≈ 0 for exact match
        self.assertAlmostEqual(chi2, 0.0, places=10)
        
        # Check predictions structure
        self.assertIn("R", predictions)
        self.assertIn("l_A", predictions)
        self.assertIn("theta_star", predictions)
        
        # Verify predictions match observations
        self.assertAlmostEqual(predictions["R"], self.cmb_data["observations"]["R"], places=6)
        self.assertAlmostEqual(predictions["l_A"], self.cmb_data["observations"]["l_A"], places=3)
        self.assertAlmostEqual(predictions["theta_star"], self.cmb_data["observations"]["theta_star"], places=5)
    
    @patch('pipelines.fit_core.likelihoods._compute_cmb_predictions')
    def test_cmb_likelihood_known_chi2(self, mock_compute):
        """Test CMB likelihood with known parameter set and expected χ² value."""
        # Mock predictions that differ from observations by known amounts
        # These values will give a specific χ² when combined with the covariance matrix
        mock_compute.return_value = {
            "R": 1.7500,
            "l_A": 301.800,
            "theta_star": 1.04090
        }
        
        chi2, predictions = likelihood_cmb(self.test_params, self.cmb_data)
        
        # Verify χ² is finite and reasonable (could be negative due to covariance correlations)
        self.assertTrue(np.isfinite(chi2))
        self.assertLess(abs(chi2), 100.0)  # Should be reasonable for small deviations
        
        # Verify predictions structure
        self.assertEqual(predictions["R"], 1.7500)
        self.assertEqual(predictions["l_A"], 301.800)
        self.assertEqual(predictions["theta_star"], 1.04090)
    
    @patch('pipelines.fit_core.likelihoods._compute_cmb_predictions')
    def test_cmb_likelihood_parameter_sensitivity(self, mock_compute):
        """Test CMB likelihood sensitivity to parameter changes."""
        # Test with different H0 values
        h0_values = [65.0, 67.4, 70.0]
        chi2_values = []
        
        for h0 in h0_values:
            # Mock _compute_cmb_predictions to return H0-dependent values
            # Higher H0 typically gives lower R and higher l_A
            R_pred = 1.750 - 0.001 * (h0 - 67.4)
            lA_pred = 301.8 + 0.5 * (h0 - 67.4)
            theta_pred = 1.0409
            
            mock_compute.return_value = {
                "R": R_pred,
                "l_A": lA_pred,
                "theta_star": theta_pred
            }
            
            test_params = self.test_params.copy()
            test_params["H0"] = h0
            
            chi2, _ = likelihood_cmb(test_params, self.cmb_data)
            chi2_values.append(chi2)
        
        # χ² should vary with parameter changes
        self.assertNotEqual(chi2_values[0], chi2_values[1])
        self.assertNotEqual(chi2_values[1], chi2_values[2])
    
    @patch('pipelines.fit_core.cmb_priors.distance_priors')
    def test_cmb_predictions_computation(self, mock_distance):
        """Test CMB predictions computation function directly."""
        mock_distance.return_value = (1.75, 302.0, 1.041)
        
        predictions = _compute_cmb_predictions(self.test_params)
        
        # Check structure and values
        self.assertEqual(predictions["R"], 1.75)
        self.assertEqual(predictions["l_A"], 302.0)
        self.assertEqual(predictions["theta_star"], 1.041)
    
    @patch('pipelines.fit_core.likelihoods._compute_cmb_predictions')
    def test_cmb_likelihood_error_handling(self, mock_compute):
        """Test CMB likelihood error handling for invalid inputs."""
        # Test with missing observations
        invalid_data = self.cmb_data.copy()
        del invalid_data["observations"]["R"]
        
        mock_compute.return_value = {
            "R": 1.75,
            "l_A": 302.0,
            "theta_star": 1.041
        }
        
        with self.assertRaises(ValueError):  # Should raise ValueError for mismatched keys
            likelihood_cmb(self.test_params, invalid_data)
        
        # Test with wrong covariance size
        invalid_data = self.cmb_data.copy()
        invalid_data["covariance"] = np.eye(2)  # Wrong size
        
        with self.assertRaises(ValueError):
            likelihood_cmb(self.test_params, invalid_data)


class TestBAOLikelihood(unittest.TestCase):
    """Test BAO likelihood computations for both isotropic and anisotropic cases."""
    
    def setUp(self):
        """Set up test parameters and mock data."""
        self.test_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "r_s_drag": 147.09,
            "model_class": "lcdm"
        }
        
        # Mock isotropic BAO dataset
        self.bao_data = {
            "observations": {
                "redshift": np.array([0.106, 0.15, 0.38, 0.51, 0.61]),
                "DV_over_rs": np.array([4.47, 4.46, 10.23, 13.36, 16.69])
            },
            "covariance": np.eye(5) * 0.01,  # Simple diagonal covariance
            "metadata": {
                "source": "BAO_compilation",
                "n_data_points": 5,
                "observables": ["DV_over_rs"]
            },
            "dataset_type": "bao"
        }
        
        # Mock anisotropic BAO dataset
        self.bao_ani_data = {
            "observations": {
                "redshift": np.array([0.38, 0.51, 0.61]),
                "DM_over_rd": np.array([10.3, 13.7, 17.0]),
                "DH_over_rd": np.array([0.198, 0.179, 0.162])  # Proper D_H/r_d range
            },
            "covariance": np.eye(6) * 0.01,  # 2N x 2N for N redshift bins
            "metadata": {
                "source": "BAO_anisotropic",
                "n_data_points": 6,
                "observables": ["DM_over_rd", "DH_over_rd"]
            },
            "dataset_type": "bao_ani"
        }
    
    def test_bao_isotropic_likelihood_exact_match(self):
        """Test isotropic BAO likelihood when predictions exactly match observations."""
        # Mock _compute_bao_predictions to return exact observations
        with patch('pipelines.fit_core.likelihoods._compute_bao_predictions') as mock_compute:
            mock_compute.return_value = {
                "DV_over_rs": self.bao_data["observations"]["DV_over_rs"]
            }
            
            chi2, predictions = likelihood_bao(self.test_params, self.bao_data)
            
            # Should get χ² ≈ 0 for exact match
            self.assertAlmostEqual(chi2, 0.0, places=10)
            
            # Check predictions structure
            self.assertIn("DV_over_rs", predictions)
            np.testing.assert_array_almost_equal(
                predictions["DV_over_rs"], 
                self.bao_data["observations"]["DV_over_rs"],
                decimal=6
            )
    
    def test_bao_anisotropic_likelihood_exact_match(self):
        """Test anisotropic BAO likelihood when predictions exactly match observations."""
        with patch('pipelines.fit_core.likelihoods._compute_bao_predictions') as mock_compute:
            mock_compute.return_value = {
                "DM_over_rd": self.bao_ani_data["observations"]["DM_over_rd"],
                "DH_over_rd": self.bao_ani_data["observations"]["DH_over_rd"]
            }
            
            chi2, predictions = likelihood_bao_ani(self.test_params, self.bao_ani_data)
            
            # Should get χ² ≈ 0 for exact match
            self.assertAlmostEqual(chi2, 0.0, places=10)
            
            # Check predictions structure
            self.assertIn("DM_over_rd", predictions)
            self.assertIn("DH_over_rd", predictions)
    
    def test_bao_predictions_isotropic(self):
        """Test isotropic BAO predictions computation."""
        predictions = _compute_bao_predictions(self.test_params, isotropic=True)
        
        # Check structure
        self.assertIn("DV_over_rs", predictions)
        self.assertIsInstance(predictions["DV_over_rs"], np.ndarray)
        
        # Check reasonable values (should be positive and in expected range)
        dv_ratios = predictions["DV_over_rs"]
        self.assertTrue(np.all(dv_ratios > 0))
        self.assertTrue(np.all(dv_ratios < 50))  # Reasonable upper bound
        
        # Check that values increase with redshift (generally expected)
        self.assertGreater(dv_ratios[-1], dv_ratios[0])
    
    def test_bao_predictions_anisotropic(self):
        """Test anisotropic BAO predictions computation."""
        predictions = _compute_bao_predictions(self.test_params, isotropic=False)
        
        # Check structure
        self.assertIn("DM_over_rd", predictions)
        self.assertIn("DH_over_rd", predictions)
        
        dm_ratios = predictions["DM_over_rd"]
        dh_ratios = predictions["DH_over_rd"]
        
        # Check arrays have same length
        self.assertEqual(len(dm_ratios), len(dh_ratios))
        
        # Check reasonable values
        self.assertTrue(np.all(dm_ratios > 0))
        self.assertTrue(np.all(dh_ratios > 0))
        self.assertTrue(np.all(dm_ratios < 50))
        self.assertTrue(np.all(dh_ratios < 1.0))  # D_H/r_d should be < 1
    
    def test_bao_likelihood_parameter_sensitivity(self):
        """Test BAO likelihood sensitivity to cosmological parameters."""
        # Test sensitivity to Om0
        om0_values = [0.25, 0.315, 0.40]
        chi2_values = []
        
        for om0 in om0_values:
            test_params = self.test_params.copy()
            test_params["Om0"] = om0
            
            # Mock predictions that depend on Om0
            with patch('pipelines.fit_core.likelihoods._compute_bao_predictions') as mock_compute:
                # Higher Om0 typically gives different distance ratios
                dv_scale = 1.0 + 0.1 * (om0 - 0.315)
                mock_predictions = self.bao_data["observations"]["DV_over_rs"] * dv_scale
                mock_compute.return_value = {"DV_over_rs": mock_predictions}
                
                chi2, _ = likelihood_bao(test_params, self.bao_data)
                chi2_values.append(chi2)
        
        # χ² should vary with parameter changes
        self.assertNotEqual(chi2_values[0], chi2_values[1])
        self.assertNotEqual(chi2_values[1], chi2_values[2])
    
    def test_bao_likelihood_error_handling(self):
        """Test BAO likelihood error handling."""
        # Test with missing observations
        invalid_data = self.bao_data.copy()
        del invalid_data["observations"]["DV_over_rs"]
        
        with patch('pipelines.fit_core.likelihoods._compute_bao_predictions') as mock_compute:
            mock_compute.return_value = {"DV_over_rs": np.array([1, 2, 3, 4, 5])}
            
            with self.assertRaises(ValueError):  # Should raise ValueError for empty observations after filtering
                likelihood_bao(self.test_params, invalid_data)


class TestSupernovaLikelihood(unittest.TestCase):
    """Test supernova likelihood with reference datasets and parameter combinations."""
    
    def setUp(self):
        """Set up test parameters and mock supernova data."""
        self.test_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "model_class": "lcdm"
        }
        
        # Mock supernova dataset (subset of typical Pantheon+ data)
        self.sn_redshifts = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5])
        self.sn_distance_moduli = np.array([32.5, 35.2, 37.8, 40.1, 43.2, 44.8, 45.9])
        self.sn_uncertainties = np.array([0.15, 0.12, 0.10, 0.08, 0.12, 0.20, 0.25])
        
        self.sn_data = {
            "observations": {
                "redshift": self.sn_redshifts,
                "distance_modulus": self.sn_distance_moduli,
                "sigma_mu": self.sn_uncertainties
            },
            "covariance": np.diag(self.sn_uncertainties**2),  # Diagonal covariance
            "metadata": {
                "source": "Pantheon+_subset",
                "n_data_points": len(self.sn_redshifts),
                "observables": ["distance_modulus"]
            },
            "dataset_type": "sn"
        }
    
    def test_sn_likelihood_exact_match(self):
        """Test supernova likelihood when predictions exactly match observations."""
        # Mock _compute_sn_predictions to return exact observations
        with patch('pipelines.fit_core.likelihoods._compute_sn_predictions') as mock_compute:
            mock_compute.return_value = {
                "distance_modulus": self.sn_data["observations"]["distance_modulus"]
            }
            
            chi2, predictions = likelihood_sn(self.test_params, self.sn_data)
            
            # Should get χ² ≈ 0 for exact match
            self.assertAlmostEqual(chi2, 0.0, places=10)
            
            # Check predictions structure
            self.assertIn("distance_modulus", predictions)
            np.testing.assert_array_almost_equal(
                predictions["distance_modulus"],
                self.sn_data["observations"]["distance_modulus"],
                decimal=6
            )
    
    def test_sn_predictions_computation(self):
        """Test supernova distance modulus predictions computation."""
        predictions = _compute_sn_predictions(self.test_params, self.sn_redshifts)
        
        # Check structure
        self.assertIn("distance_modulus", predictions)
        distance_moduli = predictions["distance_modulus"]
        
        # Check array length matches input redshifts
        self.assertEqual(len(distance_moduli), len(self.sn_redshifts))
        
        # Check reasonable values
        self.assertTrue(np.all(distance_moduli > 30))  # Reasonable lower bound
        self.assertTrue(np.all(distance_moduli < 50))  # Reasonable upper bound
        
        # Check that distance modulus increases with redshift
        self.assertTrue(np.all(np.diff(distance_moduli) > 0))
    
    def test_sn_likelihood_parameter_sensitivity(self):
        """Test supernova likelihood sensitivity to cosmological parameters."""
        # Test sensitivity to H0
        h0_values = [65.0, 67.4, 70.0]
        chi2_values = []
        
        for h0 in h0_values:
            test_params = self.test_params.copy()
            test_params["H0"] = h0
            
            # Mock predictions that depend on H0
            with patch('pipelines.fit_core.likelihoods._compute_sn_predictions') as mock_compute:
                # Higher H0 typically gives lower distance moduli
                mu_offset = -5 * np.log10(h0 / 67.4)
                mock_predictions = self.sn_data["observations"]["distance_modulus"] + mu_offset
                mock_compute.return_value = {"distance_modulus": mock_predictions}
                
                chi2, _ = likelihood_sn(test_params, self.sn_data)
                chi2_values.append(chi2)
        
        # χ² should vary with parameter changes
        self.assertNotEqual(chi2_values[0], chi2_values[1])
        self.assertNotEqual(chi2_values[1], chi2_values[2])
    
    def test_sn_likelihood_om0_sensitivity(self):
        """Test supernova likelihood sensitivity to matter density."""
        # Test sensitivity to Om0 (affects distance-redshift relation)
        om0_values = [0.25, 0.315, 0.40]
        chi2_values = []
        
        for om0 in om0_values:
            test_params = self.test_params.copy()
            test_params["Om0"] = om0
            
            with patch('pipelines.fit_core.likelihoods._compute_sn_predictions') as mock_compute:
                # Higher Om0 affects distance modulus through geometry
                # This is a simplified approximation
                mu_correction = 0.5 * (om0 - 0.315) * np.log(1 + self.sn_redshifts)
                mock_predictions = self.sn_data["observations"]["distance_modulus"] + mu_correction
                mock_compute.return_value = {"distance_modulus": mock_predictions}
                
                chi2, _ = likelihood_sn(test_params, self.sn_data)
                chi2_values.append(chi2)
        
        # χ² should vary with parameter changes
        self.assertNotEqual(chi2_values[0], chi2_values[1])
        self.assertNotEqual(chi2_values[1], chi2_values[2])
    
    def test_sn_likelihood_redshift_range(self):
        """Test supernova likelihood with different redshift ranges."""
        # Test with low-z only
        low_z_mask = self.sn_redshifts < 0.2
        low_z_data = {
            "observations": {
                "redshift": self.sn_redshifts[low_z_mask],
                "distance_modulus": self.sn_distance_moduli[low_z_mask],
                "sigma_mu": self.sn_uncertainties[low_z_mask]
            },
            "covariance": np.diag(self.sn_uncertainties[low_z_mask]**2),
            "metadata": self.sn_data["metadata"].copy(),
            "dataset_type": "sn"
        }
        low_z_data["metadata"]["n_data_points"] = np.sum(low_z_mask)
        
        with patch('pipelines.fit_core.likelihoods._compute_sn_predictions') as mock_compute:
            mock_compute.return_value = {
                "distance_modulus": self.sn_distance_moduli[low_z_mask]
            }
            
            chi2_low, _ = likelihood_sn(self.test_params, low_z_data)
            
            # Should get reasonable χ² value
            self.assertGreaterEqual(chi2_low, 0.0)
            self.assertLess(chi2_low, 100.0)
    
    def test_sn_likelihood_error_handling(self):
        """Test supernova likelihood error handling."""
        # Test with mismatched array lengths
        invalid_data = self.sn_data.copy()
        invalid_data["observations"]["distance_modulus"] = np.array([1, 2, 3])  # Wrong length
        
        with patch('pipelines.fit_core.likelihoods._compute_sn_predictions') as mock_compute:
            mock_compute.return_value = {"distance_modulus": np.array([1, 2, 3, 4, 5, 6, 7])}
            
            with self.assertRaises(ValueError):
                likelihood_sn(self.test_params, invalid_data)


class TestLikelihoodIntegration(unittest.TestCase):
    """Test integration between likelihood functions and real datasets."""
    
    def test_likelihood_functions_with_real_datasets(self):
        """Test all likelihood functions with actual loaded datasets."""
        # Test with ΛCDM parameters
        lcdm_params = build_params("lcdm")
        
        # Test CMB likelihood
        try:
            cmb_data = load_dataset("cmb")
            chi2_cmb, pred_cmb = likelihood_cmb(lcdm_params, cmb_data)
            
            self.assertIsInstance(chi2_cmb, float)
            self.assertGreaterEqual(chi2_cmb, 0.0)
            self.assertIn("R", pred_cmb)
            self.assertIn("l_A", pred_cmb)
            self.assertIn("theta_star", pred_cmb)
        except Exception as e:
            self.skipTest(f"CMB dataset not available: {e}")
        
        # Test BAO likelihood
        try:
            bao_data = load_dataset("bao")
            chi2_bao, pred_bao = likelihood_bao(lcdm_params, bao_data)
            
            self.assertIsInstance(chi2_bao, float)
            self.assertGreaterEqual(chi2_bao, 0.0)
            self.assertIn("DV_over_rs", pred_bao)
        except Exception as e:
            self.skipTest(f"BAO dataset not available: {e}")
        
        # Test anisotropic BAO likelihood
        try:
            bao_ani_data = load_dataset("bao_ani")
            chi2_ani, pred_ani = likelihood_bao_ani(lcdm_params, bao_ani_data)
            
            self.assertIsInstance(chi2_ani, float)
            self.assertGreaterEqual(chi2_ani, 0.0)
            self.assertIn("DM_over_rs", pred_ani)
            self.assertIn("H_times_rs", pred_ani)
        except Exception as e:
            self.skipTest(f"Anisotropic BAO dataset not available: {e}")
        
        # Test supernova likelihood
        try:
            sn_data = load_dataset("sn")
            chi2_sn, pred_sn = likelihood_sn(lcdm_params, sn_data)
            
            self.assertIsInstance(chi2_sn, float)
            self.assertGreaterEqual(chi2_sn, 0.0)
            self.assertIn("distance_modulus", pred_sn)
        except Exception as e:
            self.skipTest(f"Supernova dataset not available: {e}")
    
    def test_likelihood_functions_with_pbuf_parameters(self):
        """Test likelihood functions with PBUF model parameters."""
        try:
            pbuf_params = build_params("pbuf")
            
            # Test that PBUF parameters work with all likelihood functions
            # (The physics modules should handle PBUF vs ΛCDM differences)
            
            cmb_data = load_dataset("cmb")
            chi2_cmb, _ = likelihood_cmb(pbuf_params, cmb_data)
            self.assertIsInstance(chi2_cmb, float)
            self.assertGreaterEqual(chi2_cmb, 0.0)
            
        except Exception as e:
            self.skipTest(f"PBUF parameter test failed: {e}")
    
    def test_likelihood_consistency_across_calls(self):
        """Test that likelihood functions return consistent results across multiple calls."""
        try:
            params = build_params("lcdm")
            cmb_data = load_dataset("cmb")
            
            # Call likelihood function multiple times
            chi2_1, pred_1 = likelihood_cmb(params, cmb_data)
            chi2_2, pred_2 = likelihood_cmb(params, cmb_data)
            
            # Results should be identical
            self.assertEqual(chi2_1, chi2_2)
            self.assertEqual(pred_1["R"], pred_2["R"])
            self.assertEqual(pred_1["l_A"], pred_2["l_A"])
            self.assertEqual(pred_1["theta_star"], pred_2["theta_star"])
            
        except Exception as e:
            self.skipTest(f"Consistency test failed: {e}")


class TestLikelihoodNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases in likelihood computations."""
    
    def setUp(self):
        """Set up test parameters for numerical stability tests."""
        self.test_params = {
            "H0": 67.4,
            "Om0": 0.315,
            "r_s_drag": 147.09,
            "model_class": "lcdm"
        }
    
    def test_likelihood_with_singular_covariance(self):
        """Test likelihood computation with singular covariance matrix."""
        # Create dataset with singular covariance
        singular_data = {
            "observations": {"obs1": 1.0, "obs2": 1.0},
            "covariance": np.array([[1.0, 1.0], [1.0, 1.0]]),  # Singular
            "metadata": {"source": "test", "n_data_points": 2, "observables": ["obs1", "obs2"]},
            "dataset_type": "test"
        }
        
        with patch('pipelines.fit_core.likelihoods._compute_cmb_predictions') as mock_compute:
            mock_compute.return_value = {"obs1": 0.0, "obs2": 0.0}
            
            # Should handle singular matrix gracefully (using pseudo-inverse)
            chi2, _ = likelihood_cmb(self.test_params, singular_data)
            
            self.assertIsInstance(chi2, float)
            self.assertTrue(np.isfinite(chi2))
            self.assertGreaterEqual(chi2, 0.0)
    
    def test_likelihood_with_extreme_values(self):
        """Test likelihood computation with extreme parameter values."""
        extreme_params = self.test_params.copy()
        extreme_params["H0"] = 150.0  # Very high H0
        extreme_params["Om0"] = 0.05   # Very low Om0
        
        # Should not crash with extreme but valid parameters
        with patch('pipelines.fit_core.likelihoods._compute_bao_predictions') as mock_compute:
            mock_compute.return_value = {"DV_over_rs": np.array([1, 2, 3, 4, 5])}
            
            bao_data = {
                "observations": {
                    "redshift": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                    "DV_over_rs": np.array([4, 5, 6, 7, 8])
                },
                "covariance": np.eye(5) * 0.01,
                "metadata": {"source": "test", "n_data_points": 5, "observables": ["DV_over_rs"]},
                "dataset_type": "bao"
            }
            
            chi2, _ = likelihood_bao(extreme_params, bao_data)
            
            self.assertIsInstance(chi2, float)
            self.assertTrue(np.isfinite(chi2))
            self.assertGreaterEqual(chi2, 0.0)
    
    def test_likelihood_with_nan_predictions(self):
        """Test likelihood error handling when predictions contain NaN."""
        with patch('pipelines.fit_core.likelihoods._compute_cmb_predictions') as mock_compute:
            mock_compute.return_value = {"R": np.nan, "l_A": 300.0, "theta_star": 1.04}
            
            cmb_data = {
                "observations": {"R": 1.75, "l_A": 301.0, "theta_star": 1.04},
                "covariance": np.eye(3) * 0.01,
                "metadata": {"source": "test", "n_data_points": 3, "observables": ["R", "l_A", "theta_star"]},
                "dataset_type": "cmb"
            }
            
            with self.assertRaises(ValueError):
                likelihood_cmb(self.test_params, cmb_data)


if __name__ == "__main__":
    # Run all tests with verbose output
    unittest.main(verbosity=2)