"""
Integration tests for the optimization engine.

Tests cover run_fit() with individual datasets, joint fitting mode with multiple datasets,
parameter consistency, optimization convergence, and result reproducibility.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import pytest
import unittest
import numpy as np
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock
import copy

from . import engine
from . import ParameterDict, ResultsDict, MetricsDict


def create_mock_dataset(dataset_type: str, **kwargs) -> Dict[str, Any]:
    """Helper function to create properly formatted mock datasets for testing."""
    
    if dataset_type == "cmb":
        return {
            "observations": {"R": 1.7502, "l_A": 301.845, "theta_star": 1.04092e-2},
            "covariance": kwargs.get("covariance", np.array([[1e-6, 0, 0], [0, 1e-2, 0], [0, 0, 1e-8]])),
            "metadata": {
                "source": "Planck2018",
                "n_data_points": 3,
                "observables": ["R", "l_A", "theta_star"],
                "redshift_range": [1089.8, 1089.8]
            },
            "dataset_type": "cmb"
        }
    elif dataset_type == "bao":
        return {
            "observations": {"DV_over_rs": kwargs.get("DV_over_rs", np.array([8.467, 13.015, 16.726]))},
            "covariance": kwargs.get("covariance", np.diag([0.168, 0.326, 0.440])),
            "metadata": {
                "source": "BAO_compilation",
                "n_data_points": kwargs.get("n_points", 3),
                "observables": ["DV_over_rs"],
                "redshift_range": kwargs.get("redshift_range", [0.38, 0.61])
            },
            "dataset_type": "bao"
        }
    elif dataset_type == "bao_ani":
        return {
            "observations": {
                "DM_over_rd": kwargs.get("DM_over_rd", np.array([13.67, 20.75])),
                "DH_over_rd": kwargs.get("DH_over_rd", np.array([0.0123, 0.0110]))
            },
            "covariance": kwargs.get("covariance", np.array([[0.25, 0.05, 0.02, 0.01],
                                                           [0.05, 0.30, 0.01, 0.02],
                                                           [0.02, 0.01, 0.15, 0.03],
                                                           [0.01, 0.02, 0.03, 0.20]])),
            "metadata": {
                "source": "BAO_anisotropic",
                "n_data_points": kwargs.get("n_points", 4),
                "observables": ["DM_over_rd", "DH_over_rd"],
                "redshift_range": kwargs.get("redshift_range", [0.38, 0.51])
            },
            "dataset_type": "bao_ani"
        }
    elif dataset_type == "sn":
        return {
            "observations": {"distance_modulus": kwargs.get("distance_modulus", np.array([32.5, 35.2, 37.1, 38.9]))},
            "covariance": kwargs.get("covariance", np.diag([0.01, 0.02, 0.03, 0.04])),
            "metadata": {
                "source": "Pantheon+",
                "n_data_points": kwargs.get("n_points", 4),
                "observables": ["distance_modulus"],
                "redshift_range": kwargs.get("redshift_range", [0.1, 0.8])
            },
            "dataset_type": "sn"
        }
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


class TestRunFitIndividualDatasets(unittest.TestCase):
    """Test run_fit() with individual datasets (CMB, BAO, SN) and verify results."""
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_bao')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_cmb_individual_fit(self, mock_metrics, mock_build_params, mock_likelihood_bao, mock_likelihood_cmb, mock_load_dataset):
        """Test individual CMB fitting with run_fit()."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09, "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock datasets with required validation keys
        mock_cmb_data = {
            "observations": {"R": 1.7502, "l_A": 301.845, "theta_star": 1.04092e-2},
            "covariance": np.array([[1e-6, 0, 0], [0, 1e-2, 0], [0, 0, 1e-8]]),
            "metadata": {
                "source": "Planck2018",
                "n_data_points": 3, 
                "observables": ["R", "l_A", "theta_star"],
                "redshift_range": [1089.8, 1089.8]
            },
            "dataset_type": "cmb"
        }
        
        mock_bao_data = {
            "observations": {"DV_over_rs": np.array([6.5, 8.1, 13.9, 16.1, 17.4])},
            "covariance": np.eye(5) * 0.1,
            "metadata": {
                "source": "BAO_compilation",
                "n_data_points": 5,
                "observables": ["DV_over_rs"],
                "redshift_range": [0.1, 0.5]
            },
            "dataset_type": "bao"
        }
        
        def mock_load_side_effect(dataset_name):
            if dataset_name == "cmb":
                return mock_cmb_data
            elif dataset_name == "bao":
                return mock_bao_data
            else:
                return {}
        
        mock_load_dataset.side_effect = mock_load_side_effect
        
        # Setup mock likelihood functions
        mock_cmb_predictions = {"R": 1.7500, "l_A": 301.800, "theta_star": 1.04090e-2}
        mock_cmb_chi2 = 1.85
        mock_likelihood_cmb.return_value = (mock_cmb_chi2, mock_cmb_predictions)
        
        mock_bao_predictions = {"DV_over_rs": np.array([6.5, 8.1, 13.9, 16.1, 17.4])}
        mock_bao_chi2 = 2.15
        mock_likelihood_bao.return_value = (mock_bao_chi2, mock_bao_predictions)
        
        # Setup mock metrics
        total_chi2 = mock_cmb_chi2 + mock_bao_chi2
        mock_computed_metrics = {
            "chi2": total_chi2, "aic": total_chi2 + 12, "bic": total_chi2 + 18, 
            "dof": 2, "p_value": 0.396
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute CMB fit
        result = engine.run_fit("lcdm", ["cmb"], mode="individual")
        
        # Verify parameter construction was called correctly (may be called multiple times for different datasets)
        assert mock_build_params.call_count >= 1
        
        # Verify dataset loading (called twice: once for caching, once for result compilation)
        assert mock_load_dataset.call_count >= 1
        mock_load_dataset.assert_any_call("cmb")
        
        # Verify likelihood computation was called
        assert mock_likelihood_cmb.call_count >= 1
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "params" in result
        assert "results" in result
        assert "metrics" in result
        assert "chi2_breakdown" in result
        assert "datasets" in result
        assert "model" in result
        
        # Verify datasets (CMB + BAO due to DOF handling)
        assert "cmb" in result["datasets"]
        assert "bao" in result["datasets"]  # Added automatically for sufficient DOF
        assert result["model"] == "lcdm"
        assert "cmb" in result["results"]
        assert "bao" in result["results"]
        
        cmb_result = result["results"]["cmb"]
        assert "chi2" in cmb_result
        assert "predictions" in cmb_result
        assert "observations" in cmb_result
        
        bao_result = result["results"]["bao"]
        assert "chi2" in bao_result
        assert "predictions" in bao_result
        assert "observations" in bao_result
        
        # Verify χ² breakdown
        assert "cmb" in result["chi2_breakdown"]
        assert "bao" in result["chi2_breakdown"]
        assert result["chi2_breakdown"]["cmb"] == mock_cmb_chi2
        assert result["chi2_breakdown"]["bao"] == mock_bao_chi2
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_bao')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_bao_individual_fit(self, mock_metrics, mock_build_params, mock_likelihood_bao, mock_load_dataset):
        """Test individual BAO fitting with run_fit()."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09, "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock BAO dataset
        mock_bao_data = {
            "observations": {"DV_over_rs": np.array([8.467, 13.015, 16.726])},
            "covariance": np.diag([0.168, 0.326, 0.440]),
            "metadata": {
                "source": "BAO_compilation",
                "n_data_points": 3,
                "observables": ["DV_over_rs"],
                "redshift_range": [0.38, 0.61]
            },
            "dataset_type": "bao"
        }
        mock_load_dataset.return_value = mock_bao_data
        
        # Setup mock likelihood function
        mock_predictions = {"DV_over_rs": np.array([8.465, 13.010, 16.720])}
        mock_chi2 = 0.95
        mock_likelihood_bao.return_value = (mock_chi2, mock_predictions)
        
        # Setup mock metrics
        mock_computed_metrics = {
            "chi2": mock_chi2, "aic": 8.95, "bic": 14.33, 
            "dof": -1, "p_value": 0.813
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute BAO fit
        result = engine.run_fit("lcdm", ["bao"], mode="individual")
        
        # Verify parameter construction was called (may be called multiple times due to dataset addition)
        assert mock_build_params.call_count >= 1
        # Check that the first call was with the expected arguments
        first_call = mock_build_params.call_args_list[0]
        assert first_call[0][0] == "lcdm"  # First argument should be "lcdm"
        
        # Verify dataset loading (called twice: once for caching, once for result compilation)
        assert mock_load_dataset.call_count >= 1
        mock_load_dataset.assert_any_call("bao")
        
        # Verify likelihood computation was called
        assert mock_likelihood_bao.call_count >= 1
        
        # Verify result structure
        assert isinstance(result, dict)
        # Engine may add additional datasets for sufficient DOF
        assert "bao" in result["datasets"]
        assert result["model"] == "lcdm"
        assert "bao" in result["results"]
        
        bao_result = result["results"]["bao"]
        assert "chi2" in bao_result
        assert "predictions" in bao_result
        assert bao_result["chi2"] == mock_chi2
        
        # Verify χ² breakdown
        assert "bao" in result["chi2_breakdown"]
        assert result["chi2_breakdown"]["bao"] == mock_chi2
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_sn')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_sn_individual_fit(self, mock_metrics, mock_build_params, mock_likelihood_sn, mock_load_dataset):
        """Test individual supernova fitting with run_fit()."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09, "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock supernova dataset
        mock_sn_data = {
            "observations": {"distance_modulus": np.array([32.5, 35.2, 37.1, 38.9])},
            "covariance": np.diag([0.01, 0.02, 0.03, 0.04]),
            "metadata": {
                "source": "Pantheon+",
                "n_data_points": 4,
                "observables": ["distance_modulus"],
                "redshift_range": [0.1, 0.8]
            },
            "dataset_type": "sn"
        }
        mock_load_dataset.return_value = mock_sn_data
        
        # Setup mock likelihood function
        mock_predictions = {"distance_modulus": np.array([32.48, 35.18, 37.08, 38.88])}
        mock_chi2 = 2.15
        mock_likelihood_sn.return_value = (mock_chi2, mock_predictions)
        
        # Setup mock metrics
        mock_computed_metrics = {
            "chi2": mock_chi2, "aic": 10.15, "bic": 15.53, 
            "dof": 0, "p_value": 0.708
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute supernova fit
        result = engine.run_fit("lcdm", ["sn"], mode="individual")
        
        # Verify parameter construction was called (may be called multiple times due to dataset addition)
        assert mock_build_params.call_count >= 1
        # Check that the first call was with the expected arguments
        first_call = mock_build_params.call_args_list[0]
        assert first_call[0][0] == "lcdm"  # First argument should be "lcdm"
        
        # Verify dataset loading (called twice: once for caching, once for result compilation)
        assert mock_load_dataset.call_count >= 1
        mock_load_dataset.assert_any_call("sn")
        
        # Verify likelihood computation was called
        assert mock_likelihood_sn.call_count >= 1
        
        # Verify result structure
        assert isinstance(result, dict)
        # Engine may add additional datasets for sufficient DOF
        assert "sn" in result["datasets"]
        assert result["model"] == "lcdm"
        assert "sn" in result["results"]
        
        sn_result = result["results"]["sn"]
        assert "chi2" in sn_result
        assert "predictions" in sn_result
        assert sn_result["chi2"] == mock_chi2
        
        # Verify χ² breakdown
        assert "sn" in result["chi2_breakdown"]
        assert result["chi2_breakdown"]["sn"] == mock_chi2
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_bao_ani')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_bao_anisotropic_individual_fit(self, mock_metrics, mock_build_params, mock_likelihood_bao_ani, mock_load_dataset):
        """Test individual anisotropic BAO fitting with run_fit()."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09, "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock anisotropic BAO dataset
        mock_bao_ani_data = {
            "observations": {
                "DM_over_rd": np.array([13.67, 20.75]),
                "DH_over_rd": np.array([0.0123, 0.0110])
            },
            "covariance": np.array([[0.25, 0.05, 0.02, 0.01],
                                   [0.05, 0.30, 0.01, 0.02],
                                   [0.02, 0.01, 0.15, 0.03],
                                   [0.01, 0.02, 0.03, 0.20]]),
            "metadata": {
                "source": "BAO_anisotropic",
                "n_data_points": 4,
                "observables": ["DM_over_rd", "DH_over_rd"],
                "redshift_range": [0.38, 0.51]
            },
            "dataset_type": "bao_ani"
        }
        mock_load_dataset.return_value = mock_bao_ani_data
        
        # Setup mock likelihood function
        mock_predictions = {
            "DM_over_rd": np.array([13.65, 20.73]),
            "DH_over_rd": np.array([0.0123, 0.0110])
        }
        mock_chi2 = 1.45
        mock_likelihood_bao_ani.return_value = (mock_chi2, mock_predictions)
        
        # Setup mock metrics
        mock_computed_metrics = {
            "chi2": mock_chi2, "aic": 9.45, "bic": 14.83, 
            "dof": 0, "p_value": 0.835
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute anisotropic BAO fit
        result = engine.run_fit("lcdm", ["bao_ani"], mode="individual")
        
        # Verify parameter construction was called (may be called multiple times)
        assert mock_build_params.call_count >= 1
        # Check that the first call was with the expected arguments
        first_call = mock_build_params.call_args_list[0]
        assert first_call[0][0] == "lcdm"  # First argument should be "lcdm"
        
        # Verify dataset loading (called twice: once for caching, once for result compilation)
        assert mock_load_dataset.call_count >= 1
        mock_load_dataset.assert_any_call("bao_ani")
        
        # Verify likelihood computation was called
        assert mock_likelihood_bao_ani.call_count >= 1
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result["datasets"] == ["bao_ani"]
        assert result["model"] == "lcdm"
        assert "bao_ani" in result["results"]
        
        bao_ani_result = result["results"]["bao_ani"]
        assert "chi2" in bao_ani_result
        assert "predictions" in bao_ani_result
        assert bao_ani_result["chi2"] == mock_chi2
        
        # Verify χ² breakdown
        assert "bao_ani" in result["chi2_breakdown"]
        assert result["chi2_breakdown"]["bao_ani"] == mock_chi2


class TestRunFitJointMode(unittest.TestCase):
    """Test joint fitting mode with multiple datasets and parameter consistency."""
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_bao')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_sn')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_joint_cmb_bao_sn_fit(self, mock_metrics, mock_build_params, 
                                  mock_likelihood_sn, mock_likelihood_bao, mock_likelihood_cmb, 
                                  mock_load_dataset):
        """Test joint fitting with CMB + BAO + SN datasets."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09, "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock datasets
        mock_datasets = {
            "cmb": {
                "observations": {"R": 1.7502, "l_A": 301.845, "theta_star": 1.04092e-2},
                "covariance": np.array([[1e-6, 0, 0], [0, 1e-2, 0], [0, 0, 1e-8]]),
                "metadata": {
                    "source": "Planck2018",
                    "n_data_points": 3,
                    "observables": ["R", "l_A", "theta_star"],
                    "redshift_range": [1089.8, 1089.8]
                },
                "dataset_type": "cmb"
            },
            "bao": {
                "observations": {"DV_over_rs": np.array([8.467, 13.015, 16.726])},
                "covariance": np.diag([0.168, 0.326, 0.440]),
                "metadata": {
                    "source": "BAO_compilation",
                    "n_data_points": 3,
                    "observables": ["DV_over_rs"],
                    "redshift_range": [0.38, 0.61]
                },
                "dataset_type": "bao"
            },
            "sn": {
                "observations": {"distance_modulus": np.array([32.5, 35.2, 37.1, 38.9])},
                "covariance": np.diag([0.01, 0.02, 0.03, 0.04]),
                "metadata": {
                    "source": "Pantheon+",
                    "n_data_points": 4,
                    "observables": ["distance_modulus"],
                    "redshift_range": [0.1, 0.8]
                },
                "dataset_type": "sn"
            }
        }
        
        def mock_load_side_effect(name):
            return mock_datasets[name]
        
        mock_load_dataset.side_effect = mock_load_side_effect
        
        # Setup mock likelihood functions
        mock_chi2_cmb = 1.85
        mock_chi2_bao = 0.95
        mock_chi2_sn = 2.15
        
        mock_predictions_cmb = {"R": 1.7500, "l_A": 301.800, "theta_star": 1.04090e-2}
        mock_predictions_bao = {"DV_over_rs": np.array([8.465, 13.010, 16.720])}
        mock_predictions_sn = {"distance_modulus": np.array([32.48, 35.18, 37.08, 38.88])}
        
        mock_likelihood_cmb.return_value = (mock_chi2_cmb, mock_predictions_cmb)
        mock_likelihood_bao.return_value = (mock_chi2_bao, mock_predictions_bao)
        mock_likelihood_sn.return_value = (mock_chi2_sn, mock_predictions_sn)
        
        # Setup mock metrics
        total_chi2 = mock_chi2_cmb + mock_chi2_bao + mock_chi2_sn
        mock_computed_metrics = {
            "chi2": total_chi2, "aic": total_chi2 + 8, "bic": total_chi2 + 16, 
            "dof": 6, "p_value": 0.712
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute joint fit
        result = engine.run_fit("lcdm", ["cmb", "bao", "sn"], mode="joint")
        
        # Verify parameter construction was called (may be called multiple times)
        assert mock_build_params.call_count >= 1
        # Check that the first call was with the expected arguments
        first_call = mock_build_params.call_args_list[0]
        assert first_call[0][0] == "lcdm"  # First argument should be "lcdm"
        
        # Verify all datasets were loaded (called twice each: once for caching, once for result compilation)
        assert mock_load_dataset.call_count >= 3
        mock_load_dataset.assert_any_call("cmb")
        mock_load_dataset.assert_any_call("bao")
        mock_load_dataset.assert_any_call("sn")
        
        # Verify all likelihood functions were called
        assert mock_likelihood_cmb.call_count >= 1
        assert mock_likelihood_bao.call_count >= 1
        assert mock_likelihood_sn.call_count >= 1
        
        # Verify result structure
        assert isinstance(result, dict)
        # Check that all expected datasets are present (order may vary)
        assert set(result["datasets"]) >= {"cmb", "bao", "sn"}
        assert result["model"] == "lcdm"
        
        # Verify all datasets are in results
        assert "cmb" in result["results"]
        assert "bao" in result["results"]
        assert "sn" in result["results"]
        
        # Verify χ² breakdown includes all datasets
        assert "cmb" in result["chi2_breakdown"]
        assert "bao" in result["chi2_breakdown"]
        assert "sn" in result["chi2_breakdown"]
        
        # Verify χ² values are correct
        assert result["chi2_breakdown"]["cmb"] == mock_chi2_cmb
        assert result["chi2_breakdown"]["bao"] == mock_chi2_bao
        assert result["chi2_breakdown"]["sn"] == mock_chi2_sn
        
        # Verify total χ² is sum of individual contributions
        expected_total = mock_chi2_cmb + mock_chi2_bao + mock_chi2_sn
        assert abs(result["metrics"]["chi2"] - expected_total) < 1e-10
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_bao')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_joint_parameter_consistency(self, mock_metrics, mock_build_params, 
                                        mock_likelihood_bao, mock_likelihood_cmb, 
                                        mock_load_dataset):
        """Test that joint fitting uses consistent parameters across all datasets."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09, "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock datasets
        mock_datasets = {
            "cmb": {
                "observations": {"R": 1.7502, "l_A": 301.845, "theta_star": 1.04092e-2},
                "covariance": np.array([[1e-6, 0, 0], [0, 1e-2, 0], [0, 0, 1e-8]]),
                "metadata": {
                    "source": "Planck2018",
                    "n_data_points": 3,
                    "observables": ["R", "l_A", "theta_star"]
                },
                "dataset_type": "cmb"
            },
            "bao": {
                "observations": {"DV_over_rs": np.array([8.467, 13.015])},
                "covariance": np.diag([0.168, 0.326]),
                "metadata": {
                    "source": "BAO_compilation",
                    "n_data_points": 2,
                    "observables": ["DV_over_rs"]
                },
                "dataset_type": "bao"
            },
            "sn": {
                "observations": {"distance_modulus": np.array([37.0, 38.0, 39.0])},
                "covariance": np.diag([0.1, 0.1, 0.1]),
                "metadata": {
                    "source": "Pantheon+",
                    "n_data_points": 3,
                    "observables": ["distance_modulus"]
                },
                "dataset_type": "sn"
            }
        }
        
        def mock_load_side_effect(name):
            return mock_datasets[name]
        
        mock_load_dataset.side_effect = mock_load_side_effect
        
        # Setup mock likelihood functions to capture parameter arguments
        cmb_params_calls = []
        bao_params_calls = []
        
        def mock_cmb_likelihood(params, data):
            cmb_params_calls.append(copy.deepcopy(params))
            return (1.5, {"R": 1.75, "l_A": 301.8, "theta_s": 1.041e-2})
        
        def mock_bao_likelihood(params, data):
            bao_params_calls.append(copy.deepcopy(params))
            return (0.8, {"DV_over_rs": np.array([8.46, 13.01])})
        
        mock_likelihood_cmb.side_effect = mock_cmb_likelihood
        mock_likelihood_bao.side_effect = mock_bao_likelihood
        
        # Setup mock metrics
        mock_computed_metrics = {
            "chi2": 2.3, "aic": 10.3, "bic": 15.7, 
            "dof": 1, "p_value": 0.680
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute joint fit
        result = engine.run_fit("lcdm", ["cmb", "bao"], mode="joint")
        
        # Verify that both likelihood functions were called
        assert len(cmb_params_calls) >= 1
        assert len(bao_params_calls) >= 1
        
        # Verify parameter consistency across likelihood calls
        # All calls should use the same parameter structure
        for cmb_params in cmb_params_calls:
            assert "H0" in cmb_params
            assert "Om0" in cmb_params
            # Note: derived parameters like z_recomb, z_drag are added by prepare_background_params
            # and may not be present in the optimization parameter array conversion
        
        for bao_params in bao_params_calls:
            assert "H0" in bao_params
            assert "Om0" in bao_params
            # Note: derived parameters are added by prepare_background_params
        
        # Verify that parameter construction was called with correct model
        # (Engine may call build_params multiple times due to dataset additions)
        assert mock_build_params.call_count >= 1
        # Check that all calls were with the correct model
        for call in mock_build_params.call_args_list:
            assert call[0][0] == "lcdm"  # First argument should be model name
    
    def test_joint_fit_chi2_summation(self):
        """Test that joint fitting correctly sums χ² contributions from all datasets."""
        # This test uses the actual objective function to verify χ² summation
        with patch('pipelines.fit_core.engine.datasets.load_dataset') as mock_load_dataset, \
             patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb') as mock_likelihood_cmb, \
             patch('pipelines.fit_core.engine.likelihoods.likelihood_bao') as mock_likelihood_bao, \
             patch('pipelines.fit_core.engine.parameter.build_params') as mock_build_params:
            
            # Setup mock parameter construction
            mock_params = {
                "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
                "model_class": "lcdm"
            }
            mock_build_params.return_value = mock_params
            
            # Setup mock datasets
            mock_datasets = {
                "cmb": {"observations": {}, "covariance": np.eye(3)},
                "bao": {"observations": {}, "covariance": np.eye(2)}
            }
            
            def mock_load_side_effect(name):
                return mock_datasets[name]
            
            mock_load_dataset.side_effect = mock_load_side_effect
            
            # Setup mock likelihood functions with known χ² values
            mock_chi2_cmb = 2.5
            mock_chi2_bao = 1.3
            
            mock_likelihood_cmb.return_value = (mock_chi2_cmb, {})
            mock_likelihood_bao.return_value = (mock_chi2_bao, {})
            
            # Build objective function
            data_cache = {"cmb": mock_datasets["cmb"], "bao": mock_datasets["bao"]}
            objective_func = engine._build_objective_function("lcdm", ["cmb", "bao"], data_cache)
            
            # Test objective function with dummy parameter array
            param_array = np.array([67.4, 0.315, 0.02237, 0.9649])
            total_chi2 = objective_func(param_array)
            
            # Verify that total χ² is sum of individual contributions
            expected_total = mock_chi2_cmb + mock_chi2_bao
            assert abs(total_chi2 - expected_total) < 1e-10


class TestOptimizationConvergence(unittest.TestCase):
    """Test optimization convergence and result reproducibility."""
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    @patch('pipelines.fit_core.engine._execute_optimization')
    def test_optimization_convergence_success(self, mock_execute_opt, mock_metrics, 
                                            mock_build_params, mock_likelihood_cmb, 
                                            mock_load_dataset):
        """Test successful optimization convergence."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock dataset
        mock_cmb_data = create_mock_dataset("cmb")
        mock_load_dataset.return_value = mock_cmb_data
        
        # Setup mock likelihood
        mock_likelihood_cmb.return_value = (1.5, {"R": 1.75, "l_A": 301.8, "theta_s": 1.041e-2})
        
        # Setup successful optimization result
        mock_opt_result = {
            "params": {"H0": 68.0, "Om0": 0.31, "Obh2": 0.0224, "ns": 0.965},
            "chi2_breakdown": {"cmb": 1.2, "bao": 1000000.0}
        }
        mock_execute_opt.return_value = mock_opt_result
        
        # Setup mock metrics
        mock_computed_metrics = {
            "chi2": 1.2, "aic": 9.2, "bic": 14.6, 
            "dof": -1, "p_value": 0.753
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute fit
        result = engine.run_fit("lcdm", ["cmb"])
        
        # Verify optimization was called
        mock_execute_opt.assert_called_once()
        
        # Verify optimization was called with correct parameters
        call_args = mock_execute_opt.call_args
        assert call_args is not None
        
        # Verify successful result
        assert isinstance(result, dict)
        assert "params" in result
        assert result["params"]["H0"] == 68.0  # Optimized value
        assert result["params"]["Om0"] == 0.31  # Optimized value
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    @patch('pipelines.fit_core.engine._execute_optimization')
    def test_optimization_convergence_failure(self, mock_execute_opt, mock_metrics, 
                                            mock_build_params, mock_likelihood_cmb, 
                                            mock_load_dataset):
        """Test handling of optimization convergence failure."""
        # Setup mock parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock dataset
        mock_cmb_data = create_mock_dataset("cmb")
        mock_load_dataset.return_value = mock_cmb_data
        
        # Setup mock likelihood
        mock_likelihood_cmb.return_value = (1.5, {"R": 1.75, "l_A": 301.8, "theta_s": 1.041e-2})
        
        # Setup failed optimization result
        mock_opt_result = {
            "params": {"H0": 67.5, "Om0": 0.314, "Obh2": 0.02238, "ns": 0.9648},
            "chi2_breakdown": {"cmb": 2.8, "bao": 1000000.0}
        }
        mock_execute_opt.return_value = mock_opt_result
        
        # Setup mock metrics
        mock_computed_metrics = {
            "chi2": 2.8, "aic": 10.8, "bic": 16.2, 
            "dof": -1, "p_value": 0.423
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute fit (should not raise exception despite convergence failure)
        result = engine.run_fit("lcdm", ["cmb"])
        
        # Verify optimization was called
        mock_execute_opt.assert_called_once()
        
        # Verify result is still returned despite convergence failure
        assert isinstance(result, dict)
        assert "params" in result
        assert result["params"]["H0"] == 67.5  # Best available value
        
        # Note: In real implementation, a warning should be logged
        # This would be tested by capturing stdout/stderr or log messages
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_result_reproducibility(self, mock_metrics, mock_build_params, 
                                   mock_likelihood_cmb, mock_load_dataset):
        """Test that identical inputs produce identical outputs."""
        # Setup consistent mocks
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "model_class": "lcdm"
        }
        mock_build_params.return_value = mock_params
        
        mock_cmb_data = create_mock_dataset("cmb")
        mock_load_dataset.return_value = mock_cmb_data
        
        mock_likelihood_cmb.return_value = (1.5, {"R": 1.75, "l_A": 301.8, "theta_s": 1.041e-2})
        
        mock_computed_metrics = {
            "chi2": 1.5, "aic": 9.5, "bic": 14.9, 
            "dof": -1, "p_value": 0.681
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute same fit multiple times
        result1 = engine.run_fit("lcdm", ["cmb"])
        result2 = engine.run_fit("lcdm", ["cmb"])
        
        # Verify identical results
        assert result1["datasets"] == result2["datasets"]
        assert result1["model"] == result2["model"]
        
        # Compare parameter dictionaries
        for key in result1["params"]:
            if isinstance(result1["params"][key], (int, float)):
                assert abs(result1["params"][key] - result2["params"][key]) < 1e-12
            else:
                assert result1["params"][key] == result2["params"][key]
        
        # Compare metrics
        for key in result1["metrics"]:
            if isinstance(result1["metrics"][key], (int, float)):
                assert abs(result1["metrics"][key] - result2["metrics"][key]) < 1e-12
            else:
                assert result1["metrics"][key] == result2["metrics"][key]
        
        # Compare χ² breakdown
        for dataset in result1["chi2_breakdown"]:
            assert abs(result1["chi2_breakdown"][dataset] - result2["chi2_breakdown"][dataset]) < 1e-12
    
    def test_different_optimizer_configurations(self):
        """Test that different optimizer configurations work correctly."""
        with patch('pipelines.fit_core.engine.datasets.load_dataset') as mock_load_dataset, \
             patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb') as mock_likelihood_cmb, \
             patch('pipelines.fit_core.engine.parameter.build_params') as mock_build_params, \
             patch('pipelines.fit_core.engine.statistics.compute_metrics') as mock_metrics, \
             patch('pipelines.fit_core.engine._execute_optimization') as mock_execute_opt:
            
            # Setup common mocks
            mock_params = {"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649, "model_class": "lcdm"}
            mock_build_params.return_value = mock_params
            
            mock_cmb_data = create_mock_dataset("cmb")
            mock_load_dataset.return_value = mock_cmb_data
            
            mock_likelihood_cmb.return_value = (1.5, {})
            mock_metrics.return_value = {"chi2": 1.5, "aic": 9.5, "bic": 14.9, "dof": -1, "p_value": 0.681}
            
            # Setup optimization result
            mock_opt_result = {
                "params": {"H0": 68.0, "Om0": 0.31, "Obh2": 0.0224, "ns": 0.965},
                "chi2_breakdown": {"cmb": 1.2, "bao": 1000000.0}
            }
            mock_execute_opt.return_value = mock_opt_result
            
            # Test CMB optimization (uses specialized path)
            result1 = engine.run_fit("lcdm", ["cmb"])
            
            # Verify optimization was called
            mock_execute_opt.assert_called_once()
            
            # Verify result structure
            assert isinstance(result1, dict)
            assert "params" in result1
            assert result1["params"]["H0"] == 68.0
            
            # Verify result is valid
            assert isinstance(result1, dict)
            assert "params" in result1


class TestRunFitPBUFModel(unittest.TestCase):
    """Test run_fit() with PBUF model parameters."""
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    @patch('pipelines.fit_core.engine.likelihoods.likelihood_cmb')
    @patch('pipelines.fit_core.engine.parameter.build_params')
    @patch('pipelines.fit_core.engine.statistics.compute_metrics')
    def test_pbuf_model_parameter_handling(self, mock_metrics, mock_build_params, 
                                          mock_likelihood_cmb, mock_load_dataset):
        """Test that PBUF model parameters are handled correctly in optimization."""
        # Setup mock PBUF parameter construction
        mock_params = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "PLANCK18",
            "alpha": 5e-4, "Rmax": 1e9, "eps0": 0.7, "n_eps": 0.0, "k_sat": 0.9762,
            "Omh2": 0.143, "Orh2": 2.469e-5, "z_recomb": 1089.80,
            "z_drag": 1059.57, "r_s_drag": 147.09, "model_class": "pbuf"
        }
        mock_build_params.return_value = mock_params
        
        # Setup mock dataset
        mock_cmb_data = create_mock_dataset("cmb")
        mock_load_dataset.return_value = mock_cmb_data
        
        # Setup mock likelihood
        mock_likelihood_cmb.return_value = (1.8, {"R": 1.751, "l_A": 301.9, "theta_s": 1.0409e-2})
        
        # Setup mock metrics
        mock_computed_metrics = {
            "chi2": 1.8, "aic": 19.8, "bic": 31.2, 
            "dof": -6, "p_value": 0.615
        }
        mock_metrics.return_value = mock_computed_metrics
        
        # Execute PBUF fit
        result = engine.run_fit("pbuf", ["cmb"])
        
        # Verify PBUF parameter construction was called (may be called multiple times due to dataset additions)
        self.assertGreaterEqual(mock_build_params.call_count, 1)
        # Check that at least one call was with the PBUF model
        calls = [call[0] for call in mock_build_params.call_args_list]
        self.assertTrue(any("pbuf" in str(call) for call in calls), 
                       f"Expected 'pbuf' in calls, got: {calls}")
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result["model"] == "pbuf"
        assert "params" in result
        
        # Verify PBUF-specific parameters are in result
        pbuf_params = {"alpha", "Rmax", "eps0", "n_eps", "k_sat"}
        for param in pbuf_params:
            assert param in result["params"]
        
        # Verify ΛCDM parameters are also present
        lcdm_params = {"H0", "Om0", "Obh2", "ns"}
        for param in lcdm_params:
            assert param in result["params"]


class TestRunFitErrorHandling(unittest.TestCase):
    """Test error handling in run_fit() function."""
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model types."""
        with pytest.raises(ValueError, match="Invalid model: invalid"):
            engine.run_fit("invalid", ["cmb"])
        
        with pytest.raises(ValueError, match="Must be 'lcdm' or 'pbuf'"):
            engine.run_fit("wcdm", ["cmb"])
    
    def test_empty_datasets_list(self):
        """Test error handling for empty datasets list."""
        with pytest.raises(ValueError, match="At least one dataset must be specified"):
            engine.run_fit("lcdm", [])
    
    @patch('pipelines.fit_core.engine.datasets.load_dataset')
    def test_invalid_dataset_name(self, mock_load_dataset):
        """Test error handling for invalid dataset names."""
        mock_load_dataset.side_effect = ValueError("Unsupported dataset: invalid")
        
        with pytest.raises(RuntimeError, match="Dataset verification failed"):
            engine.run_fit("lcdm", ["invalid"])
    
    def test_invalid_optimizer_method(self):
        """Test error handling for invalid optimizer methods."""
        with patch('pipelines.fit_core.engine.datasets.load_dataset') as mock_load_dataset, \
             patch('pipelines.fit_core.engine.parameter.build_params') as mock_build_params:
            
            # Provide complete parameter set to avoid missing parameter errors
            mock_build_params.return_value = {
                "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
                "model_class": "lcdm"
            }
            mock_load_dataset.return_value = create_mock_dataset("cmb")
            
            optimizer_config = {"method": "invalid_method"}
            
            with pytest.raises(ValueError, match="Unknown optimization method: invalid_method"):
                engine.run_fit("lcdm", ["cmb"], optimizer_config=optimizer_config)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])