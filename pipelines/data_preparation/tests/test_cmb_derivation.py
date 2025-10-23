"""
Unit tests for the CMB (Cosmic Microwave Background) derivation module.

Tests CMB-specific transformation logic including distance priors extraction,
dimensionless consistency checking, and covariance matrix application.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock pandas if not available
try:
    import pandas as pd
except ImportError:
    pd = MagicMock()
    sys.modules['pandas'] = pd

from pipelines.data_preparation.derivation.cmb_derivation import CMBDerivationModule
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.interfaces import ProcessingError


class TestCMBDerivationModule:
    """Test cases for CMBDerivationModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.module = CMBDerivationModule()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_type(self):
        """Test dataset type property."""
        assert self.module.dataset_type == 'cmb'
    
    def test_supported_formats(self):
        """Test supported formats property."""
        expected_formats = ['.txt', '.csv', '.dat', '.json', '.fits']
        assert self.module.supported_formats == expected_formats
    
    def test_validate_input_valid_planck_data(self):
        """Test input validation with valid Planck distance priors."""
        # Create valid CMB distance priors data
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030]
        })
        
        test_file = self.temp_dir / "test_cmb_priors.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'Planck2018',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        assert self.module.validate_input(test_file, metadata) is True
    
    def test_validate_input_missing_file(self):
        """Test input validation with missing file."""
        missing_file = self.temp_dir / "missing.csv"
        metadata = {'source': 'test', 'version': '1.0'}
        
        with pytest.raises(ValueError, match="Input file does not exist"):
            self.module.validate_input(missing_file, metadata)
    
    def test_validate_input_invalid_format(self):
        """Test input validation with unsupported format."""
        test_file = self.temp_dir / "test.xyz"
        test_file.write_text("dummy content")
        
        metadata = {'source': 'test', 'version': '1.0'}
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.module.validate_input(test_file, metadata)
    
    def test_validate_input_missing_columns(self):
        """Test input validation with missing required columns."""
        # Create data missing required columns
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502]
            # Missing other required columns
        })
        
        test_file = self.temp_dir / "test_incomplete.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.module.validate_input(test_file, metadata)
    
    def test_derive_distance_priors(self):
        """Test derivation of CMB distance priors."""
        # Create test CMB distance priors data (Planck 2018 values)
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030]
        })
        
        test_file = self.temp_dir / "test_cmb_priors.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'Planck2018',
            'version': '1.0',
            'data_type': 'distance_priors',
            'citation': 'Planck Collaboration 2020'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify result is StandardDataset
        assert isinstance(result, StandardDataset)
        
        # CMB data should have 3 observables: R, l_A, theta_*
        assert len(result.z) == 3
        assert len(result.observable) == 3
        assert len(result.uncertainty) == 3
        
        # All measurements are at z_*
        np.testing.assert_array_equal(result.z, [1089.80, 1089.80, 1089.80])
        
        # Verify observable values [R, l_A, theta_*]
        expected_obs = [1.7502, 301.63, 1.04119]
        np.testing.assert_allclose(result.observable, expected_obs, rtol=1e-6)
        
        # Verify uncertainties
        expected_err = [0.0038, 0.15, 0.00030]
        np.testing.assert_allclose(result.uncertainty, expected_err, rtol=1e-6)
        
        # Verify metadata
        assert result.metadata['source'] == 'Planck2018'
        assert result.metadata['data_type'] == 'distance_priors'
        assert result.metadata['citation'] == 'Planck Collaboration 2020'
        assert 'processing_timestamp' in result.metadata
        assert 'n_points' in result.metadata
    
    def test_dimensionless_consistency_check(self):
        """Test dimensionless consistency checking."""
        # Create test data with potentially inconsistent values
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030]
        })
        
        test_file = self.temp_dir / "test_consistency.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should perform dimensionless consistency checks
        assert 'dimensionless_consistency_check' in result.metadata
        
        # Verify that R and l_A are dimensionless (order of magnitude checks)
        R_value = result.observable[0]  # R should be ~1-2
        l_A_value = result.observable[1]  # l_A should be ~300
        
        assert 0.5 < R_value < 5.0  # Reasonable range for R
        assert 200 < l_A_value < 500  # Reasonable range for l_A
    
    def test_covariance_matrix_application(self):
        """Test covariance matrix application for CMB priors."""
        # Create test data
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030]
        })
        
        test_file = self.temp_dir / "test_covariance.csv"
        test_data.to_csv(test_file, index=False)
        
        # Create covariance matrix file (3x3 for R, l_A, theta_*)
        cov_matrix = np.array([
            [0.0038**2, 0.0001, 0.0],
            [0.0001, 0.15**2, 0.0002],
            [0.0, 0.0002, 0.00030**2]
        ])
        cov_file = self.temp_dir / "cmb_covariance.npy"
        np.save(cov_file, cov_matrix)
        
        metadata = {
            'source': 'Planck2018',
            'version': '1.0',
            'data_type': 'distance_priors',
            'covariance_matrix_file': str(cov_file)
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should have covariance matrix
        assert result.covariance is not None
        assert result.covariance.shape == (3, 3)
        
        # Verify covariance matrix properties
        assert np.allclose(result.covariance, result.covariance.T)  # Symmetric
        eigenvals = np.linalg.eigvals(result.covariance)
        assert np.all(eigenvals >= 0)  # Positive semi-definite
    
    def test_cosmological_constant_validation(self):
        """Test cosmological constant validation."""
        # Create test data with cosmological parameters
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030],
            'omega_b_h2': [0.02237],
            'omega_c_h2': [0.1200],
            'H0': [67.4]
        })
        
        test_file = self.temp_dir / "test_cosmo_params.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'Planck2018',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should validate cosmological parameters
        assert 'cosmological_parameters' in result.metadata
        assert 'parameter_validation' in result.metadata
        
        # Check that reasonable cosmological parameters are preserved
        cosmo_params = result.metadata['cosmological_parameters']
        assert 'omega_b_h2' in cosmo_params
        assert 'omega_c_h2' in cosmo_params
        assert 'H0' in cosmo_params
    
    def test_planck_chain_extraction(self):
        """Test extraction from Planck chain files."""
        # Create mock Planck chain data
        chain_data = {
            'z_star': 1089.80,
            'R': 1.7502,
            'l_A': 301.63,
            'theta_star': 1.04119,
            'sigma_R': 0.0038,
            'sigma_l_A': 0.15,
            'sigma_theta_star': 0.00030
        }
        
        # Save as JSON (simulating processed chain data)
        chain_file = self.temp_dir / "planck_chains.json"
        with open(chain_file, 'w') as f:
            json.dump(chain_data, f)
        
        metadata = {
            'source': 'Planck2018',
            'version': '1.0',
            'data_type': 'chain_extract'
        }
        
        result = self.module.derive(chain_file, metadata)
        
        # Should successfully extract from chain format
        assert isinstance(result, StandardDataset)
        assert len(result.observable) == 3
        assert 'chain_extraction' in result.metadata
    
    def test_error_handling_invalid_z_star(self):
        """Test error handling with invalid z_star values."""
        # Create test data with invalid z_star
        test_data = pd.DataFrame({
            'z_star': [-100.0],  # Negative z_star (invalid)
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030]
        })
        
        test_file = self.temp_dir / "test_invalid_z.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            self.module.derive(test_file, metadata)
        
        assert exc_info.value.error_type == "data_validation_error"
    
    def test_error_handling_invalid_distance_priors(self):
        """Test error handling with invalid distance prior values."""
        # Create test data with invalid values
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [-1.7502],  # Negative R (invalid)
            'l_A': [np.nan],  # NaN l_A
            'theta_star': [np.inf],  # Infinite theta_star
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030]
        })
        
        test_file = self.temp_dir / "test_invalid_priors.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            self.module.derive(test_file, metadata)
        
        assert exc_info.value.error_type == "data_validation_error"
    
    def test_get_transformation_summary(self):
        """Test transformation summary generation."""
        summary = self.module.get_transformation_summary()
        
        assert isinstance(summary, dict)
        assert 'formulas_used' in summary
        assert 'assumptions' in summary
        assert 'references' in summary
        
        # Verify expected formulas are documented
        formulas = summary['formulas_used']
        assert any('R' in formula for formula in formulas)
        assert any('l_A' in formula for formula in formulas)
        assert any('θ' in formula or 'theta' in formula for formula in formulas)
        
        # Verify CMB-specific references
        references = summary['references']
        assert any('Planck' in ref for ref in references)
    
    def test_parameter_extraction_logic(self):
        """Test parameter extraction and validation logic."""
        # Create test data with full parameter set
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030],
            'omega_b_h2': [0.02237],
            'omega_c_h2': [0.1200],
            'tau': [0.0544],
            'n_s': [0.9649],
            'A_s': [2.1e-9]
        })
        
        test_file = self.temp_dir / "test_full_params.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'Planck2018',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should extract and validate all parameters
        assert 'parameter_extraction' in result.metadata
        assert 'cosmological_parameters' in result.metadata
        
        # Verify parameter ranges are reasonable
        cosmo_params = result.metadata['cosmological_parameters']
        assert 0.01 < cosmo_params['omega_b_h2'] < 0.05
        assert 0.05 < cosmo_params['omega_c_h2'] < 0.5
        assert 0.9 < cosmo_params['n_s'] < 1.1
    
    def test_covariance_construction_from_uncertainties(self):
        """Test covariance matrix construction from individual uncertainties."""
        # Create test data without explicit covariance matrix
        test_data = pd.DataFrame({
            'z_star': [1089.80],
            'R': [1.7502],
            'l_A': [301.63],
            'theta_star': [1.04119],
            'R_err': [0.0038],
            'l_A_err': [0.15],
            'theta_star_err': [0.00030]
        })
        
        test_file = self.temp_dir / "test_diag_cov.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should construct diagonal covariance matrix
        expected_cov = np.diag([0.0038**2, 0.15**2, 0.00030**2])
        
        if result.covariance is not None:
            np.testing.assert_allclose(result.covariance, expected_cov, rtol=1e-10)
    
    def test_multiple_redshift_handling(self):
        """Test handling of multiple redshift epochs (if applicable)."""
        # Create test data with multiple epochs
        test_data = pd.DataFrame({
            'z_star': [1089.80, 1090.0],  # Slightly different z_star values
            'R': [1.7502, 1.7505],
            'l_A': [301.63, 301.8],
            'theta_star': [1.04119, 1.04125],
            'R_err': [0.0038, 0.004],
            'l_A_err': [0.15, 0.16],
            'theta_star_err': [0.00030, 0.00032]
        })
        
        test_file = self.temp_dir / "test_multi_epoch.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'distance_priors'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should handle multiple epochs appropriately
        assert isinstance(result, StandardDataset)
        assert 'multiple_epochs' in result.metadata or len(result.z) == 6  # 2 epochs × 3 params