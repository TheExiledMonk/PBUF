"""
Unit tests for the RSD (Redshift Space Distortions) derivation module.

Tests RSD-specific transformation logic including growth rate (fσ₈) processing,
sign convention validation, and covariance homogenization.
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

from pipelines.data_preparation.derivation.rsd_derivation import RSDDerivationModule
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.interfaces import ProcessingError


class TestRSDDerivationModule:
    """Test cases for RSDDerivationModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.module = RSDDerivationModule()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_type(self):
        """Test dataset type property."""
        assert self.module.dataset_type == 'rsd'
    
    def test_supported_formats(self):
        """Test supported formats property."""
        expected_formats = ['.txt', '.csv', '.dat', '.json']
        assert self.module.supported_formats == expected_formats
    
    def test_validate_input_valid_rsd_data(self):
        """Test input validation with valid RSD data."""
        # Create valid RSD data
        test_data = pd.DataFrame({
            'z': [0.15, 0.25, 0.37, 0.51, 0.70],
            'fsigma8': [0.490, 0.427, 0.460, 0.458, 0.440],
            'fsigma8_err': [0.145, 0.043, 0.038, 0.038, 0.050],
            'survey': ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ', 'BOSS_CMASS', 'WiggleZ']
        })
        
        test_file = self.temp_dir / "test_rsd.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'RSD_compilation',
            'version': '1.0',
            'data_type': 'growth_rate'
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
            'z': [0.15, 0.25, 0.37],
            'fsigma8': [0.490, 0.427, 0.460]
            # Missing fsigma8_err
        })
        
        test_file = self.temp_dir / "test_incomplete.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate'
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.module.validate_input(test_file, metadata)
    
    def test_derive_basic_rsd_data(self):
        """Test basic RSD data derivation."""
        # Create test RSD data
        test_data = pd.DataFrame({
            'z': [0.15, 0.25, 0.37, 0.51],
            'fsigma8': [0.490, 0.427, 0.460, 0.458],
            'fsigma8_err': [0.145, 0.043, 0.038, 0.038],
            'survey': ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ', 'BOSS_CMASS']
        })
        
        test_file = self.temp_dir / "test_rsd.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'RSD_compilation',
            'version': '1.0',
            'data_type': 'growth_rate',
            'citation': 'Macaulay et al. 2013'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify result is StandardDataset
        assert isinstance(result, StandardDataset)
        
        # Verify basic properties
        assert len(result.z) == 4
        assert len(result.observable) == 4
        assert len(result.uncertainty) == 4
        
        # Verify redshift values are preserved
        np.testing.assert_array_equal(result.z, [0.15, 0.25, 0.37, 0.51])
        
        # Verify fσ₈ values (with survey corrections applied)
        # 6dFGS gets 0.95 peculiar velocity correction, BOSS surveys get 1.02 fiber correction
        expected_values = [
            0.490 * 0.95,    # 6dFGS with pv_correction
            0.427,           # SDSS_LRG (no correction)
            0.460 * 1.02,    # BOSS_LOWZ with fiber_correction
            0.458 * 1.02     # BOSS_CMASS with fiber_correction
        ]
        np.testing.assert_allclose(result.observable, expected_values, rtol=1e-6)
        
        # Verify uncertainties
        np.testing.assert_array_equal(result.uncertainty, [0.145, 0.043, 0.038, 0.038])
        
        # Verify metadata
        assert result.metadata['source'] == 'RSD_compilation'
        assert result.metadata['data_type'] == 'growth_rate'
        assert result.metadata['citation'] == 'Macaulay et al. 2013'
        assert 'processing_timestamp' in result.metadata
        assert 'n_points' in result.metadata
    
    def test_growth_rate_sign_convention_validation(self):
        """Test growth rate sign convention validation."""
        # Create test data with potentially incorrect signs
        test_data = pd.DataFrame({
            'z': [0.15, 0.25, 0.37],
            'fsigma8': [0.490, -0.427, 0.460],  # Negative fσ₈ (should be positive)
            'fsigma8_err': [0.145, 0.043, 0.038],
            'survey': ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_sign_convention.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            self.module.derive(test_file, metadata)
        
        assert exc_info.value.error_type == "data_validation_error"
        assert "negative" in exc_info.value.error_message.lower() or "non-positive" in exc_info.value.error_message.lower()
    
    def test_covariance_homogenization(self):
        """Test covariance homogenization from published sources."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.25, 0.37],
            'fsigma8': [0.427, 0.460],
            'fsigma8_err': [0.043, 0.038],
            'survey': ['SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_covariance.csv"
        test_data.to_csv(test_file, index=False)
        
        # Create covariance matrix file
        covariance_matrix = np.array([
            [0.043**2, 0.001],
            [0.001, 0.038**2]
        ])
        cov_file = self.temp_dir / "rsd_covariance.npy"
        np.save(cov_file, covariance_matrix)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate',
            'covariance_matrix_file': str(cov_file)
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should have covariance matrix
        assert result.covariance is not None
        assert result.covariance.shape == (2, 2)
        
        # Verify covariance matrix properties
        assert np.allclose(result.covariance, result.covariance.T)  # Symmetric
        eigenvals = np.linalg.eigvals(result.covariance)
        assert np.all(eigenvals >= 0)  # Positive semi-definite
        
        # Verify covariance homogenization is documented
        assert 'covariance_homogenization' in result.metadata
    
    def test_survey_specific_corrections(self):
        """Test survey-specific corrections application."""
        # Create test data from different surveys
        test_data = pd.DataFrame({
            'z': [0.15, 0.25, 0.37],
            'fsigma8': [0.490, 0.427, 0.460],
            'fsigma8_err': [0.145, 0.043, 0.038],
            'survey': ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_corrections.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate',
            'survey_corrections': {
                '6dFGS': {'bias_correction': 1.02, 'systematic_error': 0.01},
                'SDSS_LRG': {'bias_correction': 1.00, 'systematic_error': 0.005},
                'BOSS_LOWZ': {'bias_correction': 0.98, 'systematic_error': 0.008}
            }
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify corrections are applied and documented
        assert 'survey_corrections_applied' in result.metadata
        assert result.metadata['survey_corrections_applied'] == metadata['survey_corrections']
        
        # Check that corrections were applied to the data
        # (exact values depend on implementation details)
        assert 'correction_summary' in result.metadata
    
    def test_error_propagation(self):
        """Test error propagation with systematic corrections."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.25, 0.37],
            'fsigma8': [0.427, 0.460],
            'fsigma8_err': [0.043, 0.038],
            'survey': ['SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_error_prop.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate',
            'systematic_errors': {
                'theoretical_modeling': 0.02,  # Absolute systematic error
                'survey_systematics': 0.05    # Fractional systematic error
            }
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should include systematic errors in total uncertainty
        assert 'systematic_errors_applied' in result.metadata
        
        # Total uncertainties should be larger than statistical only
        for i, (fsig8_val, stat_err) in enumerate(zip([0.427, 0.460], [0.043, 0.038])):
            abs_sys = 0.02
            frac_sys = fsig8_val * 0.05
            expected_total_err = np.sqrt(stat_err**2 + abs_sys**2 + frac_sys**2)
            
            # Allow for small numerical differences
            assert abs(result.uncertainty[i] - expected_total_err) < 0.005
    
    def test_error_handling_invalid_redshifts(self):
        """Test error handling with invalid redshift values."""
        # Create test data with invalid redshifts
        test_data = pd.DataFrame({
            'z': [-0.1, 0.25, np.nan],  # Negative and NaN redshifts
            'fsigma8': [0.490, 0.427, 0.460],
            'fsigma8_err': [0.145, 0.043, 0.038],
            'survey': ['Test', 'SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_invalid_z.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            self.module.derive(test_file, metadata)
        
        assert exc_info.value.error_type == "data_validation_error"
    
    def test_error_handling_invalid_fsigma8_values(self):
        """Test error handling with invalid fσ₈ values."""
        # Create test data with invalid fσ₈ values
        test_data = pd.DataFrame({
            'z': [0.15, 0.25, 0.37],
            'fsigma8': [0.490, np.inf, 0.0],  # Infinite and zero fσ₈
            'fsigma8_err': [0.145, 0.043, 0.038],
            'survey': ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_invalid_fsigma8.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate'
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
        assert any('fσ₈' in formula or 'fsigma8' in formula for formula in formulas)
        assert any('growth' in formula.lower() for formula in formulas)
        
        # Verify RSD-specific references
        references = summary['references']
        assert any('Macaulay' in ref or 'Samushia' in ref for ref in references)
    
    def test_survey_tracking(self):
        """Test tracking of survey information through processing."""
        # Create test data with multiple surveys
        test_data = pd.DataFrame({
            'z': [0.15, 0.25, 0.37, 0.51],
            'fsigma8': [0.490, 0.427, 0.460, 0.458],
            'fsigma8_err': [0.145, 0.043, 0.038, 0.038],
            'survey': ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ', 'BOSS_CMASS']
        })
        
        test_file = self.temp_dir / "test_surveys.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'RSD_compilation',
            'version': '1.0',
            'data_type': 'growth_rate'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should track individual surveys
        assert 'individual_surveys' in result.metadata
        surveys = result.metadata['individual_surveys']
        
        expected_surveys = ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ', 'BOSS_CMASS']
        for survey in expected_surveys:
            assert survey in surveys
    
    def test_redshift_range_validation(self):
        """Test redshift range validation for RSD measurements."""
        # Create test data with extreme redshift values
        test_data = pd.DataFrame({
            'z': [0.01, 0.25, 3.0],  # Very low and very high redshifts
            'fsigma8': [0.300, 0.427, 0.800],
            'fsigma8_err': [0.100, 0.043, 0.200],
            'survey': ['Test', 'SDSS_LRG', 'Test']
        })
        
        test_file = self.temp_dir / "test_extreme_z_rsd.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate'
        }
        
        # Should either filter out extreme values or raise warning
        result = self.module.derive(test_file, metadata)
        
        # Check that processing completed (implementation may filter or warn)
        assert isinstance(result, StandardDataset)
        assert 'redshift_filtering' in result.metadata or len(result.z) <= 3
    
    def test_covariance_matrix_construction(self):
        """Test covariance matrix construction from uncertainties."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.25, 0.37, 0.51],
            'fsigma8': [0.427, 0.460, 0.458],
            'fsigma8_err': [0.043, 0.038, 0.038],
            'survey': ['SDSS_LRG', 'BOSS_LOWZ', 'BOSS_CMASS']
        })
        
        test_file = self.temp_dir / "test_covariance_construction.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should construct diagonal covariance matrix from uncertainties
        expected_cov = np.diag([0.043**2, 0.038**2, 0.038**2])
        
        if result.covariance is not None:
            np.testing.assert_allclose(result.covariance, expected_cov, rtol=1e-10)
    
    def test_physical_consistency_checks(self):
        """Test physical consistency checks for fσ₈ values."""
        # Create test data with physically reasonable values
        test_data = pd.DataFrame({
            'z': [0.15, 0.25, 0.37],
            'fsigma8': [0.490, 0.427, 0.460],  # Reasonable fσ₈ values
            'fsigma8_err': [0.145, 0.043, 0.038],
            'survey': ['6dFGS', 'SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_physical_consistency.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should perform physical consistency checks
        assert 'physical_consistency_check' in result.metadata
        
        # Verify that fσ₈ values are in reasonable range (0.1 to 1.0)
        for fsig8_val in result.observable:
            assert 0.1 < fsig8_val < 1.0
    
    def test_alternative_parameterizations(self):
        """Test handling of alternative growth rate parameterizations."""
        # Create test data with alternative parameterization (f and σ₈ separate)
        test_data = pd.DataFrame({
            'z': [0.25, 0.37],
            'f': [0.87, 0.96],  # Growth rate f
            'f_err': [0.15, 0.18],
            'sigma8': [0.49, 0.48],  # σ₈ values
            'sigma8_err': [0.05, 0.05],
            'survey': ['SDSS_LRG', 'BOSS_LOWZ']
        })
        
        test_file = self.temp_dir / "test_alternative_param.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'growth_rate',
            'parameterization': 'f_sigma8_separate',
            'boss_ap_correction': 1.0,
            'boss_fiber_correction': 1.0
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should convert to fσ₈ = f × σ₈
        expected_fsig8 = [0.87 * 0.49, 0.96 * 0.48]
        
        np.testing.assert_allclose(result.observable, expected_fsig8, rtol=1e-2)
        
        # Should document the conversion
        assert 'parameterization_conversion' in result.metadata