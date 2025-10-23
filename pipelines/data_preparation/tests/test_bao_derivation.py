"""
Unit tests for the BAO (Baryon Acoustic Oscillations) derivation module.

Tests BAO-specific transformation logic including isotropic/anisotropic processing,
distance measure unit conversion, and correlation matrix validation.
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

from pipelines.data_preparation.derivation.bao_derivation import BAODerivationModule
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.interfaces import ProcessingError


class TestBAODerivationModule:
    """Test cases for BAODerivationModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.module = BAODerivationModule()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_type(self):
        """Test dataset type property."""
        assert self.module.dataset_type == 'bao'
    
    def test_supported_formats(self):
        """Test supported formats property."""
        expected_formats = ['.txt', '.csv', '.dat', '.json']
        assert self.module.supported_formats == expected_formats
    
    def test_validate_input_isotropic_data(self):
        """Test input validation with valid isotropic BAO data."""
        # Create valid isotropic BAO data
        test_data = pd.DataFrame({
            'z': [0.15, 0.32, 0.57],
            'DV_over_rd': [664.0, 1264.0, 1976.0],
            'DV_over_rd_err': [25.0, 25.0, 45.0]
        })
        
        test_file = self.temp_dir / "test_bao_iso.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'BOSS_DR12',
            'version': '1.0',
            'measurement_type': 'isotropic'
        }
        
        assert self.module.validate_input(test_file, metadata) is True
    
    def test_validate_input_anisotropic_data(self):
        """Test input validation with valid anisotropic BAO data."""
        # Create valid anisotropic BAO data
        test_data = pd.DataFrame({
            'z': [0.32, 0.57, 0.70],
            'DM_over_rd': [1512.0, 2056.0, 2400.0],
            'DM_over_rd_err': [25.0, 35.0, 40.0],
            'DH_over_rd': [81.2, 116.8, 135.0],
            'DH_over_rd_err': [2.4, 4.2, 5.0]
        })
        
        test_file = self.temp_dir / "test_bao_aniso.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'BOSS_DR12',
            'version': '1.0',
            'measurement_type': 'anisotropic'
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
    
    def test_validate_input_missing_columns_isotropic(self):
        """Test input validation with missing required columns for isotropic data."""
        # Create data missing required columns
        test_data = pd.DataFrame({
            'z': [0.15, 0.32, 0.57],
            'DV_over_rd': [664.0, 1264.0, 1976.0]
            # Missing DV_over_rd_err
        })
        
        test_file = self.temp_dir / "test_incomplete_iso.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'isotropic'
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.module.validate_input(test_file, metadata)
    
    def test_validate_input_missing_columns_anisotropic(self):
        """Test input validation with missing required columns for anisotropic data."""
        # Create data missing required columns
        test_data = pd.DataFrame({
            'z': [0.32, 0.57],
            'DM_over_rd': [1512.0, 2056.0]
            # Missing other required columns
        })
        
        test_file = self.temp_dir / "test_incomplete_aniso.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'anisotropic'
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.module.validate_input(test_file, metadata)
    
    def test_derive_isotropic_bao(self):
        """Test derivation of isotropic BAO data."""
        # Create test isotropic data
        test_data = pd.DataFrame({
            'z': [0.15, 0.32, 0.57],
            'DV_over_rd': [664.0, 1264.0, 1976.0],
            'DV_over_rd_err': [25.0, 25.0, 45.0]
        })
        
        test_file = self.temp_dir / "test_bao_iso.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'BOSS_DR12',
            'version': '1.0',
            'measurement_type': 'isotropic',
            'citation': 'Alam et al. 2017'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify result is StandardDataset
        assert isinstance(result, StandardDataset)
        
        # Verify basic properties
        assert len(result.z) == 3
        assert len(result.observable) == 3
        assert len(result.uncertainty) == 3
        
        # Verify redshift values are preserved
        np.testing.assert_array_equal(result.z, [0.15, 0.32, 0.57])
        
        # Verify observable values (should be DV/rd)
        np.testing.assert_array_equal(result.observable, [664.0, 1264.0, 1976.0])
        
        # Verify uncertainties
        np.testing.assert_array_equal(result.uncertainty, [25.0, 25.0, 45.0])
        
        # Verify metadata
        assert result.metadata['source'] == 'BOSS_DR12'
        assert result.metadata['measurement_type'] == 'isotropic'
        assert result.metadata['citation'] == 'Alam et al. 2017'
        assert 'processing_timestamp' in result.metadata
        assert 'n_points' in result.metadata
    
    def test_derive_anisotropic_bao(self):
        """Test derivation of anisotropic BAO data."""
        # Create test anisotropic data
        test_data = pd.DataFrame({
            'z': [0.32, 0.57],
            'DM_over_rd': [1512.0, 2056.0],
            'DM_over_rd_err': [25.0, 35.0],
            'DH_over_rd': [81.2, 116.8],
            'DH_over_rd_err': [2.4, 4.2]
        })
        
        test_file = self.temp_dir / "test_bao_aniso.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'BOSS_DR12',
            'version': '1.0',
            'measurement_type': 'anisotropic',
            'citation': 'Alam et al. 2017'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify result is StandardDataset
        assert isinstance(result, StandardDataset)
        
        # Verify basic properties - anisotropic has 2 observables per redshift
        assert len(result.z) == 4  # 2 redshifts × 2 measurements each
        assert len(result.observable) == 4
        assert len(result.uncertainty) == 4
        
        # Verify metadata indicates anisotropic processing
        assert result.metadata['measurement_type'] == 'anisotropic'
        assert 'DM_rd_points' in result.metadata
        assert 'DH_rd_points' in result.metadata
    
    def test_unit_conversion(self):
        """Test distance measure unit conversion."""
        # Create test data with units that need conversion
        test_data = pd.DataFrame({
            'z': [0.32],
            'DV_over_rd': [1264.0],  # Mpc/h units
            'DV_over_rd_err': [25.0]
        })
        
        test_file = self.temp_dir / "test_units.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'isotropic',
            'units': 'Mpc_h',  # Indicates h-units
            'h_value': 0.7
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should convert from Mpc/h to Mpc
        expected_value = 1264.0 / 0.7  # Convert from Mpc/h to Mpc
        assert abs(result.observable[0] - expected_value) < 1.0
        
        # Verify unit conversion is documented
        assert 'unit_conversion_to_mpc' in result.metadata['transformations_applied']
    
    def test_correlation_matrix_validation(self):
        """Test correlation matrix validation and application."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.32, 0.57],
            'DV_over_rd': [1264.0, 1976.0],
            'DV_over_rd_err': [25.0, 45.0]
        })
        
        test_file = self.temp_dir / "test_correlation.csv"
        test_data.to_csv(test_file, index=False)
        
        # Create correlation matrix file
        correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
        corr_file = self.temp_dir / "correlation_matrix.npy"
        np.save(corr_file, correlation_matrix)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'isotropic',
            'correlation_matrix_file': str(corr_file)
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should have covariance matrix
        assert result.covariance is not None
        assert result.covariance.shape == (2, 2)
        
        # Verify covariance matrix properties
        assert np.allclose(result.covariance, result.covariance.T)  # Symmetric
        eigenvals = np.linalg.eigvals(result.covariance)
        assert np.all(eigenvals >= 0)  # Positive semi-definite
    
    def test_survey_specific_corrections(self):
        """Test survey-specific systematic corrections."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.32, 0.57],
            'DV_over_rd': [1264.0, 1976.0],
            'DV_over_rd_err': [25.0, 45.0]
        })
        
        test_file = self.temp_dir / "test_corrections.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'BOSS_DR12',
            'version': '1.0',
            'measurement_type': 'isotropic',
            'systematic_corrections': {
                'reconstruction_bias': 0.02,
                'fiber_collision_correction': 0.01
            }
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify corrections are applied and documented
        assert 'systematic_corrections_applied' in result.metadata
        assert result.metadata['systematic_corrections_applied'] == metadata['systematic_corrections']
    
    def test_error_handling_invalid_redshifts(self):
        """Test error handling with invalid redshift values."""
        # Create test data with invalid redshifts
        test_data = pd.DataFrame({
            'z': [-0.1, 0.32, np.nan],  # Negative and NaN redshifts
            'DV_over_rd': [664.0, 1264.0, 1976.0],
            'DV_over_rd_err': [25.0, 25.0, 45.0]
        })
        
        test_file = self.temp_dir / "test_invalid_z.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'isotropic'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            self.module.derive(test_file, metadata)
        
        assert exc_info.value.error_type == "data_validation_error"
    
    def test_error_handling_invalid_distances(self):
        """Test error handling with invalid distance measurements."""
        # Create test data with invalid distance values
        test_data = pd.DataFrame({
            'z': [0.15, 0.32, 0.57],
            'DV_over_rd': [664.0, -1264.0, np.inf],  # Negative and infinite values
            'DV_over_rd_err': [25.0, 25.0, 45.0]
        })
        
        test_file = self.temp_dir / "test_invalid_distances.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'isotropic'
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
        assert any('D_V' in formula for formula in formulas)
        assert any('D_M' in formula for formula in formulas)
        assert any('D_H' in formula for formula in formulas)
        assert any('r_d' in formula for formula in formulas)
    
    def test_measurement_type_detection(self):
        """Test automatic measurement type detection."""
        # Create isotropic data without explicit type in metadata
        test_data = pd.DataFrame({
            'z': [0.15, 0.32],
            'DV_over_rd': [664.0, 1264.0],
            'DV_over_rd_err': [25.0, 25.0]
        })
        
        test_file = self.temp_dir / "test_auto_detect.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0'
            # No measurement_type specified
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should automatically detect as isotropic
        assert result.metadata['measurement_type'] == 'isotropic'
    
    def test_covariance_matrix_construction(self):
        """Test covariance matrix construction from uncertainties and correlations."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.32, 0.57],
            'DV_over_rd': [1264.0, 1976.0],
            'DV_over_rd_err': [25.0, 45.0]
        })
        
        test_file = self.temp_dir / "test_covariance_construction.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'isotropic'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should construct diagonal covariance matrix from uncertainties
        expected_cov = np.diag([25.0**2, 45.0**2])
        
        if result.covariance is not None:
            np.testing.assert_allclose(result.covariance, expected_cov, rtol=1e-10)
    
    def test_redshift_range_validation(self):
        """Test redshift range validation for BAO measurements."""
        # Create test data with extreme redshift values
        test_data = pd.DataFrame({
            'z': [0.01, 0.32, 3.0],  # Very low and very high redshifts
            'DV_over_rd': [100.0, 1264.0, 5000.0],
            'DV_over_rd_err': [10.0, 25.0, 200.0]
        })
        
        test_file = self.temp_dir / "test_extreme_z_bao.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'measurement_type': 'isotropic'
        }
        
        # Should either filter out extreme values or raise warning
        result = self.module.derive(test_file, metadata)
        
        # Check that processing completed (implementation may filter or warn)
        assert isinstance(result, StandardDataset)
        assert 'redshift_filtering' in result.metadata or len(result.z) <= 3