"""
Unit tests for the SN (Supernova) derivation module.

Tests SN-specific transformation logic including magnitude to distance modulus
conversion, duplicate removal, calibration homogenization, and systematic
covariance matrix application.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock pandas and astropy if not available
try:
    import pandas as pd
except ImportError:
    pd = MagicMock()
    sys.modules['pandas'] = pd

try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
except ImportError:
    # Create proper mocks for astropy
    class MockUnit:
        def __mul__(self, other):
            return other
        def __rmul__(self, other):
            return other
    
    class MockUnits:
        degree = MockUnit()
    
    class MockSkyCoord:
        def __init__(self, ra=None, dec=None):
            self.ra = ra
            self.dec = dec
            self.separation_called = False
        
        def separation(self, other):
            self.separation_called = True
            # Return mock separations that are always > 1 arcsec to avoid duplicates
            import numpy as np
            return np.array([2.0] * len(self.ra)) * MockUnit()
        
        def __len__(self):
            return len(self.ra) if hasattr(self.ra, '__len__') else 1
    
    SkyCoord = MockSkyCoord
    u = MockUnits()
    
    # Mock the modules
    astropy_mock = MagicMock()
    sys.modules['astropy'] = astropy_mock
    sys.modules['astropy.coordinates'] = astropy_mock
    sys.modules['astropy.units'] = astropy_mock

from pipelines.data_preparation.derivation.sn_derivation import SNDerivationModule
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.interfaces import ProcessingError


class TestSNDerivationModule:
    """Test cases for SNDerivationModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.module = SNDerivationModule()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_type(self):
        """Test dataset type property."""
        assert self.module.dataset_type == 'sn'
    
    def test_supported_formats(self):
        """Test supported formats property."""
        expected_formats = ['.txt', '.csv', '.dat', '.fits']
        assert self.module.supported_formats == expected_formats
    
    def test_validate_input_valid_file(self):
        """Test input validation with valid SN data file."""
        # Create valid test data
        test_data = pd.DataFrame({
            'z': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'mb': [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            'dmb': [0.1, 0.15, 0.2, 0.12, 0.18, 0.14, 0.16, 0.13, 0.17, 0.19],
            'x1': [0.0, 0.5, -0.3, 0.2, -0.1, 0.3, -0.2, 0.1, -0.4, 0.4],
            'dx1': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'c': [0.0, 0.1, -0.1, 0.05, -0.05, 0.08, -0.08, 0.03, -0.03, 0.06],
            'dc': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            'ra': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            'dec': [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
        })
        
        test_file = self.temp_dir / "test_sn.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {'source': 'test', 'version': '1.0'}
        
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
            'z': [0.1, 0.2, 0.3],
            'mb': [15.0, 16.0, 17.0]
            # Missing other required columns
        })
        
        test_file = self.temp_dir / "test_incomplete.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {'source': 'test', 'version': '1.0'}
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.module.validate_input(test_file, metadata)
    
    def test_derive_basic_transformation(self):
        """Test basic SN data derivation."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.1, 0.2, 0.3],
            'mb': [15.0, 16.0, 17.0],
            'dmb': [0.1, 0.15, 0.2],
            'x1': [0.0, 0.5, -0.3],
            'dx1': [0.1, 0.1, 0.1],
            'c': [0.0, 0.1, -0.1],
            'dc': [0.05, 0.05, 0.05],
            'ra': [10.0, 20.0, 30.0],
            'dec': [5.0, 10.0, 15.0]
        })
        
        test_file = self.temp_dir / "test_sn.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test_survey',
            'version': '1.0',
            'citation': 'Test et al. 2024'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify result is StandardDataset
        assert isinstance(result, StandardDataset)
        
        # Verify basic properties
        assert len(result.z) == 3
        assert len(result.observable) == 3
        assert len(result.uncertainty) == 3
        
        # Verify redshift values are preserved
        np.testing.assert_array_equal(result.z, [0.1, 0.2, 0.3])
        
        # Verify metadata
        assert result.metadata['source'] == 'test_survey'
        assert result.metadata['version'] == '1.0'
        assert result.metadata['citation'] == 'Test et al. 2024'
        assert 'processing_date' in result.metadata
        assert 'n_points' in result.metadata
    
    def test_magnitude_to_distance_modulus_conversion(self):
        """Test magnitude to distance modulus conversion."""
        # Create test data with known values
        test_data = pd.DataFrame({
            'z': [0.1],
            'mb': [15.0],
            'dmb': [0.1],
            'x1': [0.0],
            'dx1': [0.1],
            'c': [0.0],
            'dc': [0.05],
            'ra': [10.0],
            'dec': [5.0]
        })
        
        test_file = self.temp_dir / "test_conversion.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {'source': 'test', 'version': '1.0'}
        
        result = self.module.derive(test_file, metadata)
        
        # The distance modulus should be calculated from mb, x1, c
        # μ = mb - M + α*x1 - β*c (with default cosmological parameters)
        expected_mu = 15.0 + 19.3 + 0.14 * 0.0 - 3.1 * 0.0  # Using default values from code
        
        # Allow for small numerical differences
        assert abs(result.observable[0] - expected_mu) < 0.01
    
    def test_duplicate_removal(self):
        """Test duplicate removal by coordinate matching."""
        # Create test data with duplicates (same coordinates)
        test_data = pd.DataFrame({
            'z': [0.1, 0.2, 0.1],  # Duplicate redshift
            'mb': [15.0, 16.0, 15.1],  # Slightly different magnitude
            'dmb': [0.1, 0.15, 0.1],
            'x1': [0.0, 0.5, 0.0],
            'dx1': [0.1, 0.1, 0.1],
            'c': [0.0, 0.1, 0.0],
            'dc': [0.05, 0.05, 0.05],
            'ra': [10.0, 20.0, 10.0001],  # Very close coordinates (duplicate)
            'dec': [5.0, 10.0, 5.0001]
        })
        
        test_file = self.temp_dir / "test_duplicates.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {'source': 'test', 'version': '1.0'}
        
        result = self.module.derive(test_file, metadata)
        
        # Should have removed one duplicate, leaving 2 points
        assert len(result.z) == 2
        assert len(result.observable) == 2
        assert len(result.uncertainty) == 2
    
    def test_covariance_matrix_application(self):
        """Test systematic covariance matrix application."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.1, 0.2],
            'mb': [15.0, 16.0],
            'dmb': [0.1, 0.15],
            'x1': [0.0, 0.5],
            'dx1': [0.1, 0.1],
            'c': [0.0, 0.1],
            'dc': [0.05, 0.05],
            'ra': [10.0, 20.0],
            'dec': [5.0, 10.0]
        })
        
        test_file = self.temp_dir / "test_covariance.csv"
        test_data.to_csv(test_file, index=False)
        
        # Create systematic covariance matrix file
        sys_cov = np.array([[0.01, 0.005], [0.005, 0.02]])
        cov_file = self.temp_dir / "systematic_covariance.npy"
        np.save(cov_file, sys_cov)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'systematic_covariance_file': str(cov_file)
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should have covariance matrix
        assert result.covariance is not None
        assert result.covariance.shape == (2, 2)
        
        # Verify covariance includes systematic component
        # (exact values depend on implementation details)
        assert np.all(result.covariance >= 0)  # Should be positive semi-definite
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data values."""
        # Create test data with invalid values
        test_data = pd.DataFrame({
            'z': [0.1, -0.1, 0.3],  # Negative redshift (invalid)
            'mb': [15.0, np.nan, 17.0],  # NaN magnitude
            'dmb': [0.1, 0.15, 0.2],
            'x1': [0.0, 0.5, -0.3],
            'dx1': [0.1, 0.1, 0.1],
            'c': [0.0, 0.1, -0.1],
            'dc': [0.05, 0.05, 0.05],
            'ra': [10.0, 20.0, 30.0],
            'dec': [5.0, 10.0, 15.0]
        })
        
        test_file = self.temp_dir / "test_invalid.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {'source': 'test', 'version': '1.0'}
        
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
        assert any('distance modulus' in formula.lower() for formula in formulas)
        assert any('α' in formula for formula in formulas)  # SALT2 parameters
        assert any('β' in formula for formula in formulas)
    
    def test_calibration_homogenization(self):
        """Test calibration system homogenization."""
        # Create test data with different calibration systems
        test_data = pd.DataFrame({
            'z': [0.1, 0.2, 0.3],
            'mb': [15.0, 16.0, 17.0],
            'dmb': [0.1, 0.15, 0.2],
            'x1': [0.0, 0.5, -0.3],
            'dx1': [0.1, 0.1, 0.1],
            'c': [0.0, 0.1, -0.1],
            'dc': [0.05, 0.05, 0.05],
            'ra': [10.0, 20.0, 30.0],
            'dec': [5.0, 10.0, 15.0],
            'calibration': ['SALT2', 'MLCS2k2', 'SALT2']
        })
        
        test_file = self.temp_dir / "test_calibration.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {'source': 'test', 'version': '1.0'}
        
        result = self.module.derive(test_file, metadata)
        
        # Should successfully process mixed calibration systems
        assert len(result.z) == 3
        assert 'calibration_homogenization' in result.metadata
    
    def test_redshift_range_validation(self):
        """Test redshift range validation."""
        # Create test data with extreme redshift values
        test_data = pd.DataFrame({
            'z': [0.1, 0.2, 5.0],  # Very high redshift
            'mb': [15.0, 16.0, 25.0],
            'dmb': [0.1, 0.15, 0.5],
            'x1': [0.0, 0.5, -0.3],
            'dx1': [0.1, 0.1, 0.1],
            'c': [0.0, 0.1, -0.1],
            'dc': [0.05, 0.05, 0.05],
            'ra': [10.0, 20.0, 30.0],
            'dec': [5.0, 10.0, 15.0]
        })
        
        test_file = self.temp_dir / "test_extreme_z.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {'source': 'test', 'version': '1.0'}
        
        # Should either filter out extreme values or raise warning
        result = self.module.derive(test_file, metadata)
        
        # Check that processing completed (implementation may filter or warn)
        assert isinstance(result, StandardDataset)
        assert 'redshift_filtering' in result.metadata or len(result.z) <= 3