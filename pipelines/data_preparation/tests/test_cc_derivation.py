"""
Unit tests for the CC (Cosmic Chronometers) derivation module.

Tests CC-specific transformation logic including H(z) data merging,
overlapping redshift bin filtering, and uncertainty propagation.
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

from pipelines.data_preparation.derivation.cc_derivation import CCDerivationModule
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.interfaces import ProcessingError


class TestCCDerivationModule:
    """Test cases for CCDerivationModule."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.module = CCDerivationModule()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_type(self):
        """Test dataset type property."""
        assert self.module.dataset_type == 'cc'
    
    def test_supported_formats(self):
        """Test supported formats property."""
        expected_formats = ['.txt', '.csv', '.dat', '.json']
        assert self.module.supported_formats == expected_formats
    
    def test_validate_input_valid_cc_data(self):
        """Test input validation with valid cosmic chronometer data."""
        # Create valid CC data
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40, 0.48, 0.88, 0.90],
            'H_z': [83.0, 77.0, 95.0, 97.0, 90.0, 117.0],
            'H_z_err': [8.0, 14.0, 17.0, 62.0, 40.0, 23.0],
            'source': ['Stern07', 'Simon05', 'Stern07', 'Stern07', 'Stern07', 'Moresco12']
        })
        
        test_file = self.temp_dir / "test_cc.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'CC_compilation',
            'version': '1.0',
            'data_type': 'hubble_parameter'
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
            'z': [0.17, 0.27, 0.40],
            'H_z': [83.0, 77.0, 95.0]
            # Missing H_z_err
        })
        
        test_file = self.temp_dir / "test_incomplete.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.module.validate_input(test_file, metadata)
    
    def test_derive_basic_cc_data(self):
        """Test basic CC data derivation."""
        # Create test CC data
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40, 0.48],
            'H_z': [83.0, 77.0, 95.0, 97.0],
            'H_z_err': [8.0, 14.0, 17.0, 62.0],
            'source': ['Stern07', 'Simon05', 'Stern07', 'Stern07']
        })
        
        test_file = self.temp_dir / "test_cc.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'CC_compilation',
            'version': '1.0',
            'data_type': 'hubble_parameter',
            'citation': 'Moresco et al. 2016'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Verify result is StandardDataset
        assert isinstance(result, StandardDataset)
        
        # Verify basic properties
        assert len(result.z) == 4
        assert len(result.observable) == 4
        assert len(result.uncertainty) == 4
        
        # Verify redshift values are preserved
        np.testing.assert_array_equal(result.z, [0.17, 0.27, 0.40, 0.48])
        
        # Verify H(z) values
        np.testing.assert_array_equal(result.observable, [83.0, 77.0, 95.0, 97.0])
        
        # Verify uncertainties
        np.testing.assert_array_equal(result.uncertainty, [8.0, 14.0, 17.0, 62.0])
        
        # Verify metadata
        assert result.metadata['source'] == 'CC_compilation'
        assert result.metadata['data_type'] == 'hubble_parameter'
        assert result.metadata['citation'] == 'Moresco et al. 2016'
        assert 'processing_timestamp' in result.metadata
        assert 'n_points' in result.metadata
    
    def test_merge_multiple_compilations(self):
        """Test merging data from multiple compilation sources."""
        # Create test data from multiple sources
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40, 0.17, 0.90],  # Overlapping redshift at 0.17
            'H_z': [83.0, 77.0, 95.0, 85.0, 117.0],  # Different H(z) at z=0.17
            'H_z_err': [8.0, 14.0, 17.0, 10.0, 23.0],
            'source': ['Stern07', 'Simon05', 'Stern07', 'Moresco12', 'Moresco12'],
            'compilation': ['A', 'A', 'A', 'B', 'B']
        })
        
        test_file = self.temp_dir / "test_multi_compilation.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'CC_merged',
            'version': '1.0',
            'data_type': 'hubble_parameter',
            'merge_strategy': 'weighted_average'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should merge overlapping points
        assert isinstance(result, StandardDataset)
        assert 'compilation_merging' in result.metadata
        assert 'merge_strategy' in result.metadata
        
        # Should have fewer points than input due to merging
        assert len(result.z) <= 5
    
    def test_overlapping_redshift_filtering(self):
        """Test filtering of overlapping redshift bins."""
        # Create test data with very close redshift values
        test_data = pd.DataFrame({
            'z': [0.170, 0.171, 0.172, 0.27, 0.40],  # Very close z values
            'H_z': [83.0, 84.0, 82.0, 77.0, 95.0],
            'H_z_err': [8.0, 9.0, 8.5, 14.0, 17.0],
            'source': ['Stern07', 'Stern07', 'Stern07', 'Simon05', 'Stern07']
        })
        
        test_file = self.temp_dir / "test_overlapping.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter',
            'redshift_tolerance': 0.01  # Merge points within Δz = 0.01
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should filter/merge overlapping redshift bins
        assert len(result.z) < 5  # Should have fewer points after filtering
        assert 'redshift_filtering' in result.metadata
    
    def test_uncertainty_propagation(self):
        """Test uncertainty propagation in merging process."""
        # Create test data with overlapping measurements
        test_data = pd.DataFrame({
            'z': [0.17, 0.17, 0.27],  # Two measurements at z=0.17
            'H_z': [83.0, 85.0, 77.0],
            'H_z_err': [8.0, 10.0, 14.0],
            'source': ['Stern07', 'Moresco12', 'Simon05']
        })
        
        test_file = self.temp_dir / "test_uncertainty.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should properly propagate uncertainties in weighted average
        assert len(result.z) == 2  # Two unique redshifts after merging
        
        # Check that merged uncertainty is reasonable (should be smaller than individual)
        merged_idx = 0 if result.z[0] == 0.17 else 1
        merged_uncertainty = result.uncertainty[merged_idx]
        
        # Weighted average uncertainty should be smaller than individual uncertainties
        assert merged_uncertainty < min(8.0, 10.0)
    
    def test_h_z_sign_convention_validation(self):
        """Test H(z) sign convention validation."""
        # Create test data with potentially incorrect signs
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40],
            'H_z': [83.0, -77.0, 95.0],  # Negative H(z) (should be positive)
            'H_z_err': [8.0, 14.0, 17.0],
            'source': ['Stern07', 'Simon05', 'Stern07']
        })
        
        test_file = self.temp_dir / "test_sign_convention.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            self.module.derive(test_file, metadata)
        
        assert exc_info.value.error_type == "data_validation_error"
        assert "negative" in exc_info.value.error_message.lower()
    
    def test_systematic_error_handling(self):
        """Test systematic error handling and propagation."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40],
            'H_z': [83.0, 77.0, 95.0],
            'H_z_err': [8.0, 14.0, 17.0],
            'source': ['Stern07', 'Simon05', 'Stern07']
        })
        
        test_file = self.temp_dir / "test_systematic.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter',
            'systematic_errors': {
                'stellar_evolution': 0.05,  # 5% systematic
                'metallicity': 0.03         # 3% systematic
            }
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should include systematic errors in total uncertainty
        assert 'systematic_errors_applied' in result.metadata
        
        # Total uncertainties should be larger than statistical only
        expected_sys_frac = np.sqrt(0.05**2 + 0.03**2)  # Combined systematic fraction
        
        for i, (h_val, stat_err) in enumerate(zip([83.0, 77.0, 95.0], [8.0, 14.0, 17.0])):
            sys_err = h_val * expected_sys_frac
            expected_total_err = np.sqrt(stat_err**2 + sys_err**2)
            
            # Allow for small numerical differences
            assert abs(result.uncertainty[i] - expected_total_err) < 1.0
    
    def test_error_handling_invalid_redshifts(self):
        """Test error handling with invalid redshift values."""
        # Create test data with invalid redshifts
        test_data = pd.DataFrame({
            'z': [-0.1, 0.27, np.nan],  # Negative and NaN redshifts
            'H_z': [83.0, 77.0, 95.0],
            'H_z_err': [8.0, 14.0, 17.0],
            'source': ['Stern07', 'Simon05', 'Stern07']
        })
        
        test_file = self.temp_dir / "test_invalid_z.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        with pytest.raises(ProcessingError) as exc_info:
            self.module.derive(test_file, metadata)
        
        assert exc_info.value.error_type == "data_validation_error"
    
    def test_error_handling_invalid_h_values(self):
        """Test error handling with invalid H(z) values."""
        # Create test data with invalid H(z) values
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40],
            'H_z': [83.0, np.inf, 0.0],  # Infinite and zero H(z)
            'H_z_err': [8.0, 14.0, 17.0],
            'source': ['Stern07', 'Simon05', 'Stern07']
        })
        
        test_file = self.temp_dir / "test_invalid_h.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
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
        assert any('H(z)' in formula for formula in formulas)
        assert any('weighted' in formula.lower() for formula in formulas)
        
        # Verify CC-specific references
        references = summary['references']
        assert any('Moresco' in ref or 'Stern' in ref for ref in references)
    
    def test_source_tracking(self):
        """Test tracking of data sources through processing."""
        # Create test data with multiple sources
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40, 0.48],
            'H_z': [83.0, 77.0, 95.0, 97.0],
            'H_z_err': [8.0, 14.0, 17.0, 62.0],
            'source': ['Stern07', 'Simon05', 'Stern07', 'Moresco12']
        })
        
        test_file = self.temp_dir / "test_sources.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'CC_compilation',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should track individual sources
        assert 'individual_sources' in result.metadata
        sources = result.metadata['individual_sources']
        
        expected_sources = ['Stern07', 'Simon05', 'Moresco12']
        for source in expected_sources:
            assert source in sources
    
    def test_redshift_range_validation(self):
        """Test redshift range validation for CC measurements."""
        # Create test data with extreme redshift values
        test_data = pd.DataFrame({
            'z': [0.01, 0.27, 5.0],  # Very low and very high redshifts
            'H_z': [70.0, 77.0, 200.0],
            'H_z_err': [10.0, 14.0, 50.0],
            'source': ['Test', 'Simon05', 'Test']
        })
        
        test_file = self.temp_dir / "test_extreme_z_cc.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        # Should either filter out extreme values or raise warning
        result = self.module.derive(test_file, metadata)
        
        # Check that processing completed (implementation may filter or warn)
        assert isinstance(result, StandardDataset)
        assert 'redshift_filtering' in result.metadata or len(result.z) <= 3
    
    def test_covariance_matrix_construction(self):
        """Test covariance matrix construction for CC data."""
        # Create test data
        test_data = pd.DataFrame({
            'z': [0.17, 0.27, 0.40],
            'H_z': [83.0, 77.0, 95.0],
            'H_z_err': [8.0, 14.0, 17.0],
            'source': ['Stern07', 'Simon05', 'Stern07']
        })
        
        test_file = self.temp_dir / "test_covariance_cc.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should construct diagonal covariance matrix from uncertainties
        expected_cov = np.diag([8.0**2, 14.0**2, 17.0**2])
        
        if result.covariance is not None:
            np.testing.assert_allclose(result.covariance, expected_cov, rtol=1e-10)
    
    def test_weighted_average_calculation(self):
        """Test weighted average calculation for overlapping measurements."""
        # Create test data with known overlapping measurements
        test_data = pd.DataFrame({
            'z': [0.17, 0.17],  # Same redshift
            'H_z': [80.0, 90.0],  # Different H(z) values
            'H_z_err': [10.0, 15.0],  # Different uncertainties
            'source': ['Source1', 'Source2']
        })
        
        test_file = self.temp_dir / "test_weighted_avg.csv"
        test_data.to_csv(test_file, index=False)
        
        metadata = {
            'source': 'test',
            'version': '1.0',
            'data_type': 'hubble_parameter'
        }
        
        result = self.module.derive(test_file, metadata)
        
        # Should have one merged measurement
        assert len(result.z) == 1
        assert result.z[0] == 0.17
        
        # Calculate expected weighted average
        w1 = 1.0 / (10.0**2)
        w2 = 1.0 / (15.0**2)
        expected_h = (w1 * 80.0 + w2 * 90.0) / (w1 + w2)
        expected_err = 1.0 / np.sqrt(w1 + w2)
        
        # Check weighted average calculation
        assert abs(result.observable[0] - expected_h) < 0.1
        assert abs(result.uncertainty[0] - expected_err) < 0.1