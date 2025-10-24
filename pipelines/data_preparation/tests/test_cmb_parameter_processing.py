"""
Unit tests for CMB raw parameter processing functionality.

Tests parameter detection, parsing, validation, and covariance matrix handling
for the CMB raw parameter integration feature.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock pandas if not available
try:
    import pandas as pd
except ImportError:
    pd = MagicMock()
    sys.modules['pandas'] = pd

from pipelines.data_preparation.derivation.cmb_derivation import (
    detect_raw_parameters, classify_parameter_format, validate_parameter_completeness,
    parse_parameter_file, normalize_parameter_names, extract_covariance_matrix,
    ParameterFormat, RawParameterInfo, PARAMETER_ALIASES
)
from pipelines.data_preparation.derivation.cmb_models import (
    ParameterSet, DistancePriors, CMBConfig
)
from pipelines.data_preparation.derivation.cmb_exceptions import (
    ParameterDetectionError, ParameterValidationError, CovarianceError
)


class TestParameterDetection:
    """Test cases for raw parameter detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detect_raw_parameters_cmb_dataset(self):
        """Test detection of raw parameters in CMB dataset registry entry."""
        registry_entry = {
            'metadata': {
                'dataset_type': 'cmb',
                'name': 'planck2018_raw_params',
                'source': 'Planck Collaboration'
            },
            'sources': {
                'primary': {
                    'url': 'https://example.com/planck_params.csv',
                    'extraction': {
                        'target_files': ['planck_cosmological_params.csv']
                    }
                }
            }
        }
        
        result = detect_raw_parameters(registry_entry)
        
        assert result is not None
        assert isinstance(result, RawParameterInfo)
        assert result.file_path == 'planck_cosmological_params.csv'
        assert result.format_type == ParameterFormat.CSV
    
    def test_detect_raw_parameters_non_cmb_dataset(self):
        """Test that non-CMB datasets return None."""
        registry_entry = {
            'metadata': {
                'dataset_type': 'sn',  # Not CMB
                'name': 'pantheon_sn',
                'source': 'Pantheon Collaboration'
            },
            'sources': {
                'primary': {
                    'url': 'https://example.com/pantheon.csv',
                    'extraction': {
                        'target_files': ['pantheon_data.csv']
                    }
                }
            }
        }
        
        result = detect_raw_parameters(registry_entry)
        assert result is None
    
    def test_detect_raw_parameters_metadata_file_reference(self):
        """Test detection from metadata file references."""
        registry_entry = {
            'metadata': {
                'dataset_type': 'cmb',
                'name': 'planck2018',
                'parameter_file': '/path/to/planck_params.json'
            },
            'sources': {}
        }
        
        result = detect_raw_parameters(registry_entry)
        
        assert result is not None
        assert result.file_path == '/path/to/planck_params.json'
        assert result.format_type == ParameterFormat.JSON
    
    def test_detect_raw_parameters_no_parameters_found(self):
        """Test behavior when no parameter files are found."""
        registry_entry = {
            'metadata': {
                'dataset_type': 'cmb',
                'name': 'cmb_distance_priors_only'
            },
            'sources': {
                'primary': {
                    'url': 'https://example.com/distance_priors.csv',
                    'extraction': {
                        'target_files': ['cmb_distance_priors.csv']
                    }
                }
            }
        }
        
        result = detect_raw_parameters(registry_entry)
        assert result is None
    
    def test_detect_raw_parameters_error_handling(self):
        """Test error handling in parameter detection."""
        # Invalid registry entry structure
        invalid_entry = {
            'invalid_structure': True
        }
        
        with pytest.raises(ParameterDetectionError) as exc_info:
            detect_raw_parameters(invalid_entry)
        
        assert "Failed to detect raw parameters" in str(exc_info.value)
        assert exc_info.value.error_code == "CMB_PARAM_DETECTION_ERROR"


class TestParameterFormatClassification:
    """Test cases for parameter file format classification."""
    
    def test_classify_json_format(self):
        """Test JSON format classification."""
        test_cases = [
            'planck_params.json',
            'cosmological_parameters.JSON',
            '/path/to/mcmc_json_output.json'
        ]
        
        for file_path in test_cases:
            result = classify_parameter_format(file_path)
            assert result == ParameterFormat.JSON
    
    def test_classify_csv_format(self):
        """Test CSV format classification."""
        test_cases = [
            'planck_params.csv',
            'cosmological_parameters.CSV',
            '/path/to/mcmc_output.csv'
        ]
        
        for file_path in test_cases:
            result = classify_parameter_format(file_path)
            assert result == ParameterFormat.CSV
    
    def test_classify_numpy_format(self):
        """Test NumPy format classification."""
        test_cases = [
            'planck_params.npy',
            'cosmological_parameters.npz',
            '/path/to/mcmc_numpy_output.NPY'
        ]
        
        for file_path in test_cases:
            result = classify_parameter_format(file_path)
            assert result == ParameterFormat.NUMPY
    
    def test_classify_text_format(self):
        """Test text format classification."""
        test_cases = [
            'planck_params.txt',
            'cosmological_parameters.dat',
            'mcmc_chain.param',
            'output.chain'
        ]
        
        for file_path in test_cases:
            result = classify_parameter_format(file_path)
            assert result == ParameterFormat.TEXT
    
    def test_classify_format_by_name_patterns(self):
        """Test format classification by filename patterns."""
        # JSON pattern in filename
        assert classify_parameter_format('planck_json_output') == ParameterFormat.JSON
        
        # CSV pattern in filename
        assert classify_parameter_format('planck_csv_data') == ParameterFormat.CSV
        
        # NumPy pattern in filename
        assert classify_parameter_format('planck_numpy_array') == ParameterFormat.NUMPY
        
        # Default to text for unknown patterns
        assert classify_parameter_format('unknown_format_file') == ParameterFormat.TEXT


class TestParameterNameNormalization:
    """Test cases for parameter name normalization and fuzzy matching."""
    
    def test_normalize_exact_matches(self):
        """Test normalization with exact parameter name matches."""
        raw_params = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544,
            'A_s': 2.1e-9
        }
        
        normalized = normalize_parameter_names(raw_params)
        
        # Should preserve exact matches
        assert normalized == raw_params
    
    def test_normalize_case_variations(self):
        """Test normalization with case variations."""
        raw_params = {
            'h0': 67.4,
            'omega_m': 0.315,
            'OMEGA_B_H2': 0.02237,
            'N_S': 0.9649,
            'TAU': 0.0544
        }
        
        normalized = normalize_parameter_names(raw_params)
        
        expected = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544
        }
        
        assert normalized == expected
    
    def test_normalize_alias_variations(self):
        """Test normalization with parameter aliases."""
        raw_params = {
            'hubble': 67.4,
            'Om0': 0.315,
            'omegabh2': 0.02237,
            'ns': 0.9649,
            'tau_reio': 0.0544,
            'As': 2.1e-9
        }
        
        normalized = normalize_parameter_names(raw_params)
        
        expected = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544,
            'A_s': 2.1e-9
        }
        
        assert normalized == expected
    
    def test_normalize_unicode_variations(self):
        """Test normalization with Unicode parameter names."""
        raw_params = {
            'H₀': 67.4,
            'Ωm': 0.315,
            'Ωbh²': 0.02237,
            'τ': 0.0544
        }
        
        normalized = normalize_parameter_names(raw_params)
        
        # Should handle Unicode variations
        expected_keys = {'H0', 'Omega_m', 'Omega_b_h2', 'tau'}
        assert set(normalized.keys()).intersection(expected_keys) == expected_keys
    
    def test_normalize_unrecognized_parameters(self):
        """Test handling of unrecognized parameter names."""
        raw_params = {
            'H0': 67.4,
            'unknown_param': 123.45,
            'another_unknown': 678.90
        }
        
        normalized = normalize_parameter_names(raw_params)
        
        # Should preserve recognized parameters
        assert 'H0' in normalized
        assert normalized['H0'] == 67.4
        
        # Unrecognized parameters may or may not be included
        # depending on fuzzy matching results


class TestParameterValidation:
    """Test cases for parameter completeness validation."""
    
    def test_validate_complete_parameters(self):
        """Test validation with all required parameters present."""
        params = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544
        }
        
        result = validate_parameter_completeness(params)
        
        assert result['valid'] is True
        assert len(result['missing']) == 0
        assert set(result['found']) == set(['H0', 'Omega_m', 'Omega_b_h2', 'n_s', 'tau'])
        assert 'normalized_params' in result
    
    def test_validate_parameters_with_optional(self):
        """Test validation with optional parameters included."""
        params = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544,
            'A_s': 2.1e-9  # Optional parameter
        }
        
        result = validate_parameter_completeness(params)
        
        assert result['valid'] is True
        assert 'A_s' in result['found']
        assert 'A_s' in result['normalized_params']
    
    def test_validate_missing_required_parameters(self):
        """Test validation with missing required parameters."""
        params = {
            'H0': 67.4,
            'Omega_m': 0.315,
            # Missing Omega_b_h2, n_s, tau
        }
        
        with pytest.raises(ParameterDetectionError) as exc_info:
            validate_parameter_completeness(params)
        
        assert "Required CMB parameters not found" in str(exc_info.value)
        assert exc_info.value.error_code == "CMB_PARAM_DETECTION_ERROR"
    
    def test_validate_case_insensitive_matching(self):
        """Test validation with case-insensitive parameter matching."""
        params = {
            'h0': 67.4,
            'omega_m': 0.315,
            'OMEGA_B_H2': 0.02237,
            'N_S': 0.9649,
            'TAU': 0.0544
        }
        
        result = validate_parameter_completeness(params)
        
        assert result['valid'] is True
        assert len(result['missing']) == 0
        
        # Check that mappings are recorded
        mappings = result['mapped']
        assert 'h0' in mappings
        assert mappings['h0'] == 'H0'


class TestParameterFileParsing:
    """Test cases for parameter file parsing across different formats."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parse_json_parameters(self):
        """Test parsing parameters from JSON file."""
        params_data = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544,
            'A_s': 2.1e-9
        }
        
        json_file = self.temp_dir / 'test_params.json'
        with open(json_file, 'w') as f:
            json.dump(params_data, f)
        
        result = parse_parameter_file(str(json_file), ParameterFormat.JSON)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
        assert result.Omega_b_h2 == 0.02237
        assert result.n_s == 0.9649
        assert result.tau == 0.0544
        assert result.A_s == 2.1e-9
    
    def test_parse_nested_json_parameters(self):
        """Test parsing parameters from nested JSON structure."""
        nested_data = {
            'metadata': {'version': '1.0'},
            'parameters': {
                'H0': 67.4,
                'Omega_m': 0.315,
                'Omega_b_h2': 0.02237,
                'n_s': 0.9649,
                'tau': 0.0544
            }
        }
        
        json_file = self.temp_dir / 'test_nested_params.json'
        with open(json_file, 'w') as f:
            json.dump(nested_data, f)
        
        result = parse_parameter_file(str(json_file), ParameterFormat.JSON)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
    
    @pytest.mark.skipif(pd is MagicMock, reason="pandas not available")
    def test_parse_csv_parameters_single_row(self):
        """Test parsing parameters from CSV file with single row."""
        import pandas as pd
        
        params_df = pd.DataFrame({
            'H0': [67.4],
            'Omega_m': [0.315],
            'Omega_b_h2': [0.02237],
            'n_s': [0.9649],
            'tau': [0.0544]
        })
        
        csv_file = self.temp_dir / 'test_params.csv'
        params_df.to_csv(csv_file, index=False)
        
        result = parse_parameter_file(str(csv_file), ParameterFormat.CSV)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
    
    @pytest.mark.skipif(pd is MagicMock, reason="pandas not available")
    def test_parse_csv_parameters_multiple_rows(self):
        """Test parsing parameters from CSV file with multiple rows (MCMC chain)."""
        import pandas as pd
        
        # Simulate MCMC chain with multiple samples
        n_samples = 100
        params_df = pd.DataFrame({
            'H0': np.random.normal(67.4, 0.5, n_samples),
            'Omega_m': np.random.normal(0.315, 0.01, n_samples),
            'Omega_b_h2': np.random.normal(0.02237, 0.0001, n_samples),
            'n_s': np.random.normal(0.9649, 0.005, n_samples),
            'tau': np.random.normal(0.0544, 0.008, n_samples)
        })
        
        csv_file = self.temp_dir / 'test_chain_params.csv'
        params_df.to_csv(csv_file, index=False)
        
        result = parse_parameter_file(str(csv_file), ParameterFormat.CSV)
        
        assert isinstance(result, ParameterSet)
        # Should use mean values from chain
        assert abs(result.H0 - 67.4) < 0.2  # Within reasonable range of true mean
        assert abs(result.Omega_m - 0.315) < 0.05
    
    def test_parse_numpy_parameters_1d_array(self):
        """Test parsing parameters from 1D NumPy array."""
        # Standard parameter order: H0, Omega_m, Omega_b_h2, n_s, tau, A_s
        params_array = np.array([67.4, 0.315, 0.02237, 0.9649, 0.0544, 2.1e-9])
        
        npy_file = self.temp_dir / 'test_params.npy'
        np.save(npy_file, params_array)
        
        result = parse_parameter_file(str(npy_file), ParameterFormat.NUMPY)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
        assert result.A_s == 2.1e-9
    
    def test_parse_numpy_parameters_2d_array(self):
        """Test parsing parameters from 2D NumPy array (MCMC chain)."""
        # Simulate MCMC chain: (n_samples, n_params)
        n_samples = 50
        n_params = 5
        
        # Generate samples around Planck values
        true_values = np.array([67.4, 0.315, 0.02237, 0.9649, 0.0544])
        params_array = np.random.multivariate_normal(
            true_values, 
            np.diag([0.5**2, 0.01**2, 0.0001**2, 0.005**2, 0.008**2]),
            n_samples
        )
        
        npy_file = self.temp_dir / 'test_chain_params.npy'
        np.save(npy_file, params_array)
        
        result = parse_parameter_file(str(npy_file), ParameterFormat.NUMPY)
        
        assert isinstance(result, ParameterSet)
        # Should use mean values from chain
        assert abs(result.H0 - 67.4) < 0.2
        assert abs(result.Omega_m - 0.315) < 0.05
    
    def test_parse_npz_parameters(self):
        """Test parsing parameters from NPZ file with named arrays."""
        params_dict = {
            'H0': np.array([67.4]),
            'Omega_m': np.array([0.315]),
            'Omega_b_h2': np.array([0.02237]),
            'n_s': np.array([0.9649]),
            'tau': np.array([0.0544])
        }
        
        npz_file = self.temp_dir / 'test_params.npz'
        np.savez(npz_file, **params_dict)
        
        result = parse_parameter_file(str(npz_file), ParameterFormat.NUMPY)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
    
    def test_parse_text_parameters_key_value_format(self):
        """Test parsing parameters from text file with key=value format."""
        text_content = """
        # Planck 2018 cosmological parameters
        H0 = 67.4
        Omega_m = 0.315
        Omega_b_h2 = 0.02237
        n_s = 0.9649
        tau = 0.0544
        """
        
        text_file = self.temp_dir / 'test_params.txt'
        with open(text_file, 'w') as f:
            f.write(text_content)
        
        result = parse_parameter_file(str(text_file), ParameterFormat.TEXT)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
    
    def test_parse_text_parameters_whitespace_separated(self):
        """Test parsing parameters from whitespace-separated text file."""
        text_content = """
        # Parameter file
        H0          67.4
        Omega_m     0.315
        Omega_b_h2  0.02237
        n_s         0.9649
        tau         0.0544
        """
        
        text_file = self.temp_dir / 'test_params_ws.txt'
        with open(text_file, 'w') as f:
            f.write(text_content)
        
        result = parse_parameter_file(str(text_file), ParameterFormat.TEXT)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
    
    def test_parse_text_parameters_csv_like(self):
        """Test parsing parameters from CSV-like text file."""
        text_content = "67.4, 0.315, 0.02237, 0.9649, 0.0544"
        
        text_file = self.temp_dir / 'test_params_csv.txt'
        with open(text_file, 'w') as f:
            f.write(text_content)
        
        result = parse_parameter_file(str(text_file), ParameterFormat.TEXT)
        
        assert isinstance(result, ParameterSet)
        assert result.H0 == 67.4
        assert result.Omega_m == 0.315
    
    def test_parse_parameter_file_error_handling(self):
        """Test error handling in parameter file parsing."""
        # Test with non-existent file
        with pytest.raises(ParameterDetectionError):
            parse_parameter_file('/nonexistent/file.json', ParameterFormat.JSON)
        
        # Test with invalid JSON
        invalid_json_file = self.temp_dir / 'invalid.json'
        with open(invalid_json_file, 'w') as f:
            f.write('{ invalid json content')
        
        with pytest.raises(ParameterDetectionError):
            parse_parameter_file(str(invalid_json_file), ParameterFormat.JSON)
    
    def test_parse_parameter_file_missing_parameters(self):
        """Test parsing file with missing required parameters."""
        incomplete_params = {
            'H0': 67.4,
            'Omega_m': 0.315
            # Missing required parameters
        }
        
        json_file = self.temp_dir / 'incomplete_params.json'
        with open(json_file, 'w') as f:
            json.dump(incomplete_params, f)
        
        with pytest.raises(ParameterDetectionError):
            parse_parameter_file(str(json_file), ParameterFormat.JSON)


class TestCovarianceMatrixHandling:
    """Test cases for covariance matrix extraction and validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_extract_covariance_matrix_npy(self):
        """Test extracting covariance matrix from .npy file."""
        # Create 5x5 covariance matrix for 5 parameters
        cov_matrix = np.array([
            [0.5**2, 0.01, 0.0, 0.0, 0.0],
            [0.01, 0.01**2, 0.0001, 0.0, 0.0],
            [0.0, 0.0001, 0.0001**2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.005**2, 0.0001],
            [0.0, 0.0, 0.0, 0.0001, 0.008**2]
        ])
        
        npy_file = self.temp_dir / 'covariance.npy'
        np.save(npy_file, cov_matrix)
        
        result = extract_covariance_matrix(str(npy_file))
        
        assert result is not None
        np.testing.assert_array_equal(result, cov_matrix)
        assert result.shape == (5, 5)
    
    def test_extract_covariance_matrix_npz(self):
        """Test extracting covariance matrix from .npz file."""
        cov_matrix = np.eye(5) * 0.01  # Simple diagonal covariance
        
        npz_file = self.temp_dir / 'covariance.npz'
        np.savez(npz_file, covariance=cov_matrix)
        
        result = extract_covariance_matrix(str(npz_file))
        
        assert result is not None
        np.testing.assert_array_equal(result, cov_matrix)
    
    def test_extract_covariance_matrix_npz_multiple_keys(self):
        """Test extracting covariance from .npz file with multiple arrays."""
        cov_matrix = np.eye(5) * 0.01
        other_data = np.random.randn(10, 10)
        
        npz_file = self.temp_dir / 'data.npz'
        np.savez(npz_file, cov=cov_matrix, other=other_data)
        
        result = extract_covariance_matrix(str(npz_file))
        
        assert result is not None
        np.testing.assert_array_equal(result, cov_matrix)
    
    def test_extract_covariance_matrix_json(self):
        """Test extracting covariance matrix from JSON file."""
        cov_matrix = np.eye(5) * 0.01
        cov_data = {
            'covariance_matrix': cov_matrix.tolist(),
            'metadata': {'units': 'dimensionless'}
        }
        
        json_file = self.temp_dir / 'covariance.json'
        with open(json_file, 'w') as f:
            json.dump(cov_data, f)
        
        result = extract_covariance_matrix(str(json_file))
        
        assert result is not None
        np.testing.assert_allclose(result, cov_matrix)
    
    def test_extract_covariance_matrix_text(self):
        """Test extracting covariance matrix from text file."""
        cov_matrix = np.eye(3) * 0.01
        
        text_file = self.temp_dir / 'covariance.txt'
        np.savetxt(text_file, cov_matrix)
        
        result = extract_covariance_matrix(str(text_file))
        
        assert result is not None
        np.testing.assert_allclose(result, cov_matrix)
    
    def test_extract_covariance_matrix_nonexistent_file(self):
        """Test behavior with non-existent covariance file."""
        result = extract_covariance_matrix('/nonexistent/covariance.npy')
        assert result is None
    
    def test_extract_covariance_matrix_invalid_dimensions(self):
        """Test error handling with invalid matrix dimensions."""
        # Create non-square matrix
        invalid_matrix = np.random.randn(5, 3)  # Not square
        
        npy_file = self.temp_dir / 'invalid_cov.npy'
        np.save(npy_file, invalid_matrix)
        
        with pytest.raises(ParameterDetectionError) as exc_info:
            extract_covariance_matrix(str(npy_file))
        
        assert "must be square" in str(exc_info.value)
    
    def test_extract_covariance_matrix_not_2d(self):
        """Test error handling with non-2D array."""
        # Create 1D array instead of 2D matrix
        invalid_array = np.array([1, 2, 3, 4, 5])
        
        npy_file = self.temp_dir / 'invalid_1d.npy'
        np.save(npy_file, invalid_array)
        
        with pytest.raises(ParameterDetectionError) as exc_info:
            extract_covariance_matrix(str(npy_file))
        
        assert "must be 2D" in str(exc_info.value)
    
    def test_validate_covariance_matrix_symmetry(self):
        """Test covariance matrix symmetry validation."""
        # Create symmetric matrix
        symmetric_matrix = np.array([
            [1.0, 0.1, 0.05],
            [0.1, 2.0, 0.2],
            [0.05, 0.2, 3.0]
        ])
        
        npy_file = self.temp_dir / 'symmetric_cov.npy'
        np.save(npy_file, symmetric_matrix)
        
        result = extract_covariance_matrix(str(npy_file))
        
        # Should not raise error for symmetric matrix
        assert result is not None
        np.testing.assert_allclose(result, result.T)
    
    def test_validate_covariance_matrix_positive_definiteness(self):
        """Test covariance matrix positive definiteness validation."""
        # Create positive definite matrix
        A = np.random.randn(5, 5)
        pos_def_matrix = A @ A.T  # Guaranteed positive definite
        
        npy_file = self.temp_dir / 'pos_def_cov.npy'
        np.save(npy_file, pos_def_matrix)
        
        result = extract_covariance_matrix(str(npy_file))
        
        # Should not raise error for positive definite matrix
        assert result is not None
        eigenvals = np.linalg.eigvals(result)
        assert np.all(eigenvals >= 0)


class TestParameterSetModel:
    """Test cases for ParameterSet data model."""
    
    def test_parameter_set_creation(self):
        """Test creating ParameterSet with valid parameters."""
        params = ParameterSet(
            H0=67.4,
            Omega_m=0.315,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544,
            A_s=2.1e-9
        )
        
        assert params.H0 == 67.4
        assert params.Omega_m == 0.315
        assert params.A_s == 2.1e-9
    
    def test_parameter_set_validation_valid_values(self):
        """Test parameter validation with valid values."""
        params = ParameterSet(
            H0=67.4,
            Omega_m=0.315,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        
        # Should not raise exception
        assert params.validate() is True
    
    def test_parameter_set_validation_invalid_h0(self):
        """Test parameter validation with invalid H0."""
        with pytest.raises(ValueError) as exc_info:
            ParameterSet(
                H0=150.0,  # Too high
                Omega_m=0.315,
                Omega_b_h2=0.02237,
                n_s=0.9649,
                tau=0.0544
            )
        
        assert "H0" in str(exc_info.value)
        assert "outside physical bounds" in str(exc_info.value)
    
    def test_parameter_set_validation_invalid_omega_m(self):
        """Test parameter validation with invalid Omega_m."""
        with pytest.raises(ValueError) as exc_info:
            ParameterSet(
                H0=67.4,
                Omega_m=0.8,  # Too high
                Omega_b_h2=0.02237,
                n_s=0.9649,
                tau=0.0544
            )
        
        assert "Omega_m" in str(exc_info.value)
    
    def test_parameter_set_validation_nan_values(self):
        """Test parameter validation with NaN values."""
        with pytest.raises(ValueError) as exc_info:
            ParameterSet(
                H0=np.nan,  # Invalid
                Omega_m=0.315,
                Omega_b_h2=0.02237,
                n_s=0.9649,
                tau=0.0544
            )
        
        assert "not finite" in str(exc_info.value)
    
    def test_parameter_set_validation_infinite_values(self):
        """Test parameter validation with infinite values."""
        with pytest.raises(ValueError) as exc_info:
            ParameterSet(
                H0=67.4,
                Omega_m=np.inf,  # Invalid
                Omega_b_h2=0.02237,
                n_s=0.9649,
                tau=0.0544
            )
        
        assert "not finite" in str(exc_info.value)
    
    def test_parameter_set_consistency_validation(self):
        """Test parameter consistency validation."""
        # Test with inconsistent baryon fraction
        with pytest.raises(ValueError) as exc_info:
            ParameterSet(
                H0=67.4,
                Omega_m=0.1,  # Very low matter density
                Omega_b_h2=0.05,  # High baryon density
                n_s=0.9649,
                tau=0.0544
            )
        
        assert "Baryon fraction" in str(exc_info.value)
    
    def test_parameter_set_to_dict(self):
        """Test converting ParameterSet to dictionary."""
        params = ParameterSet(
            H0=67.4,
            Omega_m=0.315,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544,
            A_s=2.1e-9
        )
        
        result = params.to_dict()
        
        expected = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544,
            'A_s': 2.1e-9
        }
        
        assert result == expected
    
    def test_parameter_set_from_dict(self):
        """Test creating ParameterSet from dictionary."""
        data = {
            'H0': 67.4,
            'Omega_m': 0.315,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544
        }
        
        params = ParameterSet.from_dict(data)
        
        assert params.H0 == 67.4
        assert params.Omega_m == 0.315
        assert params.A_s is None  # Optional parameter not provided
    
    def test_parameter_set_from_dict_missing_required(self):
        """Test creating ParameterSet from incomplete dictionary."""
        incomplete_data = {
            'H0': 67.4,
            'Omega_m': 0.315
            # Missing required parameters
        }
        
        with pytest.raises(ValueError) as exc_info:
            ParameterSet.from_dict(incomplete_data)
        
        assert "Missing required parameters" in str(exc_info.value)
    
    def test_parameter_set_copy(self):
        """Test copying ParameterSet."""
        original = ParameterSet(
            H0=67.4,
            Omega_m=0.315,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        
        copy = original.copy()
        
        assert copy.H0 == original.H0
        assert copy.Omega_m == original.Omega_m
        assert copy is not original  # Different objects
    
    def test_parameter_set_get_parameter_names(self):
        """Test getting parameter names."""
        params = ParameterSet(
            H0=67.4,
            Omega_m=0.315,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544,
            A_s=2.1e-9
        )
        
        names = params.get_parameter_names()
        
        expected = ['H0', 'Omega_m', 'Omega_b_h2', 'n_s', 'tau', 'A_s']
        assert names == expected
        
        # Test without optional parameter
        params_no_as = ParameterSet(
            H0=67.4,
            Omega_m=0.315,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        
        names_no_as = params_no_as.get_parameter_names()
        expected_no_as = ['H0', 'Omega_m', 'Omega_b_h2', 'n_s', 'tau']
        assert names_no_as == expected_no_as


class TestCMBConfigModel:
    """Test cases for CMBConfig data model."""
    
    def test_cmb_config_default_creation(self):
        """Test creating CMBConfig with default values."""
        config = CMBConfig()
        
        assert config.use_raw_parameters is True
        assert config.z_recombination == 1089.8
        assert config.jacobian_step_size == 1e-6
        assert config.validation_tolerance == 1e-8
        assert config.fallback_to_legacy is True
        assert config.cache_computations is True
        assert config.performance_monitoring is False
    
    def test_cmb_config_custom_creation(self):
        """Test creating CMBConfig with custom values."""
        config = CMBConfig(
            use_raw_parameters=False,
            z_recombination=1090.0,
            jacobian_step_size=1e-5,
            validation_tolerance=1e-7,
            fallback_to_legacy=False,
            cache_computations=False,
            performance_monitoring=True
        )
        
        assert config.use_raw_parameters is False
        assert config.z_recombination == 1090.0
        assert config.jacobian_step_size == 1e-5
        assert config.validation_tolerance == 1e-7
        assert config.fallback_to_legacy is False
        assert config.cache_computations is False
        assert config.performance_monitoring is True
    
    def test_cmb_config_validation_valid_values(self):
        """Test CMBConfig validation with valid values."""
        config = CMBConfig(
            z_recombination=1089.8,
            jacobian_step_size=1e-6,
            validation_tolerance=1e-8
        )
        
        # Should not raise exception
        assert config.validate() is True
    
    def test_cmb_config_validation_invalid_z_recombination(self):
        """Test CMBConfig validation with invalid z_recombination."""
        with pytest.raises(ValueError) as exc_info:
            CMBConfig(z_recombination=500.0)  # Too low
        
        assert "z_recombination" in str(exc_info.value)
        assert "outside reasonable range" in str(exc_info.value)
    
    def test_cmb_config_validation_invalid_jacobian_step_size(self):
        """Test CMBConfig validation with invalid jacobian_step_size."""
        with pytest.raises(ValueError) as exc_info:
            CMBConfig(jacobian_step_size=0.1)  # Too large
        
        assert "jacobian_step_size" in str(exc_info.value)
    
    def test_cmb_config_validation_invalid_tolerance(self):
        """Test CMBConfig validation with invalid validation_tolerance."""
        with pytest.raises(ValueError) as exc_info:
            CMBConfig(validation_tolerance=1e-2)  # Too large
        
        assert "validation_tolerance" in str(exc_info.value)
    
    def test_cmb_config_validation_non_boolean_flags(self):
        """Test CMBConfig validation with non-boolean flags."""
        with pytest.raises(ValueError) as exc_info:
            CMBConfig(use_raw_parameters="true")  # String instead of bool
        
        assert "must be boolean" in str(exc_info.value)
    
    def test_cmb_config_to_dict(self):
        """Test converting CMBConfig to dictionary."""
        config = CMBConfig(
            use_raw_parameters=False,
            z_recombination=1090.0,
            performance_monitoring=True
        )
        
        result = config.to_dict()
        
        assert result['use_raw_parameters'] is False
        assert result['z_recombination'] == 1090.0
        assert result['performance_monitoring'] is True
        assert 'jacobian_step_size' in result
        assert 'validation_tolerance' in result
    
    def test_cmb_config_from_dict(self):
        """Test creating CMBConfig from dictionary."""
        data = {
            'use_raw_parameters': False,
            'z_recombination': 1090.0,
            'jacobian_step_size': 1e-5,
            'performance_monitoring': True
        }
        
        config = CMBConfig.from_dict(data)
        
        assert config.use_raw_parameters is False
        assert config.z_recombination == 1090.0
        assert config.jacobian_step_size == 1e-5
        assert config.performance_monitoring is True
        # Should use defaults for missing values
        assert config.validation_tolerance == 1e-8
    
    def test_cmb_config_get_default_config(self):
        """Test getting default configuration."""
        config = CMBConfig.get_default_config()
        
        assert config.use_raw_parameters is True
        assert config.z_recombination == 1089.8
        assert config.fallback_to_legacy is True
        assert config.performance_monitoring is False
    
    def test_cmb_config_get_development_config(self):
        """Test getting development configuration."""
        config = CMBConfig.get_development_config()
        
        assert config.use_raw_parameters is True
        assert config.cache_computations is False  # Disabled for testing
        assert config.performance_monitoring is True  # Enabled for development
        assert config.jacobian_step_size == 1e-5  # Larger for faster computation
    
    def test_cmb_config_copy(self):
        """Test copying CMBConfig."""
        original = CMBConfig(
            use_raw_parameters=False,
            z_recombination=1090.0
        )
        
        copy = original.copy()
        
        assert copy.use_raw_parameters == original.use_raw_parameters
        assert copy.z_recombination == original.z_recombination
        assert copy is not original  # Different objects


if __name__ == '__main__':
    pytest.main([__file__])