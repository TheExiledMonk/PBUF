"""
Integration tests for CMB raw parameter processing pipeline.

Tests the complete workflow from registry entry to StandardDataset output,
backward compatibility, and error handling mechanisms.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from pipelines.data_preparation.derivation.cmb_derivation import (
    process_cmb_dataset, derive_cmb_data
)
from pipelines.data_preparation.derivation.cmb_models import (
    ParameterSet, DistancePriors, CMBConfig
)
from pipelines.data_preparation.derivation.cmb_exceptions import (
    ParameterDetectionError, ParameterValidationError, DerivationError
)
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.interfaces import ProcessingError


class TestCompleteWorkflowIntegration:
    """Test cases for complete CMB processing workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock registry entry with raw parameters
        self.raw_param_registry = {
            'metadata': {
                'dataset_type': 'cmb',
                'name': 'planck2018_raw_params',
                'source': 'Planck Collaboration',
                'citation': 'Planck Collaboration 2020'
            },
            'sources': {
                'primary': {
                    'url': 'https://example.com/planck_params.json',
                    'extraction': {
                        'target_files': ['planck_cosmological_params.json']
                    }
                }
            }
        }
        
        # Create mock registry entry with distance priors (legacy)
        self.distance_prior_registry = {
            'metadata': {
                'dataset_type': 'cmb',
                'name': 'planck2018_distance_priors',
                'source': 'Planck Collaboration',
                'data_type': 'distance_priors'
            },
            'sources': {
                'primary': {
                    'url': 'https://example.com/planck_priors.csv',
                    'extraction': {
                        'target_files': ['planck_distance_priors.csv']
                    }
                }
            }
        }
        
        # Planck 2018 parameters for testing
        self.planck_params = {
            'H0': 67.36,
            'Omega_m': 0.3153,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544,
            'A_s': 2.1e-9
        }
        
        # Configuration for testing
        self.config = CMBConfig.get_development_config()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)   
 
    def test_full_workflow_raw_parameters_to_standard_dataset(self):
        """Test complete workflow from raw parameters to StandardDataset."""
        # Create parameter file
        param_file = self.temp_dir / 'planck_cosmological_params.json'
        with open(param_file, 'w') as f:
            json.dump(self.planck_params, f)
        
        # Mock file system access
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.parse_parameter_file') as mock_parse, \
             patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            # Set up mocks
            from pipelines.data_preparation.derivation.cmb_derivation import RawParameterInfo, ParameterFormat
            mock_detect.return_value = RawParameterInfo(
                file_path=str(param_file),
                format_type=ParameterFormat.JSON
            )
            
            mock_parse.return_value = ParameterSet.from_dict(self.planck_params)
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_bg.angular_diameter_distance.return_value = 12.86
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Execute workflow
            result = process_cmb_dataset(self.raw_param_registry, self.config)
            
            # Verify result
            assert isinstance(result, StandardDataset)
            assert len(result.z) == 3  # CMB has 3 observables
            assert len(result.observable) == 3
            assert len(result.uncertainty) == 3
            
            # Check metadata
            assert result.metadata['dataset_type'] == 'cmb'
            assert result.metadata['source'] == 'Planck Collaboration'
            assert 'processing_timestamp' in result.metadata
            assert 'parameters_used' in result.metadata
    
    def test_workflow_with_covariance_matrix(self):
        """Test workflow including covariance matrix processing."""
        # Create parameter file
        param_file = self.temp_dir / 'planck_params.json'
        with open(param_file, 'w') as f:
            json.dump(self.planck_params, f)
        
        # Create covariance matrix file
        cov_matrix = np.eye(5) * 0.01  # Simple diagonal covariance
        cov_file = self.temp_dir / 'planck_covariance.npy'
        np.save(cov_file, cov_matrix)
        
        # Mock file system and background integrator
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.parse_parameter_file') as mock_parse, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.extract_covariance_matrix') as mock_extract_cov, \
             patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            # Set up mocks
            from pipelines.data_preparation.derivation.cmb_derivation import RawParameterInfo, ParameterFormat
            mock_detect.return_value = RawParameterInfo(
                file_path=str(param_file),
                format_type=ParameterFormat.JSON,
                covariance_file=str(cov_file)
            )
            
            mock_parse.return_value = ParameterSet.from_dict(self.planck_params)
            mock_extract_cov.return_value = cov_matrix
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Execute workflow
            result = process_cmb_dataset(self.raw_param_registry, self.config)
            
            # Verify covariance matrix is included
            assert result.covariance is not None
            assert result.covariance.shape == (3, 3)  # 3x3 for CMB observables
            
            # Check that uncertainties match covariance diagonal
            expected_uncertainties = np.sqrt(np.diag(result.covariance))
            np.testing.assert_allclose(result.uncertainty, expected_uncertainties, rtol=1e-10)
    
    def test_backward_compatibility_distance_priors(self):
        """Test backward compatibility with legacy distance-prior datasets."""
        # This test ensures that existing distance-prior datasets still work
        
        # Mock the legacy CMB derivation module
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.CMBDerivationModule') as mock_legacy_module:
            
            # No raw parameters detected (legacy mode)
            mock_detect.return_value = None
            
            # Mock legacy module behavior
            mock_legacy = Mock()
            mock_legacy.derive.return_value = StandardDataset(
                z=np.array([1089.8, 1089.8, 1089.8]),
                observable=np.array([1.7502, 301.76, 1.04119]),
                uncertainty=np.array([0.0023, 0.091, 0.00031]),
                covariance=None,
                metadata={
                    'dataset_type': 'cmb',
                    'source': 'Planck Collaboration',
                    'processing': 'legacy_distance_priors'
                }
            )
            mock_legacy_module.return_value = mock_legacy
            
            # Execute workflow (should fall back to legacy)
            result = process_cmb_dataset(self.distance_prior_registry, self.config)
            
            # Verify legacy processing was used
            assert isinstance(result, StandardDataset)
            assert result.metadata.get('processing') == 'legacy_distance_priors'
            
            # Verify legacy module was called
            mock_legacy_module.assert_called_once()
            mock_legacy.derive.assert_called_once()
    
    def test_automatic_fallback_mechanism(self):
        """Test automatic fallback from raw parameters to legacy mode."""
        # Test scenario where raw parameters are detected but processing fails
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.parse_parameter_file') as mock_parse, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.CMBDerivationModule') as mock_legacy_module:
            
            # Raw parameters detected but parsing fails
            from pipelines.data_preparation.derivation.cmb_derivation import RawParameterInfo, ParameterFormat
            mock_detect.return_value = RawParameterInfo(
                file_path='invalid_file.json',
                format_type=ParameterFormat.JSON
            )
            
            # Parsing fails
            mock_parse.side_effect = ParameterDetectionError(
                message="Failed to parse parameter file",
                file_path='invalid_file.json'
            )
            
            # Mock successful legacy processing
            mock_legacy = Mock()
            mock_legacy.derive.return_value = StandardDataset(
                z=np.array([1089.8, 1089.8, 1089.8]),
                observable=np.array([1.7502, 301.76, 1.04119]),
                uncertainty=np.array([0.0023, 0.091, 0.00031]),
                covariance=None,
                metadata={
                    'dataset_type': 'cmb',
                    'source': 'Planck Collaboration',
                    'processing': 'fallback_to_legacy'
                }
            )
            mock_legacy_module.return_value = mock_legacy
            
            # Execute workflow (should fall back to legacy)
            result = process_cmb_dataset(self.raw_param_registry, self.config)
            
            # Verify fallback occurred
            assert isinstance(result, StandardDataset)
            mock_legacy_module.assert_called_once()
            mock_legacy.derive.assert_called_once()
    
    def test_error_handling_invalid_parameters(self):
        """Test error handling with invalid cosmological parameters."""
        # Create invalid parameter file
        invalid_params = {
            'H0': -67.36,  # Invalid negative H0
            'Omega_m': 0.3153,
            'Omega_b_h2': 0.02237,
            'n_s': 0.9649,
            'tau': 0.0544
        }
        
        param_file = self.temp_dir / 'invalid_params.json'
        with open(param_file, 'w') as f:
            json.dump(invalid_params, f)
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.parse_parameter_file') as mock_parse:
            
            # Set up mocks
            from pipelines.data_preparation.derivation.cmb_derivation import RawParameterInfo, ParameterFormat
            mock_detect.return_value = RawParameterInfo(
                file_path=str(param_file),
                format_type=ParameterFormat.JSON
            )
            
            # This should raise an exception during parameter validation
            mock_parse.side_effect = ValueError("H0 = -67.36 outside physical bounds [50.0, 80.0]")
            
            # Should raise appropriate error
            with pytest.raises((ParameterValidationError, ProcessingError, ValueError)):
                process_cmb_dataset(self.raw_param_registry, self.config)
    
    def test_error_handling_integration_failure(self):
        """Test error handling when background integration fails."""
        param_file = self.temp_dir / 'planck_params.json'
        with open(param_file, 'w') as f:
            json.dump(self.planck_params, f)
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.parse_parameter_file') as mock_parse, \
             patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class:
            
            # Set up mocks
            from pipelines.data_preparation.derivation.cmb_derivation import RawParameterInfo, ParameterFormat
            mock_detect.return_value = RawParameterInfo(
                file_path=str(param_file),
                format_type=ParameterFormat.JSON
            )
            
            mock_parse.return_value = ParameterSet.from_dict(self.planck_params)
            
            # Background integrator fails
            mock_integrator_class.side_effect = RuntimeError("Integration failed")
            
            # Should raise appropriate error
            with pytest.raises((DerivationError, ProcessingError, RuntimeError)):
                process_cmb_dataset(self.raw_param_registry, self.config)
    
    def test_configuration_driven_processing(self):
        """Test that configuration properly controls processing behavior."""
        param_file = self.temp_dir / 'planck_params.json'
        with open(param_file, 'w') as f:
            json.dump(self.planck_params, f)
        
        # Test with raw parameters disabled
        config_no_raw = CMBConfig(use_raw_parameters=False)
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.CMBDerivationModule') as mock_legacy_module:
            
            mock_detect.return_value = None  # Should not be called when disabled
            
            # Mock legacy processing
            mock_legacy = Mock()
            mock_legacy.derive.return_value = StandardDataset(
                z=np.array([1089.8, 1089.8, 1089.8]),
                observable=np.array([1.7502, 301.76, 1.04119]),
                uncertainty=np.array([0.0023, 0.091, 0.00031]),
                covariance=None,
                metadata={'dataset_type': 'cmb', 'processing': 'legacy_by_config'}
            )
            mock_legacy_module.return_value = mock_legacy
            
            # Execute with raw parameters disabled
            result = process_cmb_dataset(self.raw_param_registry, config_no_raw)
            
            # Should use legacy processing
            assert isinstance(result, StandardDataset)
            mock_legacy_module.assert_called_once()
    
    def test_derive_cmb_data_entry_point(self):
        """Test the main derive_cmb_data entry point function."""
        param_file = self.temp_dir / 'planck_params.json'
        with open(param_file, 'w') as f:
            json.dump(self.planck_params, f)
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.detect_raw_parameters') as mock_detect, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.parse_parameter_file') as mock_parse, \
             patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            # Set up mocks
            from pipelines.data_preparation.derivation.cmb_derivation import RawParameterInfo, ParameterFormat
            mock_detect.return_value = RawParameterInfo(
                file_path=str(param_file),
                format_type=ParameterFormat.JSON
            )
            
            mock_parse.return_value = ParameterSet.from_dict(self.planck_params)
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Test with explicit configuration
            result = derive_cmb_data(
                self.raw_param_registry,
                use_raw_parameters=True,
                config=self.config
            )
            
            # Verify result
            assert isinstance(result, StandardDataset)
            assert result.metadata['dataset_type'] == 'cmb'
            
            # Test with raw parameters disabled
            with patch('pipelines.data_preparation.derivation.cmb_derivation.CMBDerivationModule') as mock_legacy_module:
                mock_legacy = Mock()
                mock_legacy.derive.return_value = StandardDataset(
                    z=np.array([1089.8]),
                    observable=np.array([1.7502]),
                    uncertainty=np.array([0.0023]),
                    covariance=None,
                    metadata={'dataset_type': 'cmb'}
                )
                mock_legacy_module.return_value = mock_legacy
                
                result_legacy = derive_cmb_data(
                    self.raw_param_registry,
                    use_raw_parameters=False,
                    config=self.config
                )
                
                assert isinstance(result_legacy, StandardDataset)
                mock_legacy_module.assert_called_once()


class TestFittingPipelineCompatibility:
    """Test compatibility with existing fitting pipelines."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Sample StandardDataset that should be compatible
        self.sample_dataset = StandardDataset(
            z=np.array([1089.8, 1089.8, 1089.8]),
            observable=np.array([1.7502, 301.76, 1.04119]),
            uncertainty=np.array([0.0023, 0.091, 0.00031]),
            covariance=np.diag([0.0023**2, 0.091**2, 0.00031**2]),
            metadata={
                'dataset_type': 'cmb',
                'source': 'Planck2018',
                'processing': 'derived_from_raw_parameters',
                'parameters_used': ['H0', 'Omega_m', 'Omega_b_h2', 'n_s', 'tau'],
                'z_recombination': 1089.8
            }
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_standard_dataset_structure_compatibility(self):
        """Test that StandardDataset structure matches fitting pipeline expectations."""
        # Verify required attributes exist
        assert hasattr(self.sample_dataset, 'z')
        assert hasattr(self.sample_dataset, 'observable')
        assert hasattr(self.sample_dataset, 'uncertainty')
        assert hasattr(self.sample_dataset, 'covariance')
        assert hasattr(self.sample_dataset, 'metadata')
        
        # Verify array shapes are consistent
        assert len(self.sample_dataset.z) == len(self.sample_dataset.observable)
        assert len(self.sample_dataset.observable) == len(self.sample_dataset.uncertainty)
        
        # Verify covariance matrix shape
        n_obs = len(self.sample_dataset.observable)
        assert self.sample_dataset.covariance.shape == (n_obs, n_obs)
        
        # Verify metadata structure
        assert isinstance(self.sample_dataset.metadata, dict)
        assert 'dataset_type' in self.sample_dataset.metadata
        assert self.sample_dataset.metadata['dataset_type'] == 'cmb'
    
    def test_observable_array_ordering(self):
        """Test that observable array follows expected ordering."""
        # CMB observables should be in order: [R, ℓ_A, θ*]
        # (Note: Ω_b h² is excluded from fitting observables)
        
        R, l_A, theta_star = self.sample_dataset.observable
        
        # Verify reasonable ranges for each observable
        assert 1.0 < R < 3.0  # Shift parameter
        assert 200 < l_A < 400  # Acoustic scale
        assert 0.5 < theta_star < 2.0  # Angular scale
        
        # Verify uncertainties are positive
        assert np.all(self.sample_dataset.uncertainty > 0)
    
    def test_covariance_matrix_properties(self):
        """Test covariance matrix properties expected by fitting code."""
        cov = self.sample_dataset.covariance
        
        # Should be symmetric
        np.testing.assert_allclose(cov, cov.T, rtol=1e-10)
        
        # Should be positive definite
        eigenvals = np.linalg.eigvals(cov)
        assert np.all(eigenvals >= 0)
        
        # Diagonal should match uncertainties squared
        expected_diag = self.sample_dataset.uncertainty ** 2
        np.testing.assert_allclose(np.diag(cov), expected_diag, rtol=1e-10)
    
    def test_metadata_fields_for_fitting(self):
        """Test that metadata contains fields used by fitting pipelines."""
        metadata = self.sample_dataset.metadata
        
        # Essential fields for fitting
        required_fields = ['dataset_type', 'source']
        for field in required_fields:
            assert field in metadata
        
        # CMB-specific fields
        cmb_fields = ['z_recombination']
        for field in cmb_fields:
            assert field in metadata
        
        # Processing provenance fields
        provenance_fields = ['processing']
        for field in provenance_fields:
            assert field in metadata
    
    def test_numerical_precision_compatibility(self):
        """Test numerical precision is adequate for fitting algorithms."""
        # Check that values are finite
        assert np.all(np.isfinite(self.sample_dataset.z))
        assert np.all(np.isfinite(self.sample_dataset.observable))
        assert np.all(np.isfinite(self.sample_dataset.uncertainty))
        assert np.all(np.isfinite(self.sample_dataset.covariance))
        
        # Check that uncertainties are not too small (numerical precision)
        min_uncertainty = np.min(self.sample_dataset.uncertainty)
        assert min_uncertainty > 1e-10
        
        # Check that covariance matrix is well-conditioned
        condition_number = np.linalg.cond(self.sample_dataset.covariance)
        assert condition_number < 1e12  # Reasonable condition number
    
    def test_mock_fitting_pipeline_integration(self):
        """Test integration with a mock fitting pipeline."""
        # Mock a simple fitting pipeline that uses StandardDataset
        
        def mock_chi_squared(dataset: StandardDataset, model_predictions: np.ndarray) -> float:
            """Mock chi-squared calculation."""
            residuals = dataset.observable - model_predictions
            if dataset.covariance is not None:
                # Use full covariance matrix
                inv_cov = np.linalg.inv(dataset.covariance)
                chi2 = residuals.T @ inv_cov @ residuals
            else:
                # Use diagonal uncertainties
                chi2 = np.sum((residuals / dataset.uncertainty) ** 2)
            return float(chi2)
        
        # Test with mock model predictions
        mock_predictions = np.array([1.75, 302.0, 1.04])  # Close to observed values
        
        # Should not raise exception
        chi2 = mock_chi_squared(self.sample_dataset, mock_predictions)
        
        # Should return reasonable chi-squared value
        assert isinstance(chi2, float)
        assert chi2 >= 0
        assert np.isfinite(chi2)
    
    def test_serialization_compatibility(self):
        """Test that StandardDataset can be serialized for pipeline communication."""
        # Test JSON serialization of metadata
        import json
        
        # Metadata should be JSON serializable
        metadata_json = json.dumps(self.sample_dataset.metadata, default=str)
        reconstructed_metadata = json.loads(metadata_json)
        
        assert isinstance(reconstructed_metadata, dict)
        assert reconstructed_metadata['dataset_type'] == 'cmb'
        
        # Test NumPy array serialization
        # Arrays should be convertible to lists for JSON
        z_list = self.sample_dataset.z.tolist()
        obs_list = self.sample_dataset.observable.tolist()
        
        assert isinstance(z_list, list)
        assert isinstance(obs_list, list)
        assert len(z_list) == len(obs_list)


if __name__ == '__main__':
    pytest.main([__file__])