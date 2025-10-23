"""
Unit tests for the StandardDataset schema.

Tests schema validation, numerical integrity checks, and covariance
matrix validation functionality.
"""

import pytest
import numpy as np
from pipelines.data_preparation.core.schema import StandardDataset


class TestStandardDataset:
    """Test cases for StandardDataset class."""
    
    def test_valid_dataset_creation(self):
        """Test creation of valid StandardDataset."""
        z = np.array([0.1, 0.2, 0.3])
        observable = np.array([1.0, 2.0, 3.0])
        uncertainty = np.array([0.1, 0.2, 0.3])
        metadata = {'source': 'test', 'version': '1.0'}
        
        dataset = StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=None,
            metadata=metadata
        )
        
        assert np.array_equal(dataset.z, z)
        assert np.array_equal(dataset.observable, observable)
        assert np.array_equal(dataset.uncertainty, uncertainty)
        assert dataset.covariance is None
        assert dataset.metadata == metadata
    
    def test_dataset_with_covariance(self):
        """Test dataset creation with covariance matrix."""
        z = np.array([0.1, 0.2])
        observable = np.array([1.0, 2.0])
        uncertainty = np.array([0.1, 0.2])
        covariance = np.array([[0.01, 0.005], [0.005, 0.04]])
        metadata = {'source': 'test'}
        
        dataset = StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=covariance,
            metadata=metadata
        )
        
        assert np.array_equal(dataset.covariance, covariance)
    
    def test_array_shape_validation(self):
        """Test validation of array shapes."""
        z = np.array([0.1, 0.2, 0.3])
        observable = np.array([1.0, 2.0])  # Wrong length
        uncertainty = np.array([0.1, 0.2, 0.3])
        metadata = {}
        
        with pytest.raises(ValueError, match="Observable array length"):
            StandardDataset(
                z=z,
                observable=observable,
                uncertainty=uncertainty,
                covariance=None,
                metadata=metadata
            )
    
    def test_covariance_shape_validation(self):
        """Test validation of covariance matrix shape."""
        z = np.array([0.1, 0.2])
        observable = np.array([1.0, 2.0])
        uncertainty = np.array([0.1, 0.2])
        covariance = np.array([[0.01, 0.005, 0.001], [0.005, 0.04, 0.002]])  # Wrong shape
        metadata = {}
        
        with pytest.raises(ValueError, match="Covariance matrix shape"):
            StandardDataset(
                z=z,
                observable=observable,
                uncertainty=uncertainty,
                covariance=covariance,
                metadata=metadata
            )
    
    def test_schema_validation_success(self):
        """Test successful schema validation."""
        dataset = self._create_valid_dataset()
        assert dataset.validate_schema() is True
    
    def test_schema_validation_non_array(self):
        """Test schema validation with non-array data."""
        # Schema validation now happens during construction
        with pytest.raises(ValueError, match="must be numpy array"):
            dataset = StandardDataset(
                z=[0.1, 0.2],  # List instead of numpy array
                observable=np.array([1.0, 2.0]),
                uncertainty=np.array([0.1, 0.2]),
                covariance=None,
                metadata={}
            )
    
    def test_numerical_validation_success(self):
        """Test successful numerical validation."""
        dataset = self._create_valid_dataset()
        assert dataset.validate_numerical() is True
    
    def test_numerical_validation_nan(self):
        """Test numerical validation with NaN values."""
        dataset = StandardDataset(
            z=np.array([0.1, np.nan, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="NaN values found"):
            dataset.validate_numerical()
    
    def test_numerical_validation_infinity(self):
        """Test numerical validation with infinite values."""
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, np.inf, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="Infinite values found"):
            dataset.validate_numerical()
    
    def test_numerical_validation_negative_uncertainty(self):
        """Test numerical validation with negative uncertainties."""
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, -0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="Negative uncertainty values"):
            dataset.validate_numerical()
    
    def test_covariance_validation_success(self):
        """Test successful covariance validation."""
        dataset = self._create_valid_dataset_with_covariance()
        assert dataset.validate_covariance() is True
    
    def test_covariance_validation_asymmetric(self):
        """Test covariance validation with asymmetric matrix."""
        z = np.array([0.1, 0.2])
        observable = np.array([1.0, 2.0])
        uncertainty = np.array([0.1, 0.2])
        covariance = np.array([[0.01, 0.005], [0.006, 0.04]])  # Asymmetric
        
        dataset = StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=covariance,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="not symmetric"):
            dataset.validate_covariance()
    
    def test_covariance_validation_not_positive_definite(self):
        """Test covariance validation with non-positive-definite matrix."""
        z = np.array([0.1, 0.2])
        observable = np.array([1.0, 2.0])
        uncertainty = np.array([0.1, 0.2])
        covariance = np.array([[0.01, 0.02], [0.02, 0.01]])  # Not positive definite
        
        dataset = StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=covariance,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="not positive-definite"):
            dataset.validate_covariance()
    
    def test_redshift_range_validation_success(self):
        """Test successful redshift range validation."""
        dataset = self._create_valid_dataset()
        assert dataset.validate_redshift_range(0.0, 1.0) is True
    
    def test_redshift_range_validation_below_minimum(self):
        """Test redshift range validation with values below minimum."""
        dataset = StandardDataset(
            z=np.array([-0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="below minimum"):
            dataset.validate_redshift_range(0.0, 1.0)
    
    def test_redshift_range_validation_above_maximum(self):
        """Test redshift range validation with values above maximum."""
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 1.5]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ValueError, match="above maximum"):
            dataset.validate_redshift_range(0.0, 1.0)
    
    def test_validate_all_success(self):
        """Test successful validation of all checks."""
        dataset = self._create_valid_dataset_with_covariance()
        assert dataset.validate_all(0.0, 1.0) is True
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        dataset = self._create_valid_dataset()
        data_dict = dataset.to_dict()
        
        assert 'z' in data_dict
        assert 'observable' in data_dict
        assert 'uncertainty' in data_dict
        assert 'covariance' in data_dict
        assert 'metadata' in data_dict
        
        assert np.array_equal(data_dict['z'], dataset.z)
        assert np.array_equal(data_dict['observable'], dataset.observable)
        assert np.array_equal(data_dict['uncertainty'], dataset.uncertainty)
        assert data_dict['covariance'] is dataset.covariance
        assert data_dict['metadata'] == dataset.metadata
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        original_dataset = self._create_valid_dataset()
        data_dict = original_dataset.to_dict()
        
        reconstructed_dataset = StandardDataset.from_dict(data_dict)
        
        assert np.array_equal(reconstructed_dataset.z, original_dataset.z)
        assert np.array_equal(reconstructed_dataset.observable, original_dataset.observable)
        assert np.array_equal(reconstructed_dataset.uncertainty, original_dataset.uncertainty)
        assert reconstructed_dataset.covariance is original_dataset.covariance
        assert reconstructed_dataset.metadata == original_dataset.metadata
    
    def _create_valid_dataset(self):
        """Helper method to create a valid dataset for testing."""
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test', 'version': '1.0'}
        )
    
    def _create_valid_dataset_with_covariance(self):
        """Helper method to create a valid dataset with covariance matrix."""
        covariance = np.array([
            [0.01, 0.005, 0.002],
            [0.005, 0.04, 0.01],
            [0.002, 0.01, 0.09]
        ])
        
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=covariance,
            metadata={'source': 'test', 'version': '1.0'}
        )


class TestStandardDatasetEdgeCases:
    """Test edge cases and additional scenarios for StandardDataset."""
    
    def test_dataset_with_minimal_metadata(self):
        """Test dataset creation with minimal metadata."""
        dataset = StandardDataset(
            z=np.array([0.1, 0.2]),
            observable=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.2]),
            covariance=None,
            metadata={'source': 'test'}  # Minimal required metadata
        )
        
        assert dataset.validate_schema() is True
        assert dataset.validate_numerical() is True
    
    def test_dataset_with_extensive_metadata(self):
        """Test dataset creation with extensive metadata."""
        extensive_metadata = {
            'source': 'test_survey',
            'version': '2.1.0',
            'citation': 'Test et al. 2024',
            'processing_timestamp': '2024-01-01T00:00:00Z',
            'n_points': 2,
            'redshift_range': [0.1, 0.2],
            'observable_type': 'distance_modulus',
            'units': 'mag',
            'systematic_corrections': ['bias_correction', 'calibration'],
            'quality_flags': [0, 0],
            'survey_area': 1000.0,
            'completeness': 0.95,
            'selection_function': 'magnitude_limited',
            'photometric_system': 'AB',
            'filter_set': ['g', 'r', 'i', 'z'],
            'data_release': 'DR3',
            'processing_pipeline': 'v2.1',
            'validation_status': 'passed',
            'checksum': 'abc123def456'
        }
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2]),
            observable=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.2]),
            covariance=None,
            metadata=extensive_metadata
        )
        
        assert dataset.validate_schema() is True
        assert len(dataset.metadata) == len(extensive_metadata)
        assert dataset.metadata['source'] == 'test_survey'
    
    def test_dataset_with_single_point(self):
        """Test dataset with single data point."""
        dataset = StandardDataset(
            z=np.array([0.5]),
            observable=np.array([2.5]),
            uncertainty=np.array([0.25]),
            covariance=None,
            metadata={'source': 'single_point_test'}
        )
        
        assert dataset.validate_schema() is True
        assert dataset.validate_numerical() is True
        assert len(dataset.z) == 1
    
    def test_dataset_with_1x1_covariance(self):
        """Test dataset with 1x1 covariance matrix."""
        dataset = StandardDataset(
            z=np.array([0.5]),
            observable=np.array([2.5]),
            uncertainty=np.array([0.25]),
            covariance=np.array([[0.0625]]),  # 1x1 covariance
            metadata={'source': 'single_cov_test'}
        )
        
        assert dataset.validate_schema() is True
        assert dataset.validate_numerical() is True
        assert dataset.validate_covariance() is True
    
    def test_dataset_serialization_compatibility(self):
        """Test dataset compatibility with serialization."""
        import pickle
        import json
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=np.eye(3) * 0.01,
            metadata={'source': 'serialization_test', 'version': '1.0'}
        )
        
        # Test pickle serialization
        pickled_data = pickle.dumps(dataset)
        unpickled_dataset = pickle.loads(pickled_data)
        
        np.testing.assert_array_equal(dataset.z, unpickled_dataset.z)
        np.testing.assert_array_equal(dataset.observable, unpickled_dataset.observable)
        assert dataset.metadata == unpickled_dataset.metadata
        
        # Test JSON serialization of metadata
        json_metadata = json.dumps(dataset.metadata)
        restored_metadata = json.loads(json_metadata)
        assert restored_metadata == dataset.metadata
    
    def test_dataset_memory_efficiency(self):
        """Test dataset memory efficiency with large arrays."""
        # Create large dataset to test memory usage
        n_points = 100000
        
        dataset = StandardDataset(
            z=np.linspace(0.1, 5.0, n_points),
            observable=np.random.normal(1.0, 0.1, n_points),
            uncertainty=np.random.uniform(0.05, 0.2, n_points),
            covariance=None,  # Don't create large covariance matrix
            metadata={'source': 'memory_test', 'n_points': n_points}
        )
        
        # Verify dataset is created successfully
        assert len(dataset.z) == n_points
        assert dataset.validate_schema() is True
        
        # Check memory usage is reasonable (arrays should share memory efficiently)
        import sys
        dataset_size = sys.getsizeof(dataset)
        assert dataset_size < 10 * 1024 * 1024  # Less than 10MB for the object itself
    
    def test_dataset_copy_and_modification(self):
        """Test dataset copying and modification behavior."""
        original = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'original', 'version': '1.0'}
        )
        
        # Test shallow copy behavior
        import copy
        shallow_copy = copy.copy(original)
        
        # Modify original metadata
        original.metadata['version'] = '2.0'
        
        # Shallow copy should share metadata reference
        assert shallow_copy.metadata['version'] == '2.0'
        
        # Test deep copy behavior
        deep_copy = copy.deepcopy(original)
        original.metadata['version'] = '3.0'
        
        # Deep copy should have independent metadata
        assert deep_copy.metadata['version'] == '2.0'
        assert original.metadata['version'] == '3.0'
    
    def test_dataset_array_views_and_copies(self):
        """Test dataset behavior with array views and copies."""
        base_z = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        base_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base_err = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Create dataset with array slices (views)
        dataset = StandardDataset(
            z=base_z[1:4],  # View of original array
            observable=base_obs[1:4],
            uncertainty=base_err[1:4],
            covariance=None,
            metadata={'source': 'view_test'}
        )
        
        # Modify original array
        base_z[2] = 0.25
        
        # Dataset should reflect the change (since it's a view)
        assert dataset.z[1] == 0.25
        
        # Verify dataset still validates
        assert dataset.validate_schema() is True
        assert dataset.validate_numerical() is True
    
    def test_dataset_with_different_dtypes(self):
        """Test dataset with different numpy dtypes."""
        # Test with different float precisions
        dataset_float32 = StandardDataset(
            z=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            observable=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            uncertainty=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            covariance=None,
            metadata={'source': 'float32_test'}
        )
        
        dataset_float64 = StandardDataset(
            z=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            observable=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            uncertainty=np.array([0.1, 0.2, 0.3], dtype=np.float64),
            covariance=None,
            metadata={'source': 'float64_test'}
        )
        
        # Both should validate successfully
        assert dataset_float32.validate_schema() is True
        assert dataset_float64.validate_schema() is True
        
        # Values should be approximately equal
        np.testing.assert_allclose(dataset_float32.z, dataset_float64.z, rtol=1e-6)
    
    def test_dataset_validation_error_messages(self):
        """Test detailed error messages from validation methods."""
        # Test schema validation error messages
        with pytest.raises(ValueError) as exc_info:
            StandardDataset(
                z="not_an_array",  # Invalid type
                observable=np.array([1.0, 2.0]),
                uncertainty=np.array([0.1, 0.2]),
                covariance=None,
                metadata={}
            )
        
        assert "numpy array" in str(exc_info.value).lower()
        
        # Test numerical validation error messages
        dataset = StandardDataset(
            z=np.array([0.1, np.nan, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ValueError) as exc_info:
            dataset.validate_numerical()
        
        assert "NaN" in str(exc_info.value)
    
    def test_dataset_boundary_conditions(self):
        """Test dataset with boundary condition values."""
        # Test with very small positive values
        dataset_small = StandardDataset(
            z=np.array([1e-10, 1e-9, 1e-8]),
            observable=np.array([1e-15, 1e-14, 1e-13]),
            uncertainty=np.array([1e-16, 1e-15, 1e-14]),
            covariance=None,
            metadata={'source': 'small_values_test'}
        )
        
        assert dataset_small.validate_schema() is True
        assert dataset_small.validate_numerical() is True
        
        # Test with very large values
        dataset_large = StandardDataset(
            z=np.array([1e10, 1e11, 1e12]),
            observable=np.array([1e15, 1e16, 1e17]),
            uncertainty=np.array([1e14, 1e15, 1e16]),
            covariance=None,
            metadata={'source': 'large_values_test'}
        )
        
        assert dataset_large.validate_schema() is True
        assert dataset_large.validate_numerical() is True


class TestStandardDatasetIntegration:
    """Test StandardDataset integration with other components."""
    
    def test_dataset_with_validation_engine(self):
        """Test dataset integration with ValidationEngine."""
        from pipelines.data_preparation.core.validation import ValidationEngine
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=np.eye(3) * 0.01,
            metadata={'source': 'integration_test', 'version': '1.0'}
        )
        
        engine = ValidationEngine()
        results = engine.validate_dataset(dataset, "integration_test")
        
        assert results['validation_passed'] is True
        assert 'summary' in results
        assert results['summary']['n_points'] == 3
    
    def test_dataset_conversion_to_dict(self):
        """Test dataset conversion to dictionary format."""
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=np.eye(3) * 0.01,
            metadata={'source': 'dict_test', 'version': '1.0'}
        )
        
        # Convert to dictionary (for compatibility with existing code)
        dataset_dict = {
            'redshifts': dataset.z,
            'observations': dataset.observable,
            'uncertainties': dataset.uncertainty,
            'covariance': dataset.covariance,
            'metadata': dataset.metadata
        }
        
        # Verify conversion
        np.testing.assert_array_equal(dataset_dict['redshifts'], dataset.z)
        np.testing.assert_array_equal(dataset_dict['observations'], dataset.observable)
        assert dataset_dict['metadata'] == dataset.metadata
    
    def test_dataset_from_dict_creation(self):
        """Test creating StandardDataset from dictionary."""
        input_dict = {
            'z': [0.1, 0.2, 0.3],
            'observable': [1.0, 2.0, 3.0],
            'uncertainty': [0.1, 0.2, 0.3],
            'metadata': {'source': 'from_dict_test'}
        }
        
        dataset = StandardDataset(
            z=np.array(input_dict['z']),
            observable=np.array(input_dict['observable']),
            uncertainty=np.array(input_dict['uncertainty']),
            covariance=None,
            metadata=input_dict['metadata']
        )
        
        assert dataset.validate_schema() is True
        np.testing.assert_array_equal(dataset.z, [0.1, 0.2, 0.3])
        assert dataset.metadata['source'] == 'from_dict_test'