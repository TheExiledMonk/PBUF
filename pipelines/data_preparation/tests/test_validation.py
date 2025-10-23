"""
Unit tests for the validation engine and validation rules.

Tests comprehensive validation including schema compliance,
numerical integrity, covariance matrix validation, and custom rules.
"""

import pytest
import numpy as np

from pipelines.data_preparation.core.validation import (
    ValidationEngine, SchemaValidationRule, NumericalIntegrityRule,
    CovarianceValidationRule, RedshiftRangeRule, MonotonicityRule
)
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.interfaces import ProcessingError


class TestSchemaValidationRule:
    """Test cases for SchemaValidationRule."""
    
    def test_rule_name(self):
        """Test rule name property."""
        rule = SchemaValidationRule()
        assert rule.rule_name == "Schema Compliance"
    
    def test_valid_schema(self):
        """Test validation with valid schema."""
        rule = SchemaValidationRule()
        dataset = self._create_valid_dataset()
        
        assert rule.validate(dataset) is True
    
    def test_invalid_schema(self):
        """Test validation with invalid schema."""
        rule = SchemaValidationRule()
        
        # Schema validation now happens during construction, so test with ProcessingError
        with pytest.raises(ValueError, match="must be numpy array"):
            dataset = StandardDataset(
                z=[0.1, 0.2],  # List instead of numpy array
                observable=np.array([1.0, 2.0]),
                uncertainty=np.array([0.1, 0.2]),
                covariance=None,
                metadata={}
            )
    
    def _create_valid_dataset(self):
        """Helper to create valid dataset."""
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )


class TestNumericalIntegrityRule:
    """Test cases for NumericalIntegrityRule."""
    
    def test_rule_name(self):
        """Test rule name property."""
        rule = NumericalIntegrityRule()
        assert rule.rule_name == "Numerical Integrity"
    
    def test_valid_numerical_data(self):
        """Test validation with valid numerical data."""
        rule = NumericalIntegrityRule()
        dataset = self._create_valid_dataset()
        
        assert rule.validate(dataset) is True
    
    def test_nan_values(self):
        """Test validation with NaN values."""
        rule = NumericalIntegrityRule()
        
        dataset = StandardDataset(
            z=np.array([0.1, np.nan, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "numerical_error"
        assert "NaN values" in exc_info.value.error_message
    
    def test_infinite_values(self):
        """Test validation with infinite values."""
        rule = NumericalIntegrityRule()
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, np.inf, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "numerical_error"
        assert "Infinite values" in exc_info.value.error_message
    
    def test_negative_uncertainties(self):
        """Test validation with negative uncertainties."""
        rule = NumericalIntegrityRule()
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, -0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "numerical_error"
        assert "Negative uncertainty" in exc_info.value.error_message
    
    def _create_valid_dataset(self):
        """Helper to create valid dataset."""
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )


class TestCovarianceValidationRule:
    """Test cases for CovarianceValidationRule."""
    
    def test_rule_name(self):
        """Test rule name property."""
        rule = CovarianceValidationRule()
        assert rule.rule_name == "Covariance Matrix Validation"
    
    def test_no_covariance(self):
        """Test validation with no covariance matrix."""
        rule = CovarianceValidationRule()
        dataset = self._create_dataset_without_covariance()
        
        assert rule.validate(dataset) is True
    
    def test_valid_covariance(self):
        """Test validation with valid covariance matrix."""
        rule = CovarianceValidationRule()
        dataset = self._create_dataset_with_valid_covariance()
        
        assert rule.validate(dataset) is True
    
    def test_asymmetric_covariance(self):
        """Test validation with asymmetric covariance matrix."""
        rule = CovarianceValidationRule()
        
        covariance = np.array([[0.01, 0.005], [0.006, 0.04]])  # Asymmetric
        dataset = StandardDataset(
            z=np.array([0.1, 0.2]),
            observable=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.2]),
            covariance=covariance,
            metadata={}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "covariance_error"
        assert "not symmetric" in exc_info.value.error_message
    
    def test_non_positive_definite_covariance(self):
        """Test validation with non-positive-definite covariance matrix."""
        rule = CovarianceValidationRule()
        
        covariance = np.array([[0.01, 0.02], [0.02, 0.01]])  # Not positive definite
        dataset = StandardDataset(
            z=np.array([0.1, 0.2]),
            observable=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.2]),
            covariance=covariance,
            metadata={}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "covariance_error"
        assert "not positive-definite" in exc_info.value.error_message
    
    def _create_dataset_without_covariance(self):
        """Helper to create dataset without covariance."""
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )
    
    def _create_dataset_with_valid_covariance(self):
        """Helper to create dataset with valid covariance."""
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
            metadata={'source': 'test'}
        )


class TestRedshiftRangeRule:
    """Test cases for RedshiftRangeRule."""
    
    def test_rule_name(self):
        """Test rule name property."""
        rule = RedshiftRangeRule(0.0, 5.0)
        assert rule.rule_name == "Redshift Range (0.0 ≤ z ≤ 5.0)"
    
    def test_valid_redshift_range(self):
        """Test validation with valid redshift range."""
        rule = RedshiftRangeRule(0.0, 1.0)
        dataset = self._create_dataset_with_redshifts([0.1, 0.5, 0.9])
        
        assert rule.validate(dataset) is True
    
    def test_redshift_below_minimum(self):
        """Test validation with redshift below minimum."""
        rule = RedshiftRangeRule(0.0, 1.0)
        dataset = self._create_dataset_with_redshifts([-0.1, 0.5, 0.9])
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "redshift_range_error"
        assert "below minimum" in exc_info.value.error_message
    
    def test_redshift_above_maximum(self):
        """Test validation with redshift above maximum."""
        rule = RedshiftRangeRule(0.0, 1.0)
        dataset = self._create_dataset_with_redshifts([0.1, 0.5, 1.5])
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "redshift_range_error"
        assert "above maximum" in exc_info.value.error_message
    
    def _create_dataset_with_redshifts(self, redshifts):
        """Helper to create dataset with specific redshifts."""
        z = np.array(redshifts)
        n = len(z)
        
        return StandardDataset(
            z=z,
            observable=np.ones(n),
            uncertainty=np.ones(n) * 0.1,
            covariance=None,
            metadata={'source': 'test'}
        )


class TestMonotonicityRule:
    """Test cases for MonotonicityRule."""
    
    def test_rule_name(self):
        """Test rule name property."""
        rule = MonotonicityRule(check_monotonic=True, observable_name="test_obs")
        assert rule.rule_name == "Monotonicity and Physical Sanity (test_obs)"
    
    def test_monotonic_disabled(self):
        """Test validation with monotonicity check disabled."""
        rule = MonotonicityRule(check_monotonic=False)
        dataset = self._create_dataset_with_redshifts([0.3, 0.1, 0.2])  # Non-monotonic
        
        assert rule.validate(dataset) is True
    
    def test_monotonic_redshifts(self):
        """Test validation with monotonic redshifts."""
        rule = MonotonicityRule(check_monotonic=True)
        dataset = self._create_dataset_with_redshifts([0.1, 0.2, 0.3])
        
        assert rule.validate(dataset) is True
    
    def test_non_monotonic_redshifts(self):
        """Test validation with non-monotonic redshifts."""
        rule = MonotonicityRule(check_monotonic=True)
        dataset = self._create_dataset_with_redshifts([0.3, 0.1, 0.2])
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "monotonicity_error"
        assert "not monotonically increasing" in exc_info.value.error_message
    
    def _create_dataset_with_redshifts(self, redshifts):
        """Helper to create dataset with specific redshifts."""
        z = np.array(redshifts)
        n = len(z)
        
        return StandardDataset(
            z=z,
            observable=np.ones(n),
            uncertainty=np.ones(n) * 0.1,
            covariance=None,
            metadata={'source': 'test'}
        )


class TestValidationEngine:
    """Test cases for ValidationEngine."""
    
    def test_initialization(self):
        """Test validation engine initialization."""
        engine = ValidationEngine()
        
        assert len(engine.rules) == 5  # Default rules (including MonotonicityRule)
        rule_names = [rule.rule_name for rule in engine.rules]
        assert "Schema Compliance" in rule_names
        assert "Numerical Integrity" in rule_names
        assert "Covariance Matrix Validation" in rule_names
        assert "Redshift Range" in rule_names[3]  # Partial match for range rule
    
    def test_add_rule(self):
        """Test adding custom validation rule."""
        engine = ValidationEngine()
        initial_count = len(engine.rules)
        
        custom_rule = MonotonicityRule(check_monotonic=True)
        engine.add_rule(custom_rule)
        
        assert len(engine.rules) == initial_count + 1
        assert custom_rule in engine.rules
    
    def test_remove_rule(self):
        """Test removing validation rule."""
        engine = ValidationEngine()
        initial_count = len(engine.rules)
        
        engine.remove_rule("Schema Compliance")
        
        assert len(engine.rules) == initial_count - 1
        rule_names = [rule.rule_name for rule in engine.rules]
        assert "Schema Compliance" not in rule_names
    
    def test_validate_dataset_success(self):
        """Test successful dataset validation."""
        engine = ValidationEngine()
        dataset = self._create_valid_dataset()
        
        results = engine.validate_dataset(dataset, "test_dataset")
        
        assert results['validation_passed'] is True
        assert results['dataset_name'] == "test_dataset"
        assert len(results['rules_checked']) == len(engine.rules)
        assert 'summary' in results
    
    def test_validate_dataset_failure(self):
        """Test dataset validation failure."""
        engine = ValidationEngine()
        
        # Create invalid dataset
        dataset = StandardDataset(
            z=np.array([0.1, np.nan, 0.3]),  # Contains NaN
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ProcessingError):
            engine.validate_dataset(dataset, "test_dataset")
    
    def test_validate_multiple_datasets(self):
        """Test validation of multiple datasets."""
        engine = ValidationEngine()
        
        valid_dataset = self._create_valid_dataset()
        invalid_dataset = StandardDataset(
            z=np.array([0.1, np.nan, 0.3]),  # Contains NaN
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        datasets = {
            'valid': valid_dataset,
            'invalid': invalid_dataset
        }
        
        results = engine.validate_multiple_datasets(datasets)
        
        assert 'valid' in results
        assert 'invalid' in results
        assert results['valid']['validation_passed'] is True
        assert results['invalid']['validation_passed'] is False
        assert 'error' in results['invalid']
    
    def test_generate_summary(self):
        """Test validation summary generation."""
        engine = ValidationEngine()
        dataset = self._create_valid_dataset_with_covariance()
        
        results = engine.validate_dataset(dataset, "test_dataset")
        summary = results['summary']
        
        assert summary['dataset_name'] == "test_dataset"
        assert summary['n_points'] == 3
        assert 'redshift_range' in summary
        assert 'observable_range' in summary
        assert summary['has_covariance'] is True
        assert 'covariance_shape' in summary
        assert 'covariance_condition_number' in summary
        assert 'metadata_keys' in summary
    
    def _create_valid_dataset(self):
        """Helper to create valid dataset."""
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test', 'version': '1.0'}
        )
    
    def _create_valid_dataset_with_covariance(self):
        """Helper to create valid dataset with covariance."""
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


class TestValidationEngineEdgeCases:
    """Test edge cases and additional scenarios for ValidationEngine."""
    
    def test_empty_dataset(self):
        """Test validation with empty dataset."""
        engine = ValidationEngine()
        
        dataset = StandardDataset(
            z=np.array([]),
            observable=np.array([]),
            uncertainty=np.array([]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            engine.validate_dataset(dataset, "empty_dataset")
        
        assert exc_info.value.error_type == "schema_error"
        assert "empty" in exc_info.value.error_message.lower()
    
    def test_mismatched_array_lengths(self):
        """Test validation with mismatched array lengths."""
        # Schema validation should fail during StandardDataset construction
        with pytest.raises(ValueError) as exc_info:
            dataset = StandardDataset(
                z=np.array([0.1, 0.2, 0.3]),
                observable=np.array([1.0, 2.0]),  # Different length
                uncertainty=np.array([0.1, 0.2, 0.3]),
                covariance=None,
                metadata={}
            )
        
        assert "length" in str(exc_info.value).lower()
    
    def test_covariance_size_mismatch(self):
        """Test validation with covariance matrix size mismatch."""
        # Schema validation should fail during StandardDataset construction
        with pytest.raises(ValueError) as exc_info:
            covariance = np.array([[0.01, 0.005], [0.005, 0.04]])  # 2x2 matrix
            dataset = StandardDataset(
                z=np.array([0.1, 0.2, 0.3]),  # 3 points
                observable=np.array([1.0, 2.0, 3.0]),
                uncertainty=np.array([0.1, 0.2, 0.3]),
                covariance=covariance,
                metadata={'source': 'test'}
            )
        
        assert "shape" in str(exc_info.value).lower()
    
    def test_zero_uncertainties(self):
        """Test validation with zero uncertainties."""
        engine = ValidationEngine()
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.0, 0.3]),  # Zero uncertainty
            covariance=None,
            metadata={'source': 'test'}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            engine.validate_dataset(dataset, "zero_uncertainty_dataset")
        
        assert exc_info.value.error_type == "numerical_error"
        assert "zero" in exc_info.value.error_message.lower()
    
    def test_very_large_values(self):
        """Test validation with very large values."""
        engine = ValidationEngine()
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1e20, 2.0, 3.0]),  # Very large value
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )
        
        # Should pass validation (large values are not necessarily invalid)
        results = engine.validate_dataset(dataset, "large_values_dataset")
        assert results['validation_passed'] is True
    
    def test_correlation_coefficient_validation(self):
        """Test validation of correlation coefficients in covariance matrix."""
        engine = ValidationEngine()
        
        # Create covariance matrix with invalid correlation (> 1)
        covariance = np.array([[0.01, 0.02], [0.02, 0.01]])  # Correlation > 1
        dataset = StandardDataset(
            z=np.array([0.1, 0.2]),
            observable=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.1]),
            covariance=covariance,
            metadata={'source': 'test'}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            engine.validate_dataset(dataset, "invalid_correlation_dataset")
        
        assert exc_info.value.error_type == "covariance_error"
    
    def test_custom_validation_rule(self):
        """Test adding and using custom validation rule."""
        from pipelines.data_preparation.core.interfaces import ValidationRule
        
        class CustomRule(ValidationRule):
            @property
            def rule_name(self) -> str:
                return "Custom Test Rule"
            
            def validate(self, dataset: StandardDataset) -> bool:
                if np.any(dataset.observable > 10.0):
                    raise ProcessingError(
                        dataset_name="unknown",
                        stage="validation",
                        error_type="custom_error",
                        error_message="Observable values too large",
                        context={'max_allowed': 10.0}
                    )
                return True
        
        engine = ValidationEngine()
        engine.add_rule(CustomRule())
        
        # Test with valid data
        valid_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )
        
        results = engine.validate_dataset(valid_dataset, "valid_custom")
        assert results['validation_passed'] is True
        
        # Test with invalid data
        invalid_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 15.0, 3.0]),  # Value > 10
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            engine.validate_dataset(invalid_dataset, "invalid_custom")
        
        assert exc_info.value.error_type == "custom_error"


class TestValidationRuleEdgeCases:
    """Test edge cases for individual validation rules."""
    
    def test_schema_validation_with_wrong_types(self):
        """Test schema validation with wrong data types."""
        rule = SchemaValidationRule()
        
        # Test with string arrays
        dataset = StandardDataset(
            z=np.array(['0.1', '0.2', '0.3']),  # String array
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "schema_error"
        assert "numeric" in exc_info.value.error_message.lower()
    
    def test_numerical_integrity_with_complex_numbers(self):
        """Test numerical integrity with complex numbers."""
        rule = NumericalIntegrityRule()
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0+1j, 2.0, 3.0]),  # Complex number
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "numerical_error"
        assert "complex" in exc_info.value.error_message.lower()
    
    def test_covariance_validation_with_nan(self):
        """Test covariance validation with NaN values."""
        rule = CovarianceValidationRule()
        
        covariance = np.array([[0.01, np.nan], [np.nan, 0.04]])
        dataset = StandardDataset(
            z=np.array([0.1, 0.2]),
            observable=np.array([1.0, 2.0]),
            uncertainty=np.array([0.1, 0.2]),
            covariance=covariance,
            metadata={'source': 'test'}
        )
        
        with pytest.raises(ProcessingError) as exc_info:
            rule.validate(dataset)
        
        assert exc_info.value.error_type == "covariance_error"
        assert "NaN" in exc_info.value.error_message
    
    def test_redshift_range_with_edge_values(self):
        """Test redshift range validation with edge values."""
        rule = RedshiftRangeRule(0.0, 1.0)
        
        # Test exactly at boundaries
        dataset = StandardDataset(
            z=np.array([0.0, 0.5, 1.0]),  # Exactly at min and max
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        assert rule.validate(dataset) is True
    
    def test_monotonicity_with_duplicate_redshifts(self):
        """Test monotonicity validation with duplicate redshift values."""
        rule = MonotonicityRule(check_monotonic=True, allow_duplicates=True)
        
        dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.2, 0.3]),  # Duplicate redshift
            observable=np.array([1.0, 2.0, 2.1, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.2, 0.3]),
            covariance=None,
            metadata={}
        )
        
        # Should pass (non-decreasing is acceptable)
        assert rule.validate(dataset) is True


class TestValidationPerformance:
    """Test validation performance with large datasets."""
    
    def test_large_dataset_validation(self):
        """Test validation performance with large dataset."""
        engine = ValidationEngine()
        
        # Create large dataset
        n_points = 10000
        dataset = StandardDataset(
            z=np.linspace(0.1, 2.0, n_points),
            observable=np.random.normal(1.0, 0.1, n_points),
            uncertainty=np.random.uniform(0.05, 0.2, n_points),
            covariance=None,
            metadata={'source': 'large_test'}
        )
        
        import time
        start_time = time.time()
        results = engine.validate_dataset(dataset, "large_dataset")
        end_time = time.time()
        
        assert results['validation_passed'] is True
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
    
    def test_large_covariance_matrix_validation(self):
        """Test validation with large covariance matrix."""
        engine = ValidationEngine()
        
        # Create dataset with large covariance matrix
        n_points = 1000
        z = np.linspace(0.1, 2.0, n_points)
        observable = np.random.normal(1.0, 0.1, n_points)
        uncertainty = np.random.uniform(0.05, 0.2, n_points)
        
        # Create positive definite covariance matrix
        A = np.random.randn(n_points, n_points)
        covariance = np.dot(A, A.T) * 0.001  # Ensure positive definite
        
        dataset = StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=covariance,
            metadata={'source': 'large_cov_test'}
        )
        
        import time
        start_time = time.time()
        results = engine.validate_dataset(dataset, "large_covariance_dataset")
        end_time = time.time()
        
        assert results['validation_passed'] is True
        assert end_time - start_time < 10.0  # Should complete within 10 seconds