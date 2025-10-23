"""
Unit tests for the core interfaces and abstract base classes.

Tests the DerivationModule interface, ValidationRule interface,
and ProcessingError functionality.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from pipelines.data_preparation.core.interfaces import (
    DerivationModule, ValidationRule, ProcessingError
)
from pipelines.data_preparation.core.schema import StandardDataset


class MockDerivationModule(DerivationModule):
    """Mock derivation module for testing."""
    
    @property
    def dataset_type(self) -> str:
        return "test"
    
    @property
    def supported_formats(self) -> List[str]:
        return ['.txt', '.csv']
    
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        return raw_data_path.exists()
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        # Create a simple test dataset
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata=metadata
        )
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        return {
            'transformation_steps': ['Load data', 'Apply test transformation'],
            'formulas_used': ['test_formula = x * 2'],
            'assumptions': ['Test assumption'],
            'references': ['Test reference']
        }


class MockValidationRule(ValidationRule):
    """Mock validation rule for testing."""
    
    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass
    
    @property
    def rule_name(self) -> str:
        return "Mock Validation Rule"
    
    def validate(self, dataset: StandardDataset) -> bool:
        if not self.should_pass:
            raise ValueError("Mock validation failure")
        return True


class TestDerivationModule:
    """Test cases for DerivationModule interface."""
    
    def test_derivation_module_properties(self):
        """Test derivation module properties."""
        module = MockDerivationModule()
        
        assert module.dataset_type == "test"
        assert module.supported_formats == ['.txt', '.csv']
    
    def test_expected_output_schema(self):
        """Test expected output schema method."""
        module = MockDerivationModule()
        schema = module.get_expected_output_schema()
        
        assert 'z' in schema
        assert 'observable' in schema
        assert 'uncertainty' in schema
        assert 'covariance' in schema
        assert 'metadata' in schema
        assert schema['observable'] == 'TEST observable'
    
    def test_validate_input(self, tmp_path):
        """Test input validation method."""
        module = MockDerivationModule()
        
        # Create a test file
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("test data")
        
        # Test with existing file
        assert module.validate_input(test_file, {}) is True
        
        # Test with non-existing file
        non_existing_file = tmp_path / "non_existing.txt"
        assert module.validate_input(non_existing_file, {}) is False
    
    def test_derive_method(self, tmp_path):
        """Test derive method."""
        module = MockDerivationModule()
        
        # Create a test file
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("test data")
        
        metadata = {'source': 'test', 'version': '1.0'}
        dataset = module.derive(test_file, metadata)
        
        assert isinstance(dataset, StandardDataset)
        assert len(dataset.z) == 3
        assert dataset.metadata == metadata
    
    def test_transformation_summary(self):
        """Test transformation summary method."""
        module = MockDerivationModule()
        summary = module.get_transformation_summary()
        
        assert 'transformation_steps' in summary
        assert 'formulas_used' in summary
        assert 'assumptions' in summary
        assert 'references' in summary
        
        assert isinstance(summary['transformation_steps'], list)
        assert len(summary['transformation_steps']) > 0


class TestValidationRule:
    """Test cases for ValidationRule interface."""
    
    def test_validation_rule_properties(self):
        """Test validation rule properties."""
        rule = MockValidationRule()
        assert rule.rule_name == "Mock Validation Rule"
    
    def test_validation_success(self):
        """Test successful validation."""
        rule = MockValidationRule(should_pass=True)
        dataset = self._create_test_dataset()
        
        assert rule.validate(dataset) is True
    
    def test_validation_failure(self):
        """Test validation failure."""
        rule = MockValidationRule(should_pass=False)
        dataset = self._create_test_dataset()
        
        with pytest.raises(ValueError, match="Mock validation failure"):
            rule.validate(dataset)
    
    def _create_test_dataset(self):
        """Helper method to create test dataset."""
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata={'source': 'test'}
        )


class TestProcessingError:
    """Test cases for ProcessingError class."""
    
    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = ProcessingError(
            dataset_name="test_dataset",
            stage="transformation",
            error_type="test_error",
            error_message="Test error message"
        )
        
        assert error.dataset_name == "test_dataset"
        assert error.stage == "transformation"
        assert error.error_type == "test_error"
        assert error.error_message == "Test error message"
        assert error.context == {}
        assert error.suggested_actions == []
    
    def test_error_with_context_and_actions(self):
        """Test error creation with context and suggested actions."""
        context = {'key': 'value', 'number': 42}
        actions = ['Action 1', 'Action 2']
        
        error = ProcessingError(
            dataset_name="test_dataset",
            stage="validation",
            error_type="validation_error",
            error_message="Validation failed",
            context=context,
            suggested_actions=actions
        )
        
        assert error.context == context
        assert error.suggested_actions == actions
    
    def test_error_message_generation(self):
        """Test error message generation."""
        error = ProcessingError(
            dataset_name="test_dataset",
            stage="transformation",
            error_type="test_error",
            error_message="Test error message",
            context={'key': 'value'},
            suggested_actions=['Action 1', 'Action 2']
        )
        
        message = error.generate_message()
        
        assert "test_dataset" in message
        assert "transformation" in message
        assert "test_error" in message
        assert "Test error message" in message
        assert "Context:" in message
        assert "Suggested Actions:" in message
        assert "1. Action 1" in message
        assert "2. Action 2" in message
    
    def test_error_to_dict(self):
        """Test error serialization to dictionary."""
        context = {'key': 'value'}
        actions = ['Action 1']
        
        error = ProcessingError(
            dataset_name="test_dataset",
            stage="transformation",
            error_type="test_error",
            error_message="Test error message",
            context=context,
            suggested_actions=actions
        )
        
        error_dict = error.to_dict()
        
        assert error_dict['dataset_name'] == "test_dataset"
        assert error_dict['stage'] == "transformation"
        assert error_dict['error_type'] == "test_error"
        assert error_dict['error_message'] == "Test error message"
        assert error_dict['context'] == context
        assert error_dict['suggested_actions'] == actions
    
    def test_error_inheritance(self):
        """Test that ProcessingError inherits from Exception."""
        error = ProcessingError(
            dataset_name="test",
            stage="test",
            error_type="test",
            error_message="test"
        )
        
        assert isinstance(error, Exception)
        
        # Test that it can be raised and caught
        with pytest.raises(ProcessingError):
            raise error