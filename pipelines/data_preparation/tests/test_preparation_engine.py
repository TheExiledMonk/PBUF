"""
Unit tests for the main preparation engine.

Tests the DataPreparationFramework orchestration, module registration,
and error handling functionality.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
from pipelines.data_preparation.core.interfaces import DerivationModule, ProcessingError
from pipelines.data_preparation.core.schema import StandardDataset


class MockDerivationModule(DerivationModule):
    """Mock derivation module for testing."""
    
    def __init__(self, dataset_type: str, should_fail: str = None):
        self._dataset_type = dataset_type
        self.should_fail = should_fail  # Stage where module should fail
    
    @property
    def dataset_type(self) -> str:
        return self._dataset_type
    
    @property
    def supported_formats(self) -> List[str]:
        return ['.txt', '.csv']
    
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        if self.should_fail == "input_validation":
            return False
        return True
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        if self.should_fail == "transformation":
            raise ValueError("Mock transformation failure")
        
        return StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.2, 0.3]),
            covariance=None,
            metadata=metadata
        )
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        return {
            'transformation_steps': [f'Mock {self._dataset_type} transformation'],
            'formulas_used': [f'{self._dataset_type}_formula = x * 2'],
            'assumptions': [f'{self._dataset_type} assumption'],
            'references': [f'{self._dataset_type} reference']
        }


class TestDataPreparationFramework:
    """Test cases for DataPreparationFramework."""
    
    def test_initialization(self):
        """Test framework initialization."""
        framework = DataPreparationFramework()
        
        assert framework.registry_manager is None
        assert isinstance(framework.derivation_modules, dict)
        assert len(framework.derivation_modules) == 0
        assert framework.validation_engine is not None
    
    def test_initialization_with_registry(self):
        """Test framework initialization with registry manager."""
        class MockRegistryManager:
            def __init__(self):
                self.registry_path = "/tmp/mock_registry"
        
        mock_registry = MockRegistryManager()
        framework = DataPreparationFramework(registry_manager=mock_registry)
        
        assert framework.registry_manager == mock_registry
    
    def test_register_derivation_module(self):
        """Test derivation module registration."""
        framework = DataPreparationFramework()
        module = MockDerivationModule("test")
        
        framework.register_derivation_module(module)
        
        assert "test" in framework.derivation_modules
        assert framework.derivation_modules["test"] == module
    
    def test_get_available_dataset_types(self):
        """Test getting available dataset types."""
        framework = DataPreparationFramework()
        
        # Initially empty
        assert framework.get_available_dataset_types() == []
        
        # After registering modules
        framework.register_derivation_module(MockDerivationModule("sn"))
        framework.register_derivation_module(MockDerivationModule("bao"))
        
        available_types = framework.get_available_dataset_types()
        assert "sn" in available_types
        assert "bao" in available_types
        assert len(available_types) == 2
    
    def test_infer_dataset_type(self):
        """Test dataset type inference from name."""
        framework = DataPreparationFramework()
        
        assert framework._infer_dataset_type("sn_pantheon") == "sn"
        assert framework._infer_dataset_type("supernova_data") == "sn"
        assert framework._infer_dataset_type("bao_boss") == "bao"
        assert framework._infer_dataset_type("cmb_planck") == "cmb"
        assert framework._infer_dataset_type("planck_2018") == "cmb"
        assert framework._infer_dataset_type("cc_moresco") == "cc"
        assert framework._infer_dataset_type("chronometer_data") == "cc"
        assert framework._infer_dataset_type("rsd_growth") == "rsd"
        assert framework._infer_dataset_type("growth_rate") == "rsd"
        assert framework._infer_dataset_type("unknown_dataset") == "unknown"
    
    def test_get_derivation_module_success(self):
        """Test successful derivation module retrieval."""
        framework = DataPreparationFramework()
        module = MockDerivationModule("sn")
        framework.register_derivation_module(module)
        
        retrieved_module = framework._get_derivation_module("sn_pantheon")
        assert retrieved_module == module
    
    def test_get_derivation_module_not_found(self):
        """Test derivation module retrieval when module not found."""
        framework = DataPreparationFramework()
        
        with pytest.raises(ProcessingError) as exc_info:
            framework._get_derivation_module("sn_pantheon")
        
        assert exc_info.value.error_type == "unsupported_dataset_type"
        assert "No derivation module available" in exc_info.value.error_message
    
    def test_prepare_dataset_no_registry(self, tmp_path):
        """Test dataset preparation without registry manager."""
        framework = DataPreparationFramework()
        module = MockDerivationModule("sn")
        framework.register_derivation_module(module)
        
        # Create test file
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("test data")
        
        metadata = {'source': 'test', 'version': '1.0'}
        
        dataset = framework.prepare_dataset(
            "sn_pantheon",
            raw_data_path=test_file,
            metadata=metadata
        )
        
        assert isinstance(dataset, StandardDataset)
        assert len(dataset.z) == 3
        assert dataset.metadata == metadata
    
    def test_prepare_dataset_input_validation_failure(self, tmp_path):
        """Test dataset preparation with input validation failure."""
        framework = DataPreparationFramework()
        module = MockDerivationModule("sn", should_fail="input_validation")
        framework.register_derivation_module(module)
        
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("test data")
        
        with pytest.raises(ProcessingError) as exc_info:
            framework.prepare_dataset(
                "sn_pantheon",
                raw_data_path=test_file,
                metadata={}
            )
        
        assert exc_info.value.stage == "input_validation"
        assert exc_info.value.error_type == "input_validation_failed"
    
    def test_prepare_dataset_transformation_failure(self, tmp_path):
        """Test dataset preparation with transformation failure."""
        framework = DataPreparationFramework()
        module = MockDerivationModule("sn", should_fail="transformation")
        framework.register_derivation_module(module)
        
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("test data")
        
        with pytest.raises(ProcessingError) as exc_info:
            framework.prepare_dataset(
                "sn_pantheon",
                raw_data_path=test_file,
                metadata={}
            )
        
        assert exc_info.value.stage == "transformation"
        assert exc_info.value.error_type == "transformation_failed"
        assert "Dataset transformation failed: Mock transformation failure" in exc_info.value.error_message
    
    def test_prepare_dataset_output_validation_failure(self, tmp_path):
        """Test dataset preparation with output validation failure."""
        framework = DataPreparationFramework()
        
        # Create module that produces invalid output
        class InvalidOutputModule(MockDerivationModule):
            def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
                return StandardDataset(
                    z=np.array([0.1, np.nan, 0.3]),  # Contains NaN
                    observable=np.array([1.0, 2.0, 3.0]),
                    uncertainty=np.array([0.1, 0.2, 0.3]),
                    covariance=None,
                    metadata=metadata
                )
        
        module = InvalidOutputModule("sn")
        framework.register_derivation_module(module)
        
        test_file = tmp_path / "test_data.txt"
        test_file.write_text("test data")
        
        with pytest.raises(ProcessingError) as exc_info:
            framework.prepare_dataset(
                "sn_pantheon",
                raw_data_path=test_file,
                metadata={}
            )
        
        assert exc_info.value.stage == "output_validation"
    
    def test_retrieve_from_registry_no_manager(self):
        """Test registry retrieval without registry manager."""
        framework = DataPreparationFramework()
        
        with pytest.raises(ProcessingError) as exc_info:
            framework._retrieve_from_registry_with_recovery("test_dataset")
        
        assert exc_info.value.error_type == "no_registry_integration"
    
    def test_retrieve_from_registry_not_implemented(self):
        """Test registry retrieval when not yet implemented."""
        # Create a mock registry manager with required attributes
        class MockRegistryManager:
            def __init__(self):
                self.registry_path = "/tmp/mock_registry"
        
        mock_registry = MockRegistryManager()
        framework = DataPreparationFramework(registry_manager=mock_registry)
        
        with pytest.raises(ProcessingError) as exc_info:
            framework._retrieve_from_registry_with_recovery("test_dataset")
        
        # The error will be from the registry integration trying to access the dataset
        assert exc_info.value.error_type in ["dataset_not_found", "registry_access_error"]
    
    def test_validate_framework_setup_empty(self):
        """Test framework setup validation with no modules."""
        framework = DataPreparationFramework()
        
        results = framework.validate_framework_setup()
        
        assert results['framework_ready'] is False
        assert "No derivation modules registered" in results['issues']
        assert "No registry manager configured - manual data paths required" in results['warnings']
        assert results['available_modules'] == []
        assert len(results['validation_rules']) > 0
    
    def test_validate_framework_setup_with_modules(self):
        """Test framework setup validation with registered modules."""
        class MockRegistryManager:
            def __init__(self):
                self.registry_path = "/tmp/mock_registry"
        
        mock_registry = MockRegistryManager()
        framework = DataPreparationFramework(registry_manager=mock_registry)
        framework.register_derivation_module(MockDerivationModule("sn"))
        framework.register_derivation_module(MockDerivationModule("bao"))
        
        results = framework.validate_framework_setup()
        
        assert results['framework_ready'] is True
        assert len(results['issues']) == 0
        assert "sn" in results['available_modules']
        assert "bao" in results['available_modules']
        assert len(results['validation_rules']) > 0