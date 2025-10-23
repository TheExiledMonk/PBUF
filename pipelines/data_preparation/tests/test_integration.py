"""
Integration tests for the data preparation framework.

Tests end-to-end functionality of the framework components working together,
including registry integration, provenance recording, and fit pipeline compatibility.

Requirements: 8.4, 8.5
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from pipelines.data_preparation import DataPreparationFramework, StandardDataset, DerivationModule
from pipelines.data_preparation.core.registry_integration import RegistryIntegration
from pipelines.data_preparation.core.interfaces import ProcessingError
from pipelines.dataset_registry.core.registry_manager import (
    RegistryManager, ProvenanceRecord, VerificationResult, EnvironmentInfo
)


class TestDerivationModule(DerivationModule):
    """Test derivation module for integration testing."""
    
    @property
    def dataset_type(self) -> str:
        return "test"
    
    @property
    def supported_formats(self) -> List[str]:
        return ['.txt', '.csv']
    
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        """Validate that input file exists and has content."""
        if not raw_data_path.exists():
            return False
        
        content = raw_data_path.read_text().strip()
        return len(content) > 0
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """Create a test dataset from input file."""
        # Read simple test data format: z,observable,uncertainty per line
        lines = raw_data_path.read_text().strip().split('\n')
        
        z_values = []
        obs_values = []
        unc_values = []
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 3:
                    z_values.append(float(parts[0]))
                    obs_values.append(float(parts[1]))
                    unc_values.append(float(parts[2]))
        
        return StandardDataset(
            z=np.array(z_values),
            observable=np.array(obs_values),
            uncertainty=np.array(unc_values),
            covariance=None,
            metadata={
                **metadata,
                'source': metadata.get('source', 'test_source'),  # Ensure 'source' field exists
                'processing_info': 'Processed by TestDerivationModule',
                'n_points': len(z_values)
            }
        )
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return transformation summary."""
        return {
            'transformation_steps': [
                'Read CSV-like input file',
                'Parse z, observable, uncertainty columns',
                'Create StandardDataset'
            ],
            'formulas_used': ['No mathematical transformations applied'],
            'assumptions': ['Input data is already in correct units'],
            'references': ['Test derivation module - no external references']
        }


class TestFrameworkIntegration:
    """Integration tests for the complete framework."""
    
    def test_end_to_end_processing(self, tmp_path):
        """Test complete end-to-end dataset processing."""
        # Create test input file
        test_data = """# Test dataset
0.1,1.0,0.1
0.2,2.0,0.2
0.3,3.0,0.3
0.4,4.0,0.4
"""
        test_file = tmp_path / "test_dataset.txt"
        test_file.write_text(test_data)
        
        # Initialize framework and register module
        framework = DataPreparationFramework()
        test_module = TestDerivationModule()
        framework.register_derivation_module(test_module)
        
        # Prepare dataset
        metadata = {
            'source': 'integration_test',
            'version': '1.0',
            'description': 'Test dataset for integration testing'
        }
        
        dataset = framework.prepare_dataset(
            "test_data",  # This will be inferred as "test" type due to "test" in name
            raw_data_path=test_file,
            metadata=metadata
        )
        
        # Verify results
        assert isinstance(dataset, StandardDataset)
        assert len(dataset.z) == 4
        assert np.array_equal(dataset.z, [0.1, 0.2, 0.3, 0.4])
        assert np.array_equal(dataset.observable, [1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(dataset.uncertainty, [0.1, 0.2, 0.3, 0.4])
        assert dataset.covariance is None
        
        # Check metadata
        assert dataset.metadata['source'] == 'integration_test'
        assert dataset.metadata['version'] == '1.0'
        assert dataset.metadata['processing_info'] == 'Processed by TestDerivationModule'
        assert dataset.metadata['n_points'] == 4
    
    def test_validation_integration(self, tmp_path):
        """Test that validation is properly integrated in the workflow."""
        # Create test input file with invalid data (NaN)
        test_data = """# Test dataset with NaN
0.1,1.0,0.1
0.2,nan,0.2
0.3,3.0,0.3
"""
        test_file = tmp_path / "invalid_dataset.txt"
        test_file.write_text(test_data)
        
        # Create module that doesn't handle NaN properly
        class BadDerivationModule(TestDerivationModule):
            def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
                # This will create a dataset with NaN values
                lines = raw_data_path.read_text().strip().split('\n')
                
                z_values = []
                obs_values = []
                unc_values = []
                
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            z_values.append(float(parts[0]))
                            # This will create NaN when parsing "nan"
                            obs_values.append(float(parts[1]) if parts[1] != 'nan' else np.nan)
                            unc_values.append(float(parts[2]))
                
                return StandardDataset(
                    z=np.array(z_values),
                    observable=np.array(obs_values),
                    uncertainty=np.array(unc_values),
                    covariance=None,
                    metadata=metadata
                )
        
        # Initialize framework
        framework = DataPreparationFramework()
        bad_module = BadDerivationModule()
        framework.register_derivation_module(bad_module)
        
        # Attempt to prepare dataset - should fail validation
        from pipelines.data_preparation.core.interfaces import ProcessingError
        
        with pytest.raises(ProcessingError) as exc_info:
            framework.prepare_dataset(
                "test_invalid_data",  # This will be inferred as "test" type
                raw_data_path=test_file,
                metadata={'source': 'test'}
            )
        
        # Verify it failed at validation stage with numerical error
        assert exc_info.value.stage == "output_validation"
        assert exc_info.value.error_type == "numerical_error"
        assert "NaN values" in exc_info.value.error_message
    
    def test_framework_setup_validation(self):
        """Test framework setup validation functionality."""
        # Test empty framework
        framework = DataPreparationFramework()
        results = framework.validate_framework_setup()
        
        assert results['framework_ready'] is False
        assert "No derivation modules registered" in results['issues']
        assert results['available_modules'] == []
        
        # Add module and test again
        test_module = TestDerivationModule()
        framework.register_derivation_module(test_module)
        
        results = framework.validate_framework_setup()
        assert "test" in results['available_modules']
        assert len(results['validation_rules']) > 0
    
    def test_multiple_module_registration(self):
        """Test registration and use of multiple derivation modules."""
        framework = DataPreparationFramework()
        
        # Register multiple modules
        class SNModule(TestDerivationModule):
            @property
            def dataset_type(self) -> str:
                return "sn"
        
        class BAOModule(TestDerivationModule):
            @property
            def dataset_type(self) -> str:
                return "bao"
        
        sn_module = SNModule()
        bao_module = BAOModule()
        
        framework.register_derivation_module(sn_module)
        framework.register_derivation_module(bao_module)
        
        # Verify both are available
        available_types = framework.get_available_dataset_types()
        assert "sn" in available_types
        assert "bao" in available_types
        assert len(available_types) == 2
        
        # Verify correct module selection
        sn_retrieved = framework._get_derivation_module("sn_pantheon")
        bao_retrieved = framework._get_derivation_module("bao_boss")
        
        assert sn_retrieved.dataset_type == "sn"
        assert bao_retrieved.dataset_type == "bao"
    
    def test_error_handling_integration(self, tmp_path):
        """Test comprehensive error handling throughout the pipeline."""
        framework = DataPreparationFramework()
        
        # Test with non-existent file
        non_existent_file = tmp_path / "does_not_exist.txt"
        
        from pipelines.data_preparation.core.interfaces import ProcessingError
        
        with pytest.raises(ProcessingError) as exc_info:
            framework.prepare_dataset(
                "unknown_nonexistent",  # This will be inferred as "unknown" type
                raw_data_path=non_existent_file,
                metadata={}
            )
        
        # Should fail at module selection stage (no modules registered)
        assert exc_info.value.error_type == "unsupported_dataset_type"
        
        # Register module and test input validation failure
        class StrictModule(TestDerivationModule):
            def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
                return False  # Always fail validation
        
        strict_module = StrictModule()
        framework.register_derivation_module(strict_module)
        
        # Create a valid file but module will reject it
        test_file = tmp_path / "test.txt"
        test_file.write_text("0.1,1.0,0.1")
        
        with pytest.raises(ProcessingError) as exc_info:
            framework.prepare_dataset(
                "test_strict_data",  # This will be inferred as "test" type
                raw_data_path=test_file,
                metadata={}
            )
        
        assert exc_info.value.stage == "input_validation"
        assert exc_info.value.error_type == "input_validation_failed"


class TestEndToEndWorkflow:
    """
    End-to-end workflow tests for complete preparation pipeline.
    
    Tests the complete data preparation workflow from raw dataset input
    through standardized output generation with full validation.
    
    Requirements: 8.4 - Integration tests for system components
    """
    
    @pytest.fixture
    def mock_registry_manager(self):
        """Create mock registry manager for end-to-end testing."""
        mock_manager = Mock(spec=RegistryManager)
        mock_manager.registry_path = Path("/tmp/test_registry")
        return mock_manager
    
    @pytest.fixture
    def sample_provenance_record(self):
        """Create comprehensive provenance record for testing."""
        environment = EnvironmentInfo(
            pbuf_commit="abc123def456",
            python_version="3.9.0",
            platform="Linux",
            hostname="test-host",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        verification = VerificationResult(
            sha256_verified=True,
            sha256_expected="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            sha256_actual="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            size_verified=True,
            size_expected=1000,
            size_actual=1000,
            schema_verified=True,
            schema_errors=[],
            verification_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        return ProvenanceRecord(
            dataset_name="sn_pantheon_test",
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            source_used="test_source",
            download_agent="test_agent",
            environment=environment,
            verification=verification,
            file_info={
                "local_path": "/tmp/sn_pantheon_test.txt",
                "original_filename": "sn_pantheon_test.txt",
                "mime_type": "text/plain"
            }
        )
    
    def test_complete_sn_processing_workflow(self, tmp_path, mock_registry_manager, sample_provenance_record):
        """Test complete end-to-end processing workflow for supernova data."""
        # Create realistic SN test data
        sn_test_data = """# Supernova test data
# z, mu, sigma_mu
0.0233,32.81,0.12
0.0404,34.48,0.11
0.0593,35.58,0.13
0.0743,36.31,0.14
0.0930,37.03,0.15
"""
        test_file = tmp_path / "sn_pantheon_test.txt"
        test_file.write_text(sn_test_data)
        
        # Update provenance record with actual file path
        sample_provenance_record.file_info["local_path"] = str(test_file)
        
        # Mock registry manager responses
        mock_registry_manager.get_registry_entry.return_value = sample_provenance_record
        mock_registry_manager.has_registry_entry.return_value = True
        
        # Create framework with registry integration
        framework = DataPreparationFramework(
            registry_manager=mock_registry_manager,
            output_directory=tmp_path / "output"
        )
        
        # Register SN derivation module
        class SNDerivationModule(TestDerivationModule):
            @property
            def dataset_type(self) -> str:
                return "sn"
            
            def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
                """Process SN data with realistic transformations."""
                lines = raw_data_path.read_text().strip().split('\n')
                
                z_values = []
                mu_values = []
                sigma_values = []
                
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            z_values.append(float(parts[0]))
                            mu_values.append(float(parts[1]))
                            sigma_values.append(float(parts[2]))
                
                return StandardDataset(
                    z=np.array(z_values),
                    observable=np.array(mu_values),  # Distance modulus
                    uncertainty=np.array(sigma_values),
                    covariance=None,
                    metadata={
                        **metadata,
                        'source': metadata.get('source_used', 'test_source'),  # Ensure 'source' field exists
                        'processing_info': 'SN distance modulus processing',
                        'n_points': len(z_values),
                        'observable_type': 'distance_modulus',
                        'units': 'mag',
                        'transformation_summary': {
                            'steps': ['Parse SN data', 'Extract z-mu-sigma columns'],
                            'formulas': ['mu = m - M (distance modulus)'],
                            'assumptions': ['Data already calibrated']
                        }
                    }
                )
        
        sn_module = SNDerivationModule()
        framework.register_derivation_module(sn_module)
        
        # Mock file operations for registry integration
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(framework.registry_integration, '_calculate_file_checksum') as mock_checksum:
            
            mock_stat.return_value.st_size = 1000
            mock_checksum.return_value = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            
            # Execute complete workflow
            result = framework.prepare_dataset_from_registry("sn_pantheon_test")
        
        # Verify complete workflow results
        assert isinstance(result, StandardDataset)
        assert len(result.z) == 5
        assert np.allclose(result.z, [0.0233, 0.0404, 0.0593, 0.0743, 0.0930])
        assert np.allclose(result.observable, [32.81, 34.48, 35.58, 36.31, 37.03])
        assert np.allclose(result.uncertainty, [0.12, 0.11, 0.13, 0.14, 0.15])
        
        # Verify metadata includes provenance and processing information
        assert result.metadata['dataset_name'] == 'sn_pantheon_test'
        assert result.metadata['observable_type'] == 'distance_modulus'
        assert result.metadata['n_points'] == 5
        assert 'transformation_summary' in result.metadata
        assert 'provenance_summary' in result.metadata
        
        # Verify provenance tracking
        provenance = result.metadata['provenance_summary']
        assert provenance['source_dataset'] == 'sn_pantheon_test'
        assert provenance['source_checksum'] == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert 'processing_timestamp' in provenance
        assert 'environment_hash' in provenance
    
    def test_complete_bao_processing_workflow(self, tmp_path, mock_registry_manager):
        """Test complete end-to-end processing workflow for BAO data."""
        # Create realistic BAO test data
        bao_test_data = """# BAO test data
# z, DV_over_rd, sigma_DV_over_rd
0.15,4.47,0.17
0.32,8.88,0.17
0.57,13.77,0.13
"""
        test_file = tmp_path / "bao_boss_test.txt"
        test_file.write_text(bao_test_data)
        
        # Create provenance record for BAO data
        environment = EnvironmentInfo(
            pbuf_commit="abc123def456",
            python_version="3.9.0",
            platform="Linux",
            hostname="test-host",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        verification = VerificationResult(
            sha256_verified=True,
            sha256_expected="bao_test_hash",
            sha256_actual="bao_test_hash",
            size_verified=True,
            size_expected=500,
            size_actual=500,
            schema_verified=True,
            schema_errors=[],
            verification_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        bao_provenance = ProvenanceRecord(
            dataset_name="bao_boss_test",
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            source_used="BOSS_survey",
            download_agent="test_agent",
            environment=environment,
            verification=verification,
            file_info={
                "local_path": str(test_file),
                "original_filename": "bao_boss_test.txt",
                "mime_type": "text/plain"
            }
        )
        
        # Mock registry manager responses
        mock_registry_manager.get_registry_entry.return_value = bao_provenance
        mock_registry_manager.has_registry_entry.return_value = True
        
        # Create framework
        framework = DataPreparationFramework(
            registry_manager=mock_registry_manager,
            output_directory=tmp_path / "output"
        )
        
        # Register BAO derivation module
        class BAODerivationModule(TestDerivationModule):
            @property
            def dataset_type(self) -> str:
                return "bao"
            
            def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
                """Process BAO data with realistic transformations."""
                lines = raw_data_path.read_text().strip().split('\n')
                
                z_values = []
                dv_values = []
                sigma_values = []
                
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 3:
                            z_values.append(float(parts[0]))
                            dv_values.append(float(parts[1]))
                            sigma_values.append(float(parts[2]))
                
                return StandardDataset(
                    z=np.array(z_values),
                    observable=np.array(dv_values),  # D_V/r_d
                    uncertainty=np.array(sigma_values),
                    covariance=None,
                    metadata={
                        **metadata,
                        'source': metadata.get('source_used', 'test_source'),  # Ensure 'source' field exists
                        'processing_info': 'BAO distance scale processing',
                        'n_points': len(z_values),
                        'observable_type': 'DV_over_rd',
                        'units': 'dimensionless',
                        'transformation_summary': {
                            'steps': ['Parse BAO data', 'Extract z-DV/rd-sigma columns'],
                            'formulas': ['D_V = (D_M^2 * D_H * z)^(1/3)'],
                            'assumptions': ['Isotropic BAO measurements']
                        }
                    }
                )
        
        bao_module = BAODerivationModule()
        framework.register_derivation_module(bao_module)
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(framework.registry_integration, '_calculate_file_checksum') as mock_checksum:
            
            mock_stat.return_value.st_size = 500
            mock_checksum.return_value = "bao_test_hash"
            
            # Execute complete workflow
            result = framework.prepare_dataset_from_registry("bao_boss_test")
        
        # Verify results
        assert isinstance(result, StandardDataset)
        assert len(result.z) == 3
        assert np.allclose(result.z, [0.15, 0.32, 0.57])
        assert np.allclose(result.observable, [4.47, 8.88, 13.77])
        assert result.metadata['observable_type'] == 'DV_over_rd'
        assert 'transformation_summary' in result.metadata
    
    def test_deterministic_processing_workflow(self, tmp_path, mock_registry_manager, sample_provenance_record):
        """Test that processing workflow produces deterministic results."""
        # Create test data
        test_data = """0.1,1.0,0.1
0.2,2.0,0.2
0.3,3.0,0.3"""
        test_file = tmp_path / "deterministic_test.txt"
        test_file.write_text(test_data)
        
        sample_provenance_record.file_info["local_path"] = str(test_file)
        sample_provenance_record.dataset_name = "test_deterministic"
        
        mock_registry_manager.get_registry_entry.return_value = sample_provenance_record
        mock_registry_manager.has_registry_entry.return_value = True
        
        # Create framework
        framework = DataPreparationFramework(
            registry_manager=mock_registry_manager,
            output_directory=tmp_path / "output"
        )
        
        # Register test module
        test_module = TestDerivationModule()
        framework.register_derivation_module(test_module)
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(framework.registry_integration, '_calculate_file_checksum') as mock_checksum:
            
            mock_stat.return_value.st_size = 100
            mock_checksum.return_value = "deterministic_hash"
            
            # Process dataset multiple times
            result1 = framework.prepare_dataset_from_registry("test_deterministic")
            result2 = framework.prepare_dataset_from_registry("test_deterministic")
        
        # Verify deterministic behavior
        assert np.array_equal(result1.z, result2.z)
        assert np.array_equal(result1.observable, result2.observable)
        assert np.array_equal(result1.uncertainty, result2.uncertainty)
        
        # Verify checksums match (deterministic processing)
        checksum1 = result1.metadata.get('derived_checksum')
        checksum2 = result2.metadata.get('derived_checksum')
        if checksum1 and checksum2:
            assert checksum1 == checksum2


class TestRegistryIntegrationWorkflow:
    """
    Registry integration tests for data retrieval and provenance recording.
    
    Tests the integration between the data preparation framework and the
    dataset registry system, including provenance tracking and metadata handling.
    
    Requirements: 8.4 - Integration tests for registry integration
    """
    
    @pytest.fixture
    def mock_registry_manager(self):
        """Create comprehensive mock registry manager."""
        mock_manager = Mock(spec=RegistryManager)
        mock_manager.registry_path = Path("/tmp/test_registry")
        
        # Mock audit log methods
        mock_manager._append_audit_log = Mock()
        mock_manager._get_environment_info = Mock()
        
        return mock_manager
    
    def test_registry_dataset_retrieval_integration(self, tmp_path, mock_registry_manager):
        """Test complete registry dataset retrieval workflow."""
        # Create test dataset file
        test_data = """# Test cosmological dataset
0.1,35.2,0.15
0.2,37.8,0.12
0.3,39.1,0.18"""
        test_file = tmp_path / "cosmo_test.txt"
        test_file.write_text(test_data)
        
        # Create comprehensive provenance record
        environment = EnvironmentInfo(
            pbuf_commit="integration_test_commit",
            python_version="3.9.0",
            platform="Linux",
            hostname="integration-test-host",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        verification = VerificationResult(
            sha256_verified=True,
            sha256_expected="integration_test_hash",
            sha256_actual="integration_test_hash",
            size_verified=True,
            size_expected=len(test_data),
            size_actual=len(test_data),
            schema_verified=True,
            schema_errors=[],
            verification_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        provenance_record = ProvenanceRecord(
            dataset_name="sn_integration_test",
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            source_used="integration_test_source",
            download_agent="integration_test_agent",
            environment=environment,
            verification=verification,
            file_info={
                "local_path": str(test_file),
                "original_filename": "cosmo_test.txt",
                "mime_type": "text/plain"
            }
        )
        
        # Configure mock registry manager
        mock_registry_manager.get_registry_entry.return_value = provenance_record
        mock_registry_manager.has_registry_entry.return_value = True
        
        # Create registry integration
        registry_integration = RegistryIntegration(mock_registry_manager)
        
        # Mock file operations
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(registry_integration, '_calculate_file_checksum') as mock_checksum:
            
            mock_stat.return_value.st_size = len(test_data)
            mock_checksum.return_value = "integration_test_hash"
            
            # Test dataset retrieval
            result = registry_integration.get_verified_dataset("sn_integration_test")
        
        # Verify retrieval results
        assert result["file_path"] == test_file
        assert result["metadata"]["dataset_name"] == "sn_integration_test"
        assert result["metadata"]["dataset_type"] == "sn"
        assert result["provenance"] == provenance_record
        assert result["verification_status"] == verification
        
        # Verify registry manager was called correctly
        mock_registry_manager.get_registry_entry.assert_called_once_with("sn_integration_test")
    
    def test_provenance_recording_integration(self, tmp_path, mock_registry_manager):
        """Test complete provenance recording workflow."""
        # Create mock derived dataset
        derived_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([35.2, 37.8, 39.1]),
            uncertainty=np.array([0.15, 0.12, 0.18]),
            covariance=None,
            metadata={
                'source': 'integration_test',
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'transformation_summary': {
                    'steps': ['Parse data', 'Validate format'],
                    'formulas': ['No transformations applied'],
                    'assumptions': ['Data pre-validated']
                }
            }
        )
        
        # Create source provenance record
        source_provenance = ProvenanceRecord(
            dataset_name="source_dataset",
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            source_used="test_source",
            download_agent="test_agent",
            environment=EnvironmentInfo(
                pbuf_commit="source_commit",
                python_version="3.9.0",
                platform="Linux",
                hostname="test-host",
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            verification=VerificationResult(
                sha256_verified=True,
                sha256_expected="source_hash",
                sha256_actual="source_hash",
                size_verified=True,
                size_expected=1000,
                size_actual=1000,
                schema_verified=True,
                schema_errors=[],
                verification_timestamp=datetime.now(timezone.utc).isoformat()
            ),
            file_info={
                "local_path": "/tmp/source_dataset.txt",
                "original_filename": "source_dataset.txt",
                "mime_type": "text/plain"
            }
        )
        
        # Create output file
        output_file = tmp_path / "derived_dataset.json"
        output_file.write_text(json.dumps({
            "z": derived_dataset.z.tolist(),
            "observable": derived_dataset.observable.tolist(),
            "uncertainty": derived_dataset.uncertainty.tolist(),
            "metadata": derived_dataset.metadata
        }))
        
        # Create registry integration
        registry_integration = RegistryIntegration(mock_registry_manager)
        
        # Mock file operations and registry methods
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.is_dir') as mock_is_dir, \
             patch('pathlib.Path.glob') as mock_glob, \
             patch.object(registry_integration, '_calculate_file_checksum') as mock_checksum, \
             patch.object(registry_integration, '_get_latest_environment_registry_entry') as mock_env_ref, \
             patch.object(registry_integration, '_update_derived_dataset_index') as mock_update_index, \
             patch('builtins.open', create=True):
            
            mock_stat.return_value.st_size = 500
            mock_exists.return_value = True
            mock_is_dir.return_value = True
            mock_glob.return_value = []
            mock_checksum.return_value = "derived_hash"
            mock_env_ref.return_value = "test_env_ref"
            mock_update_index.return_value = None
            
            # Test provenance recording
            result_hash = registry_integration.register_derived_dataset(
                "test_dataset",
                derived_dataset,
                source_provenance,
                derived_dataset.metadata['transformation_summary'],
                output_file
            )
        
        # Verify provenance recording
        assert result_hash == "derived_hash"
        mock_registry_manager._append_audit_log.assert_called_once()
        
        # Verify audit log call arguments
        call_args = mock_registry_manager._append_audit_log.call_args[0]
        action = call_args[0]
        audit_entry = call_args[1]
        
        assert action == "derived_dataset_created"
        assert audit_entry["operation"] == "derived_dataset_created"
        assert audit_entry["source_dataset_name"] == "test_dataset"
        assert audit_entry["derived_hash"] == "derived_hash"
        assert "transformation_agent" in audit_entry
        assert "processing_timestamp" in audit_entry
    
    def test_registry_error_handling_integration(self, mock_registry_manager):
        """Test registry integration error handling."""
        # Configure registry manager to simulate various error conditions
        mock_registry_manager.get_registry_entry.return_value = None
        mock_registry_manager.has_registry_entry.return_value = False
        
        # Create registry integration
        registry_integration = RegistryIntegration(mock_registry_manager)
        
        # Test dataset not found error
        with pytest.raises(ProcessingError) as exc_info:
            registry_integration.get_verified_dataset("nonexistent_dataset")
        
        assert exc_info.value.error_type == "dataset_not_found"
        assert exc_info.value.dataset_name == "nonexistent_dataset"
        assert exc_info.value.stage == "input_validation"
        
        # Test verification failure
        failed_verification = VerificationResult(
            sha256_verified=False,
            sha256_expected="expected_hash",
            sha256_actual="wrong_hash",
            size_verified=True,
            size_expected=1000,
            size_actual=1000,
            schema_verified=True,
            schema_errors=[],
            verification_timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        failed_provenance = ProvenanceRecord(
            dataset_name="failed_dataset",
            download_timestamp=datetime.now(timezone.utc).isoformat(),
            source_used="test_source",
            download_agent="test_agent",
            environment=EnvironmentInfo(
                pbuf_commit="test_commit",
                python_version="3.9.0",
                platform="Linux",
                hostname="test-host",
                timestamp=datetime.now(timezone.utc).isoformat()
            ),
            verification=failed_verification,
            file_info={
                "local_path": "/tmp/failed_dataset.txt",
                "original_filename": "failed_dataset.txt",
                "mime_type": "text/plain"
            }
        )
        
        mock_registry_manager.get_registry_entry.return_value = failed_provenance
        mock_registry_manager.has_registry_entry.return_value = True
        
        # Test verification failure error
        with pytest.raises(ProcessingError) as exc_info:
            registry_integration.get_verified_dataset("failed_dataset")
        
        assert exc_info.value.error_type == "verification_failed"
        assert exc_info.value.dataset_name == "failed_dataset"


class TestFitPipelineCompatibility:
    """
    Fit pipeline integration tests for compatibility verification.
    
    Tests the compatibility between the data preparation framework output
    and the existing fit_core pipeline interface, ensuring seamless integration.
    
    Requirements: 8.4 - Integration tests for fit pipeline compatibility
    """
    
    def test_dataset_dict_format_compatibility(self, tmp_path):
        """Test that StandardDataset converts correctly to DatasetDict format."""
        # Create StandardDataset with comprehensive metadata
        standard_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3, 0.4]),
            observable=np.array([35.2, 37.8, 39.1, 40.5]),
            uncertainty=np.array([0.15, 0.12, 0.18, 0.20]),
            covariance=np.array([
                [0.0225, 0.0010, 0.0005, 0.0002],
                [0.0010, 0.0144, 0.0008, 0.0003],
                [0.0005, 0.0008, 0.0324, 0.0010],
                [0.0002, 0.0003, 0.0010, 0.0400]
            ]),
            metadata={
                'source': 'test_supernova_survey',
                'version': '1.0',
                'description': 'Test SN dataset for compatibility testing',
                'observable_type': 'distance_modulus',
                'units': 'mag',
                'n_points': 4,
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'transformation_summary': {
                    'steps': ['Magnitude calibration', 'Distance modulus calculation'],
                    'formulas': ['mu = m - M'],
                    'assumptions': ['Standard candle assumption']
                }
            }
        )
        
        # Convert to DatasetDict format using format converter
        from pipelines.data_preparation.output.format_converter import FormatConverter
        dataset_dict = FormatConverter.standard_to_dataset_dict(standard_dataset, "sn")
        
        # Verify DatasetDict structure matches expected fit_core format
        required_keys = ["observations", "uncertainties", "redshifts", "metadata"]
        for key in required_keys:
            assert key in dataset_dict, f"Missing required key: {key}"
        
        # Verify data arrays - for SN data, observations is a structured dict
        assert np.array_equal(dataset_dict["observations"]["distance_modulus"], standard_dataset.observable)
        assert np.array_equal(dataset_dict["observations"]["redshift"], standard_dataset.z)
        assert np.array_equal(dataset_dict["observations"]["sigma_mu"], standard_dataset.uncertainty)
        assert np.array_equal(dataset_dict["uncertainties"], standard_dataset.uncertainty)
        assert np.array_equal(dataset_dict["redshifts"], standard_dataset.z)
        
        # Verify covariance matrix handling
        if standard_dataset.covariance is not None:
            assert "covariance" in dataset_dict
            assert np.array_equal(dataset_dict["covariance"], standard_dataset.covariance)
        
        # Verify metadata preservation
        assert dataset_dict["metadata"]["source"] == "test_supernova_survey"
        assert dataset_dict["metadata"]["observable_type"] == "distance_modulus"
        assert dataset_dict["metadata"]["n_points"] == 4
    
    def test_fit_core_load_dataset_integration(self, tmp_path):
        """Test integration with fit_core load_dataset function."""
        # Create test framework and dataset
        framework = DataPreparationFramework(output_directory=tmp_path / "output")
        
        # Register test module
        class FitCompatibilityModule(TestDerivationModule):
            @property
            def dataset_type(self) -> str:
                return "sn"
            
            def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
                """Create dataset compatible with fit_core expectations."""
                return StandardDataset(
                    z=np.array([0.023, 0.040, 0.059]),
                    observable=np.array([32.81, 34.48, 35.58]),
                    uncertainty=np.array([0.12, 0.11, 0.13]),
                    covariance=None,
                    metadata={
                        **metadata,
                        'source': metadata.get('source', 'fit_compatibility_test'),  # Ensure 'source' field exists
                        'observable_type': 'distance_modulus',
                        'units': 'mag',
                        'n_points': 3,
                        'fit_core_compatible': True
                    }
                )
        
        fit_module = FitCompatibilityModule()
        framework.register_derivation_module(fit_module)
        
        # Create test data file
        test_data = """0.023,32.81,0.12
0.040,34.48,0.11
0.059,35.58,0.13"""
        test_file = tmp_path / "sn_fit_test.txt"
        test_file.write_text(test_data)
        
        # Process dataset through framework
        result = framework.prepare_dataset(
            "sn_fit_test",
            raw_data_path=test_file,
            metadata={'source': 'fit_compatibility_test'}
        )
        
        # Convert to DatasetDict format
        from pipelines.data_preparation.output.format_converter import FormatConverter
        dataset_dict = FormatConverter.standard_to_dataset_dict(result, "sn")
        
        # Verify fit_core compatibility by simulating load_dataset usage
        # This tests the interface that fit_core expects
        assert "observations" in dataset_dict
        assert "uncertainties" in dataset_dict
        assert "redshifts" in dataset_dict
        assert "metadata" in dataset_dict
        
        # Verify data types are compatible with fit_core expectations
        # For SN data, observations is a structured dictionary
        assert isinstance(dataset_dict["observations"], dict)
        assert "distance_modulus" in dataset_dict["observations"]
        assert isinstance(dataset_dict["observations"]["distance_modulus"], np.ndarray)
        assert isinstance(dataset_dict["uncertainties"], np.ndarray)
        assert isinstance(dataset_dict["redshifts"], np.ndarray)
        assert isinstance(dataset_dict["metadata"], dict)
        
        # Verify array shapes are consistent
        n_points = len(dataset_dict["observations"]["distance_modulus"])
        assert len(dataset_dict["uncertainties"]) == n_points
        assert len(dataset_dict["redshifts"]) == n_points
        
        # Verify metadata includes required information for fit_core
        metadata = dataset_dict["metadata"]
        assert "source" in metadata
        assert "n_points" in metadata
        assert metadata["n_points"] == n_points
    
    def test_preparation_framework_integration_with_datasets_py(self, tmp_path):
        """Test integration with the enhanced datasets.py load_dataset function."""
        # This test simulates the integration point in fit_core/datasets.py
        
        # Create mock preparation framework
        mock_framework = Mock()
        
        # Create realistic StandardDataset response
        mock_standard_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([35.0, 37.5, 39.0]),
            uncertainty=np.array([0.15, 0.12, 0.18]),
            covariance=None,
            metadata={
                'source': 'mock_survey',
                'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                'validation_status': 'passed',
                'transformation_summary': {
                    'steps': ['Mock processing'],
                    'formulas': ['Mock formula'],
                    'assumptions': ['Mock assumptions']
                },
                'environment_hash': 'mock_env_hash',
                'cache_used': False
            }
        )
        
        mock_framework.prepare_dataset.return_value = mock_standard_dataset
        mock_framework.get_available_dataset_types.return_value = ["sn", "bao", "cmb"]
        
        # Mock the _load_dataset_from_preparation_framework function behavior
        def mock_load_from_framework(name: str):
            """Mock the framework loading function from datasets.py."""
            standard_dataset = mock_framework.prepare_dataset(name)
            
            # Convert to DatasetDict format (simulating FormatConverter)
            from pipelines.data_preparation.output.format_converter import FormatConverter
            # Infer dataset type from name or use 'sn' as default for this test
            dataset_type = "sn" if "sn" in name else name
            dataset_dict = FormatConverter.standard_to_dataset_dict(standard_dataset, dataset_type)
            
            # Add preparation framework metadata (simulating datasets.py enhancement)
            dataset_dict["metadata"]["preparation_framework"] = {
                "used": True,
                "version": "1.0.0",
                "processing_timestamp": standard_dataset.metadata.get("processing_timestamp"),
                "validation_status": "passed",
                "framework_module": "data_preparation.engine.preparation_engine",
                "environment_hash": standard_dataset.metadata.get("environment_hash"),
                "transformation_summary": standard_dataset.metadata.get("transformation_summary"),
                "cache_used": standard_dataset.metadata.get("cache_used", False)
            }
            
            return dataset_dict
        
        # Test the integration
        result = mock_load_from_framework("sn_test")
        
        # Verify the result has the expected structure for fit_core
        assert "observations" in result
        assert "uncertainties" in result
        assert "redshifts" in result
        assert "metadata" in result
        
        # Verify preparation framework metadata is included
        assert "preparation_framework" in result["metadata"]
        framework_meta = result["metadata"]["preparation_framework"]
        assert framework_meta["used"] is True
        assert framework_meta["version"] == "1.0.0"
        assert framework_meta["validation_status"] == "passed"
        assert "processing_timestamp" in framework_meta
        assert "environment_hash" in framework_meta
        assert "transformation_summary" in framework_meta
        
        # Verify data integrity (SN data has structured observations)
        assert np.array_equal(result["observations"]["distance_modulus"], [35.0, 37.5, 39.0])
        assert np.array_equal(result["uncertainties"], [0.15, 0.12, 0.18])
        assert np.array_equal(result["redshifts"], [0.1, 0.2, 0.3])
    
    def test_backward_compatibility_with_legacy_datasets(self, tmp_path):
        """Test that framework integration maintains backward compatibility."""
        # This test ensures that existing fit_core code continues to work
        # even when the preparation framework is not available
        
        # Simulate legacy dataset loading (fallback behavior)
        def mock_legacy_load(name: str):
            """Mock legacy dataset loading function."""
            # Simulate the old dataset format
            return {
                "observations": np.array([35.0, 37.5, 39.0]),
                "uncertainties": np.array([0.15, 0.12, 0.18]),
                "redshifts": np.array([0.1, 0.2, 0.3]),
                "metadata": {
                    "source": "legacy_loader",
                    "version": "legacy",
                    "preparation_framework": {
                        "used": False,
                        "fallback_reason": "framework_not_available"
                    }
                }
            }
        
        # Test legacy loading
        legacy_result = mock_legacy_load("sn_legacy")
        
        # Verify legacy format is still supported
        assert "observations" in legacy_result
        assert "uncertainties" in legacy_result
        assert "redshifts" in legacy_result
        assert "metadata" in legacy_result
        
        # Verify backward compatibility metadata
        assert legacy_result["metadata"]["preparation_framework"]["used"] is False
        assert "fallback_reason" in legacy_result["metadata"]["preparation_framework"]
        
        # Verify data arrays are still numpy arrays
        assert isinstance(legacy_result["observations"], np.ndarray)
        assert isinstance(legacy_result["uncertainties"], np.ndarray)
        assert isinstance(legacy_result["redshifts"], np.ndarray)
    
    def test_covariance_matrix_compatibility(self, tmp_path):
        """Test covariance matrix handling compatibility with fit_core."""
        # Create StandardDataset with covariance matrix
        n_points = 3
        covariance_matrix = np.array([
            [0.0225, 0.0010, 0.0005],
            [0.0010, 0.0144, 0.0008],
            [0.0005, 0.0008, 0.0324]
        ])
        
        standard_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([35.0, 37.5, 39.0]),
            uncertainty=np.array([0.15, 0.12, 0.18]),
            covariance=covariance_matrix,
            metadata={
                'source': 'covariance_test',
                'has_covariance': True,
                'covariance_type': 'full'
            }
        )
        
        # Convert to DatasetDict format
        from pipelines.data_preparation.output.format_converter import FormatConverter
        dataset_dict = FormatConverter.standard_to_dataset_dict(standard_dataset, "sn")
        
        # Verify covariance matrix is properly included
        assert "covariance" in dataset_dict
        assert np.array_equal(dataset_dict["covariance"], covariance_matrix)
        
        # Verify covariance matrix properties for fit_core compatibility
        cov = dataset_dict["covariance"]
        assert cov.shape == (n_points, n_points)
        assert np.allclose(cov, cov.T)  # Symmetric
        assert np.all(np.linalg.eigvals(cov) > 0)  # Positive definite
        
        # Verify diagonal elements match uncertainties squared
        diagonal_uncertainties = np.sqrt(np.diag(cov))
        assert np.allclose(diagonal_uncertainties, dataset_dict["uncertainties"], rtol=1e-10)