"""
Tests for registry integration module.

This module tests the RegistryIntegration class functionality including
dataset retrieval, metadata extraction, and provenance tracking.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipelines.data_preparation.core.registry_integration import RegistryIntegration
from pipelines.data_preparation.core.interfaces import ProcessingError
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.dataset_registry.core.registry_manager import (
    RegistryManager, 
    ProvenanceRecord, 
    VerificationResult,
    EnvironmentInfo
)
import numpy as np


class TestRegistryIntegration:
    """Test cases for RegistryIntegration class."""
    
    @pytest.fixture
    def mock_registry_manager(self):
        """Create mock RegistryManager for testing."""
        mock_manager = Mock(spec=RegistryManager)
        mock_manager.registry_path = Path("/tmp/test_registry")
        return mock_manager
    
    @pytest.fixture
    def registry_integration(self, mock_registry_manager):
        """Create RegistryIntegration instance for testing."""
        return RegistryIntegration(mock_registry_manager)
    
    @pytest.fixture
    def sample_provenance_record(self):
        """Create sample provenance record for testing."""
        environment = EnvironmentInfo(
            pbuf_commit="abc123",
            python_version="3.9.0",
            platform="Linux",
            hostname="test-host",
            timestamp="2023-01-01T00:00:00Z"
        )
        
        verification = VerificationResult(
            sha256_verified=True,
            sha256_expected="test_hash",
            sha256_actual="test_hash",
            size_verified=True,
            size_expected=1000,
            size_actual=1000,
            schema_verified=True,
            schema_errors=[],
            verification_timestamp="2023-01-01T00:00:00Z"
        )
        
        return ProvenanceRecord(
            dataset_name="test_sn_dataset",
            download_timestamp="2023-01-01T00:00:00Z",
            source_used="test_source",
            download_agent="test_agent",
            environment=environment,
            verification=verification,
            file_info={
                "local_path": "/tmp/test_dataset.txt",
                "original_filename": "test_dataset.txt",
                "mime_type": "text/plain"
            }
        )
    
    def test_init(self, mock_registry_manager):
        """Test RegistryIntegration initialization."""
        integration = RegistryIntegration(mock_registry_manager)
        
        assert integration.registry_manager == mock_registry_manager
        assert integration._derived_datasets_path.name == "derived"
    
    def test_get_verified_dataset_success(self, registry_integration, sample_provenance_record):
        """Test successful dataset retrieval."""
        # Mock registry manager methods
        registry_integration.registry_manager.get_registry_entry.return_value = sample_provenance_record
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            result = registry_integration.get_verified_dataset("test_sn_dataset")
        
        assert result["file_path"] == Path("/tmp/test_dataset.txt")
        assert result["metadata"]["dataset_name"] == "test_sn_dataset"
        assert result["metadata"]["dataset_type"] == "sn"
        assert result["provenance"] == sample_provenance_record
    
    def test_get_verified_dataset_not_found(self, registry_integration):
        """Test dataset not found error."""
        registry_integration.registry_manager.get_registry_entry.return_value = None
        
        with pytest.raises(ProcessingError) as exc_info:
            registry_integration.get_verified_dataset("nonexistent_dataset")
        
        assert exc_info.value.error_type == "dataset_not_found"
        assert exc_info.value.dataset_name == "nonexistent_dataset"
    
    def test_get_verified_dataset_verification_failed(self, registry_integration, sample_provenance_record):
        """Test dataset with failed verification."""
        # Modify verification to fail
        sample_provenance_record.verification.sha256_verified = False
        registry_integration.registry_manager.get_registry_entry.return_value = sample_provenance_record
        
        with pytest.raises(ProcessingError) as exc_info:
            registry_integration.get_verified_dataset("test_sn_dataset")
        
        assert exc_info.value.error_type == "verification_failed"
    
    def test_infer_dataset_type(self, registry_integration):
        """Test dataset type inference from names."""
        test_cases = [
            ("sn_pantheon", "sn"),
            ("supernova_data", "sn"),
            ("bao_boss", "bao"),
            ("baryon_oscillations", "bao"),
            ("cmb_planck", "cmb"),
            ("cosmic_microwave_background", "cmb"),
            ("cc_hubble", "cc"),
            ("chronometer_data", "cc"),
            ("rsd_growth", "rsd"),
            ("redshift_space_distortions", "rsd"),
            ("unknown_dataset", None)
        ]
        
        for dataset_name, expected_type in test_cases:
            result = registry_integration._infer_dataset_type(dataset_name)
            assert result == expected_type
    
    def test_validate_raw_dataset_integrity_success(self, registry_integration, sample_provenance_record):
        """Test successful dataset integrity validation."""
        registry_integration.registry_manager.get_registry_entry.return_value = sample_provenance_record
        
        test_file = Path("/tmp/test_dataset.txt")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(registry_integration, '_calculate_file_checksum', return_value="test_hash"):
            
            mock_stat.return_value.st_size = 1000
            
            result = registry_integration.validate_raw_dataset_integrity("test_sn_dataset", test_file)
            assert result is True
    
    def test_validate_raw_dataset_integrity_file_not_found(self, registry_integration):
        """Test integrity validation with missing file."""
        test_file = Path("/tmp/nonexistent.txt")
        
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ProcessingError) as exc_info:
                registry_integration.validate_raw_dataset_integrity("test_dataset", test_file)
            
            assert exc_info.value.error_type == "file_not_found"
    
    def test_validate_raw_dataset_integrity_checksum_mismatch(self, registry_integration, sample_provenance_record):
        """Test integrity validation with checksum mismatch."""
        registry_integration.registry_manager.get_registry_entry.return_value = sample_provenance_record
        
        test_file = Path("/tmp/test_dataset.txt")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(registry_integration, '_calculate_file_checksum', return_value="wrong_hash"):
            
            mock_stat.return_value.st_size = 1000
            
            with pytest.raises(ProcessingError) as exc_info:
                registry_integration.validate_raw_dataset_integrity("test_sn_dataset", test_file)
            
            assert exc_info.value.error_type == "checksum_mismatch"
    
    def test_register_derived_dataset(self, registry_integration, sample_provenance_record):
        """Test derived dataset registration."""
        # Create mock StandardDataset
        mock_dataset = StandardDataset(
            z=np.array([0.1, 0.2, 0.3]),
            observable=np.array([1.0, 2.0, 3.0]),
            uncertainty=np.array([0.1, 0.1, 0.1]),
            covariance=None,
            metadata={"source": "test"}
        )
        
        transformation_summary = {"steps": ["test_transformation"]}
        output_file = Path("/tmp/derived_dataset.json")
        
        with patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.is_dir') as mock_is_dir, \
             patch('pathlib.Path.glob') as mock_glob, \
             patch.object(registry_integration, '_calculate_file_checksum', return_value="derived_hash"), \
             patch.object(registry_integration, '_get_latest_environment_registry_entry', return_value="test_env_ref"), \
             patch.object(registry_integration, '_update_derived_dataset_index'), \
             patch('builtins.open', create=True) as mock_open, \
             patch.object(registry_integration.registry_manager, '_append_audit_log'):
            
            mock_stat.return_value.st_size = 500
            mock_exists.return_value = True
            mock_is_dir.return_value = True
            mock_glob.return_value = []
            mock_open.return_value.__enter__.return_value = Mock()
            
            result = registry_integration.register_derived_dataset(
                "test_dataset",
                mock_dataset,
                sample_provenance_record,
                transformation_summary,
                output_file
            )
            
            assert result == "derived_hash"
    
    def test_get_dataset_processing_requirements(self, registry_integration, sample_provenance_record):
        """Test processing requirements extraction."""
        registry_integration.registry_manager.get_registry_entry.return_value = sample_provenance_record
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch.object(registry_integration, 'get_verified_dataset') as mock_get_verified:
            
            mock_get_verified.return_value = {
                "file_path": Path("/tmp/test_dataset.txt"),
                "metadata": {
                    "dataset_name": "test_sn_dataset",
                    "dataset_type": "sn",
                    "source_used": "test_source"
                },
                "provenance": sample_provenance_record,
                "verification_status": sample_provenance_record.verification
            }
            
            result = registry_integration.get_dataset_processing_requirements("test_sn_dataset")
            
            assert result["dataset_type"] == "sn"
            assert result["input_file_path"] == Path("/tmp/test_dataset.txt")
            assert "validation_config" in result
            assert "transformation_config" in result
    
    def test_check_registry_entry_status(self, registry_integration, sample_provenance_record):
        """Test registry entry status checking."""
        registry_integration.registry_manager.has_registry_entry.return_value = True
        registry_integration.registry_manager.get_registry_entry.return_value = sample_provenance_record
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            
            result = registry_integration.check_registry_entry_status("test_sn_dataset")
            
            assert result["exists"] is True
            assert result["verified"] is True
            assert result["accessible"] is True
            assert result["dataset_type"] == "sn"
            assert len(result["errors"]) == 0
    
    def test_extract_detailed_metadata(self, registry_integration, sample_provenance_record):
        """Test detailed metadata extraction."""
        with patch.object(registry_integration, 'get_verified_dataset') as mock_get_verified:
            mock_get_verified.return_value = {
                "file_path": Path("/tmp/test_dataset.txt"),
                "metadata": {
                    "dataset_name": "test_sn_dataset",
                    "dataset_type": "sn"
                },
                "provenance": sample_provenance_record,
                "verification_status": sample_provenance_record.verification
            }
            
            result = registry_integration.extract_detailed_metadata("test_sn_dataset")
            
            assert "basic_info" in result
            assert "file_info" in result
            assert "provenance" in result
            assert "processing_hints" in result
            assert "validation_params" in result
            assert result["basic_info"]["dataset_type"] == "sn"
    
    def test_validate_dataset_for_processing(self, registry_integration):
        """Test comprehensive dataset validation for processing."""
        with patch.object(registry_integration, 'check_registry_entry_status') as mock_status, \
             patch.object(registry_integration, 'extract_detailed_metadata') as mock_metadata, \
             patch.object(registry_integration, 'get_dataset_processing_requirements') as mock_reqs, \
             patch.object(registry_integration, '_validate_file_integrity') as mock_integrity, \
             patch.object(registry_integration, '_validate_file_format') as mock_format:
            
            # Mock successful status
            mock_status.return_value = {
                "exists": True,
                "verified": True,
                "accessible": True,
                "dataset_type": "sn",
                "errors": [],
                "warnings": []
            }
            
            # Mock successful metadata
            mock_metadata.return_value = {
                "basic_info": {"dataset_type": "sn"},
                "file_info": {"local_path": "/tmp/test.txt", "file_size": 1000},
                "provenance": {"pbuf_commit": "abc123"},
                "processing_hints": {"special_considerations": []}
            }
            
            # Mock successful requirements
            mock_reqs.return_value = {"dataset_type": "sn"}
            
            # Mock successful integrity
            mock_integrity.return_value = {"is_valid": True, "errors": []}
            
            # Mock successful format
            mock_format.return_value = {"is_compatible": True, "issues": []}
            
            result = registry_integration.validate_dataset_for_processing("test_sn_dataset")
            
            assert result["is_ready"] is True
            assert len(result["blocking_issues"]) == 0
    
    def test_generate_processing_report(self, registry_integration):
        """Test processing report generation."""
        with patch.object(registry_integration, 'check_registry_entry_status') as mock_status, \
             patch.object(registry_integration, 'extract_detailed_metadata') as mock_metadata, \
             patch.object(registry_integration, 'validate_dataset_for_processing') as mock_validate, \
             patch.object(registry_integration, 'get_dataset_processing_requirements') as mock_reqs:
            
            # Mock successful responses
            mock_status.return_value = {"exists": True}
            mock_metadata.return_value = {"basic_info": {"dataset_type": "sn"}}
            mock_validate.return_value = {"is_ready": True, "blocking_issues": []}
            mock_reqs.return_value = {"dataset_type": "sn"}
            
            result = registry_integration.generate_processing_report("test_sn_dataset")
            
            assert result["dataset_name"] == "test_sn_dataset"
            assert result["overall_status"] == "ready"
            assert result["ready_for_processing"] is True
            assert "sections" in result
            assert "registry_status" in result["sections"]
            assert "metadata" in result["sections"]
            assert "validation" in result["sections"]
            assert "processing_requirements" in result["sections"]


if __name__ == "__main__":
    pytest.main([__file__])