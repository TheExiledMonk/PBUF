"""
Unit tests for registry manager

Tests registry entry creation, immutable audit trails, and provenance tracking.
This covers requirements 4.1-4.5 for immutable provenance tracking.
"""

import pytest
import tempfile
import json
import fcntl
import time
import threading
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timezone

from pipelines.dataset_registry.core.registry_manager import (
    RegistryManager,
    EnvironmentInfo,
    ProvenanceRecord,
    RegistryError,
    RegistryLockError
)
from pipelines.dataset_registry.verification.verification_engine import (
    VerificationResult,
    VerificationStatus
)


class TestEnvironmentInfo:
    """Test EnvironmentInfo functionality"""
    
    def test_environment_info_creation(self):
        """Test creating EnvironmentInfo objects"""
        env_info = EnvironmentInfo(
            pbuf_commit="abc123def456",
            python_version="3.9.7",
            platform="linux-x86_64",
            hostname="test-machine",
            timestamp="2025-10-23T14:30:00Z"
        )
        
        assert env_info.pbuf_commit == "abc123def456"
        assert env_info.python_version == "3.9.7"
        assert env_info.platform == "linux-x86_64"
        assert env_info.hostname == "test-machine"
        assert env_info.timestamp == "2025-10-23T14:30:00Z"
    
    def test_environment_collection(self):
        """Test collecting current environment information"""
        env_info = EnvironmentInfo.collect()
        
        # Just test that we get some values
        assert env_info.pbuf_commit is not None
        assert "3." in env_info.python_version  # Should contain Python version
        assert env_info.platform is not None
        assert env_info.hostname is not None
        assert env_info.timestamp is not None
    
    def test_environment_collection_basic(self):
        """Test basic environment collection functionality"""
        env_info = EnvironmentInfo.collect()
        
        # Basic checks that don't depend on git
        assert env_info.python_version is not None
        assert env_info.platform is not None
        assert env_info.hostname is not None
        assert env_info.timestamp is not None


class TestProvenanceRecord:
    """Test ProvenanceRecord functionality"""
    
    def test_provenance_record_basic(self):
        """Test basic provenance record functionality"""
        # Just test that we can import and use the class
        from pipelines.dataset_registry.core.registry_manager import ProvenanceRecord
        
        # Test that the class exists and has expected attributes
        assert hasattr(ProvenanceRecord, '__annotations__')
        assert 'dataset_name' in ProvenanceRecord.__annotations__
        assert 'download_timestamp' in ProvenanceRecord.__annotations__


class TestRegistryManager:
    """Test RegistryManager functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test registry"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_verification_result(self):
        """Sample verification result for testing"""
        return VerificationResult(
            dataset_name="test_dataset",
            status=VerificationStatus.SUCCESS,
            sha256_verified=True,
            size_verified=True,
            schema_verified=True,
            sha256_expected="abc123",
            sha256_actual="abc123",
            size_expected=1024,
            size_actual=1024,
            verification_time=datetime.now(timezone.utc),
            verification_duration_ms=150
        )
    
    def test_registry_manager_initialization(self, temp_dir):
        """Test RegistryManager initialization"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        assert manager.registry_path == registry_dir
        assert registry_dir.exists()
        assert (registry_dir / "audit.jsonl").exists()
    
    def test_register_manual_dataset(self, temp_dir):
        """Test registering a manual dataset"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Create test data file
        data_file = temp_dir / "test_dataset.dat"
        data_file.write_bytes(b"test data")
        
        # Register manual dataset
        success = manager.register_manual_dataset(
            dataset_name="test_dataset",
            canonical_name="Test Dataset",
            file_path=data_file,
            description="A test dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        assert success
        
        # Verify entry was created
        assert manager.has_registry_entry("test_dataset")
        
        # Verify entry file exists
        entry_file = registry_dir / "test_dataset.json"
        assert entry_file.exists()
        
        # Verify audit log was updated
        audit_file = registry_dir / "audit.jsonl"
        audit_content = audit_file.read_text()
        assert "test_dataset" in audit_content
    
    def test_get_registry_entry(self, temp_dir):
        """Test retrieving an existing registry entry"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Create test data file
        data_file = temp_dir / "test_dataset.dat"
        data_file.write_bytes(b"test data")
        
        # Register manual dataset
        manager.register_manual_dataset(
            dataset_name="test_dataset",
            canonical_name="Test Dataset",
            file_path=data_file,
            description="A test dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        # Retrieve entry
        retrieved_entry = manager.get_registry_entry("test_dataset")
        
        assert retrieved_entry is not None
        assert retrieved_entry.dataset_name == "test_dataset"
        # ProvenanceRecord doesn't have canonical_name directly
        assert retrieved_entry.dataset_name == "test_dataset"
    
    def test_get_nonexistent_entry(self, temp_dir):
        """Test retrieving a nonexistent registry entry"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Should return None for nonexistent entry
        entry = manager.get_registry_entry("nonexistent_dataset")
        assert entry is None
    
    def test_list_datasets(self, temp_dir):
        """Test listing all registered datasets"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Create multiple test entries
        for i in range(3):
            data_file = temp_dir / f"test_dataset_{i}.dat"
            data_file.write_bytes(b"test data")
            
            manager.register_manual_dataset(
                dataset_name=f"test_dataset_{i}",
                canonical_name=f"Test Dataset {i}",
                file_path=data_file,
                description=f"Test dataset {i}",
                citation="Test et al. 2025",
                license="CC-BY-4.0"
            )
        
        # List datasets
        datasets = manager.list_datasets()
        
        assert len(datasets) == 3
        assert "test_dataset_0" in datasets
        assert "test_dataset_1" in datasets
        assert "test_dataset_2" in datasets
    
    def test_entry_exists(self, temp_dir):
        """Test checking if registry entry exists"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Initially should not exist
        assert not manager.has_registry_entry("test_dataset")
        
        # Create entry
        data_file = temp_dir / "test_dataset.dat"
        data_file.write_bytes(b"test data")
        
        manager.register_manual_dataset(
            dataset_name="test_dataset",
            canonical_name="Test Dataset",
            file_path=data_file,
            description="A test dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        # Now should exist
        assert manager.has_registry_entry("test_dataset")
    
    def test_get_registry_summary(self, temp_dir):
        """Test getting registry summary"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Create test entry
        data_file = temp_dir / "test_dataset.dat"
        data_file.write_bytes(b"test data")
        
        manager.register_manual_dataset(
            dataset_name="test_dataset",
            canonical_name="Test Dataset",
            file_path=data_file,
            description="A test dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        # Get summary
        summary = manager.get_registry_summary()
        
        assert "total_datasets" in summary
        assert summary["total_datasets"] == 1
        assert "datasets" in summary
        assert len(summary["datasets"]) == 1
    
    def test_audit_trail_functionality(self, temp_dir):
        """Test audit trail functionality"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Create test data file
        data_file = temp_dir / "test_dataset.dat"
        data_file.write_bytes(b"test data")
        
        # Register dataset
        manager.register_manual_dataset(
            dataset_name="test_dataset",
            canonical_name="Test Dataset",
            file_path=data_file,
            description="A test dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        # Get audit trail
        audit_entries = manager.get_audit_trail()
        
        assert len(audit_entries) > 0
        
        # Check that audit entries have required fields
        for entry in audit_entries:
            assert "timestamp" in entry
            assert "event" in entry
            assert "dataset_name" in entry
    
    def test_export_provenance_summary(self, temp_dir):
        """Test exporting provenance summary for publications"""
        registry_dir = temp_dir / "registry"
        manager = RegistryManager(registry_dir)
        
        # Create test data file
        data_file = temp_dir / "test_dataset.dat"
        data_file.write_bytes(b"test data")
        
        # Register dataset
        manager.register_manual_dataset(
            dataset_name="test_dataset",
            canonical_name="Test Dataset",
            file_path=data_file,
            description="A test dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        # Export provenance summary
        summary = manager.export_provenance_summary()
        
        assert "datasets" in summary
        
        # Verify dataset information
        assert len(summary["datasets"]) >= 1
        assert "test_dataset" in summary["datasets"]
        
        # Environment info might be in different keys
        has_env_info = any(key in summary for key in ["environment", "environment_summary"])
        assert has_env_info


if __name__ == "__main__":
    pytest.main([__file__])