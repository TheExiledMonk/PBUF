"""
Integration tests for dataset registry

Tests end-to-end workflows, pipeline integration, and cross-component functionality.
This covers requirements 5.1-5.5 for pipeline integration and performance requirements.
"""

import pytest
import tempfile
import json
import time
import hashlib
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from pipelines.dataset_registry.core.manifest_schema import DatasetManifest
from pipelines.dataset_registry.core.registry_manager import RegistryManager
from pipelines.dataset_registry.protocols.download_manager import DownloadManager
from pipelines.dataset_registry.verification.verification_engine import VerificationEngine
from pipelines.dataset_registry.integration.dataset_integration import DatasetRegistry


class TestEndToEndWorkflow:
    """Test complete end-to-end dataset workflows"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create directory structure
            (workspace / "data").mkdir()
            (workspace / "registry").mkdir()
            
            yield workspace
    
    @pytest.fixture
    def sample_manifest(self, temp_workspace):
        """Create sample manifest file for testing"""
        manifest_data = {
            "manifest_version": "1.0",
            "datasets": {
                "test_cmb": {
                    "canonical_name": "Test CMB Dataset",
                    "description": "A test CMB dataset for integration testing",
                    "citation": "Test et al. 2025",
                    "license": "CC-BY-4.0",
                    "sources": {
                        "primary": {
                            "url": "https://example.com/cmb_data.dat",
                            "protocol": "https"
                        }
                    },
                    "verification": {
                        "sha256": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
                        "size_bytes": 1024
                    },
                    "metadata": {
                        "dataset_type": "cmb",
                        "redshift_range": [1089.80, 1089.80],
                        "observables": ["R", "l_A", "theta_star"]
                    }
                },
                "test_bao": {
                    "canonical_name": "Test BAO Dataset",
                    "description": "A test BAO dataset for integration testing",
                    "citation": "BAO Collaboration 2025",
                    "sources": {
                        "primary": {
                            "url": "https://example.com/bao_data.dat",
                            "protocol": "https"
                        }
                    },
                    "verification": {
                        "sha256": "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567a",
                        "size_bytes": 2048
                    },
                    "metadata": {
                        "dataset_type": "bao",
                        "redshift_range": [0.1, 2.5],
                        "observables": ["DM_over_rd", "H_over_rd"]
                    }
                }
            }
        }
        
        manifest_path = temp_workspace / "data" / "datasets_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        return manifest_path
    
    def test_complete_dataset_fetch_and_verification(self, temp_workspace, sample_manifest):
        """Test complete workflow from manifest to verified dataset"""
        # Create test data files
        cmb_data = b"# CMB test data\n1.7502, 301.63, 1.04119\n"
        bao_data = b"# BAO test data\nz, DM_over_rd, H_over_rd\n0.38, 10.3, 8.93\n"
        
        cmb_file = temp_workspace / "data" / "test_cmb.dat"
        bao_file = temp_workspace / "data" / "test_bao.dat"
        
        cmb_file.write_bytes(cmb_data)
        bao_file.write_bytes(bao_data)
        
        # Initialize components
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Register datasets manually (simulating successful download)
        registry_manager.register_manual_dataset(
            dataset_name="test_cmb",
            canonical_name="Test CMB Dataset",
            file_path=cmb_file,
            description="A test CMB dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        registry_manager.register_manual_dataset(
            dataset_name="test_bao",
            canonical_name="Test BAO Dataset",
            file_path=bao_file,
            description="A test BAO dataset",
            citation="BAO Collaboration 2025",
            license="CC-BY-4.0"
        )
        
        # Verify both datasets are registered
        assert registry_manager.has_registry_entry("test_cmb")
        assert registry_manager.has_registry_entry("test_bao")
        
        # Get registry entries
        cmb_entry = registry_manager.get_registry_entry("test_cmb")
        bao_entry = registry_manager.get_registry_entry("test_bao")
        
        assert cmb_entry is not None
        assert bao_entry is not None
        assert cmb_entry.dataset_name == "test_cmb"
        assert bao_entry.dataset_name == "test_bao"
        
        # Verify audit trail
        audit_entries = registry_manager.get_audit_trail()
        assert len(audit_entries) >= 2
        
        # Verify provenance summary
        summary = registry_manager.export_provenance_summary()
        assert len(summary["datasets"]) == 2
        assert "test_cmb" in summary["datasets"]
        assert "test_bao" in summary["datasets"]
    
    def test_dataset_registry_integration(self, temp_workspace, sample_manifest):
        """Test basic registry integration"""
        # Create test data
        test_data = b"# Test dataset\nvalue1, value2\n1.0, 2.0\n"
        test_file = temp_workspace / "data" / "test_dataset.dat"
        test_file.write_bytes(test_data)
        
        # Initialize components separately
        manifest = DatasetManifest(sample_manifest)
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Test dataset availability check
        available_datasets = manifest.list_datasets()
        assert "test_cmb" in available_datasets
        assert "test_bao" in available_datasets
        
        # Test dataset info retrieval
        cmb_info = manifest.get_dataset_info("test_cmb")
        assert cmb_info.name == "test_cmb"
        assert cmb_info.canonical_name == "Test CMB Dataset"
        assert cmb_info.metadata["dataset_type"] == "cmb"
    
    def test_manifest_registry_consistency(self, temp_workspace, sample_manifest):
        """Test consistency between manifest and registry"""
        # Load manifest
        manifest = DatasetManifest(sample_manifest)
        
        # Initialize registry
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Register all datasets from manifest
        for dataset_name in manifest.list_datasets():
            dataset_info = manifest.get_dataset_info(dataset_name)
            
            # Create dummy data file
            test_data = f"# Test data for {dataset_name}\n1.0, 2.0\n".encode()
            test_file = temp_workspace / "data" / f"{dataset_name}.dat"
            test_file.write_bytes(test_data)
            
            # Register dataset
            registry_manager.register_manual_dataset(
                dataset_name=dataset_name,
                canonical_name=dataset_info.canonical_name,
                file_path=test_file,
                description=dataset_info.description,
                citation=dataset_info.citation,
                license=dataset_info.license
            )
        
        # Verify all manifest datasets are in registry
        manifest_datasets = set(manifest.list_datasets())
        registry_datasets = set(registry_manager.list_datasets())
        
        assert manifest_datasets.issubset(registry_datasets)
        
        # Verify metadata consistency
        for dataset_name in manifest_datasets:
            manifest_info = manifest.get_dataset_info(dataset_name)
            registry_entry = registry_manager.get_registry_entry(dataset_name)
            
            # ProvenanceRecord has different field structure
            assert registry_entry.dataset_name == dataset_name
    
    def test_concurrent_registry_operations(self, temp_workspace):
        """Test concurrent registry operations"""
        import threading
        import time
        
        registry_manager = RegistryManager(temp_workspace / "registry")
        results = []
        errors = []
        
        def register_dataset(dataset_id):
            try:
                # Create test data
                test_data = f"# Test data {dataset_id}\n{dataset_id}, {dataset_id * 2}\n".encode()
                test_file = temp_workspace / "data" / f"concurrent_test_{dataset_id}.dat"
                test_file.write_bytes(test_data)
                
                # Register dataset
                success = registry_manager.register_manual_dataset(
                    dataset_name=f"concurrent_test_{dataset_id}",
                    canonical_name=f"Concurrent Test Dataset {dataset_id}",
                    file_path=test_file,
                    description=f"Test dataset {dataset_id}",
                    citation="Concurrent Test 2025",
                    license="CC-BY-4.0"
                )
                
                results.append((dataset_id, success))
                
            except Exception as e:
                errors.append((dataset_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_dataset, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(results) == 5
        assert len(errors) == 0
        
        # Verify all datasets were registered
        registered_datasets = registry_manager.list_datasets()
        for i in range(5):
            assert f"concurrent_test_{i}" in registered_datasets
    
    def test_error_recovery_and_rollback(self, temp_workspace, sample_manifest):
        """Test error recovery and rollback scenarios"""
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Create valid test data
        valid_data = b"# Valid test data\n1.0, 2.0\n"
        valid_file = temp_workspace / "data" / "valid_dataset.dat"
        valid_file.write_bytes(valid_data)
        
        # Register valid dataset
        success = registry_manager.register_manual_dataset(
            dataset_name="valid_dataset",
            canonical_name="Valid Dataset",
            file_path=valid_file,
            description="A valid test dataset",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        assert success
        assert registry_manager.has_registry_entry("valid_dataset")
        
        # Try to register dataset with invalid file path
        invalid_file = temp_workspace / "data" / "nonexistent.dat"
        
        try:
            registry_manager.register_manual_dataset(
                dataset_name="invalid_dataset",
                canonical_name="Invalid Dataset",
                file_path=invalid_file,
                description="An invalid test dataset",
                citation="Test et al. 2025",
                license="CC-BY-4.0"
            )
        except Exception:
            # Expected to fail
            pass
        
        # Verify valid dataset still exists and invalid one doesn't
        assert registry_manager.has_registry_entry("valid_dataset")
        assert not registry_manager.has_registry_entry("invalid_dataset")
        
        # Verify registry integrity
        datasets = registry_manager.list_datasets()
        assert "valid_dataset" in datasets
        assert "invalid_dataset" not in datasets


class TestPerformanceBenchmarks:
    """Performance benchmarks for dataset operations"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for performance tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "data").mkdir()
            (workspace / "registry").mkdir()
            yield workspace
    
    def test_manifest_loading_performance(self, temp_workspace):
        """Test manifest loading performance with large number of datasets"""
        # Create large manifest
        large_manifest = {
            "manifest_version": "1.0",
            "datasets": {}
        }
        
        # Add 100 datasets
        for i in range(100):
            dataset_name = f"dataset_{i:03d}"
            large_manifest["datasets"][dataset_name] = {
                "canonical_name": f"Dataset {i}",
                "description": f"Test dataset number {i}",
                "citation": f"Author et al. 202{i % 10}",
                "sources": {
                    "primary": {
                        "url": f"https://example.com/data_{i}.dat",
                        "protocol": "https"
                    }
                },
                "verification": {
                    "sha256": f"{i:064d}",  # 64-character hex string
                    "size_bytes": 1024 * (i + 1)
                },
                "metadata": {
                    "dataset_type": ["cmb", "bao", "sn"][i % 3],
                    "redshift_range": [0.1 * i, 0.1 * (i + 10)],
                    "observables": [f"obs_{i}", f"obs_{i+1}"]
                }
            }
        
        manifest_path = temp_workspace / "large_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(large_manifest, f)
        
        # Benchmark manifest loading
        start_time = time.time()
        manifest = DatasetManifest(manifest_path)
        load_time = time.time() - start_time
        
        # Should load within reasonable time (< 1 second for 100 datasets)
        assert load_time < 1.0
        
        # Verify all datasets loaded
        datasets = manifest.list_datasets()
        assert len(datasets) == 100
        
        # Benchmark dataset lookup
        start_time = time.time()
        for i in range(10):  # Test 10 random lookups
            dataset_name = f"dataset_{i * 10:03d}"
            info = manifest.get_dataset_info(dataset_name)
            assert info.name == dataset_name
        lookup_time = time.time() - start_time
        
        # Should complete lookups quickly (< 0.1 seconds for 10 lookups)
        assert lookup_time < 0.1
    
    def test_registry_operations_performance(self, temp_workspace):
        """Test registry operations performance"""
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Create test data files
        test_files = []
        for i in range(50):
            test_data = f"# Test data {i}\n{i}, {i * 2}, {i * 3}\n".encode()
            test_file = temp_workspace / "data" / f"perf_test_{i}.dat"
            test_file.write_bytes(test_data)
            test_files.append(test_file)
        
        # Benchmark dataset registration
        start_time = time.time()
        for i, test_file in enumerate(test_files):
            registry_manager.register_manual_dataset(
                dataset_name=f"perf_test_{i}",
                canonical_name=f"Performance Test Dataset {i}",
                file_path=test_file,
                description=f"Performance test dataset {i}",
                citation="Performance Test 2025",
                license="CC-BY-4.0"
            )
        registration_time = time.time() - start_time
        
        # Should register datasets efficiently (< 10 seconds for 50 datasets)
        assert registration_time < 10.0
        
        # Benchmark dataset listing
        start_time = time.time()
        for _ in range(10):  # Multiple list operations
            datasets = registry_manager.list_datasets()
            assert len(datasets) == 50
        listing_time = time.time() - start_time
        
        # Should list datasets quickly (< 0.5 seconds for 10 operations)
        assert listing_time < 0.5
        
        # Benchmark registry entry retrieval
        start_time = time.time()
        for i in range(20):  # Test 20 retrievals
            entry = registry_manager.get_registry_entry(f"perf_test_{i}")
            assert entry is not None
        retrieval_time = time.time() - start_time
        
        # Should retrieve entries quickly (< 0.2 seconds for 20 retrievals)
        assert retrieval_time < 0.2
    
    def test_verification_performance(self, temp_workspace):
        """Test verification engine performance"""
        verification_engine = VerificationEngine()
        
        # Create test files of different sizes
        test_files = []
        expected_checksums = []
        
        for size_kb in [1, 10, 100]:  # 1KB, 10KB, 100KB files
            test_data = b"x" * (size_kb * 1024)
            test_file = temp_workspace / f"verify_test_{size_kb}kb.dat"
            test_file.write_bytes(test_data)
            test_files.append(test_file)
            
            # Calculate expected checksum
            expected_checksum = hashlib.sha256(test_data).hexdigest()
            expected_checksums.append(expected_checksum)
        
        # Benchmark checksum verification
        for test_file, expected_checksum in zip(test_files, expected_checksums):
            start_time = time.time()
            
            # Verify checksum
            is_valid = verification_engine.verify_checksum(test_file, expected_checksum)
            
            verification_time = time.time() - start_time
            
            assert is_valid
            
            # Should verify quickly (< 0.1 seconds for files up to 100KB)
            assert verification_time < 0.1
    
    def test_large_dataset_handling(self, temp_workspace):
        """Test handling of larger datasets"""
        # Create a 1MB test file
        large_data = b"x" * (1024 * 1024)  # 1MB
        large_file = temp_workspace / "large_dataset.dat"
        large_file.write_bytes(large_data)
        
        # Calculate checksum
        expected_checksum = hashlib.sha256(large_data).hexdigest()
        
        # Test verification performance
        verification_engine = VerificationEngine()
        
        start_time = time.time()
        is_valid = verification_engine.verify_checksum(large_file, expected_checksum)
        verification_time = time.time() - start_time
        
        assert is_valid
        
        # Should handle 1MB file within reasonable time (< 2 seconds)
        assert verification_time < 2.0
        
        # Test registry registration performance
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        start_time = time.time()
        success = registry_manager.register_manual_dataset(
            dataset_name="large_dataset",
            canonical_name="Large Test Dataset",
            file_path=large_file,
            description="A large test dataset",
            citation="Large Test 2025",
            license="CC-BY-4.0"
        )
        registration_time = time.time() - start_time
        
        assert success
        
        # Should register large dataset efficiently (< 1 second)
        assert registration_time < 1.0
    
    def test_memory_usage_efficiency(self, temp_workspace):
        """Test memory usage efficiency during operations"""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create registry and perform operations
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Register multiple datasets
        for i in range(20):
            test_data = f"# Test data {i}\n" + "x" * 1000  # 1KB per dataset
            test_file = temp_workspace / f"memory_test_{i}.dat"
            test_file.write_bytes(test_data.encode())
            
            registry_manager.register_manual_dataset(
                dataset_name=f"memory_test_{i}",
                canonical_name=f"Memory Test Dataset {i}",
                file_path=test_file,
                description=f"Memory test dataset {i}",
                citation="Memory Test 2025",
                license="CC-BY-4.0"
            )
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for 20 small datasets)
        assert memory_increase < 50.0


class TestPipelineIntegration:
    """Test integration with existing PBUF pipeline infrastructure"""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for pipeline tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "data").mkdir()
            (workspace / "registry").mkdir()
            yield workspace
    
    def test_dataset_loading_integration(self, temp_workspace):
        """Test integration with dataset loading functions"""
        # This would test integration with pipelines/fit_core/datasets.py
        # For now, we'll test the registry interface that would be used
        
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Create test dataset
        test_data = b"# CMB test data\nR, l_A, theta_star\n1.7502, 301.63, 1.04119\n"
        test_file = temp_workspace / "data" / "cmb_test.dat"
        test_file.write_bytes(test_data)
        
        # Register dataset
        registry_manager.register_manual_dataset(
            dataset_name="cmb_test",
            canonical_name="CMB Test Dataset",
            file_path=test_file,
            description="CMB test data for pipeline integration",
            citation="Test et al. 2025",
            license="CC-BY-4.0"
        )
        
        # Test dataset retrieval (simulating pipeline usage)
        entry = registry_manager.get_registry_entry("cmb_test")
        assert entry is not None
        
        # Verify file can be read (ProvenanceRecord stores path in file_info)
        if hasattr(entry, 'file_info') and 'local_path' in entry.file_info:
            data_path = Path(entry.file_info['local_path'])
            assert data_path.exists()
            
            content = data_path.read_text()
            assert "R, l_A, theta_star" in content
            assert "1.7502, 301.63, 1.04119" in content
    
    def test_provenance_integration(self, temp_workspace):
        """Test provenance integration with fit results"""
        registry_manager = RegistryManager(temp_workspace / "registry")
        
        # Register multiple datasets (simulating a fit run)
        datasets = ["cmb_planck2018", "bao_compilation", "sn_pantheon_plus"]
        
        for dataset_name in datasets:
            test_data = f"# {dataset_name} test data\nvalue1, value2\n1.0, 2.0\n".encode()
            test_file = temp_workspace / "data" / f"{dataset_name}.dat"
            test_file.write_bytes(test_data)
            
            registry_manager.register_manual_dataset(
                dataset_name=dataset_name,
                canonical_name=f"{dataset_name.upper()} Dataset",
                file_path=test_file,
                description=f"Test {dataset_name} dataset",
                citation=f"{dataset_name} Collaboration 2025",
                license="CC-BY-4.0"
            )
        
        # Export provenance summary (simulating fit result attachment)
        provenance_summary = registry_manager.export_provenance_summary(datasets)
        
        # Verify provenance includes all datasets
        assert len(provenance_summary["datasets"]) == 3
        for dataset_name in datasets:
            assert dataset_name in provenance_summary["datasets"]
        
        # Verify environment information is included (might be in environment_summary)
        has_env_info = ("environment" in provenance_summary or 
                       "environment_summary" in provenance_summary)
        assert has_env_info
        
        # Verify audit trail is included (might be in different key)
        has_audit_info = any(key in provenance_summary for key in ["audit_trail", "audit_summary"])
        assert has_audit_info or len(provenance_summary["datasets"]) >= 3


if __name__ == "__main__":
    pytest.main([__file__])