"""
Registry integration module for the PBUF data preparation framework.

This module provides the RegistryIntegration class that interfaces with the existing
RegistryManager to retrieve verified raw datasets and track provenance for derived datasets.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from pipelines.dataset_registry.core.registry_manager import (
    RegistryManager, 
    ProvenanceRecord, 
    VerificationResult,
    EnvironmentInfo
)
from .interfaces import ProcessingError
from .schema import StandardDataset


class RegistryIntegration:
    """
    Integration layer between data preparation framework and existing RegistryManager.
    
    This class provides methods to retrieve verified raw datasets, validate their
    integrity, and register derived datasets with complete provenance tracking.
    """
    
    def __init__(self, registry_manager: RegistryManager):
        """
        Initialize registry integration.
        
        Args:
            registry_manager: Existing RegistryManager instance
        """
        self.registry_manager = registry_manager
        self._derived_datasets_path = Path(registry_manager.registry_path) / "derived"
        self._derived_datasets_path.mkdir(parents=True, exist_ok=True)
    
    def get_verified_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Retrieve verified raw dataset and metadata from registry.
        
        Args:
            dataset_name: Name of the dataset to retrieve
            
        Returns:
            Dict containing:
                - file_path: Path to verified raw dataset file
                - metadata: Complete registry metadata
                - provenance: Source provenance record
                - verification_status: Verification results
                
        Raises:
            ProcessingError: If dataset not found or verification failed
        """
        # Get registry entry
        provenance_record = self.registry_manager.get_registry_entry(dataset_name)
        
        if provenance_record is None:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="input_validation",
                error_type="dataset_not_found",
                error_message=f"Dataset '{dataset_name}' not found in registry",
                suggested_actions=[
                    "Check dataset name spelling",
                    "Verify dataset has been downloaded and registered",
                    "Run registry summary to see available datasets"
                ]
            )
        
        # Check verification status
        if not provenance_record.verification.is_valid:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="input_validation", 
                error_type="verification_failed",
                error_message=f"Dataset '{dataset_name}' failed verification",
                context={
                    "sha256_verified": provenance_record.verification.sha256_verified,
                    "size_verified": provenance_record.verification.size_verified,
                    "schema_verified": provenance_record.verification.schema_verified,
                    "schema_errors": provenance_record.verification.schema_errors
                },
                suggested_actions=[
                    "Re-download the dataset",
                    "Check for file corruption",
                    "Verify dataset source integrity"
                ]
            )
        
        # Get file path
        file_path = Path(provenance_record.file_info["local_path"])
        
        if not file_path.exists():
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="input_validation",
                error_type="file_not_found", 
                error_message=f"Dataset file not found at {file_path}",
                suggested_actions=[
                    "Re-download the dataset",
                    "Check file system permissions",
                    "Verify registry entry is correct"
                ]
            )
        
        return {
            "file_path": file_path,
            "metadata": self._extract_dataset_metadata(provenance_record),
            "provenance": provenance_record,
            "verification_status": provenance_record.verification
        }
    
    def _extract_dataset_metadata(self, provenance_record: ProvenanceRecord) -> Dict[str, Any]:
        """
        Extract dataset metadata from provenance record.
        
        Args:
            provenance_record: Registry provenance record
            
        Returns:
            Dict containing extracted metadata
        """
        metadata = {
            "dataset_name": provenance_record.dataset_name,
            "source_used": provenance_record.source_used,
            "download_timestamp": provenance_record.download_timestamp,
            "download_agent": provenance_record.download_agent,
            "file_info": provenance_record.file_info,
            "pbuf_commit": provenance_record.environment.pbuf_commit,
            "python_version": provenance_record.environment.python_version,
            "platform": provenance_record.environment.platform,
            "hostname": provenance_record.environment.hostname
        }
        
        # Extract dataset type from name or metadata
        dataset_type = self._infer_dataset_type(provenance_record.dataset_name)
        if dataset_type:
            metadata["dataset_type"] = dataset_type
        
        return metadata
    
    def _infer_dataset_type(self, dataset_name: str) -> Optional[str]:
        """
        Infer dataset type from dataset name.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset type identifier or None if cannot be inferred
        """
        name_lower = dataset_name.lower()
        
        if any(keyword in name_lower for keyword in ['sn', 'supernova', 'supernovae']):
            return 'sn'
        elif any(keyword in name_lower for keyword in ['bao', 'baryon']):
            return 'bao'
        elif any(keyword in name_lower for keyword in ['cmb', 'planck', 'cosmic_microwave']):
            return 'cmb'
        elif any(keyword in name_lower for keyword in ['cc', 'chronometer', 'hubble']):
            return 'cc'
        elif any(keyword in name_lower for keyword in ['rsd', 'redshift_space', 'growth']):
            return 'rsd'
        
        return None
    
    def validate_raw_dataset_integrity(self, dataset_name: str, file_path: Path) -> bool:
        """
        Validate raw dataset integrity before processing.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to the dataset file
            
        Returns:
            bool: True if integrity validation passes
            
        Raises:
            ProcessingError: If integrity validation fails
        """
        # Check file exists and is readable
        if not file_path.exists():
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="input_validation",
                error_type="file_not_found",
                error_message=f"Dataset file not found: {file_path}",
                suggested_actions=["Verify file path", "Check file system permissions"]
            )
        
        if not file_path.is_file():
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="input_validation", 
                error_type="invalid_file_type",
                error_message=f"Path is not a regular file: {file_path}",
                suggested_actions=["Verify file path points to a file, not directory"]
            )
        
        # Check file is not empty
        if file_path.stat().st_size == 0:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="input_validation",
                error_type="empty_file",
                error_message=f"Dataset file is empty: {file_path}",
                suggested_actions=["Re-download the dataset", "Check data source"]
            )
        
        # Verify checksum against registry
        provenance_record = self.registry_manager.get_registry_entry(dataset_name)
        if provenance_record and provenance_record.verification.sha256_expected:
            actual_checksum = self._calculate_file_checksum(file_path)
            expected_checksum = provenance_record.verification.sha256_expected
            
            if actual_checksum != expected_checksum:
                raise ProcessingError(
                    dataset_name=dataset_name,
                    stage="input_validation",
                    error_type="checksum_mismatch",
                    error_message="File checksum does not match registry",
                    context={
                        "expected_checksum": expected_checksum,
                        "actual_checksum": actual_checksum
                    },
                    suggested_actions=[
                        "Re-download the dataset",
                        "Check for file corruption",
                        "Verify file has not been modified"
                    ]
                )
        
        return True
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 checksum as hex string
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def register_derived_dataset(
        self,
        dataset_name: str,
        derived_dataset: StandardDataset,
        source_provenance: ProvenanceRecord,
        transformation_summary: Dict[str, Any],
        output_file_path: Path
    ) -> str:
        """
        Register derived dataset with complete provenance tracking.
        
        Implements comprehensive provenance tracking including:
        - Source registry entry hash references
        - Environment snapshot links
        - Complete transformation audit trail
        - Integration with existing registry audit system
        
        Args:
            dataset_name: Name of the source dataset
            derived_dataset: Processed StandardDataset
            source_provenance: Provenance record of source dataset
            transformation_summary: Summary of applied transformations
            output_file_path: Path where derived dataset is stored
            
        Returns:
            str: Hash of the derived dataset entry for reference
            
        Raises:
            ProcessingError: If registration fails
            
        Requirements: 3.1, 3.2, 3.4
        """
        try:
            # Create derived dataset name with timestamp for uniqueness
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            derived_name = f"{dataset_name}_derived_{timestamp}"
            
            # Calculate checksum of derived dataset file
            derived_checksum = self._calculate_file_checksum(output_file_path)
            
            # Collect current environment and create snapshot
            current_environment = EnvironmentInfo.collect()
            environment_snapshot = self._create_environment_snapshot(current_environment)
            
            # Reference latest environment registry entry
            environment_registry_ref = self._get_latest_environment_registry_entry()
            
            # Create comprehensive verification result for derived dataset
            verification_result = VerificationResult(
                sha256_verified=True,
                sha256_expected=derived_checksum,
                sha256_actual=derived_checksum,
                size_verified=True,
                size_expected=output_file_path.stat().st_size,
                size_actual=output_file_path.stat().st_size,
                schema_verified=True,
                schema_errors=[],
                verification_timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Create comprehensive provenance record for derived dataset
            derived_provenance = ProvenanceRecord(
                dataset_name=derived_name,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
                source_used="data_preparation_framework",
                download_agent="data-preparation-framework-v1.0",
                environment=current_environment,
                verification=verification_result,
                file_info={
                    "local_path": str(output_file_path),
                    "original_filename": output_file_path.name,
                    "mime_type": "application/json",
                    
                    # Source dataset references
                    "source_dataset": dataset_name,
                    "source_registry_hash": source_provenance.verification.sha256_actual,
                    "source_download_timestamp": source_provenance.download_timestamp,
                    "source_agent": source_provenance.download_agent,
                    
                    # Environment snapshot references
                    "environment_snapshot_hash": environment_snapshot["snapshot_hash"],
                    "environment_registry_ref": environment_registry_ref,
                    "processing_environment": environment_snapshot["environment_details"],
                    
                    # Transformation details
                    "transformation_summary": transformation_summary,
                    "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                    "framework_version": "1.0.0",
                    
                    # Provenance chain
                    "provenance_chain": self._build_provenance_chain(source_provenance, transformation_summary),
                    
                    # Reproducibility information
                    "reproducibility_info": {
                        "deterministic_processing": True,
                        "input_hash": self._calculate_input_hash(dataset_name, source_provenance),
                        "environment_hash": environment_snapshot["snapshot_hash"],
                        "transformation_hash": self._calculate_transformation_hash(transformation_summary)
                    }
                }
            )
            
            # Write derived dataset registry entry
            derived_registry_file = self._derived_datasets_path / f"{derived_name}.json"
            
            with open(derived_registry_file, 'w') as f:
                json.dump(derived_provenance.to_dict(), f, indent=2, sort_keys=True)
            
            # Create comprehensive audit trail entry
            audit_entry = {
                "operation": "derived_dataset_created",
                "derived_dataset_name": derived_name,
                "source_dataset_name": dataset_name,
                "source_registry_hash": source_provenance.verification.sha256_actual,
                "derived_hash": derived_checksum,
                "transformation_agent": "data-preparation-framework-v1.0",
                "pbuf_commit": current_environment.pbuf_commit,
                "environment_snapshot_hash": environment_snapshot["snapshot_hash"],
                "environment_registry_ref": environment_registry_ref,
                "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                "reproducibility_hash": derived_provenance.file_info["reproducibility_info"]["input_hash"],
                "transformation_modules": transformation_summary.get("modules_used", []),
                "validation_status": "passed"
            }
            
            self.registry_manager._append_audit_log("derived_dataset_created", audit_entry)
            
            # Update derived dataset index
            self._update_derived_dataset_index(derived_name, derived_provenance, audit_entry)
            
            return derived_checksum
            
        except Exception as e:
            # Enhanced error handling with context
            error_context = {
                "dataset_name": dataset_name,
                "output_file_exists": output_file_path.exists() if output_file_path else False,
                "registry_path_writable": self._derived_datasets_path.exists(),
                "error_type": type(e).__name__
            }
            
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="output_registration",
                error_type="registration_failed",
                error_message=f"Failed to register derived dataset: {str(e)}",
                context=error_context,
                suggested_actions=[
                    "Check file system permissions",
                    "Verify registry directory is writable",
                    "Check disk space availability",
                    "Verify source provenance record integrity"
                ]
            )
    
    def _create_environment_snapshot(self, environment: EnvironmentInfo) -> Dict[str, Any]:
        """
        Create comprehensive environment snapshot for reproducibility.
        
        Args:
            environment: Current environment information
            
        Returns:
            Dict containing environment snapshot and hash
            
        Requirements: 3.2, 3.4
        """
        # Collect comprehensive environment details
        environment_details = {
            "pbuf_commit": environment.pbuf_commit,
            "python_version": environment.python_version,
            "platform": environment.platform,
            "hostname": environment.hostname,
            "timestamp": environment.timestamp,
            
            # Additional environment factors
            "framework_version": "1.0.0",
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            
            # Package versions (if available)
            "package_versions": self._collect_package_versions(),
            
            # System information
            "system_info": {
                "cpu_count": os.cpu_count(),
                "memory_total": self._get_system_memory(),
                "disk_space": self._get_disk_space()
            }
        }
        
        # Calculate environment snapshot hash
        snapshot_string = json.dumps(environment_details, sort_keys=True)
        snapshot_hash = hashlib.sha256(snapshot_string.encode()).hexdigest()
        
        # Save environment snapshot
        snapshot_file = self._derived_datasets_path / f"environment_{snapshot_hash}.json"
        if not snapshot_file.exists():
            with open(snapshot_file, 'w') as f:
                json.dump({
                    "snapshot_hash": snapshot_hash,
                    "creation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "environment_details": environment_details
                }, f, indent=2, sort_keys=True)
        
        return {
            "snapshot_hash": snapshot_hash,
            "environment_details": environment_details,
            "snapshot_file": str(snapshot_file)
        }
    
    def _get_latest_environment_registry_entry(self) -> Optional[str]:
        """
        Reference latest environment registry entry.
        
        Returns:
            Path to latest environment registry entry or None if not found
            
        Requirements: 3.2
        """
        # Look for environment registry entries in data/registry/
        registry_path = Path("data/registry")
        
        # Find environment files matching pattern environment_*.json
        environment_files = list(registry_path.glob("environment_*.json"))
        
        if environment_files:
            # Return the most recent environment file
            latest_file = max(environment_files, key=lambda f: f.stat().st_mtime)
            return str(latest_file.relative_to(registry_path))
        
        # If no environment files found, create reference to current snapshot location
        return f"derived/environment_snapshots"
    
    def _build_provenance_chain(self, source_provenance: ProvenanceRecord, 
                               transformation_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build complete provenance chain from source to derived dataset.
        
        Args:
            source_provenance: Source dataset provenance
            transformation_summary: Applied transformations
            
        Returns:
            List of provenance chain entries
            
        Requirements: 3.1, 3.2
        """
        provenance_chain = []
        
        # Add source dataset entry
        source_entry = {
            "step": 1,
            "operation": "source_dataset",
            "dataset_name": source_provenance.dataset_name,
            "source_used": source_provenance.source_used,
            "download_timestamp": source_provenance.download_timestamp,
            "verification_hash": source_provenance.verification.sha256_actual,
            "agent": source_provenance.download_agent
        }
        provenance_chain.append(source_entry)
        
        # Add transformation steps
        transformation_steps = transformation_summary.get("transformation_steps", [])
        for i, step in enumerate(transformation_steps, start=2):
            transformation_entry = {
                "step": i,
                "operation": "transformation",
                "transformation_type": step.get("type", "unknown"),
                "module_used": step.get("module", "unknown"),
                "parameters": step.get("parameters", {}),
                "timestamp": step.get("timestamp", datetime.now(timezone.utc).isoformat())
            }
            provenance_chain.append(transformation_entry)
        
        # Add validation step
        validation_entry = {
            "step": len(provenance_chain) + 1,
            "operation": "validation",
            "validation_type": "comprehensive",
            "validation_results": transformation_summary.get("validation_results", {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        provenance_chain.append(validation_entry)
        
        # Add output generation step
        output_entry = {
            "step": len(provenance_chain) + 1,
            "operation": "output_generation",
            "output_format": "StandardDataset",
            "output_timestamp": datetime.now(timezone.utc).isoformat()
        }
        provenance_chain.append(output_entry)
        
        return provenance_chain
    
    def _calculate_input_hash(self, dataset_name: str, source_provenance: ProvenanceRecord) -> str:
        """
        Calculate hash of input parameters for reproducibility tracking.
        
        Args:
            dataset_name: Name of source dataset
            source_provenance: Source dataset provenance
            
        Returns:
            SHA256 hash of input parameters
        """
        input_data = {
            "dataset_name": dataset_name,
            "source_hash": source_provenance.verification.sha256_actual,
            "source_timestamp": source_provenance.download_timestamp,
            "source_agent": source_provenance.download_agent
        }
        
        input_string = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(input_string.encode()).hexdigest()
    
    def _calculate_transformation_hash(self, transformation_summary: Dict[str, Any]) -> str:
        """
        Calculate hash of transformation parameters.
        
        Args:
            transformation_summary: Summary of applied transformations
            
        Returns:
            SHA256 hash of transformation parameters
        """
        # Extract deterministic transformation parameters
        transformation_data = {
            "modules_used": transformation_summary.get("modules_used", []),
            "transformation_steps": transformation_summary.get("transformation_steps", []),
            "parameters": transformation_summary.get("parameters", {}),
            "framework_version": "1.0.0"
        }
        
        transformation_string = json.dumps(transformation_data, sort_keys=True)
        return hashlib.sha256(transformation_string.encode()).hexdigest()
    
    def _update_derived_dataset_index(self, derived_name: str, derived_provenance: ProvenanceRecord,
                                    audit_entry: Dict[str, Any]):
        """
        Update derived dataset index for efficient querying.
        
        Args:
            derived_name: Name of derived dataset
            derived_provenance: Provenance record of derived dataset
            audit_entry: Audit trail entry
            
        Requirements: 3.4
        """
        index_file = self._derived_datasets_path / "derived_datasets_index.json"
        
        # Load existing index or create new one
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {
                "created": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "datasets": {}
            }
        
        # Add new derived dataset entry
        index["datasets"][derived_name] = {
            "source_dataset": audit_entry["source_dataset_name"],
            "creation_timestamp": audit_entry["processing_timestamp"],
            "derived_hash": audit_entry["derived_hash"],
            "environment_snapshot": audit_entry["environment_snapshot_hash"],
            "transformation_agent": audit_entry["transformation_agent"],
            "registry_file": f"{derived_name}.json",
            "reproducibility_hash": audit_entry["reproducibility_hash"]
        }
        
        # Update index metadata
        index["last_updated"] = datetime.now(timezone.utc).isoformat()
        index["total_datasets"] = len(index["datasets"])
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2, sort_keys=True)
    
    def _collect_package_versions(self) -> Dict[str, str]:
        """Collect versions of key packages for reproducibility."""
        package_versions = {}
        
        key_packages = ['numpy', 'scipy', 'pandas', 'astropy', 'h5py']
        
        for package in key_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                package_versions[package] = version
            except (ImportError, Exception) as e:
                # Catch all exceptions, not just ImportError
                # Some packages like astropy can fail during import with various errors
                package_versions[package] = f'import_failed: {type(e).__name__}'
        
        return package_versions
    
    def _get_system_memory(self) -> Optional[int]:
        """Get total system memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().total
        except (ImportError, Exception):
            # Fallback to reading /proc/meminfo directly if psutil fails
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            # Extract memory in kB and convert to bytes
                            mem_kb = int(line.split()[1])
                            return mem_kb * 1024
            except:
                pass
            return None
    
    def _get_disk_space(self) -> Optional[Dict[str, int]]:
        """Get disk space information."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path.cwd())
            return {"total": total, "used": used, "free": free}
        except:
            return None
    
    def get_dataset_processing_requirements(self, dataset_name: str) -> Dict[str, Any]:
        """
        Extract dataset type and processing requirements from registry metadata.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing processing requirements and configuration
        """
        dataset_info = self.get_verified_dataset(dataset_name)
        metadata = dataset_info["metadata"]
        
        # Infer dataset type
        dataset_type = metadata.get("dataset_type") or self._infer_dataset_type(dataset_name)
        
        if not dataset_type:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="input_validation",
                error_type="unknown_dataset_type",
                error_message=f"Cannot determine dataset type for '{dataset_name}'",
                suggested_actions=[
                    "Add dataset type to registry metadata",
                    "Use standard naming convention",
                    "Manually specify dataset type"
                ]
            )
        
        # Get processing requirements based on dataset type
        processing_requirements = {
            "dataset_type": dataset_type,
            "input_file_path": dataset_info["file_path"],
            "source_metadata": metadata,
            "validation_config": self._get_validation_config(dataset_type),
            "transformation_config": self._get_transformation_config(dataset_type)
        }
        
        return processing_requirements
    
    def _get_validation_config(self, dataset_type: str) -> Dict[str, Any]:
        """
        Get validation configuration for dataset type.
        
        Args:
            dataset_type: Type of dataset (sn, bao, cmb, cc, rsd)
            
        Returns:
            Dict containing validation parameters
        """
        validation_configs = {
            "sn": {
                "min_redshift": 0.01,
                "max_redshift": 2.5,
                "required_columns": ["z", "mu", "sigma_mu"],
                "allow_negative_observables": False
            },
            "bao": {
                "min_redshift": 0.1,
                "max_redshift": 2.5,
                "required_columns": ["z", "measurement", "uncertainty"],
                "allow_negative_observables": False
            },
            "cmb": {
                "min_redshift": 1000.0,  # CMB redshift
                "max_redshift": 1200.0,
                "required_columns": ["R", "l_A", "theta_star"],
                "allow_negative_observables": False
            },
            "cc": {
                "min_redshift": 0.0,
                "max_redshift": 2.0,
                "required_columns": ["z", "H_z", "sigma_H"],
                "allow_negative_observables": False
            },
            "rsd": {
                "min_redshift": 0.0,
                "max_redshift": 1.5,
                "required_columns": ["z", "f_sigma8", "sigma_f_sigma8"],
                "allow_negative_observables": False
            }
        }
        
        return validation_configs.get(dataset_type, {
            "min_redshift": 0.0,
            "max_redshift": 10.0,
            "required_columns": [],
            "allow_negative_observables": True
        })
    
    def _get_transformation_config(self, dataset_type: str) -> Dict[str, Any]:
        """
        Get transformation configuration for dataset type.
        
        Args:
            dataset_type: Type of dataset (sn, bao, cmb, cc, rsd)
            
        Returns:
            Dict containing transformation parameters
        """
        transformation_configs = {
            "sn": {
                "output_observable_name": "distance_modulus",
                "output_units": "mag",
                "requires_covariance": True,
                "duplicate_removal": True
            },
            "bao": {
                "output_observable_name": "distance_ratio",
                "output_units": "dimensionless",
                "requires_covariance": True,
                "separate_isotropic_anisotropic": True
            },
            "cmb": {
                "output_observable_name": "distance_priors",
                "output_units": "dimensionless",
                "requires_covariance": True,
                "extract_compressed_parameters": True
            },
            "cc": {
                "output_observable_name": "hubble_parameter",
                "output_units": "km/s/Mpc",
                "requires_covariance": False,
                "merge_compilations": True
            },
            "rsd": {
                "output_observable_name": "growth_rate",
                "output_units": "dimensionless",
                "requires_covariance": True,
                "validate_sign_convention": True
            }
        }
        
        return transformation_configs.get(dataset_type, {
            "output_observable_name": "observable",
            "output_units": "unknown",
            "requires_covariance": False
        })
    
    def check_registry_entry_status(self, dataset_name: str) -> Dict[str, Any]:
        """
        Check registry entry status and provide detailed information.
        
        Args:
            dataset_name: Name of the dataset to check
            
        Returns:
            Dict containing status information
        """
        status = {
            "exists": False,
            "verified": False,
            "accessible": False,
            "dataset_type": None,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Check if registry entry exists
            if not self.registry_manager.has_registry_entry(dataset_name):
                status["errors"].append("Registry entry does not exist")
                return status
            
            status["exists"] = True
            
            # Get registry entry
            provenance_record = self.registry_manager.get_registry_entry(dataset_name)
            
            if provenance_record is None:
                status["errors"].append("Registry entry exists but cannot be loaded")
                return status
            
            # Check verification status
            if provenance_record.verification.is_valid:
                status["verified"] = True
            else:
                status["errors"].append("Dataset failed verification")
                if not provenance_record.verification.sha256_verified:
                    status["errors"].append("SHA256 checksum verification failed")
                if not provenance_record.verification.size_verified:
                    status["errors"].append("File size verification failed")
                if not provenance_record.verification.schema_verified:
                    status["errors"].append("Schema verification failed")
                    status["errors"].extend(provenance_record.verification.schema_errors)
            
            # Check file accessibility
            file_path = Path(provenance_record.file_info["local_path"])
            if file_path.exists() and file_path.is_file():
                status["accessible"] = True
            else:
                status["errors"].append(f"Dataset file not accessible: {file_path}")
            
            # Determine dataset type
            status["dataset_type"] = self._infer_dataset_type(dataset_name)
            if not status["dataset_type"]:
                status["warnings"].append("Cannot determine dataset type from name")
            
        except Exception as e:
            status["errors"].append(f"Error checking registry entry: {str(e)}")
        
        return status
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets with their status.
        
        Returns:
            List of dictionaries containing dataset information
        """
        datasets = []
        
        for dataset_name in self.registry_manager.list_datasets():
            status = self.check_registry_entry_status(dataset_name)
            
            dataset_info = {
                "name": dataset_name,
                "exists": status["exists"],
                "verified": status["verified"],
                "accessible": status["accessible"],
                "dataset_type": status["dataset_type"],
                "ready_for_processing": (
                    status["exists"] and 
                    status["verified"] and 
                    status["accessible"] and 
                    status["dataset_type"] is not None
                ),
                "errors": status["errors"],
                "warnings": status["warnings"]
            }
            
            datasets.append(dataset_info)
        
        return datasets
    
    def extract_detailed_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """
        Extract comprehensive dataset metadata for processing requirements.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing detailed metadata including:
                - basic_info: Dataset name, type, source information
                - file_info: File path, size, format, checksum
                - provenance: Download timestamp, agent, environment
                - processing_hints: Inferred processing requirements
                - validation_params: Dataset-specific validation parameters
                
        Raises:
            ProcessingError: If metadata extraction fails
        """
        try:
            # Get verified dataset info
            dataset_info = self.get_verified_dataset(dataset_name)
            provenance = dataset_info["provenance"]
            
            # Extract basic information
            basic_info = {
                "dataset_name": dataset_name,
                "dataset_type": self._infer_dataset_type(dataset_name),
                "source_used": provenance.source_used,
                "download_agent": provenance.download_agent,
                "registry_version": provenance.registry_version
            }
            
            # Extract file information
            file_info = {
                "local_path": dataset_info["file_path"],
                "file_size": provenance.verification.size_actual,
                "sha256_checksum": provenance.verification.sha256_actual,
                "mime_type": provenance.file_info.get("mime_type", "unknown"),
                "original_filename": provenance.file_info.get("original_filename", "unknown")
            }
            
            # Extract provenance information
            provenance_info = {
                "download_timestamp": provenance.download_timestamp,
                "verification_timestamp": provenance.verification.verification_timestamp,
                "pbuf_commit": provenance.environment.pbuf_commit,
                "python_version": provenance.environment.python_version,
                "platform": provenance.environment.platform,
                "hostname": provenance.environment.hostname
            }
            
            # Generate processing hints
            processing_hints = self._generate_processing_hints(basic_info["dataset_type"], file_info)
            
            # Get validation parameters
            validation_params = self._get_validation_config(basic_info["dataset_type"])
            
            return {
                "basic_info": basic_info,
                "file_info": file_info,
                "provenance": provenance_info,
                "processing_hints": processing_hints,
                "validation_params": validation_params,
                "extraction_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                dataset_name=dataset_name,
                stage="metadata_extraction",
                error_type="extraction_failed",
                error_message=f"Failed to extract metadata: {str(e)}",
                suggested_actions=[
                    "Check registry entry integrity",
                    "Verify dataset file accessibility",
                    "Check file system permissions"
                ]
            )
    
    def _generate_processing_hints(self, dataset_type: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate processing hints based on dataset type and file characteristics.
        
        Args:
            dataset_type: Type of dataset (sn, bao, cmb, cc, rsd)
            file_info: File information dictionary
            
        Returns:
            Dict containing processing hints and recommendations
        """
        hints = {
            "recommended_derivation_module": dataset_type,
            "estimated_processing_time": "unknown",
            "memory_requirements": "standard",
            "special_considerations": []
        }
        
        # File size based hints
        file_size_mb = file_info["file_size"] / (1024 * 1024)
        
        if file_size_mb > 100:
            hints["memory_requirements"] = "high"
            hints["estimated_processing_time"] = "long"
            hints["special_considerations"].append("Large file - consider chunked processing")
        elif file_size_mb > 10:
            hints["memory_requirements"] = "medium"
            hints["estimated_processing_time"] = "medium"
        else:
            hints["estimated_processing_time"] = "short"
        
        # Dataset type specific hints
        if dataset_type == "sn":
            hints["special_considerations"].extend([
                "Check for duplicate entries by coordinates",
                "Validate magnitude to distance modulus conversion",
                "Verify systematic covariance matrices"
            ])
        elif dataset_type == "bao":
            hints["special_considerations"].extend([
                "Separate isotropic and anisotropic measurements",
                "Validate distance measure units",
                "Check correlation matrix properties"
            ])
        elif dataset_type == "cmb":
            hints["special_considerations"].extend([
                "Extract compressed distance priors",
                "Validate dimensionless consistency",
                "Check Planck file format compatibility"
            ])
        elif dataset_type == "cc":
            hints["special_considerations"].extend([
                "Merge multiple compilation sources",
                "Filter overlapping redshift bins",
                "Validate H(z) sign conventions"
            ])
        elif dataset_type == "rsd":
            hints["special_considerations"].extend([
                "Validate growth rate sign conventions",
                "Homogenize covariance matrices",
                "Check survey-specific corrections"
            ])
        
        # File format hints
        file_extension = Path(file_info["local_path"]).suffix.lower()
        
        if file_extension in ['.fits', '.fit']:
            hints["special_considerations"].append("FITS format - use astropy for reading")
        elif file_extension in ['.hdf5', '.h5']:
            hints["special_considerations"].append("HDF5 format - use h5py for reading")
        elif file_extension in ['.txt', '.dat']:
            hints["special_considerations"].append("Text format - check delimiter and header structure")
        elif file_extension == '.csv':
            hints["special_considerations"].append("CSV format - validate column names and types")
        
        return hints
    
    def validate_dataset_for_processing(self, dataset_name: str) -> Dict[str, Any]:
        """
        Comprehensive validation of dataset readiness for processing.
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Dict containing validation results:
                - is_ready: Boolean indicating if dataset is ready for processing
                - validation_results: Detailed validation check results
                - blocking_issues: List of issues that prevent processing
                - warnings: List of non-blocking warnings
                - recommendations: List of recommended actions
        """
        validation_results = {
            "is_ready": False,
            "validation_results": {},
            "blocking_issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Check registry entry status
            status = self.check_registry_entry_status(dataset_name)
            validation_results["validation_results"]["registry_status"] = status
            
            if not status["exists"]:
                validation_results["blocking_issues"].append("Dataset not found in registry")
                return validation_results
            
            if not status["verified"]:
                validation_results["blocking_issues"].append("Dataset failed verification")
                validation_results["blocking_issues"].extend(status["errors"])
                return validation_results
            
            if not status["accessible"]:
                validation_results["blocking_issues"].append("Dataset file not accessible")
                return validation_results
            
            # Extract metadata
            metadata = self.extract_detailed_metadata(dataset_name)
            validation_results["validation_results"]["metadata"] = metadata
            
            # Validate dataset type
            if not metadata["basic_info"]["dataset_type"]:
                validation_results["blocking_issues"].append("Cannot determine dataset type")
                validation_results["recommendations"].append("Add dataset type to registry metadata or use standard naming convention")
                return validation_results
            
            # Validate file integrity
            file_integrity = self._validate_file_integrity(dataset_name, Path(metadata["file_info"]["local_path"]))
            validation_results["validation_results"]["file_integrity"] = file_integrity
            
            if not file_integrity["is_valid"]:
                validation_results["blocking_issues"].extend(file_integrity["errors"])
                return validation_results
            
            # Check processing requirements
            processing_reqs = self.get_dataset_processing_requirements(dataset_name)
            validation_results["validation_results"]["processing_requirements"] = processing_reqs
            
            # Validate file format compatibility
            format_validation = self._validate_file_format(metadata["file_info"]["local_path"], metadata["basic_info"]["dataset_type"])
            validation_results["validation_results"]["format_compatibility"] = format_validation
            
            if not format_validation["is_compatible"]:
                validation_results["blocking_issues"].extend(format_validation["issues"])
            
            # Add warnings for potential issues
            if metadata["file_info"]["file_size"] > 100 * 1024 * 1024:  # 100MB
                validation_results["warnings"].append("Large file size may require additional memory")
            
            if not metadata["provenance"]["pbuf_commit"]:
                validation_results["warnings"].append("Missing PBUF commit hash in provenance")
            
            # Generate recommendations
            validation_results["recommendations"].extend(metadata["processing_hints"]["special_considerations"])
            
            # Determine if ready for processing
            validation_results["is_ready"] = len(validation_results["blocking_issues"]) == 0
            
        except ProcessingError as e:
            validation_results["blocking_issues"].append(f"Validation error: {e.error_message}")
        except Exception as e:
            validation_results["blocking_issues"].append(f"Unexpected validation error: {str(e)}")
        
        return validation_results
    
    def _validate_file_integrity(self, dataset_name: str, file_path: Path) -> Dict[str, Any]:
        """
        Validate file integrity beyond basic checksum verification.
        
        Args:
            dataset_name: Name of the dataset
            file_path: Path to the dataset file
            
        Returns:
            Dict containing integrity validation results
        """
        integrity_results = {
            "is_valid": False,
            "checks_performed": [],
            "errors": [],
            "warnings": []
        }
        
        try:
            # Basic file existence and readability
            integrity_results["checks_performed"].append("file_existence")
            if not file_path.exists():
                integrity_results["errors"].append("File does not exist")
                return integrity_results
            
            if not file_path.is_file():
                integrity_results["errors"].append("Path is not a regular file")
                return integrity_results
            
            # File size validation
            integrity_results["checks_performed"].append("file_size")
            file_size = file_path.stat().st_size
            
            if file_size == 0:
                integrity_results["errors"].append("File is empty")
                return integrity_results
            
            if file_size < 100:  # Very small files might be incomplete
                integrity_results["warnings"].append("File is very small, may be incomplete")
            
            # File readability test
            integrity_results["checks_performed"].append("file_readability")
            try:
                with open(file_path, 'rb') as f:
                    # Try to read first 1KB
                    f.read(1024)
            except IOError as e:
                integrity_results["errors"].append(f"File read error: {str(e)}")
                return integrity_results
            
            # Checksum verification against registry
            integrity_results["checks_performed"].append("checksum_verification")
            try:
                self.validate_raw_dataset_integrity(dataset_name, file_path)
            except ProcessingError as e:
                if e.error_type == "checksum_mismatch":
                    integrity_results["errors"].append("Checksum mismatch with registry")
                    return integrity_results
            
            integrity_results["is_valid"] = True
            
        except Exception as e:
            integrity_results["errors"].append(f"Integrity validation failed: {str(e)}")
        
        return integrity_results
    
    def _validate_file_format(self, file_path: str, dataset_type: str) -> Dict[str, Any]:
        """
        Validate file format compatibility with dataset type.
        
        Args:
            file_path: Path to the dataset file
            dataset_type: Type of dataset (sn, bao, cmb, cc, rsd)
            
        Returns:
            Dict containing format compatibility results
        """
        format_results = {
            "is_compatible": False,
            "detected_format": None,
            "expected_formats": [],
            "issues": [],
            "recommendations": []
        }
        
        file_path_obj = Path(file_path)
        file_extension = file_path_obj.suffix.lower()
        
        # Define expected formats for each dataset type
        expected_formats = {
            "sn": [".txt", ".csv", ".dat", ".fits"],
            "bao": [".txt", ".csv", ".dat"],
            "cmb": [".fits", ".txt", ".csv"],
            "cc": [".txt", ".csv", ".dat"],
            "rsd": [".txt", ".csv", ".dat"]
        }
        
        format_results["expected_formats"] = expected_formats.get(dataset_type, [])
        format_results["detected_format"] = file_extension
        
        # Check if format is expected for dataset type
        if file_extension in format_results["expected_formats"]:
            format_results["is_compatible"] = True
        else:
            format_results["issues"].append(f"Unexpected file format '{file_extension}' for dataset type '{dataset_type}'")
            format_results["recommendations"].append(f"Expected formats: {', '.join(format_results['expected_formats'])}")
        
        # Additional format-specific checks
        if file_extension == ".fits":
            format_results["recommendations"].append("FITS format detected - ensure astropy is available")
        elif file_extension in [".txt", ".dat"]:
            format_results["recommendations"].append("Text format detected - verify delimiter and column structure")
        elif file_extension == ".csv":
            format_results["recommendations"].append("CSV format detected - validate column headers")
        
        return format_results
    
    def generate_processing_report(self, dataset_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive processing readiness report.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dict containing complete processing readiness assessment
        """
        report = {
            "dataset_name": dataset_name,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "ready_for_processing": False,
            "sections": {}
        }
        
        try:
            # Registry status section
            report["sections"]["registry_status"] = self.check_registry_entry_status(dataset_name)
            
            # Metadata extraction section
            if report["sections"]["registry_status"]["exists"]:
                report["sections"]["metadata"] = self.extract_detailed_metadata(dataset_name)
            
            # Validation section
            report["sections"]["validation"] = self.validate_dataset_for_processing(dataset_name)
            
            # Processing requirements section
            if report["sections"]["validation"]["is_ready"]:
                report["sections"]["processing_requirements"] = self.get_dataset_processing_requirements(dataset_name)
            
            # Determine overall status
            if report["sections"]["validation"]["is_ready"]:
                report["overall_status"] = "ready"
                report["ready_for_processing"] = True
            elif report["sections"]["registry_status"]["exists"]:
                if report["sections"]["validation"]["blocking_issues"]:
                    report["overall_status"] = "blocked"
                else:
                    report["overall_status"] = "warning"
            else:
                report["overall_status"] = "not_found"
            
        except Exception as e:
            report["overall_status"] = "error"
            report["sections"]["error"] = {
                "error_message": str(e),
                "error_type": type(e).__name__
            }
        
        return report
    
    def get_derived_dataset_provenance(self, derived_dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get complete provenance information for a derived dataset.
        
        Args:
            derived_dataset_name: Name of derived dataset
            
        Returns:
            Complete provenance information or None if not found
            
        Requirements: 3.1, 3.2
        """
        derived_registry_file = self._derived_datasets_path / f"{derived_dataset_name}.json"
        
        if not derived_registry_file.exists():
            return None
        
        try:
            with open(derived_registry_file, 'r') as f:
                provenance_data = json.load(f)
            
            # Load environment snapshot if available
            environment_snapshot_hash = provenance_data.get("file_info", {}).get("environment_snapshot_hash")
            environment_snapshot = None
            
            if environment_snapshot_hash:
                snapshot_file = self._derived_datasets_path / f"environment_{environment_snapshot_hash}.json"
                if snapshot_file.exists():
                    with open(snapshot_file, 'r') as f:
                        environment_snapshot = json.load(f)
            
            return {
                "provenance_record": provenance_data,
                "environment_snapshot": environment_snapshot,
                "registry_file": str(derived_registry_file),
                "query_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "error": f"Failed to load provenance: {str(e)}",
                "registry_file": str(derived_registry_file)
            }
    
    def list_derived_datasets(self, source_dataset: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all derived datasets with optional filtering by source dataset.
        
        Args:
            source_dataset: Optional source dataset name to filter by
            
        Returns:
            List of derived dataset information
            
        Requirements: 3.4
        """
        index_file = self._derived_datasets_path / "derived_datasets_index.json"
        
        if not index_file.exists():
            return []
        
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            datasets = []
            for derived_name, info in index["datasets"].items():
                if source_dataset is None or info["source_dataset"] == source_dataset:
                    datasets.append({
                        "derived_name": derived_name,
                        "source_dataset": info["source_dataset"],
                        "creation_timestamp": info["creation_timestamp"],
                        "derived_hash": info["derived_hash"],
                        "environment_snapshot": info["environment_snapshot"],
                        "transformation_agent": info["transformation_agent"],
                        "reproducibility_hash": info["reproducibility_hash"]
                    })
            
            return datasets
            
        except Exception as e:
            return [{"error": f"Failed to load derived datasets index: {str(e)}"}]
    
    def verify_derived_dataset_integrity(self, derived_dataset_name: str) -> Dict[str, Any]:
        """
        Verify integrity of a derived dataset and its provenance chain.
        
        Args:
            derived_dataset_name: Name of derived dataset to verify
            
        Returns:
            Verification results
            
        Requirements: 3.1, 3.2
        """
        verification_results = {
            "dataset_name": derived_dataset_name,
            "verification_timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "checks": {}
        }
        
        try:
            # Check if derived dataset registry entry exists
            derived_registry_file = self._derived_datasets_path / f"{derived_dataset_name}.json"
            verification_results["checks"]["registry_entry_exists"] = derived_registry_file.exists()
            
            if not derived_registry_file.exists():
                verification_results["overall_status"] = "failed"
                verification_results["error"] = "Registry entry not found"
                return verification_results
            
            # Load provenance record
            with open(derived_registry_file, 'r') as f:
                provenance_data = json.load(f)
            
            verification_results["checks"]["registry_entry_valid"] = True
            
            # Check if derived dataset file exists
            dataset_file_path = Path(provenance_data["file_info"]["local_path"])
            verification_results["checks"]["dataset_file_exists"] = dataset_file_path.exists()
            
            if dataset_file_path.exists():
                # Verify file checksum
                actual_checksum = self._calculate_file_checksum(dataset_file_path)
                expected_checksum = provenance_data["verification"]["sha256_actual"]
                verification_results["checks"]["checksum_verified"] = (actual_checksum == expected_checksum)
                
                if actual_checksum != expected_checksum:
                    verification_results["checks"]["checksum_mismatch"] = {
                        "expected": expected_checksum,
                        "actual": actual_checksum
                    }
            
            # Check environment snapshot integrity
            environment_snapshot_hash = provenance_data.get("file_info", {}).get("environment_snapshot_hash")
            if environment_snapshot_hash:
                snapshot_file = self._derived_datasets_path / f"environment_{environment_snapshot_hash}.json"
                verification_results["checks"]["environment_snapshot_exists"] = snapshot_file.exists()
            
            # Check source dataset provenance chain
            source_dataset = provenance_data["file_info"]["source_dataset"]
            source_registry_hash = provenance_data["file_info"]["source_registry_hash"]
            
            source_provenance = self.registry_manager.get_registry_entry(source_dataset)
            if source_provenance:
                verification_results["checks"]["source_dataset_exists"] = True
                verification_results["checks"]["source_hash_matches"] = (
                    source_provenance.verification.sha256_actual == source_registry_hash
                )
            else:
                verification_results["checks"]["source_dataset_exists"] = False
            
            # Determine overall status
            critical_checks = [
                "registry_entry_valid",
                "dataset_file_exists", 
                "checksum_verified",
                "source_dataset_exists"
            ]
            
            if all(verification_results["checks"].get(check, False) for check in critical_checks):
                verification_results["overall_status"] = "verified"
            else:
                verification_results["overall_status"] = "failed"
            
        except Exception as e:
            verification_results["overall_status"] = "error"
            verification_results["error"] = str(e)
        
        return verification_results
    
    def export_provenance_summary(self, dataset_name: str) -> Dict[str, Any]:
        """
        Export comprehensive provenance summary for publication materials.
        
        Args:
            dataset_name: Name of dataset (source or derived)
            
        Returns:
            Comprehensive provenance summary
            
        Requirements: 3.4
        """
        summary = {
            "dataset_name": dataset_name,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "provenance_type": "unknown"
        }
        
        try:
            # Check if it's a source dataset
            source_provenance = self.registry_manager.get_registry_entry(dataset_name)
            if source_provenance:
                summary["provenance_type"] = "source_dataset"
                summary["source_provenance"] = {
                    "download_timestamp": source_provenance.download_timestamp,
                    "source_used": source_provenance.source_used,
                    "download_agent": source_provenance.download_agent,
                    "verification_hash": source_provenance.verification.sha256_actual,
                    "pbuf_commit": source_provenance.environment.pbuf_commit,
                    "verification_status": "verified" if source_provenance.verification.is_valid else "failed"
                }
                
                # Check for derived datasets from this source
                derived_datasets = self.list_derived_datasets(source_dataset=dataset_name)
                if derived_datasets:
                    summary["derived_datasets"] = derived_datasets
            
            # Check if it's a derived dataset
            derived_provenance = self.get_derived_dataset_provenance(dataset_name)
            if derived_provenance and "error" not in derived_provenance:
                summary["provenance_type"] = "derived_dataset"
                
                provenance_record = derived_provenance["provenance_record"]
                file_info = provenance_record["file_info"]
                
                summary["derived_provenance"] = {
                    "source_dataset": file_info["source_dataset"],
                    "source_registry_hash": file_info["source_registry_hash"],
                    "processing_timestamp": file_info["processing_timestamp"],
                    "transformation_summary": file_info["transformation_summary"],
                    "environment_snapshot_hash": file_info["environment_snapshot_hash"],
                    "framework_version": file_info["framework_version"],
                    "provenance_chain": file_info["provenance_chain"],
                    "reproducibility_info": file_info["reproducibility_info"]
                }
                
                # Include environment snapshot details
                if derived_provenance["environment_snapshot"]:
                    summary["environment_snapshot"] = derived_provenance["environment_snapshot"]
            
            if summary["provenance_type"] == "unknown":
                summary["error"] = f"Dataset '{dataset_name}' not found in source or derived registries"
        
        except Exception as e:
            summary["error"] = f"Failed to export provenance summary: {str(e)}"
        
        return summary
    
    def cleanup_old_environment_snapshots(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old environment snapshots to manage disk space.
        
        Args:
            days_to_keep: Number of days to keep snapshots
            
        Returns:
            Cleanup results
            
        Requirements: 3.4
        """
        cleanup_results = {
            "cleanup_timestamp": datetime.now(timezone.utc).isoformat(),
            "days_to_keep": days_to_keep,
            "snapshots_found": 0,
            "snapshots_removed": 0,
            "snapshots_kept": 0,
            "errors": []
        }
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            # Find all environment snapshot files
            snapshot_files = list(self._derived_datasets_path.glob("environment_*.json"))
            cleanup_results["snapshots_found"] = len(snapshot_files)
            
            # Check which snapshots are still referenced by derived datasets
            referenced_snapshots = set()
            index_file = self._derived_datasets_path / "derived_datasets_index.json"
            
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                for dataset_info in index["datasets"].values():
                    if "environment_snapshot" in dataset_info:
                        referenced_snapshots.add(dataset_info["environment_snapshot"])
            
            # Remove old unreferenced snapshots
            for snapshot_file in snapshot_files:
                try:
                    # Extract snapshot hash from filename
                    snapshot_hash = snapshot_file.stem.replace("environment_", "")
                    
                    # Check if snapshot is referenced
                    if snapshot_hash in referenced_snapshots:
                        cleanup_results["snapshots_kept"] += 1
                        continue
                    
                    # Check snapshot age
                    with open(snapshot_file, 'r') as f:
                        snapshot_data = json.load(f)
                    
                    creation_time = datetime.fromisoformat(
                        snapshot_data["creation_timestamp"].replace('Z', '+00:00')
                    )
                    
                    if creation_time < cutoff_time:
                        snapshot_file.unlink()
                        cleanup_results["snapshots_removed"] += 1
                    else:
                        cleanup_results["snapshots_kept"] += 1
                        
                except Exception as e:
                    cleanup_results["errors"].append(f"Error processing {snapshot_file}: {str(e)}")
        
        except Exception as e:
            cleanup_results["errors"].append(f"Cleanup failed: {str(e)}")
        
        return cleanup_results