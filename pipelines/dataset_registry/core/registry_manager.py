"""
Registry manager for immutable provenance tracking

This module provides the RegistryManager class for maintaining dataset
registry entries with complete provenance tracking and audit trails.
"""

import json
import fcntl
import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager

from .manifest_schema import DatasetInfo
from .logging_integration import get_logging_integration, log_registry_operation


@dataclass
class EnvironmentInfo:
    """Environment fingerprint for reproducibility"""
    pbuf_commit: Optional[str]
    python_version: str
    platform: str
    hostname: str
    timestamp: str
    
    @classmethod
    def collect(cls) -> 'EnvironmentInfo':
        """Collect current environment information"""
        # Use version control integration for enhanced environment collection
        try:
            from .version_control_integration import VersionControlIntegration
            vc_integration = VersionControlIntegration()
            full_fingerprint = vc_integration.collect_environment_fingerprint()
            
            return cls(
                pbuf_commit=full_fingerprint.pbuf_commit,
                python_version=full_fingerprint.python_version,
                platform=full_fingerprint.platform,
                hostname=full_fingerprint.hostname,
                timestamp=full_fingerprint.timestamp
            )
        except ImportError:
            # Fallback to basic collection if version control integration not available
            return cls(
                pbuf_commit=cls._get_pbuf_commit(),
                python_version=sys.version,
                platform=platform.platform(),
                hostname=platform.node(),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    @staticmethod
    def _get_pbuf_commit() -> Optional[str]:
        """Get current PBUF git commit hash (fallback method)"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return None


@dataclass
class VerificationResult:
    """Results of dataset verification"""
    sha256_verified: bool
    sha256_expected: Optional[str]
    sha256_actual: Optional[str]
    size_verified: bool
    size_expected: Optional[int]
    size_actual: Optional[int]
    schema_verified: bool
    schema_errors: List[str]
    verification_timestamp: str
    
    @property
    def is_valid(self) -> bool:
        """Check if all verifications passed"""
        return (self.sha256_verified and 
                self.size_verified and 
                self.schema_verified)


@dataclass
class ProvenanceRecord:
    """Complete provenance record for a dataset"""
    dataset_name: str
    download_timestamp: str
    source_used: str
    download_agent: str
    environment: EnvironmentInfo
    verification: VerificationResult
    file_info: Dict[str, Any]
    registry_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "dataset_name": self.dataset_name,
            "registry_version": self.registry_version,
            "provenance": {
                "download_timestamp": self.download_timestamp,
                "source_used": self.source_used,
                "download_agent": self.download_agent,
                "environment": asdict(self.environment)
            },
            "verification": asdict(self.verification),
            "file_info": self.file_info,
            "status": "verified" if self.verification.is_valid else "failed",
            "created_at": self.download_timestamp,
            "last_verified": self.verification.verification_timestamp
        }


class RegistryError(Exception):
    """Base exception for registry operations"""
    pass


class RegistryLockError(RegistryError):
    """Raised when registry file locking fails"""
    pass


class RegistryManager:
    """
    Registry manager for immutable provenance tracking
    
    Manages dataset registry entries with complete metadata, immutable
    audit trails, and file locking for concurrent write safety.
    """
    
    def __init__(self, registry_path: Union[str, Path], enable_structured_logging: bool = True):
        """
        Initialize registry manager
        
        Args:
            registry_path: Path to registry directory
            enable_structured_logging: Whether to enable structured logging integration
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.enable_structured_logging = enable_structured_logging
        
        # Create audit log file if it doesn't exist
        self.audit_log_path = self.registry_path / "audit.jsonl"
        if not self.audit_log_path.exists():
            self.audit_log_path.touch()
        
        # Initialize logging integration if enabled
        if self.enable_structured_logging:
            try:
                from .logging_integration import configure_logging_integration
                configure_logging_integration(
                    registry_path=self.registry_path,
                    pbuf_log_level="INFO",
                    enable_structured_logging=True,
                    enable_pbuf_integration=True
                )
            except Exception:
                # Continue without structured logging if setup fails
                self.enable_structured_logging = False
    
    def _get_registry_file_path(self, dataset_name: str) -> Path:
        """Get path to registry file for a dataset"""
        return self.registry_path / f"{dataset_name}.json"
    
    def _get_lock_file_path(self, dataset_name: str) -> Path:
        """Get path to lock file for a dataset"""
        return self.registry_path / f"{dataset_name}.lock"
    
    @contextmanager
    def _file_lock(self, dataset_name: str):
        """
        Context manager for file locking
        
        Args:
            dataset_name: Dataset name to lock
            
        Raises:
            RegistryLockError: If lock cannot be acquired
        """
        lock_file_path = self._get_lock_file_path(dataset_name)
        
        try:
            with open(lock_file_path, 'w') as lock_file:
                try:
                    # Acquire exclusive lock with timeout
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    yield
                except BlockingIOError:
                    raise RegistryLockError(f"Could not acquire lock for dataset '{dataset_name}'")
                finally:
                    # Lock is automatically released when file is closed
                    pass
        except IOError as e:
            raise RegistryLockError(f"Failed to create lock file for dataset '{dataset_name}': {e}")
        finally:
            # Clean up lock file
            try:
                lock_file_path.unlink(missing_ok=True)
            except OSError:
                pass  # Lock file cleanup is best-effort
    
    def create_registry_entry(
        self,
        dataset_info: DatasetInfo,
        verification_result: VerificationResult,
        source_used: str,
        local_path: Path,
        download_agent: str = "dataset-registry-v1.0"
    ) -> ProvenanceRecord:
        """
        Create a new registry entry with complete metadata
        
        Args:
            dataset_info: Dataset information from manifest
            verification_result: Results of dataset verification
            source_used: Which source was used for download
            local_path: Local path to the dataset file
            download_agent: Agent identifier that performed the download
            
        Returns:
            ProvenanceRecord: Created provenance record
            
        Raises:
            RegistryError: If registry entry creation fails
            RegistryLockError: If file locking fails
        """
        with self._file_lock(dataset_info.name):
            # Check if entry already exists
            registry_file = self._get_registry_file_path(dataset_info.name)
            if registry_file.exists():
                raise RegistryError(f"Registry entry for '{dataset_info.name}' already exists")
            
            # Collect environment information
            environment = EnvironmentInfo.collect()
            
            # Create file info
            file_info = {
                "local_path": str(local_path),
                "original_filename": local_path.name,
                "mime_type": self._detect_mime_type(local_path)
            }
            
            # Create provenance record
            provenance_record = ProvenanceRecord(
                dataset_name=dataset_info.name,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
                source_used=source_used,
                download_agent=download_agent,
                environment=environment,
                verification=verification_result,
                file_info=file_info
            )
            
            # Write registry entry
            try:
                with open(registry_file, 'w') as f:
                    json.dump(provenance_record.to_dict(), f, indent=2, sort_keys=True)
            except IOError as e:
                # Log failure to structured logging system
                if self.enable_structured_logging:
                    log_registry_operation(
                        "registry_create",
                        dataset_info.name,
                        status="failed",
                        error=str(e),
                        metadata={"source": source_used, "agent": download_agent}
                    )
                raise RegistryError(f"Failed to write registry entry for '{dataset_info.name}': {e}")
            
            # Append to audit log
            self._append_audit_log("registry_create", {
                "dataset_name": dataset_info.name,
                "timestamp": provenance_record.download_timestamp,
                "agent": download_agent,
                "pbuf_commit": environment.pbuf_commit
            })
            
            # Log success to structured logging system
            if self.enable_structured_logging:
                log_registry_operation(
                    "registry_create",
                    dataset_info.name,
                    status="success",
                    metadata={
                        "source": source_used,
                        "agent": download_agent,
                        "pbuf_commit": environment.pbuf_commit,
                        "verification_status": "verified" if verification_result.is_valid else "failed",
                        "sha256_match": verification_result.sha256_verified,
                        "size_bytes": verification_result.size_actual
                    }
                )
            
            return provenance_record
    
    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type of a file"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or "application/octet-stream"
    
    def _append_audit_log(self, event_type: str, event_data: Dict[str, Any]):
        """
        Append event to immutable audit log
        
        Args:
            event_type: Type of event (e.g., 'registry_create', 'registry_update')
            event_data: Event-specific data
        """
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **event_data
        }
        
        try:
            with open(self.audit_log_path, 'a') as f:
                json.dump(audit_entry, f, sort_keys=True)
                f.write('\n')
        except IOError:
            # Audit log failures should not break the main operation
            pass
        
        # Also log to structured logging system if enabled
        if self.enable_structured_logging:
            from .structured_logging import log_dataset_event
            log_dataset_event(
                event_type=event_type,
                dataset_name=event_data.get("dataset_name"),
                level="INFO",
                metadata=event_data
            )
    
    def get_registry_entry(self, dataset_name: str) -> Optional[ProvenanceRecord]:
        """
        Get registry entry for a dataset (works for both downloaded and manual datasets)
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            ProvenanceRecord if found, None otherwise
            
        Raises:
            RegistryError: If registry entry is corrupted
        """
        registry_file = self._get_registry_file_path(dataset_name)
        
        if not registry_file.exists():
            return None
        
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
            
            # Convert back to ProvenanceRecord
            return self._dict_to_provenance_record(data)
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise RegistryError(f"Corrupted registry entry for '{dataset_name}': {e}")
    
    def _dict_to_provenance_record(self, data: Dict[str, Any]) -> ProvenanceRecord:
        """Convert dictionary back to ProvenanceRecord"""
        provenance_data = data["provenance"]
        verification_data = data["verification"]
        
        environment = EnvironmentInfo(**provenance_data["environment"])
        verification = VerificationResult(**verification_data)
        
        return ProvenanceRecord(
            dataset_name=data["dataset_name"],
            download_timestamp=provenance_data["download_timestamp"],
            source_used=provenance_data["source_used"],
            download_agent=provenance_data["download_agent"],
            environment=environment,
            verification=verification,
            file_info=data["file_info"],
            registry_version=data.get("registry_version", "1.0")
        )
    
    def has_registry_entry(self, dataset_name: str) -> bool:
        """
        Check if registry entry exists for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if registry entry exists
        """
        registry_file = self._get_registry_file_path(dataset_name)
        return registry_file.exists()
    
    def update_verification(
        self,
        dataset_name: str,
        verification_result: VerificationResult
    ) -> bool:
        """
        Update verification results for an existing registry entry (works for both downloaded and manual datasets)
        
        Args:
            dataset_name: Name of the dataset
            verification_result: New verification results
            
        Returns:
            True if update was successful
            
        Raises:
            RegistryError: If dataset not found or update fails
            RegistryLockError: If file locking fails
        """
        with self._file_lock(dataset_name):
            registry_file = self._get_registry_file_path(dataset_name)
            
            if not registry_file.exists():
                raise RegistryError(f"Registry entry for '{dataset_name}' not found")
            
            try:
                # Load existing entry
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                # Update verification data
                data["verification"] = asdict(verification_result)
                data["last_verified"] = verification_result.verification_timestamp
                data["status"] = "verified" if verification_result.is_valid else "failed"
                
                # Write updated entry
                with open(registry_file, 'w') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                
                # Determine source type for audit log
                source_type = data.get("provenance", {}).get("source_used", "unknown")
                
                # Append to audit log
                self._append_audit_log("verification_update", {
                    "dataset_name": dataset_name,
                    "verification_status": "success" if verification_result.is_valid else "failed",
                    "source_type": source_type
                })
                
                # Log to structured logging system
                if self.enable_structured_logging:
                    log_registry_operation(
                        "verification_update",
                        dataset_name,
                        status="success" if verification_result.is_valid else "warning",
                        metadata={
                            "verification_status": "verified" if verification_result.is_valid else "failed",
                            "sha256_match": verification_result.sha256_verified,
                            "size_match": verification_result.size_verified,
                            "schema_valid": verification_result.schema_verified,
                            "source_type": source_type
                        }
                    )
                
                return True
                
            except (json.JSONDecodeError, IOError) as e:
                # Log failure to structured logging system
                if self.enable_structured_logging:
                    log_registry_operation(
                        "verification_update",
                        dataset_name,
                        status="failed",
                        error=str(e)
                    )
                raise RegistryError(f"Failed to update registry entry for '{dataset_name}': {e}")
    
    def list_datasets(self) -> List[str]:
        """
        List all datasets in the registry
        
        Returns:
            List of dataset names
        """
        dataset_names = []
        
        for registry_file in self.registry_path.glob("*.json"):
            if registry_file.name != "audit.jsonl":
                # Extract dataset name from filename
                dataset_name = registry_file.stem
                dataset_names.append(dataset_name)
        
        return sorted(dataset_names)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary of all registry entries (includes both downloaded and manual datasets)
        
        Returns:
            Dictionary with registry summary information
        """
        datasets = self.list_datasets()
        summary = {
            "total_datasets": len(datasets),
            "verified_datasets": 0,
            "failed_datasets": 0,
            "downloaded_datasets": 0,
            "manual_datasets": 0,
            "datasets": []
        }
        
        for dataset_name in datasets:
            try:
                entry = self.get_registry_entry(dataset_name)
                if entry:
                    if entry.verification.is_valid:
                        summary["verified_datasets"] += 1
                    else:
                        summary["failed_datasets"] += 1
                    
                    # Count by source type
                    if entry.source_used == "manual":
                        summary["manual_datasets"] += 1
                    else:
                        summary["downloaded_datasets"] += 1
                    
                    summary["datasets"].append({
                        "name": dataset_name,
                        "status": "verified" if entry.verification.is_valid else "failed",
                        "last_verified": entry.verification.verification_timestamp,
                        "source_used": entry.source_used,
                        "source_type": "manual" if entry.source_used == "manual" else "downloaded",
                        "pbuf_commit": entry.environment.pbuf_commit
                    })
            except RegistryError:
                summary["failed_datasets"] += 1
                summary["datasets"].append({
                    "name": dataset_name,
                    "status": "corrupted",
                    "last_verified": None,
                    "source_used": None,
                    "source_type": "unknown",
                    "pbuf_commit": None
                })
        
        return summary
    
    def get_audit_trail(self, dataset_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get audit trail entries
        
        Args:
            dataset_name: If provided, filter entries for this dataset only
            
        Returns:
            List of audit trail entries
        """
        audit_entries = []
        
        if not self.audit_log_path.exists():
            return audit_entries
        
        try:
            with open(self.audit_log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if dataset_name is None or entry.get("dataset_name") == dataset_name:
                                audit_entries.append(entry)
                        except json.JSONDecodeError:
                            continue  # Skip malformed lines
        except IOError:
            pass  # Return empty list if audit log cannot be read
        
        return audit_entries
    
    def export_provenance_summary(self, dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Export provenance summary for publication materials (includes both downloaded and manual datasets)
        
        Args:
            dataset_names: If provided, export only these datasets
            
        Returns:
            Dictionary with complete provenance information
        """
        if dataset_names is None:
            dataset_names = self.list_datasets()
        
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "registry_version": "1.0",
            "total_datasets": len(dataset_names),
            "datasets": {},
            "environment_summary": {}
        }
        
        # Collect environment information from all entries
        environments = {}
        
        for dataset_name in dataset_names:
            try:
                entry = self.get_registry_entry(dataset_name)
                if entry:
                    # Get manual registration info if available
                    manual_info = self.get_manual_dataset_info(dataset_name)
                    
                    dataset_export = {
                        "canonical_name": dataset_name,  # Default to dataset name
                        "download_timestamp": entry.download_timestamp,
                        "source_used": entry.source_used,
                        "source_type": "manual" if entry.source_used == "manual" else "downloaded",
                        "verification_status": "verified" if entry.verification.is_valid else "failed",
                        "sha256": entry.verification.sha256_actual,
                        "file_size": entry.verification.size_actual,
                        "pbuf_commit": entry.environment.pbuf_commit,
                        "python_version": entry.environment.python_version,
                        "platform": entry.environment.platform
                    }
                    
                    # Add manual registration details if available
                    if manual_info:
                        dataset_export.update({
                            "canonical_name": manual_info.get("canonical_name", dataset_name),
                            "description": manual_info.get("description"),
                            "citation": manual_info.get("citation"),
                            "license": manual_info.get("license"),
                            "metadata": manual_info.get("metadata", {}),
                            "registered_at": manual_info.get("registered_at")
                        })
                    
                    export_data["datasets"][dataset_name] = dataset_export
                    
                    # Track unique environments
                    env_key = f"{entry.environment.pbuf_commit}_{entry.environment.python_version}"
                    if env_key not in environments:
                        environments[env_key] = {
                            "pbuf_commit": entry.environment.pbuf_commit,
                            "python_version": entry.environment.python_version,
                            "platform": entry.environment.platform,
                            "datasets": []
                        }
                    environments[env_key]["datasets"].append(dataset_name)
            except RegistryError:
                export_data["datasets"][dataset_name] = {
                    "status": "error",
                    "error": "Registry entry corrupted or missing"
                }
        
        export_data["environment_summary"] = environments
        return export_data
    
    def get_datasets_by_commit(self, pbuf_commit: str) -> List[str]:
        """
        Get datasets registered with a specific PBUF commit
        
        Args:
            pbuf_commit: PBUF git commit hash
            
        Returns:
            List of dataset names
        """
        matching_datasets = []
        
        for dataset_name in self.list_datasets():
            try:
                entry = self.get_registry_entry(dataset_name)
                if entry and entry.environment.pbuf_commit == pbuf_commit:
                    matching_datasets.append(dataset_name)
            except RegistryError:
                continue
        
        return matching_datasets
    
    def get_datasets_by_status(self, status: str) -> List[str]:
        """
        Get datasets by verification status
        
        Args:
            status: Status to filter by ('verified', 'failed', 'corrupted')
            
        Returns:
            List of dataset names
        """
        matching_datasets = []
        
        for dataset_name in self.list_datasets():
            try:
                entry = self.get_registry_entry(dataset_name)
                if entry:
                    if status == "verified" and entry.verification.is_valid:
                        matching_datasets.append(dataset_name)
                    elif status == "failed" and not entry.verification.is_valid:
                        matching_datasets.append(dataset_name)
            except RegistryError:
                if status == "corrupted":
                    matching_datasets.append(dataset_name)
        
        return matching_datasets
    
    def validate_registry_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of all registry entries
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "total_entries": 0,
            "valid_entries": 0,
            "corrupted_entries": 0,
            "errors": [],
            "warnings": []
        }
        
        datasets = self.list_datasets()
        results["total_entries"] = len(datasets)
        
        for dataset_name in datasets:
            try:
                entry = self.get_registry_entry(dataset_name)
                if entry:
                    results["valid_entries"] += 1
                    
                    # Check for potential issues
                    if not entry.environment.pbuf_commit:
                        results["warnings"].append(f"Dataset '{dataset_name}' missing PBUF commit hash")
                    
                    if not entry.verification.is_valid:
                        results["warnings"].append(f"Dataset '{dataset_name}' has failed verification")
                
            except RegistryError as e:
                results["valid"] = False
                results["corrupted_entries"] += 1
                results["errors"].append(f"Dataset '{dataset_name}': {str(e)}")
        
        return results
    
    def register_manual_dataset(
        self,
        dataset_name: str,
        file_path: Path,
        canonical_name: str,
        description: str,
        citation: str,
        license: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_sha256: Optional[str] = None,
        expected_size: Optional[int] = None,
        schema_config: Optional[Dict[str, Any]] = None,
        download_agent: str = "manual-registration-v1.0"
    ) -> ProvenanceRecord:
        """
        Register a manually provided dataset with checksum validation
        
        Args:
            dataset_name: Unique dataset identifier
            file_path: Path to the local dataset file
            canonical_name: Human-readable dataset name
            description: Dataset description
            citation: Citation information
            license: Dataset license (optional)
            metadata: Additional metadata (optional)
            expected_sha256: Expected SHA256 checksum (optional, will calculate if not provided)
            expected_size: Expected file size in bytes (optional, will use actual if not provided)
            schema_config: Schema validation configuration (optional)
            download_agent: Agent identifier for manual registration
            
        Returns:
            ProvenanceRecord: Created provenance record
            
        Raises:
            RegistryError: If registration fails or dataset already exists
            RegistryLockError: If file locking fails
            FileNotFoundError: If the provided file doesn't exist
            VerificationError: If checksum validation fails
        """
        # Validate inputs
        if not dataset_name or not dataset_name.strip():
            raise RegistryError("Dataset name cannot be empty")
        
        if not canonical_name or not canonical_name.strip():
            raise RegistryError("Canonical name cannot be empty")
        
        if not description or not description.strip():
            raise RegistryError("Description cannot be empty")
        
        if not citation or not citation.strip():
            raise RegistryError("Citation cannot be empty")
        
        # Validate dataset name format
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', dataset_name):
            raise RegistryError(f"Invalid dataset name format: {dataset_name}. Use only letters, numbers, underscores, and hyphens.")
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read one byte
        except PermissionError:
            raise RegistryError(f"Cannot read dataset file: {file_path}")
        
        with self._file_lock(dataset_name):
            # Check if entry already exists
            registry_file = self._get_registry_file_path(dataset_name)
            if registry_file.exists():
                raise RegistryError(f"Registry entry for '{dataset_name}' already exists")
            
            # Calculate actual file properties
            actual_size = file_path.stat().st_size
            
            # Calculate SHA256 if not provided
            if expected_sha256 is None:
                try:
                    sha256_hash = hashlib.sha256()
                    with open(file_path, 'rb') as f:
                        while chunk := f.read(8192):
                            sha256_hash.update(chunk)
                    expected_sha256 = sha256_hash.hexdigest()
                except Exception as e:
                    raise RegistryError(f"Failed to calculate SHA256 checksum: {e}")
            
            # Use actual size if expected size not provided
            if expected_size is None:
                expected_size = actual_size
            
            # Create verification configuration
            verification_config = {
                "sha256": expected_sha256,
                "size_bytes": expected_size
            }
            if schema_config:
                verification_config["schema"] = schema_config
            
            # Perform verification using the verification engine
            from ..verification.verification_engine import VerificationEngine
            verification_engine = VerificationEngine()
            
            try:
                verification_result_obj = verification_engine.verify_dataset(
                    dataset_name, file_path, verification_config
                )
                
                # Convert to registry VerificationResult format
                verification_result = VerificationResult(
                    sha256_verified=verification_result_obj.sha256_match,
                    sha256_expected=verification_result_obj.sha256_expected,
                    sha256_actual=verification_result_obj.sha256_actual,
                    size_verified=verification_result_obj.size_match,
                    size_expected=verification_result_obj.size_expected,
                    size_actual=verification_result_obj.size_actual,
                    schema_verified=verification_result_obj.schema_valid,
                    schema_errors=verification_result_obj.schema_errors,
                    verification_timestamp=verification_result_obj.verification_time.isoformat()
                )
                
                if not verification_result_obj.is_valid:
                    error_details = "; ".join(verification_result_obj.errors)
                    raise RegistryError(f"Dataset verification failed: {error_details}")
                
            except Exception as e:
                if isinstance(e, RegistryError):
                    raise
                raise RegistryError(f"Verification failed during manual registration: {e}")
            
            # Collect environment information
            environment = EnvironmentInfo.collect()
            
            # Create file info
            file_info = {
                "local_path": str(file_path),
                "original_filename": file_path.name,
                "mime_type": self._detect_mime_type(file_path),
                "registration_type": "manual"
            }
            
            # Create provenance record
            provenance_record = ProvenanceRecord(
                dataset_name=dataset_name,
                download_timestamp=datetime.now(timezone.utc).isoformat(),
                source_used="manual",
                download_agent=download_agent,
                environment=environment,
                verification=verification_result,
                file_info=file_info
            )
            
            # Write registry entry
            try:
                registry_data = provenance_record.to_dict()
                # Add manual registration specific metadata
                registry_data["manual_registration"] = {
                    "canonical_name": canonical_name,
                    "description": description,
                    "citation": citation,
                    "license": license,
                    "metadata": metadata or {},
                    "registered_at": datetime.now(timezone.utc).isoformat()
                }
                
                with open(registry_file, 'w') as f:
                    json.dump(registry_data, f, indent=2, sort_keys=True)
            except IOError as e:
                raise RegistryError(f"Failed to write registry entry for '{dataset_name}': {e}")
            
            # Append to audit log
            self._append_audit_log("manual_registration", {
                "dataset_name": dataset_name,
                "canonical_name": canonical_name,
                "file_path": str(file_path),
                "file_size": actual_size,
                "sha256": expected_sha256,
                "timestamp": provenance_record.download_timestamp,
                "agent": download_agent,
                "pbuf_commit": environment.pbuf_commit
            })
            
            return provenance_record
    
    def get_manual_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get manual registration information for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with manual registration info, or None if not manually registered
        """
        registry_file = self._get_registry_file_path(dataset_name)
        
        if not registry_file.exists():
            return None
        
        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)
            
            # Check if this is a manually registered dataset
            if data.get("provenance", {}).get("source_used") == "manual":
                return data.get("manual_registration")
            
            return None
        
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    
    def list_manual_datasets(self) -> List[str]:
        """
        List all manually registered datasets
        
        Returns:
            List of manually registered dataset names
        """
        manual_datasets = []
        
        for dataset_name in self.list_datasets():
            if self.get_manual_dataset_info(dataset_name) is not None:
                manual_datasets.append(dataset_name)
        
        return manual_datasets
    
    def update_manual_dataset_metadata(
        self,
        dataset_name: str,
        canonical_name: Optional[str] = None,
        description: Optional[str] = None,
        citation: Optional[str] = None,
        license: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update metadata for a manually registered dataset
        
        Args:
            dataset_name: Name of the dataset
            canonical_name: New canonical name (optional)
            description: New description (optional)
            citation: New citation (optional)
            license: New license (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if update was successful
            
        Raises:
            RegistryError: If dataset not found or not manually registered
            RegistryLockError: If file locking fails
        """
        with self._file_lock(dataset_name):
            registry_file = self._get_registry_file_path(dataset_name)
            
            if not registry_file.exists():
                raise RegistryError(f"Registry entry for '{dataset_name}' not found")
            
            try:
                # Load existing entry
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                # Check if this is a manually registered dataset
                if data.get("provenance", {}).get("source_used") != "manual":
                    raise RegistryError(f"Dataset '{dataset_name}' is not manually registered")
                
                # Update manual registration metadata
                manual_reg = data.get("manual_registration", {})
                
                if canonical_name is not None:
                    manual_reg["canonical_name"] = canonical_name
                if description is not None:
                    manual_reg["description"] = description
                if citation is not None:
                    manual_reg["citation"] = citation
                if license is not None:
                    manual_reg["license"] = license
                if metadata is not None:
                    manual_reg["metadata"] = metadata
                
                manual_reg["last_updated"] = datetime.now(timezone.utc).isoformat()
                data["manual_registration"] = manual_reg
                
                # Write updated entry
                with open(registry_file, 'w') as f:
                    json.dump(data, f, indent=2, sort_keys=True)
                
                # Append to audit log
                self._append_audit_log("manual_metadata_update", {
                    "dataset_name": dataset_name,
                    "updated_fields": [k for k, v in {
                        "canonical_name": canonical_name,
                        "description": description,
                        "citation": citation,
                        "license": license,
                        "metadata": metadata
                    }.items() if v is not None]
                })
                
                return True
                
            except (json.JSONDecodeError, IOError) as e:
                raise RegistryError(f"Failed to update manual dataset metadata for '{dataset_name}': {e}")
    
    def remove_registry_entry(self, dataset_name: str) -> bool:
        """
        Remove a registry entry (for cleanup purposes)
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if entry was removed, False if not found
            
        Raises:
            RegistryLockError: If file locking fails
        """
        with self._file_lock(dataset_name):
            registry_file = self._get_registry_file_path(dataset_name)
            
            if registry_file.exists():
                try:
                    registry_file.unlink()
                    
                    # Append to audit log
                    self._append_audit_log("registry_remove", {
                        "dataset_name": dataset_name
                    })
                    
                    return True
                except OSError as e:
                    raise RegistryError(f"Failed to remove registry entry for '{dataset_name}': {e}")
            
            return False