"""
Dataset integration for PBUF pipeline infrastructure

This module provides the DatasetRegistry class that integrates dataset
downloading, verification, and registry management for seamless pipeline use.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass

from ..core.manifest_schema import DatasetManifest, DatasetInfo
from ..core.registry_manager import RegistryManager, VerificationResult, ProvenanceRecord
from ..protocols.download_manager import DownloadManager
from ..verification.verification_engine import VerificationEngine
from ..core.extensible_interface import ExtensibleDatasetInterface, APIVersion, DatasetRequest
from ..core.version_control_integration import VersionControlIntegration, check_dataset_compatibility


@dataclass
class DatasetRegistryInfo:
    """Information about a dataset from registry perspective"""
    name: str
    local_path: Path
    is_verified: bool
    provenance: Optional[ProvenanceRecord]


class DatasetRegistry:
    """
    Unified dataset registry for PBUF pipeline integration
    
    Provides high-level interface for dataset fetching, verification,
    and provenance tracking that integrates seamlessly with existing pipelines.
    """
    
    def __init__(
        self,
        manifest_path: Union[str, Path] = "data/datasets_manifest.json",
        registry_path: Union[str, Path] = "data/registry"
    ):
        """
        Initialize dataset registry
        
        Args:
            manifest_path: Path to dataset manifest file
            registry_path: Path to registry directory
        """
        self.manifest_path = Path(manifest_path)
        self.registry_path = Path(registry_path)
        
        # Initialize components
        self.manifest = DatasetManifest(self.manifest_path)
        self.registry_manager = RegistryManager(self.registry_path)
        self.download_manager = DownloadManager()
        self.verification_engine = VerificationEngine()
        self.extensible_interface = ExtensibleDatasetInterface(self.registry_manager)
        self.version_control = VersionControlIntegration()
    
    def fetch_dataset(self, name: str, force_refresh: bool = False) -> DatasetRegistryInfo:
        """
        Fetch and verify dataset, creating registry entry if needed (works for both downloaded and manual datasets)
        
        Args:
            name: Dataset name
            force_refresh: If True, re-download even if dataset exists (ignored for manual datasets)
            
        Returns:
            DatasetRegistryInfo with local path and verification status
            
        Raises:
            ValueError: If dataset not found in manifest or registry
            Exception: If download or verification fails
        """
        # Check if this is a manually registered dataset
        if self.registry_manager.has_registry_entry(name):
            registry_entry = self.registry_manager.get_registry_entry(name)
            if registry_entry and registry_entry.source_used == "manual":
                # For manual datasets, just return existing info (ignore force_refresh)
                local_path = Path(registry_entry.file_info["local_path"])
                if local_path.exists():
                    return DatasetRegistryInfo(
                        name=name,
                        local_path=local_path,
                        is_verified=registry_entry.verification.is_valid,
                        provenance=registry_entry
                    )
                else:
                    raise ValueError(f"Manual dataset file not found: {local_path}")
        
        # For downloaded datasets, get dataset info from manifest
        dataset_info = self.manifest.get_dataset_info(name)
        if not dataset_info:
            raise ValueError(f"Dataset '{name}' not found in manifest")
        
        # Check if already in registry and verified
        if not force_refresh and self.registry_manager.has_registry_entry(name):
            registry_entry = self.registry_manager.get_registry_entry(name)
            if registry_entry and registry_entry.verification.is_valid:
                local_path = Path(registry_entry.file_info["local_path"])
                if local_path.exists():
                    return DatasetRegistryInfo(
                        name=name,
                        local_path=local_path,
                        is_verified=True,
                        provenance=registry_entry
                    )
        
        # Download dataset
        local_path = self._get_local_path(name)
        source_used = self.download_manager.download_dataset(dataset_info, local_path)
        
        # Verify dataset
        verification_result_obj = self.verification_engine.verify_dataset(
            name, local_path, dataset_info.verification
        )
        
        # Convert to registry VerificationResult format
        from ..core.registry_manager import VerificationResult
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
        
        # Create or update registry entry
        if self.registry_manager.has_registry_entry(name):
            self.registry_manager.update_verification(name, verification_result)
        else:
            self.registry_manager.create_registry_entry(
                dataset_info, verification_result, source_used, local_path
            )
        
        # Get updated registry entry
        registry_entry = self.registry_manager.get_registry_entry(name)
        
        return DatasetRegistryInfo(
            name=name,
            local_path=local_path,
            is_verified=verification_result.is_valid,
            provenance=registry_entry
        )
    
    def verify_dataset(self, name: str) -> VerificationResult:
        """
        Verify an existing dataset (works for both downloaded and manual datasets)
        
        Args:
            name: Dataset name
            
        Returns:
            VerificationResult with verification status
            
        Raises:
            ValueError: If dataset not found
        """
        # Check if dataset exists in registry
        if not self.registry_manager.has_registry_entry(name):
            raise ValueError(f"Dataset '{name}' not found in registry")
        
        registry_entry = self.registry_manager.get_registry_entry(name)
        if not registry_entry:
            raise ValueError(f"Failed to load registry entry for '{name}'")
        
        local_path = Path(registry_entry.file_info["local_path"])
        if not local_path.exists():
            raise ValueError(f"Dataset file not found: {local_path}")
        
        # Check if this is a manual dataset
        manual_info = self.registry_manager.get_manual_dataset_info(name)
        
        if manual_info:
            # For manual datasets, create verification config from registry data
            verification_config = {
                "sha256": registry_entry.verification.sha256_expected,
                "size_bytes": registry_entry.verification.size_expected
            }
            
            # Perform verification using verification engine
            verification_result_obj = self.verification_engine.verify_dataset(
                name, local_path, verification_config
            )
            
            # Convert to registry VerificationResult format
            from ..core.registry_manager import VerificationResult
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
        else:
            # For downloaded datasets, get dataset info from manifest
            dataset_info = self.manifest.get_dataset_info(name)
            if not dataset_info:
                raise ValueError(f"Dataset '{name}' not found in manifest")
            
            # Perform verification
            verification_result_obj = self.verification_engine.verify_dataset(
                name, local_path, dataset_info.verification
            )
            
            # Convert to registry VerificationResult format
            from ..core.registry_manager import VerificationResult
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
        
        # Update registry with new verification results
        self.registry_manager.update_verification(name, verification_result)
        
        return verification_result
    
    def get_dataset_path(self, name: str) -> Path:
        """
        Get local path to dataset file
        
        Args:
            name: Dataset name
            
        Returns:
            Path to local dataset file
            
        Raises:
            ValueError: If dataset not found or not downloaded
        """
        if not self.registry_manager.has_registry_entry(name):
            raise ValueError(f"Dataset '{name}' not found in registry")
        
        registry_entry = self.registry_manager.get_registry_entry(name)
        if not registry_entry:
            raise ValueError(f"Failed to load registry entry for '{name}'")
        
        local_path = Path(registry_entry.file_info["local_path"])
        if not local_path.exists():
            raise ValueError(f"Dataset file not found: {local_path}")
        
        return local_path
    
    def list_datasets(self) -> List[DatasetRegistryInfo]:
        """
        List all datasets in registry
        
        Returns:
            List of DatasetRegistryInfo objects
        """
        dataset_infos = []
        
        for dataset_name in self.registry_manager.list_datasets():
            try:
                registry_entry = self.registry_manager.get_registry_entry(dataset_name)
                if registry_entry:
                    local_path = Path(registry_entry.file_info["local_path"])
                    dataset_infos.append(DatasetRegistryInfo(
                        name=dataset_name,
                        local_path=local_path,
                        is_verified=registry_entry.verification.is_valid,
                        provenance=registry_entry
                    ))
            except Exception:
                # Skip corrupted entries
                continue
        
        return dataset_infos
    
    def get_provenance(self, name: str) -> Optional[ProvenanceRecord]:
        """
        Get provenance record for a dataset
        
        Args:
            name: Dataset name
            
        Returns:
            ProvenanceRecord if found, None otherwise
        """
        return self.registry_manager.get_registry_entry(name)
    
    def export_manifest_summary(self) -> Dict[str, Any]:
        """
        Export manifest summary for publication materials (includes both downloaded and manual datasets)
        
        Returns:
            Dictionary with manifest and registry summary
        """
        # Get registry summary
        registry_summary = self.registry_manager.get_registry_summary()
        
        # Get manifest datasets
        manifest_datasets = {}
        for dataset_name in self.manifest.list_datasets():
            dataset_info = self.manifest.get_dataset_info(dataset_name)
            if dataset_info:
                manifest_datasets[dataset_name] = {
                    "canonical_name": dataset_info.canonical_name,
                    "description": dataset_info.description,
                    "citation": dataset_info.citation,
                    "license": dataset_info.license,
                    "source_type": "downloaded"
                }
        
        # Get manual datasets
        manual_datasets = {}
        for dataset_name in self.registry_manager.list_manual_datasets():
            manual_info = self.registry_manager.get_manual_dataset_info(dataset_name)
            if manual_info:
                manual_datasets[dataset_name] = {
                    "canonical_name": manual_info.get("canonical_name", dataset_name),
                    "description": manual_info.get("description", ""),
                    "citation": manual_info.get("citation", ""),
                    "license": manual_info.get("license"),
                    "source_type": "manual"
                }
        
        # Combine all datasets
        all_datasets = {**manifest_datasets, **manual_datasets}
        
        return {
            "manifest_summary": {
                "total_datasets": len(manifest_datasets),
                "datasets": manifest_datasets
            },
            "manual_datasets_summary": {
                "total_datasets": len(manual_datasets),
                "datasets": manual_datasets
            },
            "combined_summary": {
                "total_datasets": len(all_datasets),
                "downloaded_datasets": len(manifest_datasets),
                "manual_datasets": len(manual_datasets),
                "datasets": all_datasets
            },
            "registry_summary": registry_summary,
            "export_timestamp": registry_summary.get("export_timestamp")
        }
    
    def register_manual_dataset(
        self,
        dataset_name: str,
        file_path: Union[str, Path],
        canonical_name: str,
        description: str,
        citation: str,
        license: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expected_sha256: Optional[str] = None,
        expected_size: Optional[int] = None,
        schema_config: Optional[Dict[str, Any]] = None
    ) -> DatasetRegistryInfo:
        """
        Register a manually provided dataset with checksum validation
        
        This function provides a high-level interface for registering datasets
        that are not available through the standard manifest sources.
        
        Args:
            dataset_name: Unique dataset identifier (letters, numbers, underscores, hyphens only)
            file_path: Path to the local dataset file
            canonical_name: Human-readable dataset name for citations
            description: Detailed description of the dataset
            citation: Citation information for the dataset
            license: Dataset license information (optional)
            metadata: Additional metadata dictionary (optional)
            expected_sha256: Expected SHA256 checksum (will calculate if not provided)
            expected_size: Expected file size in bytes (will use actual if not provided)
            schema_config: Schema validation configuration (optional)
            
        Returns:
            DatasetRegistryInfo with registration details
            
        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If the provided file doesn't exist
            Exception: If registration or verification fails
            
        Example:
            >>> registry = DatasetRegistry()
            >>> info = registry.register_manual_dataset(
            ...     dataset_name="my_custom_cmb",
            ...     file_path="/path/to/my_data.dat",
            ...     canonical_name="My Custom CMB Dataset",
            ...     description="Custom CMB distance priors from proprietary analysis",
            ...     citation="Smith et al. 2025, ApJ 123, 456",
            ...     license="Proprietary",
            ...     metadata={"dataset_type": "cmb", "n_data_points": 3}
            ... )
        """
        file_path = Path(file_path)
        
        # Validate inputs
        if not dataset_name or not dataset_name.strip():
            raise ValueError("Dataset name cannot be empty")
        
        if not canonical_name or not canonical_name.strip():
            raise ValueError("Canonical name cannot be empty")
        
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        if not citation or not citation.strip():
            raise ValueError("Citation cannot be empty")
        
        # Register the dataset using the registry manager
        provenance_record = self.registry_manager.register_manual_dataset(
            dataset_name=dataset_name,
            file_path=file_path,
            canonical_name=canonical_name,
            description=description,
            citation=citation,
            license=license,
            metadata=metadata,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            schema_config=schema_config
        )
        
        return DatasetRegistryInfo(
            name=dataset_name,
            local_path=file_path,
            is_verified=provenance_record.verification.is_valid,
            provenance=provenance_record
        )
    
    def get_manual_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get manual registration information for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with manual registration info, or None if not manually registered
        """
        return self.registry_manager.get_manual_dataset_info(dataset_name)
    
    def list_manual_datasets(self) -> List[str]:
        """
        List all manually registered datasets
        
        Returns:
            List of manually registered dataset names
        """
        return self.registry_manager.list_manual_datasets()
    
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
            ValueError: If dataset not found or not manually registered
        """
        try:
            return self.registry_manager.update_manual_dataset_metadata(
                dataset_name=dataset_name,
                canonical_name=canonical_name,
                description=description,
                citation=citation,
                license=license,
                metadata=metadata
            )
        except Exception as e:
            raise ValueError(f"Failed to update manual dataset metadata: {e}")
    
    def is_manual_dataset(self, dataset_name: str) -> bool:
        """
        Check if a dataset is manually registered
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            True if dataset is manually registered
        """
        return self.get_manual_dataset_info(dataset_name) is not None
    
    def get_dataset_by_canonical_name(self, canonical_name: str, api_version: str = "1.0") -> DatasetRegistryInfo:
        """
        Get dataset by canonical name using extensible interface
        
        Args:
            canonical_name: Human-readable dataset name
            api_version: API version to use (default: "1.0")
            
        Returns:
            DatasetRegistryInfo for the matching dataset
        """
        api_ver = APIVersion.V1_0 if api_version == "1.0" else APIVersion.V1_1
        response = self.extensible_interface.get_dataset_by_canonical_name(canonical_name, api_ver)
        
        return DatasetRegistryInfo(
            name=response.name,
            local_path=response.local_path,
            is_verified=response.is_verified,
            provenance=response.provenance
        )
    
    def request_dataset_versioned(self, name: str, api_version: str = "1.0", **kwargs) -> DatasetRegistryInfo:
        """
        Request dataset using versioned API
        
        Args:
            name: Dataset name
            api_version: API version to use
            **kwargs: Additional parameters for DatasetRequest
            
        Returns:
            DatasetRegistryInfo with dataset information
        """
        api_ver = APIVersion.V1_0 if api_version == "1.0" else APIVersion.V1_1
        request = DatasetRequest(
            name=name,
            api_version=api_ver,
            **kwargs
        )
        
        response = self.extensible_interface.request_dataset(request)
        
        return DatasetRegistryInfo(
            name=response.name,
            local_path=response.local_path,
            is_verified=response.is_verified,
            provenance=response.provenance
        )
    
    def check_dataset_compatibility(self, dataset_name: str, strict_mode: bool = False) -> Dict[str, Any]:
        """
        Check dataset compatibility with current environment
        
        Args:
            dataset_name: Name of the dataset
            strict_mode: If True, require exact environment match
            
        Returns:
            Dictionary with compatibility information
        """
        try:
            compatibility = check_dataset_compatibility(dataset_name, strict_mode)
            return {
                "compatible": compatibility.is_compatible,
                "compatibility_level": compatibility.compatibility_level,
                "issues": compatibility.compatibility_issues,
                "warnings": compatibility.compatibility_warnings,
                "recommendations": compatibility.recommendations,
                "dataset_pbuf_commit": compatibility.dataset_environment.pbuf_commit,
                "current_pbuf_commit": compatibility.current_environment.pbuf_commit
            }
        except Exception as e:
            return {
                "compatible": False,
                "error": str(e)
            }
    
    def get_datasets_by_commit(self, pbuf_commit: str) -> List[str]:
        """
        Get datasets registered with a specific PBUF commit
        
        Args:
            pbuf_commit: PBUF git commit hash
            
        Returns:
            List of dataset names
        """
        return self.version_control.get_datasets_by_commit(pbuf_commit)
    
    def validate_reproducibility(self, dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate reproducibility requirements for datasets
        
        Args:
            dataset_names: List of dataset names (defaults to all datasets)
            
        Returns:
            Dictionary with validation results
        """
        if dataset_names is None:
            dataset_names = self.registry_manager.list_datasets()
        
        return self.version_control.validate_reproducibility_requirements(dataset_names)
    
    def create_reproducibility_report(self, dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create comprehensive reproducibility report
        
        Args:
            dataset_names: List of dataset names (defaults to all datasets)
            
        Returns:
            Dictionary with complete reproducibility information
        """
        if dataset_names is None:
            dataset_names = self.registry_manager.list_datasets()
        
        return self.version_control.create_reproducibility_manifest(dataset_names)
    
    def get_current_environment_info(self) -> Dict[str, Any]:
        """
        Get current environment information
        
        Returns:
            Dictionary with current environment details
        """
        return self.version_control.export_environment_summary()
    
    def list_available_datasets_versioned(self, api_version: str = "1.0") -> List[Dict[str, Any]]:
        """
        List available datasets using versioned API
        
        Args:
            api_version: API version to use
            
        Returns:
            List of dataset information dictionaries
        """
        api_ver = APIVersion.V1_0 if api_version == "1.0" else APIVersion.V1_1
        return self.extensible_interface.list_available_datasets(api_ver)
    
    def get_supported_api_versions(self) -> List[str]:
        """
        Get list of supported API versions
        
        Returns:
            List of supported API version strings
        """
        return self.extensible_interface.get_supported_versions()
    
    def _get_local_path(self, dataset_name: str) -> Path:
        """Get local path for storing dataset file"""
        datasets_dir = Path("data/datasets")
        datasets_dir.mkdir(parents=True, exist_ok=True)
        return datasets_dir / f"{dataset_name}.dat"


# Convenience functions for pipeline integration
def load_dataset(name: str) -> DatasetRegistryInfo:
    """
    Load dataset using registry system
    
    Args:
        name: Dataset name
        
    Returns:
        DatasetRegistryInfo with local path and verification status
    """
    registry = DatasetRegistry()
    return registry.fetch_dataset(name)


def verify_all_datasets(dataset_names: List[str]) -> bool:
    """
    Verify all datasets in the list
    
    Args:
        dataset_names: List of dataset names to verify
        
    Returns:
        True if all datasets are verified, False otherwise
    """
    if not dataset_names:
        return True
    
    registry = DatasetRegistry()
    
    for name in dataset_names:
        try:
            verification_result = registry.verify_dataset(name)
            if not verification_result.is_valid:
                return False
        except Exception:
            return False
    
    return True


def register_manual_dataset(
    dataset_name: str,
    file_path: Union[str, Path],
    canonical_name: str,
    description: str,
    citation: str,
    license: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    expected_sha256: Optional[str] = None,
    expected_size: Optional[int] = None,
    schema_config: Optional[Dict[str, Any]] = None
) -> DatasetRegistryInfo:
    """
    Register a manually provided dataset with checksum validation
    
    Convenience function for manual dataset registration that creates a registry
    instance and performs the registration.
    
    Args:
        dataset_name: Unique dataset identifier
        file_path: Path to the local dataset file
        canonical_name: Human-readable dataset name
        description: Dataset description
        citation: Citation information
        license: Dataset license (optional)
        metadata: Additional metadata (optional)
        expected_sha256: Expected SHA256 checksum (optional)
        expected_size: Expected file size in bytes (optional)
        schema_config: Schema validation configuration (optional)
        
    Returns:
        DatasetRegistryInfo with registration details
        
    Example:
        >>> info = register_manual_dataset(
        ...     dataset_name="my_custom_sn",
        ...     file_path="/data/my_supernovae.dat",
        ...     canonical_name="My Supernova Sample",
        ...     description="Custom supernova distance measurements",
        ...     citation="Doe et al. 2025, MNRAS 456, 789",
        ...     metadata={"dataset_type": "sn", "n_data_points": 150}
        ... )
    """
    registry = DatasetRegistry()
    return registry.register_manual_dataset(
        dataset_name=dataset_name,
        file_path=file_path,
        canonical_name=canonical_name,
        description=description,
        citation=citation,
        license=license,
        metadata=metadata,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
        schema_config=schema_config
    )


def list_manual_datasets() -> List[str]:
    """
    List all manually registered datasets
    
    Returns:
        List of manually registered dataset names
    """
    registry = DatasetRegistry()
    return registry.list_manual_datasets()


def get_manual_dataset_info(dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Get manual registration information for a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with manual registration info, or None if not manually registered
    """
    registry = DatasetRegistry()
    return registry.get_manual_dataset_info(dataset_name)


# Versioned API convenience functions

def get_dataset_by_canonical_name(canonical_name: str, api_version: str = "1.0") -> DatasetRegistryInfo:
    """
    Get dataset by canonical name using versioned API
    
    Args:
        canonical_name: Human-readable dataset name
        api_version: API version to use
        
    Returns:
        DatasetRegistryInfo for the matching dataset
    """
    registry = DatasetRegistry()
    return registry.get_dataset_by_canonical_name(canonical_name, api_version)


def request_dataset_versioned(name: str, api_version: str = "1.0", **kwargs) -> DatasetRegistryInfo:
    """
    Request dataset using versioned API
    
    Args:
        name: Dataset name
        api_version: API version to use
        **kwargs: Additional parameters
        
    Returns:
        DatasetRegistryInfo with dataset information
    """
    registry = DatasetRegistry()
    return registry.request_dataset_versioned(name, api_version, **kwargs)


def list_available_datasets_versioned(api_version: str = "1.0") -> List[Dict[str, Any]]:
    """
    List available datasets using versioned API
    
    Args:
        api_version: API version to use
        
    Returns:
        List of dataset information dictionaries
    """
    registry = DatasetRegistry()
    return registry.list_available_datasets_versioned(api_version)


def check_dataset_compatibility(dataset_name: str, strict_mode: bool = False) -> Dict[str, Any]:
    """
    Check dataset compatibility with current environment
    
    Args:
        dataset_name: Name of the dataset
        strict_mode: If True, require exact environment match
        
    Returns:
        Dictionary with compatibility information
    """
    registry = DatasetRegistry()
    return registry.check_dataset_compatibility(dataset_name, strict_mode)


def validate_reproducibility(dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate reproducibility requirements for datasets
    
    Args:
        dataset_names: List of dataset names (defaults to all datasets)
        
    Returns:
        Dictionary with validation results
    """
    registry = DatasetRegistry()
    return registry.validate_reproducibility(dataset_names)


def create_reproducibility_report(dataset_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create comprehensive reproducibility report
    
    Args:
        dataset_names: List of dataset names (defaults to all datasets)
        
    Returns:
        Dictionary with complete reproducibility information
    """
    registry = DatasetRegistry()
    return registry.create_reproducibility_report(dataset_names)


def get_current_environment_info() -> Dict[str, Any]:
    """
    Get current environment information
    
    Returns:
        Dictionary with current environment details
    """
    registry = DatasetRegistry()
    return registry.get_current_environment_info()


def get_supported_api_versions() -> List[str]:
    """
    Get list of supported API versions
    
    Returns:
        List of supported API version strings
    """
    registry = DatasetRegistry()
    return registry.get_supported_api_versions()