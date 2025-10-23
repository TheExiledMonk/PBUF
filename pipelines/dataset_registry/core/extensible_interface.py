"""
Extensible dataset interface with versioned API and plugin architecture

This module provides the extensible interface for dataset access with support
for versioned APIs, plugin architecture for new dataset types and protocols,
and future extension capabilities.
"""

import abc
import importlib
import inspect
from typing import Dict, List, Any, Optional, Union, Type, Protocol, runtime_checkable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .manifest_schema import DatasetInfo
from .registry_manager import ProvenanceRecord, VerificationResult


class APIVersion(Enum):
    """Supported API versions"""
    V1_0 = "1.0"
    V1_1 = "1.1"  # Future version for demonstration


@dataclass
class DatasetRequest:
    """Versioned dataset request structure"""
    name: str
    api_version: APIVersion
    force_refresh: bool = False
    verification_level: str = "standard"  # standard, strict, minimal
    metadata_requirements: Optional[Dict[str, Any]] = None


@dataclass
class DatasetResponse:
    """Versioned dataset response structure"""
    name: str
    local_path: Path
    is_verified: bool
    api_version: APIVersion
    provenance: Optional[ProvenanceRecord]
    metadata: Optional[Dict[str, Any]]
    response_timestamp: datetime


@runtime_checkable
class DatasetTypePlugin(Protocol):
    """Protocol for dataset type plugins"""
    
    @property
    def supported_types(self) -> List[str]:
        """Return list of dataset types this plugin supports"""
        ...
    
    @property
    def plugin_version(self) -> str:
        """Return plugin version"""
        ...
    
    def validate_dataset_config(self, config: Dict[str, Any]) -> bool:
        """Validate dataset configuration for this type"""
        ...
    
    def process_dataset_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance dataset metadata"""
        ...
    
    def get_schema_config(self, dataset_type: str) -> Optional[Dict[str, Any]]:
        """Get schema configuration for dataset type"""
        ...


@runtime_checkable
class ProtocolPlugin(Protocol):
    """Protocol for download protocol plugins"""
    
    @property
    def supported_protocols(self) -> List[str]:
        """Return list of protocols this plugin supports"""
        ...
    
    @property
    def plugin_version(self) -> str:
        """Return plugin version"""
        ...
    
    def can_handle_url(self, url: str) -> bool:
        """Check if this plugin can handle the given URL"""
        ...
    
    def download_dataset(
        self, 
        url: str, 
        local_path: Path, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Download dataset from URL to local path"""
        ...


class PluginManager:
    """Manager for dataset type and protocol plugins"""
    
    def __init__(self):
        self._dataset_type_plugins: Dict[str, DatasetTypePlugin] = {}
        self._protocol_plugins: Dict[str, ProtocolPlugin] = {}
        self._plugin_registry: Dict[str, Dict[str, Any]] = {}
        
        # Load built-in plugins
        self._load_builtin_plugins()
    
    def _load_builtin_plugins(self):
        """Load built-in dataset type and protocol plugins"""
        # Built-in dataset type plugins
        self.register_dataset_type_plugin("cosmology", CosmologyDatasetPlugin())
        
        # Built-in protocol plugins  
        self.register_protocol_plugin("https", HTTPSProtocolPlugin())
        self.register_protocol_plugin("zenodo", ZenodoProtocolPlugin())
    
    def register_dataset_type_plugin(self, name: str, plugin: DatasetTypePlugin):
        """Register a dataset type plugin"""
        if not isinstance(plugin, DatasetTypePlugin):
            raise ValueError(f"Plugin must implement DatasetTypePlugin protocol")
        
        self._dataset_type_plugins[name] = plugin
        self._plugin_registry[f"dataset_type_{name}"] = {
            "type": "dataset_type",
            "name": name,
            "version": plugin.plugin_version,
            "supported_types": plugin.supported_types,
            "registered_at": datetime.now().isoformat()
        }
    
    def register_protocol_plugin(self, name: str, plugin: ProtocolPlugin):
        """Register a protocol plugin"""
        if not isinstance(plugin, ProtocolPlugin):
            raise ValueError(f"Plugin must implement ProtocolPlugin protocol")
        
        self._protocol_plugins[name] = plugin
        self._plugin_registry[f"protocol_{name}"] = {
            "type": "protocol",
            "name": name,
            "version": plugin.plugin_version,
            "supported_protocols": plugin.supported_protocols,
            "registered_at": datetime.now().isoformat()
        }
    
    def get_dataset_type_plugin(self, dataset_type: str) -> Optional[DatasetTypePlugin]:
        """Get plugin for a specific dataset type"""
        for plugin in self._dataset_type_plugins.values():
            if dataset_type in plugin.supported_types:
                return plugin
        return None
    
    def get_protocol_plugin(self, protocol: str) -> Optional[ProtocolPlugin]:
        """Get plugin for a specific protocol"""
        return self._protocol_plugins.get(protocol)
    
    def get_protocol_plugin_for_url(self, url: str) -> Optional[ProtocolPlugin]:
        """Get plugin that can handle a specific URL"""
        for plugin in self._protocol_plugins.values():
            if plugin.can_handle_url(url):
                return plugin
        return None
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all registered plugins"""
        return self._plugin_registry.copy()
    
    def load_external_plugin(self, module_path: str, plugin_class: str) -> bool:
        """Load an external plugin from a module"""
        try:
            module = importlib.import_module(module_path)
            plugin_cls = getattr(module, plugin_class)
            
            # Check if it's a dataset type or protocol plugin
            if hasattr(plugin_cls, 'supported_types'):
                plugin = plugin_cls()
                if isinstance(plugin, DatasetTypePlugin):
                    self.register_dataset_type_plugin(plugin_class.lower(), plugin)
                    return True
            
            if hasattr(plugin_cls, 'supported_protocols'):
                plugin = plugin_cls()
                if isinstance(plugin, ProtocolPlugin):
                    self.register_protocol_plugin(plugin_class.lower(), plugin)
                    return True
            
            return False
        except Exception:
            return False


class ExtensibleDatasetInterface:
    """
    Extensible dataset interface with versioned API
    
    Provides a versioned API for dataset access with plugin support
    for new dataset types and protocols.
    """
    
    def __init__(self, registry_manager=None):
        """Initialize extensible interface"""
        self.plugin_manager = PluginManager()
        self._registry_manager = registry_manager
        self._supported_versions = [APIVersion.V1_0, APIVersion.V1_1]
    
    def get_supported_versions(self) -> List[str]:
        """Get list of supported API versions"""
        return [v.value for v in self._supported_versions]
    
    def request_dataset(self, request: DatasetRequest) -> DatasetResponse:
        """
        Request dataset using versioned API
        
        Args:
            request: DatasetRequest with version and parameters
            
        Returns:
            DatasetResponse with dataset information
            
        Raises:
            ValueError: If API version not supported
            Exception: If dataset request fails
        """
        if request.api_version not in self._supported_versions:
            raise ValueError(f"API version {request.api_version.value} not supported")
        
        # Route to appropriate version handler
        if request.api_version == APIVersion.V1_0:
            return self._handle_v1_0_request(request)
        elif request.api_version == APIVersion.V1_1:
            return self._handle_v1_1_request(request)
        else:
            raise ValueError(f"No handler for API version {request.api_version.value}")
    
    def _handle_v1_0_request(self, request: DatasetRequest) -> DatasetResponse:
        """Handle v1.0 API requests"""
        # Import here to avoid circular imports
        from ..integration.dataset_integration import DatasetRegistry
        
        registry = DatasetRegistry()
        dataset_info = registry.fetch_dataset(request.name, request.force_refresh)
        
        return DatasetResponse(
            name=request.name,
            local_path=dataset_info.local_path,
            is_verified=dataset_info.is_verified,
            api_version=request.api_version,
            provenance=dataset_info.provenance,
            metadata=self._extract_metadata_v1_0(dataset_info),
            response_timestamp=datetime.now()
        )
    
    def _handle_v1_1_request(self, request: DatasetRequest) -> DatasetResponse:
        """Handle v1.1 API requests (future version)"""
        # Enhanced v1.1 features would go here
        # For now, delegate to v1.0 with additional metadata processing
        v1_0_request = DatasetRequest(
            name=request.name,
            api_version=APIVersion.V1_0,
            force_refresh=request.force_refresh
        )
        
        response = self._handle_v1_0_request(v1_0_request)
        
        # Enhance with v1.1 features
        enhanced_metadata = self._enhance_metadata_v1_1(response.metadata, request)
        
        return DatasetResponse(
            name=response.name,
            local_path=response.local_path,
            is_verified=response.is_verified,
            api_version=APIVersion.V1_1,
            provenance=response.provenance,
            metadata=enhanced_metadata,
            response_timestamp=datetime.now()
        )
    
    def _extract_metadata_v1_0(self, dataset_info) -> Dict[str, Any]:
        """Extract metadata for v1.0 API"""
        metadata = {}
        
        if dataset_info.provenance:
            metadata.update({
                "download_timestamp": dataset_info.provenance.download_timestamp,
                "source_used": dataset_info.provenance.source_used,
                "verification_status": "verified" if dataset_info.is_verified else "failed",
                "pbuf_commit": dataset_info.provenance.environment.pbuf_commit
            })
        
        return metadata
    
    def _enhance_metadata_v1_1(self, base_metadata: Dict[str, Any], request: DatasetRequest) -> Dict[str, Any]:
        """Enhance metadata for v1.1 API"""
        enhanced = base_metadata.copy()
        
        # Add v1.1 specific enhancements
        enhanced.update({
            "api_version": "1.1",
            "verification_level": request.verification_level,
            "metadata_requirements": request.metadata_requirements or {},
            "plugin_info": self._get_plugin_info_for_dataset(request.name)
        })
        
        return enhanced
    
    def _get_plugin_info_for_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Get plugin information for a dataset"""
        plugin_info = {}
        
        # Try to determine dataset type and get plugin info
        try:
            from ..integration.dataset_integration import DatasetRegistry
            registry = DatasetRegistry()
            
            # Check if it's a manual dataset
            manual_info = registry.get_manual_dataset_info(dataset_name)
            if manual_info and manual_info.get("metadata"):
                dataset_type = manual_info["metadata"].get("dataset_type")
                if dataset_type:
                    plugin = self.plugin_manager.get_dataset_type_plugin(dataset_type)
                    if plugin:
                        plugin_info["dataset_type_plugin"] = {
                            "version": plugin.plugin_version,
                            "supported_types": plugin.supported_types
                        }
            
            # Check manifest for dataset type
            if not plugin_info:
                try:
                    dataset_info = registry.manifest.get_dataset_info(dataset_name)
                    if dataset_info.metadata:
                        dataset_type = dataset_info.metadata.get("dataset_type")
                        if dataset_type:
                            plugin = self.plugin_manager.get_dataset_type_plugin(dataset_type)
                            if plugin:
                                plugin_info["dataset_type_plugin"] = {
                                    "version": plugin.plugin_version,
                                    "supported_types": plugin.supported_types
                                }
                except:
                    pass
        except:
            pass
        
        return plugin_info
    
    def get_dataset_by_canonical_name(self, canonical_name: str, api_version: APIVersion = APIVersion.V1_0) -> DatasetResponse:
        """
        Get dataset by canonical name instead of internal name
        
        Args:
            canonical_name: Human-readable dataset name
            api_version: API version to use
            
        Returns:
            DatasetResponse for the matching dataset
            
        Raises:
            ValueError: If no dataset found with that canonical name
        """
        # Search manifest datasets
        from ..integration.dataset_integration import DatasetRegistry
        registry = DatasetRegistry()
        
        # Search in manifest
        for dataset_name in registry.manifest.list_datasets():
            dataset_info = registry.manifest.get_dataset_info(dataset_name)
            if dataset_info.canonical_name == canonical_name:
                request = DatasetRequest(
                    name=dataset_name,
                    api_version=api_version
                )
                return self.request_dataset(request)
        
        # Search in manual datasets
        for dataset_name in registry.list_manual_datasets():
            manual_info = registry.get_manual_dataset_info(dataset_name)
            if manual_info and manual_info.get("canonical_name") == canonical_name:
                request = DatasetRequest(
                    name=dataset_name,
                    api_version=api_version
                )
                return self.request_dataset(request)
        
        raise ValueError(f"No dataset found with canonical name: {canonical_name}")
    
    def list_available_datasets(self, api_version: APIVersion = APIVersion.V1_0) -> List[Dict[str, Any]]:
        """
        List all available datasets with their canonical names
        
        Args:
            api_version: API version to use for response format
            
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        
        from ..integration.dataset_integration import DatasetRegistry
        registry = DatasetRegistry()
        
        # Add manifest datasets
        for dataset_name in registry.manifest.list_datasets():
            try:
                dataset_info = registry.manifest.get_dataset_info(dataset_name)
                datasets.append({
                    "name": dataset_name,
                    "canonical_name": dataset_info.canonical_name,
                    "description": dataset_info.description,
                    "citation": dataset_info.citation,
                    "source_type": "downloaded",
                    "dataset_type": dataset_info.metadata.get("dataset_type") if dataset_info.metadata else None
                })
            except:
                continue
        
        # Add manual datasets
        for dataset_name in registry.list_manual_datasets():
            try:
                manual_info = registry.get_manual_dataset_info(dataset_name)
                if manual_info:
                    datasets.append({
                        "name": dataset_name,
                        "canonical_name": manual_info.get("canonical_name", dataset_name),
                        "description": manual_info.get("description", ""),
                        "citation": manual_info.get("citation", ""),
                        "source_type": "manual",
                        "dataset_type": manual_info.get("metadata", {}).get("dataset_type") if manual_info.get("metadata") else None
                    })
            except:
                continue
        
        return datasets
    
    def validate_dataset_compatibility(self, dataset_name: str, pbuf_version: str) -> Dict[str, Any]:
        """
        Validate dataset compatibility with PBUF version
        
        Args:
            dataset_name: Name of the dataset
            pbuf_version: PBUF version to check compatibility against
            
        Returns:
            Dictionary with compatibility information
        """
        compatibility = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "dataset_pbuf_version": None,
            "requested_pbuf_version": pbuf_version
        }
        
        try:
            from ..integration.dataset_integration import DatasetRegistry
            registry = DatasetRegistry()
            
            provenance = registry.get_provenance(dataset_name)
            if provenance:
                dataset_pbuf_version = provenance.environment.pbuf_commit
                compatibility["dataset_pbuf_version"] = dataset_pbuf_version
                
                # Simple version compatibility check
                if dataset_pbuf_version and dataset_pbuf_version != pbuf_version:
                    compatibility["warnings"].append(
                        f"Dataset was registered with PBUF commit {dataset_pbuf_version}, "
                        f"but current version is {pbuf_version}"
                    )
            else:
                compatibility["errors"].append(f"No provenance record found for dataset {dataset_name}")
                compatibility["compatible"] = False
        
        except Exception as e:
            compatibility["errors"].append(f"Failed to check compatibility: {str(e)}")
            compatibility["compatible"] = False
        
        return compatibility


# Built-in plugins

class CosmologyDatasetPlugin:
    """Built-in plugin for cosmology dataset types"""
    
    @property
    def supported_types(self) -> List[str]:
        return ["cmb", "bao", "sn", "bao_aniso"]
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    def validate_dataset_config(self, config: Dict[str, Any]) -> bool:
        """Validate cosmology dataset configuration"""
        required_fields = ["sha256"]
        return all(field in config for field in required_fields)
    
    def process_dataset_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process cosmology dataset metadata"""
        processed = metadata.copy()
        
        # Add cosmology-specific processing
        if "redshift_range" in processed:
            z_range = processed["redshift_range"]
            if isinstance(z_range, list) and len(z_range) >= 2:
                processed["redshift_span"] = z_range[1] - z_range[0]
        
        if "n_data_points" in processed:
            processed["data_density"] = processed.get("n_data_points", 0)
        
        return processed
    
    def get_schema_config(self, dataset_type: str) -> Optional[Dict[str, Any]]:
        """Get schema configuration for cosmology dataset types"""
        schemas = {
            "cmb": {
                "format": "ascii_table",
                "expected_columns": ["R", "l_A", "theta_star"],
                "min_rows": 1,
                "max_rows": 10
            },
            "bao": {
                "format": "ascii_table", 
                "expected_columns": ["z", "DM_over_rd", "DM_over_rd_err"],
                "min_rows": 1
            },
            "sn": {
                "format": "ascii_table",
                "expected_columns": ["z", "MU", "MU_err"],
                "min_rows": 10
            },
            "bao_aniso": {
                "format": "ascii_table",
                "expected_columns": ["z", "DM_over_rd", "DH_over_rd", "cov_matrix"],
                "min_rows": 1
            }
        }
        return schemas.get(dataset_type)


class HTTPSProtocolPlugin:
    """Built-in plugin for HTTPS protocol"""
    
    @property
    def supported_protocols(self) -> List[str]:
        return ["https", "http"]
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    def can_handle_url(self, url: str) -> bool:
        """Check if this plugin can handle the URL"""
        return url.startswith(("https://", "http://"))
    
    def download_dataset(
        self, 
        url: str, 
        local_path: Path, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Download dataset via HTTPS"""
        # This would integrate with the existing download manager
        return {
            "protocol": "https",
            "url": url,
            "local_path": str(local_path),
            "plugin_version": self.plugin_version
        }


class ZenodoProtocolPlugin:
    """Built-in plugin for Zenodo protocol"""
    
    @property
    def supported_protocols(self) -> List[str]:
        return ["zenodo"]
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    def can_handle_url(self, url: str) -> bool:
        """Check if this plugin can handle the URL"""
        return "zenodo.org" in url
    
    def download_dataset(
        self, 
        url: str, 
        local_path: Path, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Download dataset from Zenodo"""
        # This would integrate with Zenodo API
        return {
            "protocol": "zenodo",
            "url": url,
            "local_path": str(local_path),
            "plugin_version": self.plugin_version
        }


# Convenience functions for versioned API access

def get_dataset_v1(name: str, force_refresh: bool = False) -> DatasetResponse:
    """Get dataset using v1.0 API"""
    interface = ExtensibleDatasetInterface()
    request = DatasetRequest(
        name=name,
        api_version=APIVersion.V1_0,
        force_refresh=force_refresh
    )
    return interface.request_dataset(request)


def get_dataset_by_canonical_name(canonical_name: str) -> DatasetResponse:
    """Get dataset by canonical name using latest API"""
    interface = ExtensibleDatasetInterface()
    return interface.get_dataset_by_canonical_name(canonical_name, APIVersion.V1_0)


def list_available_datasets() -> List[Dict[str, Any]]:
    """List all available datasets"""
    interface = ExtensibleDatasetInterface()
    return interface.list_available_datasets()


def validate_dataset_compatibility(dataset_name: str, pbuf_version: str) -> Dict[str, Any]:
    """Validate dataset compatibility with PBUF version"""
    interface = ExtensibleDatasetInterface()
    return interface.validate_dataset_compatibility(dataset_name, pbuf_version)