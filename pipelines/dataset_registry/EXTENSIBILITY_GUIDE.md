# Dataset Registry Extensibility Guide

This guide provides comprehensive documentation for extending the dataset registry system with new dataset types, protocols, and API versions.

## Overview

The dataset registry system is designed with extensibility in mind, supporting:

- **Versioned APIs**: Multiple API versions for backward compatibility
- **Plugin Architecture**: Pluggable dataset types and download protocols
- **Version Control Integration**: Automatic environment fingerprinting and compatibility checking
- **Future Dataset Types**: Support for gravitational-wave, weak-lensing, and other cosmological datasets

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                 Extensible Interface                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   API v1.0      │  │   API v1.1      │  │  Future     │ │
│  │                 │  │                 │  │  Versions   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Plugin Manager                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Dataset Type    │  │ Protocol        │  │ Version     │ │
│  │ Plugins         │  │ Plugins         │  │ Control     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Adding New Dataset Types

### 1. Create a Dataset Type Plugin

Dataset type plugins implement the `DatasetTypePlugin` protocol:

```python
from pipelines.dataset_registry.core.extensible_interface import DatasetTypePlugin

class GravitationalWavePlugin:
    """Plugin for gravitational wave datasets"""
    
    @property
    def supported_types(self) -> List[str]:
        return ["gw_strain", "gw_events", "gw_catalog"]
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    def validate_dataset_config(self, config: Dict[str, Any]) -> bool:
        """Validate GW dataset configuration"""
        required_fields = ["sha256", "sampling_rate", "duration"]
        return all(field in config for field in required_fields)
    
    def process_dataset_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process GW-specific metadata"""
        processed = metadata.copy()
        
        # Add GW-specific processing
        if "strain_data" in processed:
            processed["data_quality"] = self._assess_strain_quality(processed["strain_data"])
        
        return processed
    
    def get_schema_config(self, dataset_type: str) -> Optional[Dict[str, Any]]:
        """Get schema configuration for GW dataset types"""
        schemas = {
            "gw_strain": {
                "format": "hdf5",
                "required_datasets": ["/strain/Strain", "/meta/GPSstart"],
                "sampling_rate_hz": [4096, 16384]
            },
            "gw_events": {
                "format": "json",
                "required_fields": ["GPS", "mass1", "mass2", "distance"],
                "min_events": 1
            }
        }
        return schemas.get(dataset_type)
```

### 2. Register the Plugin

```python
from pipelines.dataset_registry.core.extensible_interface import PluginManager

# Register the plugin
plugin_manager = PluginManager()
plugin_manager.register_dataset_type_plugin("gravitational_wave", GravitationalWavePlugin())
```

### 3. Update Manifest Schema

Add the new dataset type to your manifest:

```json
{
  "manifest_version": "1.0",
  "datasets": {
    "ligo_o3_strain": {
      "canonical_name": "LIGO O3 Strain Data",
      "description": "Gravitational wave strain data from LIGO O3 run",
      "citation": "Abbott et al. 2021, PRX 11, 021053",
      "sources": {
        "primary": {
          "url": "https://gwosc.org/eventapi/html/O3_Discovery_Papers/",
          "protocol": "gwosc"
        }
      },
      "verification": {
        "sha256": "abc123...",
        "sampling_rate": 4096,
        "duration": 4096
      },
      "metadata": {
        "dataset_type": "gw_strain",
        "observables": ["h_plus", "h_cross"],
        "detectors": ["H1", "L1", "V1"],
        "frequency_range": [20, 2048]
      }
    }
  }
}
```

## Adding New Download Protocols

### 1. Create a Protocol Plugin

Protocol plugins implement the `ProtocolPlugin` protocol:

```python
from pipelines.dataset_registry.core.extensible_interface import ProtocolPlugin

class GWOSCProtocolPlugin:
    """Plugin for Gravitational Wave Open Science Center protocol"""
    
    @property
    def supported_protocols(self) -> List[str]:
        return ["gwosc"]
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    def can_handle_url(self, url: str) -> bool:
        """Check if this plugin can handle the URL"""
        return "gwosc.org" in url or "losc.ligo.org" in url
    
    def download_dataset(
        self, 
        url: str, 
        local_path: Path, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Download dataset from GWOSC"""
        import requests
        
        # GWOSC-specific download logic
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return {
            "protocol": "gwosc",
            "url": url,
            "local_path": str(local_path),
            "plugin_version": self.plugin_version,
            "gwosc_metadata": self._extract_gwosc_metadata(response.headers)
        }
    
    def _extract_gwosc_metadata(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Extract GWOSC-specific metadata from response headers"""
        return {
            "content_type": headers.get("content-type"),
            "last_modified": headers.get("last-modified"),
            "gwosc_version": headers.get("x-gwosc-version")
        }
```

### 2. Register the Protocol Plugin

```python
plugin_manager.register_protocol_plugin("gwosc", GWOSCProtocolPlugin())
```

## Adding New API Versions

### 1. Define New API Version

```python
from pipelines.dataset_registry.core.extensible_interface import APIVersion

class APIVersion(Enum):
    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"  # New version
```

### 2. Implement Version Handler

```python
def _handle_v2_0_request(self, request: DatasetRequest) -> DatasetResponse:
    """Handle v2.0 API requests with enhanced features"""
    
    # Enhanced v2.0 features
    if request.metadata_requirements:
        # Apply metadata filtering
        pass
    
    if request.verification_level == "strict":
        # Enhanced verification
        pass
    
    # Delegate to existing logic with enhancements
    base_response = self._handle_v1_0_request(request)
    
    # Add v2.0 specific enhancements
    enhanced_metadata = self._enhance_metadata_v2_0(base_response.metadata, request)
    
    return DatasetResponse(
        name=base_response.name,
        local_path=base_response.local_path,
        is_verified=base_response.is_verified,
        api_version=APIVersion.V2_0,
        provenance=base_response.provenance,
        metadata=enhanced_metadata,
        response_timestamp=datetime.now()
    )
```

### 3. Update Request Routing

```python
def request_dataset(self, request: DatasetRequest) -> DatasetResponse:
    """Request dataset using versioned API"""
    
    if request.api_version == APIVersion.V1_0:
        return self._handle_v1_0_request(request)
    elif request.api_version == APIVersion.V1_1:
        return self._handle_v1_1_request(request)
    elif request.api_version == APIVersion.V2_0:
        return self._handle_v2_0_request(request)
    else:
        raise ValueError(f"Unsupported API version: {request.api_version}")
```

## Version Control Integration

### Environment Fingerprinting

The system automatically collects environment fingerprints including:

- PBUF git commit hash and branch
- Python version and installed packages
- Platform and system information
- Environment variables

### Compatibility Checking

```python
from pipelines.dataset_registry.integration.dataset_integration import check_dataset_compatibility

# Check compatibility
compatibility = check_dataset_compatibility("my_dataset", strict_mode=False)

if not compatibility["compatible"]:
    print("Compatibility issues:")
    for issue in compatibility["issues"]:
        print(f"  - {issue}")

if compatibility["warnings"]:
    print("Compatibility warnings:")
    for warning in compatibility["warnings"]:
        print(f"  - {warning}")
```

### Reproducibility Validation

```python
from pipelines.dataset_registry.integration.dataset_integration import validate_reproducibility

# Validate reproducibility for all datasets
validation = validate_reproducibility()

if not validation["valid"]:
    print(f"Reproducibility issues found:")
    for issue in validation["issues"]:
        print(f"  - {issue}")

print(f"Datasets with commit hash: {validation['datasets_with_commit']}")
print(f"Datasets without commit hash: {validation['datasets_without_commit']}")
```

## External Plugin Loading

### 1. Create External Plugin Module

Create a Python module with your plugin:

```python
# my_custom_plugins.py

from pipelines.dataset_registry.core.extensible_interface import DatasetTypePlugin

class WeakLensingPlugin:
    """Plugin for weak lensing datasets"""
    
    @property
    def supported_types(self) -> List[str]:
        return ["weak_lensing_shear", "weak_lensing_convergence"]
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    # ... implement required methods
```

### 2. Load External Plugin

```python
from pipelines.dataset_registry.core.extensible_interface import PluginManager

plugin_manager = PluginManager()
success = plugin_manager.load_external_plugin("my_custom_plugins", "WeakLensingPlugin")

if success:
    print("Plugin loaded successfully")
else:
    print("Failed to load plugin")
```

## Best Practices

### Dataset Type Plugins

1. **Validation**: Always implement thorough validation in `validate_dataset_config()`
2. **Metadata Processing**: Use `process_dataset_metadata()` to add domain-specific enhancements
3. **Schema Definition**: Provide clear schema configurations for verification
4. **Error Handling**: Handle edge cases gracefully and provide clear error messages

### Protocol Plugins

1. **URL Validation**: Implement robust URL pattern matching in `can_handle_url()`
2. **Error Recovery**: Handle network failures and implement retry logic
3. **Progress Reporting**: Provide progress feedback for large downloads
4. **Metadata Extraction**: Extract and preserve protocol-specific metadata

### API Versioning

1. **Backward Compatibility**: Ensure new versions don't break existing functionality
2. **Graceful Degradation**: Handle missing features in older versions
3. **Clear Documentation**: Document API changes and migration paths
4. **Version Detection**: Allow clients to detect supported versions

### Version Control Integration

1. **Environment Consistency**: Ensure consistent environment fingerprinting
2. **Compatibility Levels**: Provide different levels of compatibility checking
3. **Clear Reporting**: Generate clear compatibility and reproducibility reports
4. **Automated Validation**: Integrate validation into CI/CD pipelines

## Future Extensions

### Planned Features

1. **Multi-format Support**: Enhanced support for HDF5, FITS, and other scientific formats
2. **Cloud Integration**: Direct integration with cloud storage providers
3. **Streaming Downloads**: Support for streaming large datasets
4. **Distributed Verification**: Parallel verification across multiple nodes
5. **Advanced Caching**: Intelligent caching with automatic cleanup

### Extension Points

1. **Custom Verification**: Plugin-based verification engines
2. **Metadata Enrichment**: Automatic metadata extraction from dataset contents
3. **Provenance Tracking**: Enhanced provenance with data lineage tracking
4. **Integration Hooks**: Hooks for external monitoring and alerting systems

## Examples

### Complete Example: Adding Weak Lensing Support

```python
# 1. Create the plugin
class WeakLensingPlugin:
    @property
    def supported_types(self) -> List[str]:
        return ["weak_lensing_shear", "weak_lensing_convergence", "weak_lensing_catalog"]
    
    @property
    def plugin_version(self) -> str:
        return "1.0.0"
    
    def validate_dataset_config(self, config: Dict[str, Any]) -> bool:
        required = ["sha256", "survey_area", "redshift_bins"]
        return all(field in config for field in required)
    
    def process_dataset_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        processed = metadata.copy()
        
        if "survey_area" in processed and "n_galaxies" in processed:
            processed["galaxy_density"] = processed["n_galaxies"] / processed["survey_area"]
        
        return processed
    
    def get_schema_config(self, dataset_type: str) -> Optional[Dict[str, Any]]:
        return {
            "weak_lensing_shear": {
                "format": "fits",
                "required_columns": ["RA", "DEC", "e1", "e2", "weight"],
                "min_galaxies": 1000
            }
        }.get(dataset_type)

# 2. Register the plugin
from pipelines.dataset_registry.core.extensible_interface import PluginManager
plugin_manager = PluginManager()
plugin_manager.register_dataset_type_plugin("weak_lensing", WeakLensingPlugin())

# 3. Use the new dataset type
from pipelines.dataset_registry.integration.dataset_integration import DatasetRegistry

registry = DatasetRegistry()
dataset_info = registry.fetch_dataset("des_y3_shear")
```

This extensibility guide provides the foundation for extending the dataset registry system to support new cosmological datasets, protocols, and analysis workflows while maintaining backward compatibility and reproducibility standards.