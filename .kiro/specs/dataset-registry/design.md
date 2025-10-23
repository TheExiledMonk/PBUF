# Design Document

## Overview

The Centralized Dataset Downloader & Verification Registry is designed as a modular system that integrates seamlessly with the existing PBUF cosmology pipeline architecture. The system replaces scattered dataset loading logic with a unified, declarative approach that ensures complete reproducibility and scientific integrity.

### Scope and Boundaries

This subsystem governs dataset retrieval and verification only; it does not modify analytical computations or cosmological model code. The registry manages data provenance and integrity while preserving the existing physics and fitting logic unchanged.

The design follows a layered architecture with clear separation of concerns:
- **Manifest Layer**: Declarative dataset definitions
- **Acquisition Layer**: Multi-protocol download and caching
- **Verification Layer**: Cryptographic and structural validation  
- **Registry Layer**: Immutable provenance tracking
- **Integration Layer**: Seamless pipeline integration

## Architecture

### Non-Functional Properties

- **Performance**: Dataset verification completes within 2 minutes for 100MB files; concurrent downloads supported
- **Reliability**: Automatic retry with exponential backoff; mirror fallback without data corruption
- **Security**: HTTPS-only by default; mandatory checksum verification; no arbitrary code execution
- **Portability**: Cross-platform operation (Linux, macOS, Windows); offline and cluster environment support
- **Observability**: Structured JSON logging for all operations; integration with provenance dashboards

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dataset       │    │   Download      │    │  Verification   │
│   Manifest      │───▶│   Manager       │───▶│   Engine        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Pipeline      │◀───│   Registry      │◀────────────┘
│  Integration    │    │   Manager       │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

### Core Modules

#### 1. Dataset Manifest (`dataset_manifest.py`)
- **Purpose**: Manages the central dataset definition file
- **Format**: JSON schema with validation
- **Location**: `data/datasets_manifest.json`
- **Responsibilities**:
  - Parse and validate manifest schema
  - Provide dataset metadata lookup
  - Support manifest versioning and updates
  - Validate dataset definitions against schema

#### 2. Download Manager (`download_manager.py`)
- **Purpose**: Handles multi-protocol dataset acquisition
- **Supported Protocols**: HTTPS, Zenodo API, arXiv, local mirrors, manual upload
- **Responsibilities**:
  - Execute downloads with retry logic and fallback sources
  - Manage local caching and avoid redundant downloads
  - Handle decompression and extraction
  - Provide progress reporting and cancellation
  - Support concurrent downloads

#### 3. Verification Engine (`verification_engine.py`)
- **Purpose**: Validates dataset integrity and structure
- **Validation Types**: SHA256 checksums, file size, schema validation
- **Responsibilities**:
  - Perform cryptographic verification
  - Validate dataset structure against expected schema
  - Check data consistency and format compliance
  - Generate detailed verification reports

#### 4. Registry Manager (`registry_manager.py`)
- **Purpose**: Maintains immutable provenance records with complete version tracking
- **Storage Format**: Individual JSON files per dataset
- **Location**: `data/registry/`
- **Thread Safety**: Concurrent writes serialized using advisory file locks
- **Responsibilities**:
  - Create and maintain dataset registry entries with mandatory PBUF commit hash
  - Ensure immutable audit trails with environment fingerprinting
  - Support registry queries and exports
  - Generate provenance summaries for publications
  - Emit structured log events for all registry operations

#### 5. Pipeline Integration (`dataset_integration.py`)
- **Purpose**: Seamless integration with existing PBUF pipelines
- **Integration Points**: `datasets.py`, `config.py`, fitting pipelines
- **Responsibilities**:
  - Replace existing `load_dataset()` calls
  - Provide pre-run dataset validation
  - Attach provenance to fit results
  - Support backward compatibility during transition

## Components and Interfaces

### Dataset Manifest Schema

```json
{
  "manifest_version": "1.0",
  "datasets": {
    "cmb_planck2018": {
      "canonical_name": "Planck 2018 Distance Priors",
      "description": "CMB distance priors from Planck 2018 final release",
      "citation": "Aghanim et al. 2020, A&A 641, A6",
      "license": "CC-BY-4.0",
      "sources": {
        "primary": {
          "url": "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.00.zip",
          "protocol": "https",
          "extraction": {
            "format": "zip",
            "target_files": ["base/plikHM_TTTEEE_lowl_lowE_lensing/base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.theory"]
          }
        },
        "mirror": {
          "url": "https://zenodo.org/record/4543143/files/planck2018_distance_priors.dat",
          "protocol": "https"
        }
      },
      "verification": {
        "sha256": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
        "size_bytes": 1024,
        "schema": {
          "format": "ascii_table",
          "columns": ["R", "l_A", "theta_star"],
          "expected_rows": 1
        }
      },
      "metadata": {
        "dataset_type": "cmb",
        "redshift_range": [1089.80, 1089.80],
        "observables": ["R", "l_A", "theta_star"],
        "n_data_points": 3,
        "covariance_included": true
      }
    }
  }
}
```

### Registry Entry Schema

```json
{
  "dataset_name": "cmb_planck2018",
  "registry_version": "1.0",
  "provenance": {
    "download_timestamp": "2025-10-23T14:30:00Z",
    "source_used": "primary",
    "download_agent": "dataset-registry-v1.0",
    "environment": {
      "pbuf_commit": "a1b2c3d4",
      "python_version": "3.9.7",
      "platform": "linux-x86_64"
    }
  },
  "verification": {
    "sha256_verified": true,
    "sha256_expected": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
    "sha256_actual": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
    "size_verified": true,
    "size_expected": 1024,
    "size_actual": 1024,
    "schema_verified": true,
    "verification_timestamp": "2025-10-23T14:30:15Z"
  },
  "file_info": {
    "local_path": "data/datasets/cmb_planck2018.dat",
    "original_filename": "base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.theory",
    "mime_type": "text/plain"
  },
  "status": "verified",
  "created_at": "2025-10-23T14:30:15Z",
  "last_verified": "2025-10-23T14:30:15Z"
}
```

### API Interfaces

#### Core Dataset Interface
```python
class DatasetRegistry:
    def fetch_dataset(self, name: str, force_refresh: bool = False) -> DatasetInfo
    def verify_dataset(self, name: str) -> VerificationResult
    def get_dataset_path(self, name: str) -> Path
    def list_datasets(self) -> List[DatasetInfo]
    def get_provenance(self, name: str) -> ProvenanceRecord
    def export_manifest_summary(self) -> Dict[str, Any]
```

#### Integration Interface
```python
def load_dataset(name: str) -> DatasetDict:
    """Drop-in replacement for existing load_dataset function"""
    registry = get_dataset_registry()
    dataset_info = registry.fetch_dataset(name)
    return _load_dataset_from_path(dataset_info.local_path, name)

def verify_all_datasets(dataset_names: List[str]) -> bool:
    """Pre-run verification for pipeline integration"""
    registry = get_dataset_registry()
    return all(registry.verify_dataset(name).is_valid for name in dataset_names)
```

## Data Models

### Dataset Information Model
```python
@dataclass
class DatasetInfo:
    name: str
    canonical_name: str
    description: str
    citation: str
    license: str
    local_path: Optional[Path]
    verification_status: VerificationStatus
    last_verified: Optional[datetime]
    metadata: Dict[str, Any]
```

### Verification Result Model
```python
@dataclass
class VerificationResult:
    is_valid: bool
    sha256_match: bool
    size_match: bool
    schema_valid: bool
    errors: List[str]
    warnings: List[str]
    verification_time: datetime
```

### Provenance Record Model
```python
@dataclass
class ProvenanceRecord:
    dataset_name: str
    download_timestamp: datetime
    source_used: str
    download_agent: str
    environment_info: Dict[str, str]
    verification_result: VerificationResult
    file_info: Dict[str, Any]
```

## Error Handling

### Error Categories and Responses

#### Download Errors
- **Network failures**: Automatic retry with exponential backoff, fallback to mirror sources
- **Authentication errors**: Clear error messages with resolution steps
- **File not found**: Try alternative sources, provide manual upload option
- **Disk space**: Check available space, provide cleanup suggestions

#### Verification Errors  
- **Checksum mismatch**: Halt processing, suggest re-download or manual verification
- **Schema validation failure**: Detailed error report, suggest data format fixes
- **File corruption**: Automatic re-download attempt, manual intervention if persistent

#### Integration Errors
- **Missing datasets**: Clear diagnostic with fetch instructions
- **Version conflicts**: Automatic resolution or user prompt for action
- **Permission errors**: Clear instructions for file system permissions

### Error Recovery Strategies

```python
class DatasetError(Exception):
    """Base class for dataset-related errors"""
    pass

class DownloadError(DatasetError):
    """Raised when dataset download fails"""
    def __init__(self, dataset_name: str, sources_tried: List[str], last_error: str):
        self.dataset_name = dataset_name
        self.sources_tried = sources_tried
        self.last_error = last_error
        super().__init__(f"Failed to download {dataset_name} from {len(sources_tried)} sources")

class VerificationError(DatasetError):
    """Raised when dataset verification fails"""
    def __init__(self, dataset_name: str, verification_result: VerificationResult):
        self.dataset_name = dataset_name
        self.verification_result = verification_result
        super().__init__(f"Dataset {dataset_name} failed verification")
```

## Logging and Observability

### Structured Logging Interface

All actions (download start/complete, verification result, registry write) SHALL emit structured log events in JSON Lines format with timestamps, consumed by higher-level provenance dashboards:

```json
{"timestamp": "2025-10-23T14:30:00Z", "event": "download_start", "dataset": "cmb_planck2018", "source": "primary"}
{"timestamp": "2025-10-23T14:30:15Z", "event": "verification_complete", "dataset": "cmb_planck2018", "status": "success", "sha256_match": true}
{"timestamp": "2025-10-23T14:30:16Z", "event": "registry_write", "dataset": "cmb_planck2018", "pbuf_commit": "a1b2c3d4"}
```

### Audit Trail Integration

Registry operations integrate with PBUF's existing logging infrastructure while maintaining independent audit trails for scientific reproducibility requirements.

## Testing Strategy

### Unit Testing Approach

#### Manifest Management Tests
- Schema validation with valid and invalid manifests
- Dataset lookup and metadata extraction
- Manifest versioning and migration
- Error handling for malformed manifests

#### Download Manager Tests
- Mock HTTP responses for different protocols
- Retry logic and fallback source handling
- Progress reporting and cancellation
- Concurrent download coordination
- File extraction and decompression

#### Verification Engine Tests
- Checksum calculation and validation
- Schema validation with various data formats
- File size and structure validation
- Error reporting and recovery

#### Registry Manager Tests
- Registry entry creation and persistence
- Immutable audit trail maintenance
- Query and export functionality
- Concurrent access handling

#### Integration Tests
- End-to-end dataset fetch and verification
- Pipeline integration with existing `load_dataset()` calls
- Backward compatibility during transition
- Performance impact on existing workflows

### Mock Data Strategy

```python
# Test manifest with minimal dataset definitions
TEST_MANIFEST = {
    "manifest_version": "1.0",
    "datasets": {
        "test_cmb": {
            "canonical_name": "Test CMB Dataset",
            "sources": {"primary": {"url": "http://test.example/cmb.dat"}},
            "verification": {"sha256": "test_hash", "size_bytes": 100}
        }
    }
}

# Mock HTTP responses for download testing
@pytest.fixture
def mock_http_responses():
    with responses.RequestsMock() as rsps:
        rsps.add(responses.GET, "http://test.example/cmb.dat", 
                body=b"test data", status=200)
        yield rsps
```

### Performance Testing

#### Benchmarks
- Dataset download speed across different protocols
- Verification time for various dataset sizes
- Registry query performance with large numbers of datasets
- Memory usage during concurrent operations

#### Load Testing
- Multiple simultaneous dataset downloads
- Registry access under concurrent load
- Integration performance impact on existing pipelines

## Integration Points

### Existing PBUF Integration

#### 1. Dataset Loading (`pipelines/fit_core/datasets.py`)
```python
# Current implementation
def load_dataset(name: str) -> DatasetDict:
    # Existing hardcoded loading logic
    pass

# New implementation with registry
def load_dataset(name: str) -> DatasetDict:
    registry = get_dataset_registry()
    dataset_info = registry.fetch_dataset(name)
    
    # Use existing parsing logic but with verified data
    return _parse_dataset_file(dataset_info.local_path, name)
```

#### 2. Configuration Integration (`pipelines/fit_core/config.py`)
```python
# Add dataset registry configuration section
def get_datasets_config(self) -> Dict[str, Any]:
    return {
        "registry_enabled": True,
        "manifest_path": "data/datasets_manifest.json",
        "registry_path": "data/registry/",
        "auto_fetch": True,
        "verify_on_load": True,
        **self.config_data.get('datasets', {})
    }
```

#### 3. Pipeline Pre-run Checks
```python
# Integration in fitting pipelines
def run_fit(model: str, datasets: List[str], **kwargs):
    # Add dataset verification before fitting
    if not verify_all_datasets(datasets):
        raise DatasetError("Dataset verification failed")
    
    # Continue with existing fitting logic
    return existing_run_fit(model, datasets, **kwargs)
```

### Backward Compatibility Strategy

#### Phase 1: Parallel Operation
- Registry system operates alongside existing dataset loading
- Gradual migration of datasets to registry management
- Fallback to existing loading for non-registered datasets

#### Phase 2: Default Registry
- Registry becomes default for all dataset operations
- Existing loading available as fallback option
- Deprecation warnings for direct file access

#### Phase 3: Registry Only
- Complete migration to registry-based dataset management
- Removal of legacy dataset loading code
- Full provenance tracking for all operations

### Configuration Migration

```python
# Legacy configuration
datasets_config = {
    "default_datasets": ["cmb", "bao", "sn"],
    "data_directory": "./data"
}

# New registry configuration
datasets_config = {
    "registry_enabled": True,
    "manifest_path": "data/datasets_manifest.json",
    "registry_path": "data/registry/",
    "auto_fetch": True,
    "verify_on_load": True,
    "fallback_to_legacy": True,  # During transition
    "default_datasets": ["cmb_planck2018", "bao_compilation", "sn_pantheon_plus"]
}
```

## Implementation Phases

### Phase 1: Core Infrastructure (Requirements 1, 2, 3)
- Implement manifest schema and parser
- Build download manager with basic HTTP support
- Create verification engine with SHA256 validation
- Establish registry storage and basic queries

### Phase 2: Advanced Features (Requirements 4, 5, 6)
- Add multi-protocol support (Zenodo, arXiv)
- Implement comprehensive provenance tracking
- Build pipeline integration layer
- Add manual dataset registration

### Phase 3: Management & Operations (Requirements 7, 8, 9)
- Create administrative tools and queries
- Implement CLI interface for administrative tasks (list, re-verify, export summary)
- Add optional web-dashboard hooks for registry monitoring
- Implement export and summary generation
- Add performance optimization and caching
- Build one-command reproducibility tools

### Phase 4: Production Hardening
- Comprehensive error handling and recovery
- Performance optimization and monitoring
- Security hardening and access controls
- Documentation and user training materials

## Governance and Standards Compliance

The design complies with FAIR data principles (Findable, Accessible, Interoperable, Reusable) and supports open-science provenance standards. All registry operations maintain complete audit trails suitable for scientific publication and peer review requirements.