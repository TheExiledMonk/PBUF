# Dataset Registry Deployment Guide

This guide provides comprehensive instructions for deploying the PBUF Dataset Registry system in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Environment Setup](#environment-setup)
5. [Migration from Legacy Systems](#migration-from-legacy-systems)
6. [Validation and Testing](#validation-and-testing)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Minimum 1GB available disk space for dataset cache
- Network access for dataset downloads (or pre-downloaded datasets)
- Write permissions for registry and cache directories

### Dependencies

The dataset registry system is integrated with the PBUF pipeline and uses the following dependencies:

```bash
# Core dependencies (already included in PBUF requirements)
requests>=2.25.0
pathlib
json
hashlib
```

Optional dependencies for enhanced functionality:

```bash
# For YAML configuration support
PyYAML>=5.4.0

# For advanced logging
structlog>=21.0.0
```

## Installation

### 1. Verify PBUF Installation

Ensure the PBUF pipeline is properly installed and functional:

```bash
# Test basic PBUF functionality
python -c "from pipelines.fit_core import config; print('PBUF installation OK')"
```

### 2. Verify Dataset Registry Installation

The dataset registry is included with PBUF. Verify installation:

```bash
# Test dataset registry import
python -c "from pipelines.dataset_registry.core import DatasetRegistryConfig; print('Dataset registry installation OK')"
```

### 3. Create Required Directories

Create the necessary directories for registry operation:

```bash
# Create registry directories
mkdir -p data/registry
mkdir -p data/cache
mkdir -p data/datasets

# Set appropriate permissions
chmod 755 data/registry data/cache data/datasets
```

## Configuration

### 1. Basic Configuration

Create a basic configuration file for the dataset registry:

```bash
# Generate example configuration
python -c "
from pipelines.dataset_registry.core.config import create_example_dataset_registry_config
create_example_dataset_registry_config('dataset_registry_config.json')
print('Example configuration created: dataset_registry_config.json')
"
```

### 2. Integration with PBUF Configuration

For integrated operation with PBUF, add dataset registry configuration to your PBUF configuration file:

**JSON Configuration (pbuf_config.json):**

```json
{
  "datasets": {
    "default_datasets": ["cmb", "bao", "sn"],
    "data_directory": "./data",
    "registry": {
      "enabled": true,
      "manifest_path": "data/datasets_manifest.json",
      "registry_path": "data/registry/",
      "cache_path": "data/cache/",
      "auto_fetch": true,
      "verify_on_load": true,
      "fallback_to_legacy": false,
      "download": {
        "timeout": 300,
        "max_retries": 3,
        "concurrent_downloads": 3
      },
      "verification": {
        "verify_checksums": true,
        "verify_file_sizes": true,
        "verify_schemas": true
      },
      "logging": {
        "structured_logging": true,
        "log_level": "INFO",
        "audit_trail_enabled": true
      }
    }
  }
}
```

**YAML Configuration (pbuf_config.yaml):**

```yaml
datasets:
  default_datasets:
    - cmb
    - bao
    - sn
  data_directory: ./data
  registry:
    enabled: true
    manifest_path: data/datasets_manifest.json
    registry_path: data/registry/
    cache_path: data/cache/
    auto_fetch: true
    verify_on_load: true
    fallback_to_legacy: false
    download:
      timeout: 300
      max_retries: 3
      concurrent_downloads: 3
    verification:
      verify_checksums: true
      verify_file_sizes: true
      verify_schemas: true
    logging:
      structured_logging: true
      log_level: INFO
      audit_trail_enabled: true
```

### 3. Environment-Specific Configuration

Configure different settings for different environments:

```json
{
  "datasets": {
    "registry": {
      "environment_overrides": {
        "development": {
          "log_level": "DEBUG",
          "verify_on_load": false,
          "cache_enabled": false
        },
        "production": {
          "log_level": "WARNING",
          "verify_on_load": true,
          "cache_enabled": true,
          "audit_trail_enabled": true
        },
        "testing": {
          "registry_enabled": false,
          "fallback_to_legacy": true,
          "auto_fetch": false
        }
      }
    }
  }
}
```

Set the environment using the `PBUF_ENV` environment variable:

```bash
export PBUF_ENV=production
```

## Environment Setup

### Development Environment

```bash
# Set development environment
export PBUF_ENV=development
export PBUF_LOG_LEVEL=DEBUG

# Optional: Use local cache directory
export PBUF_CACHE_PATH=./dev_cache

# Optional: Disable verification for faster development
export PBUF_VERIFY_ON_LOAD=false
```

### Testing Environment

```bash
# Set testing environment
export PBUF_ENV=testing

# Disable registry for testing (use legacy datasets)
export PBUF_REGISTRY_ENABLED=false

# Use test-specific paths
export PBUF_CACHE_PATH=./test_cache
export PBUF_REGISTRY_PATH=./test_registry
```

### Production Environment

```bash
# Set production environment
export PBUF_ENV=production
export PBUF_LOG_LEVEL=WARNING

# Enable all verification and auditing
export PBUF_VERIFY_ON_LOAD=true
export PBUF_REGISTRY_ENABLED=true

# Use production paths
export PBUF_CACHE_PATH=/var/lib/pbuf/cache
export PBUF_REGISTRY_PATH=/var/lib/pbuf/registry
export PBUF_MANIFEST_PATH=/etc/pbuf/datasets_manifest.json
```

## Migration from Legacy Systems

### 1. Automatic Migration

The dataset registry includes automatic migration tools for legacy configurations:

```python
# Run automatic migration
from pipelines.fit_core.config import ConfigurationManager
from pipelines.dataset_registry.core.config_integration import migrate_legacy_dataset_configuration

# Load existing PBUF configuration
config_manager = ConfigurationManager('pbuf_config.json')

# Perform migration
migration_results = migrate_legacy_dataset_configuration(config_manager)

print("Migration Results:")
print(f"Migrated: {migration_results['migrated']}")
print(f"Legacy settings found: {migration_results['legacy_settings_found']}")
print(f"New settings created: {migration_results['new_settings_created']}")

# Save updated configuration
config_manager.save_config('pbuf_config_migrated.json')
```

### 2. Manual Migration Steps

If automatic migration is not suitable, follow these manual steps:

#### Step 1: Identify Legacy Dataset Paths

```bash
# Find existing dataset files
find . -name "*.dat" -o -name "*.txt" | grep -E "(cmb|bao|sn|planck|pantheon)"
```

#### Step 2: Create Dataset Manifest

Create or update `data/datasets_manifest.json` with your datasets:

```json
{
  "manifest_version": "1.0",
  "datasets": {
    "cmb_planck2018": {
      "canonical_name": "Planck 2018 Distance Priors",
      "description": "CMB distance priors from Planck 2018 final release",
      "sources": {
        "manual": {
          "path": "data/cmb_planck2018.dat"
        }
      },
      "verification": {
        "sha256": "calculated_checksum_here",
        "size_bytes": 1024
      }
    }
  }
}
```

#### Step 3: Register Existing Datasets

```python
# Register existing datasets manually
from pipelines.dataset_registry.core.registry_manager import RegistryManager

registry = RegistryManager()

# Register each existing dataset
registry.register_manual_dataset(
    name="cmb_planck2018",
    file_path="data/cmb_planck2018.dat",
    description="Manually migrated CMB dataset",
    source_info={"type": "manual", "original_path": "data/cmb_planck2018.dat"}
)
```

#### Step 4: Update Configuration

Update your PBUF configuration to enable registry with legacy fallback:

```json
{
  "datasets": {
    "registry": {
      "enabled": true,
      "fallback_to_legacy": true,
      "legacy_dataset_paths": {
        "cmb": "data/cmb_planck2018.dat",
        "bao": "data/bao_compilation.dat",
        "sn": "data/sn_pantheon_plus.dat"
      }
    }
  }
}
```

### 3. Migration Validation

Validate the migration was successful:

```python
# Validate migration
from pipelines.dataset_registry.core.config_integration import validate_integrated_configuration
from pipelines.fit_core.config import ConfigurationManager

config_manager = ConfigurationManager('pbuf_config_migrated.json')
validation_results = validate_integrated_configuration(config_manager)

print("Validation Results:")
print(f"Valid: {validation_results['valid']}")
print(f"Errors: {validation_results['errors']}")
print(f"Warnings: {validation_results['warnings']}")
```

## Validation and Testing

### 1. Configuration Validation

```python
# Validate configuration
from pipelines.dataset_registry.core.config import DatasetRegistryConfigManager

config_manager = DatasetRegistryConfigManager('dataset_registry_config.json')
validation = config_manager.validate_config()

if validation['valid']:
    print("Configuration is valid")
else:
    print("Configuration errors:")
    for error in validation['errors']:
        print(f"  - {error}")
```

### 2. Registry Functionality Test

```python
# Test registry functionality
from pipelines.dataset_registry.integration.dataset_integration import DatasetIntegration

integration = DatasetIntegration()

# Test dataset listing
datasets = integration.list_available_datasets()
print(f"Available datasets: {[d.name for d in datasets]}")

# Test dataset fetching (if manifest exists)
if datasets:
    test_dataset = datasets[0].name
    try:
        dataset_info = integration.get_dataset(test_dataset)
        print(f"Successfully fetched dataset: {test_dataset}")
    except Exception as e:
        print(f"Error fetching dataset {test_dataset}: {e}")
```

### 3. Integration Test with PBUF Pipelines

```python
# Test integration with PBUF pipelines
from pipelines.fit_core.datasets import load_dataset

# Test loading a dataset through the registry
try:
    dataset = load_dataset('cmb_planck2018')
    print("Dataset loading through registry: SUCCESS")
except Exception as e:
    print(f"Dataset loading failed: {e}")
```

## Troubleshooting

### Common Issues

#### 1. Permission Errors

```bash
# Fix directory permissions
sudo chown -R $USER:$USER data/
chmod -R 755 data/
```

#### 2. Network Issues

```bash
# Test network connectivity
curl -I https://pla.esac.esa.int/

# Configure proxy if needed
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

#### 3. Configuration Errors

```python
# Debug configuration loading
from pipelines.dataset_registry.core.config import DatasetRegistryConfigManager

try:
    config_manager = DatasetRegistryConfigManager('your_config.json')
    config = config_manager.get_config()
    print("Configuration loaded successfully")
except Exception as e:
    print(f"Configuration error: {e}")
```

#### 4. Dataset Verification Failures

```python
# Debug verification issues
from pipelines.dataset_registry.verification.verification_engine import VerificationEngine

verifier = VerificationEngine()
result = verifier.verify_dataset(
    name="test_dataset",
    file_path="data/test.dat",
    verification_config={"sha256": "expected_hash"}
)

print(f"Verification result: {result}")
```

### Log Analysis

Enable debug logging to troubleshoot issues:

```bash
export PBUF_LOG_LEVEL=DEBUG
```

Check log files for detailed error information:

```bash
# Check structured logs
tail -f data/registry/structured_logs.jsonl

# Check audit trail
tail -f data/registry/audit.jsonl
```

## Maintenance

### Regular Maintenance Tasks

#### 1. Cache Cleanup

```python
# Clean old cache files
from pipelines.dataset_registry.protocols.download_manager import CacheManager

cache_manager = CacheManager()
cache_manager.cleanup_old_files(max_age_days=30)
```

#### 2. Registry Integrity Check

```python
# Verify registry integrity
from pipelines.dataset_registry.core.registry_manager import RegistryManager

registry = RegistryManager()
integrity_report = registry.verify_registry_integrity()

print(f"Registry integrity: {integrity_report}")
```

#### 3. Dataset Re-verification

```bash
# Re-verify all datasets using CLI
python -m pipelines.dataset_registry.cli verify-all
```

#### 4. Configuration Updates

```python
# Update configuration programmatically
from pipelines.dataset_registry.core.config import DatasetRegistryConfigManager

config_manager = DatasetRegistryConfigManager()
config = config_manager.get_config()

# Update settings
config.cache_max_size_gb = 20.0
config.log_level = "WARNING"

# Save updated configuration
config_manager.save_config('updated_config.json')
```

### Monitoring

#### 1. Registry Statistics

```python
# Get registry statistics
from pipelines.dataset_registry.core.registry_manager import RegistryManager

registry = RegistryManager()
stats = registry.get_registry_statistics()

print(f"Total datasets: {stats['total_datasets']}")
print(f"Verified datasets: {stats['verified_datasets']}")
print(f"Failed verifications: {stats['failed_verifications']}")
```

#### 2. Cache Usage

```python
# Monitor cache usage
from pipelines.dataset_registry.protocols.download_manager import CacheManager

cache_manager = CacheManager()
usage = cache_manager.get_cache_usage()

print(f"Cache size: {usage['size_gb']:.2f} GB")
print(f"Cache utilization: {usage['utilization']:.1%}")
```

### Backup and Recovery

#### 1. Backup Registry

```bash
# Backup registry data
tar -czf registry_backup_$(date +%Y%m%d).tar.gz data/registry/

# Backup configuration
cp pbuf_config.json pbuf_config_backup_$(date +%Y%m%d).json
```

#### 2. Restore Registry

```bash
# Restore registry from backup
tar -xzf registry_backup_20231023.tar.gz

# Restore configuration
cp pbuf_config_backup_20231023.json pbuf_config.json
```

## Security Considerations

### 1. File Permissions

Ensure appropriate file permissions for registry data:

```bash
# Set secure permissions
chmod 600 pbuf_config.json  # Configuration files
chmod 755 data/registry/     # Registry directory
chmod 644 data/registry/*.json  # Registry files
```

### 2. Network Security

- Use HTTPS for all dataset downloads
- Verify SSL certificates
- Configure proxy settings if required

### 3. Data Integrity

- Always verify checksums for downloaded datasets
- Enable audit trail for all registry operations
- Regular integrity checks of registry data

## Support and Documentation

For additional support:

1. Check the [CLI Usage Guide](CLI_USAGE.md)
2. Review the [Extensibility Guide](EXTENSIBILITY_GUIDE.md)
3. Consult the [Structured Logging Guide](STRUCTURED_LOGGING_GUIDE.md)
4. Review test implementations in the test files

For issues or questions, consult the PBUF documentation or contact the development team.