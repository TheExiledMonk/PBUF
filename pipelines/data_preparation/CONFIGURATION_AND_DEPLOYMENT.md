# Configuration and Deployment Support for CMB Raw Parameter Processing

This document describes the configuration and deployment support implemented for the CMB raw parameter processing feature.

## Overview

Task 8 "Add configuration and deployment support" has been implemented with three main components:

1. **Configuration Integration** (8.1)
2. **Logging and Monitoring** (8.2) 
3. **Deployment Validation** (8.3)

## 8.1 Configuration Integration

### Files Created
- `pipelines/data_preparation/core/cmb_config_integration.py` - Main configuration integration module
- `pipelines/data_preparation/config_validation.py` - Configuration validation script

### Features Implemented

#### Integrated Configuration Management
- `DataPreparationConfig` class that extends framework configuration with CMB-specific settings
- `DataPreparationConfigManager` for loading, validating, and managing configurations
- Seamless integration with existing `CMBConfig` from the derivation module

#### Environment Variable Support
- Comprehensive environment variable support for all configuration parameters
- Framework-level variables: `PBUF_OUTPUT_PATH`, `PBUF_CACHE_PATH`, `PBUF_LOG_LEVEL`, etc.
- CMB-specific variables: `CMB_USE_RAW_PARAMETERS`, `CMB_Z_RECOMBINATION`, etc.
- Automatic configuration updates from environment variables

#### Configuration File Support
- JSON configuration file format with nested structure
- Multiple search locations for configuration files
- Environment-specific configuration overrides
- Example configurations for development, production, and testing environments

#### Configuration Validation
- Comprehensive validation of all configuration parameters
- Deployment-specific validation checks
- Path validation and creation
- Parameter bounds checking

### Usage Examples

```bash
# Validate configuration file
python pipelines/data_preparation/config_validation.py --validate config/data_preparation.json

# Check environment configuration
python pipelines/data_preparation/config_validation.py --check-environment

# Create example configurations
python pipelines/data_preparation/config_validation.py --create-examples

# Show environment variables help
python pipelines/data_preparation/config_validation.py --help-env-vars
```

## 8.2 Logging and Monitoring

### Files Created
- `pipelines/data_preparation/core/cmb_logging.py` - Structured logging and monitoring module

### Features Implemented

#### Structured Logging
- `CMBStructuredLogger` class for comprehensive logging
- JSON-formatted structured log output
- Event-based logging with detailed context
- Integration with existing data preparation framework logging

#### Performance Monitoring
- `PerformanceMetrics` class for tracking operation performance
- Context manager for automatic performance monitoring
- Memory usage tracking (when psutil available)
- CPU usage monitoring
- Operation timing and metrics collection

#### Diagnostic Capabilities
- Comprehensive diagnostic report generation
- Processing step tracking and error context
- Numerical instability detection
- Performance warning thresholds
- Troubleshooting guide generation

#### Logging Events
- Processing start/complete events
- Parameter detection and validation events
- Distance derivation events
- Covariance propagation events
- Fallback to legacy mode events
- Performance warnings and errors

### Integration with CMB Processing

The CMB derivation module has been updated to use structured logging:
- Automatic logger initialization with integrated configuration
- Performance monitoring for key processing steps
- Detailed error context and diagnostic information
- Fallback handling when logging is not available

## 8.3 Deployment Validation

### Files Created
- `pipelines/data_preparation/deployment_validation.py` - Comprehensive deployment validation script

### Features Implemented

#### System Environment Checks
- Python version compatibility validation
- Memory availability checking
- Disk space validation for output and cache directories
- File permissions verification
- Environment variables detection

#### Dependency Validation
- Required dependencies: NumPy, PBUF background integrators
- Optional dependencies: pandas, psutil, scipy
- Version compatibility checking
- Import validation

#### Registry Integration Testing
- Mock registry entry processing
- Parameter detection functionality testing
- Legacy fallback validation
- Format classification testing

#### Pipeline Compatibility Verification
- StandardDataset format compatibility
- Metadata format validation
- Covariance matrix format testing
- Fitting pipeline integration checks

#### Performance Requirements
- Numerical stability testing
- Memory usage pattern validation
- Computation speed benchmarking
- Matrix operation performance

### Usage Examples

```bash
# Run all deployment validation checks
python pipelines/data_preparation/deployment_validation.py

# Run with specific configuration
python pipelines/data_preparation/deployment_validation.py --config config/production.json

# Generate JSON report
python pipelines/data_preparation/deployment_validation.py --json --output deployment_report.json

# Quiet mode for automated checks
python pipelines/data_preparation/deployment_validation.py --quiet
```

## Configuration Examples

### Development Configuration
```json
{
  "data_preparation": {
    "framework_enabled": true,
    "output_path": "data/prepared/",
    "cache_path": "data/cache/",
    "logging": {
      "log_level": "DEBUG",
      "performance_monitoring": true
    },
    "derivation": {
      "cmb": {
        "use_raw_parameters": true,
        "cache_computations": false,
        "performance_monitoring": true
      }
    }
  }
}
```

### Production Configuration
```json
{
  "data_preparation": {
    "framework_enabled": true,
    "output_path": "/data/pbuf/prepared/",
    "cache_path": "/data/pbuf/cache/",
    "performance": {
      "parallel_processing": true,
      "max_workers": 8,
      "memory_limit_gb": 16.0
    },
    "logging": {
      "log_level": "INFO",
      "log_file": "/var/log/pbuf/data_preparation.log"
    },
    "derivation": {
      "cmb": {
        "use_raw_parameters": true,
        "cache_computations": true,
        "jacobian_step_size": 1e-6
      }
    }
  }
}
```

## Environment Variables

### Framework Variables
- `PBUF_OUTPUT_PATH` - Path for prepared dataset outputs
- `PBUF_CACHE_PATH` - Path for caching intermediate results
- `PBUF_FRAMEWORK_ENABLED` - Enable/disable framework (true/false)
- `PBUF_LOG_LEVEL` - Logging level (DEBUG/INFO/WARNING/ERROR)
- `PBUF_PERFORMANCE_MONITORING` - Enable performance monitoring (true/false)

### CMB-Specific Variables
- `CMB_USE_RAW_PARAMETERS` - Enable raw parameter processing (true/false)
- `CMB_Z_RECOMBINATION` - Recombination redshift (float)
- `CMB_JACOBIAN_STEP_SIZE` - Numerical differentiation step size (float)
- `CMB_FALLBACK_TO_LEGACY` - Auto-fallback to legacy mode (true/false)
- `CMB_CACHE_COMPUTATIONS` - Cache expensive computations (true/false)

## Integration with Existing Code

### CMB Derivation Module Updates
- Updated `process_cmb_dataset` function to use integrated configuration
- Added structured logging throughout processing pipeline
- Performance monitoring for key operations
- Enhanced error reporting with diagnostic context

### Backward Compatibility
- All existing functionality remains unchanged
- Default configurations maintain current behavior
- Graceful fallback when new features are not available
- Optional dependency handling

## Deployment Checklist

1. **Configuration Validation**
   ```bash
   python pipelines/data_preparation/config_validation.py --deployment-check config/production.json
   ```

2. **System Validation**
   ```bash
   python pipelines/data_preparation/deployment_validation.py --config config/production.json
   ```

3. **Environment Setup**
   ```bash
   export PBUF_ENV=production
   export PBUF_LOG_LEVEL=INFO
   export CMB_USE_RAW_PARAMETERS=true
   ```

4. **Verify Integration**
   - Test with sample CMB datasets
   - Verify logging output
   - Check performance metrics
   - Validate output format compatibility

## Requirements Satisfied

This implementation satisfies the following requirements from the specification:

- **5.1, 5.2, 5.3, 5.4, 5.5**: Configuration integration with framework settings and environment variable support
- **9.1, 9.2, 9.3, 9.4, 9.5**: Structured logging, performance monitoring, and diagnostic capabilities
- **7.4, 7.5**: Deployment validation with system checks and pipeline compatibility verification

The implementation provides a robust, production-ready configuration and deployment system for the CMB raw parameter processing feature.