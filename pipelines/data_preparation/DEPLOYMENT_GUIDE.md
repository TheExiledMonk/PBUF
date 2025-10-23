# Data Preparation Framework - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the PBUF Data Preparation & Derivation Framework in production environments. It covers installation, configuration, integration procedures, and operational considerations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Integration with PBUF Infrastructure](#integration-with-pbuf-infrastructure)
5. [Production Deployment](#production-deployment)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+), macOS 10.14+, Windows 10
- **Python**: 3.8 or higher
- **Memory**: 4 GB RAM minimum, 8 GB recommended
- **Storage**: 10 GB free space for framework and logs
- **CPU**: 2 cores minimum, 4+ cores recommended

#### Recommended Requirements
- **Memory**: 16 GB RAM for large dataset processing
- **Storage**: 100 GB+ for data caching and derived datasets
- **CPU**: 8+ cores for parallel processing
- **Network**: High-speed connection for dataset downloads

### Software Dependencies

#### Core Dependencies
```bash
# Python packages (install via pip or conda)
numpy>=1.19.0
scipy>=1.5.0
pandas>=1.1.0
jsonschema>=3.2.0
pytest>=6.0.0
```

#### Optional Dependencies
```bash
# For enhanced functionality
astropy>=4.0.0          # Astronomical calculations
h5py>=2.10.0            # HDF5 file support
matplotlib>=3.3.0       # Plotting and visualization
```

### PBUF Infrastructure

The framework requires access to:
- PBUF codebase (pipelines directory)
- Dataset registry system
- Data storage directories
- Logging infrastructure

## Installation

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv pbuf_data_prep_env
source pbuf_data_prep_env/bin/activate  # Linux/macOS
# or
pbuf_data_prep_env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install numpy scipy pandas jsonschema pytest

# Install optional dependencies (recommended)
pip install astropy h5py matplotlib

# For development/testing
pip install pytest-cov black flake8
```

### Step 3: PBUF Integration

```bash
# Set PYTHONPATH to include PBUF root
export PYTHONPATH=/path/to/PBUF:$PYTHONPATH

# Add to shell profile for persistence
echo 'export PYTHONPATH=/path/to/PBUF:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Verify Installation

```python
# Test basic imports
python -c "
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
from pipelines.data_preparation.core.schema import StandardDataset
from pipelines.data_preparation.core.validation import ValidationEngine
print('✅ Data Preparation Framework installed successfully')
"
```

## Configuration

### Environment Variables

Set the following environment variables for optimal operation:

```bash
# Core configuration
export PBUF_DATA_PREPARATION_OUTPUT_DIR="/path/to/data/derived"
export PBUF_DATA_PREPARATION_LOG_LEVEL="INFO"
export PBUF_DATA_PREPARATION_CACHE_ENABLED="true"

# Performance tuning
export PBUF_DATA_PREPARATION_MAX_MEMORY_GB="8"
export PBUF_DATA_PREPARATION_PARALLEL_PROCESSING="false"

# Registry integration
export PBUF_REGISTRY_BASE_DIR="/path/to/data/registry"
export PBUF_REGISTRY_MANIFEST_FILE="/path/to/data/datasets_manifest.json"
```

### Configuration File

Create a configuration file at `~/.pbuf/data_preparation_config.json`:

```json
{
  "framework": {
    "output_directory": "/path/to/data/derived",
    "cache_directory": "/path/to/data/cache",
    "log_directory": "/path/to/data/logs"
  },
  "processing": {
    "cache_enabled": true,
    "parallel_processing": false,
    "max_memory_usage_gb": 8,
    "chunk_size": 10000,
    "timeout_seconds": 3600
  },
  "validation": {
    "strict_mode": true,
    "covariance_validation": true,
    "physical_consistency_checks": true,
    "numerical_precision_tolerance": 1e-10
  },
  "logging": {
    "level": "INFO",
    "file_logging": true,
    "console_logging": true,
    "rotation_size_mb": 100,
    "max_log_files": 10
  },
  "registry": {
    "enabled": true,
    "base_directory": "/path/to/data/registry",
    "manifest_file": "/path/to/data/datasets_manifest.json",
    "auto_register_derived": true
  }
}
```

### Directory Structure Setup

Create the required directory structure:

```bash
# Create base directories
mkdir -p /path/to/data/{raw,derived,cache,logs,registry}

# Create subdirectories
mkdir -p /path/to/data/logs/{transformations,errors,performance}
mkdir -p /path/to/data/cache/{processing,validation}
mkdir -p /path/to/data/registry/{derived,audit}

# Set appropriate permissions
chmod 755 /path/to/data
chmod 755 /path/to/data/{raw,derived,cache,logs,registry}
chmod 644 /path/to/data/logs/*
```

## Integration with PBUF Infrastructure

### Registry Integration

#### Step 1: Configure Registry Manager

```python
# In your deployment script
from pipelines.dataset_registry.core.registry_manager import RegistryManager
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework

# Initialize registry manager
registry_manager = RegistryManager(
    base_directory=Path("/path/to/data/registry"),
    manifest_file=Path("/path/to/data/datasets_manifest.json")
)

# Initialize framework with registry
framework = DataPreparationFramework(
    registry_manager=registry_manager,
    output_directory=Path("/path/to/data/derived")
)
```

#### Step 2: Register Derivation Modules

```python
# Register all required derivation modules
from pipelines.data_preparation.derivation import (
    SNDerivationModule, BAODerivationModule, CMBDerivationModule,
    CCDerivationModule, RSDDerivationModule
)

modules = [
    SNDerivationModule(),
    BAODerivationModule(),
    CMBDerivationModule(),
    CCDerivationModule(),
    RSDDerivationModule()
]

for module in modules:
    framework.register_derivation_module(module)
```

### Fit Pipeline Integration

#### Step 1: Update datasets.py

Modify `pipelines/fit_core/datasets.py` to use the preparation framework:

```python
# Add at the top of datasets.py
try:
    from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
    from pipelines.data_preparation.derivation import *
    PREPARATION_FRAMEWORK_AVAILABLE = True
except ImportError:
    PREPARATION_FRAMEWORK_AVAILABLE = False

# Global framework instance
_framework_instance = None

def _get_framework_instance():
    """Get or create framework instance."""
    global _framework_instance
    if _framework_instance is None and PREPARATION_FRAMEWORK_AVAILABLE:
        _framework_instance = DataPreparationFramework()
        # Register modules
        modules = [SNDerivationModule(), BAODerivationModule(), CMBDerivationModule()]
        for module in modules:
            _framework_instance.register_derivation_module(module)
    return _framework_instance

def load_dataset(name: str, use_preparation_framework: bool = True) -> DatasetDict:
    """
    Load dataset with optional preparation framework integration.
    
    Args:
        name: Dataset name
        use_preparation_framework: Whether to use preparation framework
        
    Returns:
        DatasetDict in standard format
    """
    if use_preparation_framework and PREPARATION_FRAMEWORK_AVAILABLE:
        framework = _get_framework_instance()
        if framework:
            try:
                standard_dataset = framework.prepare_dataset(name)
                return _convert_to_dataset_dict(standard_dataset)
            except Exception as e:
                print(f"Framework processing failed, falling back to legacy: {e}")
    
    # Fallback to legacy loading
    return _load_dataset_legacy(name)

def _convert_to_dataset_dict(standard_dataset) -> DatasetDict:
    """Convert StandardDataset to DatasetDict format."""
    return {
        "observations": standard_dataset.observable,
        "uncertainties": standard_dataset.uncertainty,
        "covariance": standard_dataset.covariance,
        "redshifts": standard_dataset.z,
        "metadata": standard_dataset.metadata
    }
```

#### Step 2: Test Integration

```python
# Test the integration
from pipelines.fit_core.datasets import load_dataset

# This should now use the preparation framework
dataset = load_dataset("sn_pantheon_plus")
print(f"Loaded {len(dataset['redshifts'])} data points")
```

## Production Deployment

### Deployment Script

Create a deployment script `deploy_data_preparation_framework.py`:

```python
#!/usr/bin/env python3
"""
Production deployment script for PBUF Data Preparation Framework.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

def setup_logging():
    """Set up production logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/path/to/data/logs/deployment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_environment() -> bool:
    """Validate deployment environment."""
    logger = logging.getLogger(__name__)
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        return False
    
    # Check required directories
    required_dirs = [
        Path("/path/to/data/raw"),
        Path("/path/to/data/derived"),
        Path("/path/to/data/logs"),
        Path("/path/to/data/registry")
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.error(f"Required directory missing: {dir_path}")
            return False
        if not dir_path.is_dir():
            logger.error(f"Path is not a directory: {dir_path}")
            return False
    
    # Check imports
    try:
        from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
        from pipelines.data_preparation.core.validation import ValidationEngine
        logger.info("✅ Framework imports successful")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False
    
    return True

def initialize_framework() -> 'DataPreparationFramework':
    """Initialize and configure the framework."""
    from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
    from pipelines.data_preparation.derivation import (
        SNDerivationModule, BAODerivationModule, CMBDerivationModule,
        CCDerivationModule, RSDDerivationModule
    )
    
    logger = logging.getLogger(__name__)
    
    # Initialize framework
    framework = DataPreparationFramework(
        output_directory=Path("/path/to/data/derived")
    )
    
    # Register derivation modules
    modules = [
        SNDerivationModule(),
        BAODerivationModule(),
        CMBDerivationModule(),
        CCDerivationModule(),
        RSDDerivationModule()
    ]
    
    for module in modules:
        framework.register_derivation_module(module)
        logger.info(f"Registered {module.dataset_type} derivation module")
    
    return framework

def run_validation_tests(framework) -> bool:
    """Run validation tests on the framework."""
    logger = logging.getLogger(__name__)
    
    try:
        # Test framework initialization
        available_types = framework.get_available_dataset_types()
        logger.info(f"Available dataset types: {available_types}")
        
        # Test validation engine
        from pipelines.data_preparation.core.validation import ValidationEngine
        validation_engine = ValidationEngine()
        logger.info("✅ Validation engine initialized")
        
        # Additional tests can be added here
        
        return True
        
    except Exception as e:
        logger.error(f"Validation test failed: {e}")
        return False

def generate_deployment_report(framework) -> Dict[str, Any]:
    """Generate deployment report."""
    return {
        "deployment_timestamp": "2023-01-01T00:00:00Z",
        "framework_version": "1.0.0",
        "available_modules": framework.get_available_dataset_types(),
        "configuration": {
            "output_directory": str(framework.output_directory),
            "cache_enabled": True,
            "validation_enabled": True
        },
        "status": "DEPLOYED"
    }

def main():
    """Main deployment function."""
    logger = setup_logging()
    logger.info("Starting PBUF Data Preparation Framework deployment")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return 1
    
    # Initialize framework
    try:
        framework = initialize_framework()
        logger.info("✅ Framework initialized successfully")
    except Exception as e:
        logger.error(f"Framework initialization failed: {e}")
        return 1
    
    # Run validation tests
    if not run_validation_tests(framework):
        logger.error("Validation tests failed")
        return 1
    
    # Generate deployment report
    report = generate_deployment_report(framework)
    report_file = Path("/path/to/data/logs/deployment_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Deployment completed successfully")
    logger.info(f"Deployment report: {report_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Service Configuration

For production environments, consider running as a service:

#### systemd Service (Linux)

Create `/etc/systemd/system/pbuf-data-preparation.service`:

```ini
[Unit]
Description=PBUF Data Preparation Framework
After=network.target

[Service]
Type=simple
User=pbuf
Group=pbuf
WorkingDirectory=/path/to/PBUF
Environment=PYTHONPATH=/path/to/PBUF
ExecStart=/path/to/pbuf_data_prep_env/bin/python -m pipelines.data_preparation.service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable pbuf-data-preparation
sudo systemctl start pbuf-data-preparation
sudo systemctl status pbuf-data-preparation
```

## Monitoring and Maintenance

### Health Checks

Implement health check endpoints:

```python
def health_check() -> Dict[str, Any]:
    """Perform framework health check."""
    from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
    
    try:
        framework = DataPreparationFramework()
        available_types = framework.get_available_dataset_types()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "available_modules": available_types,
            "framework_version": "1.0.0"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }
```

### Log Monitoring

Set up log monitoring and alerting:

```bash
# Monitor error logs
tail -f /path/to/data/logs/errors.log

# Set up log rotation
cat > /etc/logrotate.d/pbuf-data-preparation << EOF
/path/to/data/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 pbuf pbuf
}
EOF
```

### Performance Monitoring

Monitor system resources:

```python
import psutil
import time

def monitor_resources():
    """Monitor system resources during processing."""
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"CPU: {cpu_percent}%")
        print(f"Memory: {memory.percent}% ({memory.used/1e9:.1f}GB used)")
        print(f"Disk: {disk.percent}% ({disk.used/1e9:.1f}GB used)")
        
        time.sleep(60)  # Check every minute
```

### Backup and Recovery

Implement backup procedures:

```bash
#!/bin/bash
# backup_data_preparation.sh

BACKUP_DIR="/backup/pbuf/data_preparation"
DATA_DIR="/path/to/data"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup derived datasets
tar -czf "$BACKUP_DIR/$DATE/derived_datasets.tar.gz" "$DATA_DIR/derived"

# Backup registry
tar -czf "$BACKUP_DIR/$DATE/registry.tar.gz" "$DATA_DIR/registry"

# Backup configuration
cp ~/.pbuf/data_preparation_config.json "$BACKUP_DIR/$DATE/"

# Backup logs (last 7 days)
find "$DATA_DIR/logs" -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/$DATE/" \;

echo "Backup completed: $BACKUP_DIR/$DATE"
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Symptoms**: `ModuleNotFoundError` when importing framework components

**Solutions**:
```bash
# Check PYTHONPATH
echo $PYTHONPATH

# Verify PBUF directory structure
ls -la /path/to/PBUF/pipelines/data_preparation/

# Test imports manually
python -c "import sys; print(sys.path)"
```

#### 2. Permission Errors

**Symptoms**: Permission denied when accessing data directories

**Solutions**:
```bash
# Check directory permissions
ls -la /path/to/data/

# Fix permissions
sudo chown -R pbuf:pbuf /path/to/data/
sudo chmod -R 755 /path/to/data/
```

#### 3. Memory Issues

**Symptoms**: Out of memory errors during processing

**Solutions**:
```python
# Monitor memory usage
import psutil
print(f"Available memory: {psutil.virtual_memory().available/1e9:.1f} GB")

# Reduce processing chunk size
framework = DataPreparationFramework()
# Configure smaller chunk sizes in derivation modules
```

#### 4. Registry Integration Issues

**Symptoms**: Cannot connect to or access dataset registry

**Solutions**:
```python
# Test registry connection
from pipelines.dataset_registry.core.registry_manager import RegistryManager
registry = RegistryManager()
print(registry.list_available_datasets())

# Check registry configuration
print(f"Registry base: {registry.base_directory}")
print(f"Manifest file: {registry.manifest_file}")
```

### Debug Mode

Enable comprehensive debugging:

```python
import logging

# Enable debug logging for all components
logging.getLogger('data_preparation').setLevel(logging.DEBUG)
logging.getLogger('dataset_registry').setLevel(logging.DEBUG)

# Add detailed file logging
debug_handler = logging.FileHandler('/path/to/data/logs/debug.log')
debug_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
debug_handler.setFormatter(formatter)

logger = logging.getLogger('data_preparation')
logger.addHandler(debug_handler)
```

## Performance Optimization

### Memory Optimization

```python
# Configure memory limits
import os
os.environ['PBUF_DATA_PREPARATION_MAX_MEMORY_GB'] = '8'

# Use memory-efficient processing
framework = DataPreparationFramework()
# Process datasets in chunks
```

### Disk I/O Optimization

```bash
# Use SSD storage for cache and derived datasets
# Mount with appropriate options
mount -o noatime,nodiratime /dev/ssd1 /path/to/data/cache

# Configure appropriate file system
# ext4 with large_file option for large datasets
```

### CPU Optimization

```python
# Enable parallel processing (when deterministic)
os.environ['PBUF_DATA_PREPARATION_PARALLEL_PROCESSING'] = 'true'

# Set appropriate number of workers
import multiprocessing
os.environ['PBUF_DATA_PREPARATION_WORKERS'] = str(multiprocessing.cpu_count())
```

### Caching Strategy

```python
# Configure intelligent caching
framework = DataPreparationFramework()

# Cache configuration
cache_config = {
    'enabled': True,
    'max_size_gb': 10,
    'ttl_hours': 24,
    'cleanup_threshold': 0.8
}
```

This deployment guide provides comprehensive instructions for successfully deploying the data preparation framework in production environments. Follow the steps carefully and adapt the configuration to your specific infrastructure requirements.