# Data Preparation Framework - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Basic Usage](#basic-usage)
4. [Adding New Dataset Types](#adding-new-dataset-types)
5. [Integration with PBUF Infrastructure](#integration-with-pbuf-infrastructure)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Introduction

The PBUF Data Preparation & Derivation Framework provides a unified system for transforming raw cosmological datasets into standardized, analysis-ready formats. This guide will help you understand how to use the framework effectively and extend it for new dataset types.

### Key Features

- **Unified Data Format**: All datasets are transformed into a consistent `StandardDataset` format
- **Extensible Architecture**: Easy to add new dataset types through derivation modules
- **Comprehensive Validation**: Multi-stage validation ensures data quality and integrity
- **Complete Provenance**: Full traceability from raw data to final results
- **Error Recovery**: Robust error handling with detailed diagnostics
- **Performance Monitoring**: System health monitoring and performance metrics

### Supported Dataset Types

- **Supernovae (SN)**: Type Ia supernova distance measurements
- **Baryon Acoustic Oscillations (BAO)**: Isotropic and anisotropic BAO measurements
- **Cosmic Microwave Background (CMB)**: Distance priors from Planck
- **Cosmic Chronometers (CC)**: H(z) measurements from stellar chronometry
- **Redshift Space Distortions (RSD)**: Growth rate measurements (fσ₈)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- NumPy, SciPy for numerical operations
- Access to PBUF codebase and data directories

### Installation

The framework is integrated into the PBUF codebase. Ensure your Python path includes the PBUF root directory:

```bash
export PYTHONPATH=/path/to/PBUF:$PYTHONPATH
```

### Quick Start

```python
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
from pipelines.data_preparation.derivation.sn_derivation import SNDerivationModule
from pathlib import Path

# Initialize framework
framework = DataPreparationFramework()

# Register derivation modules
sn_module = SNDerivationModule()
framework.register_derivation_module(sn_module)

# Process a dataset
dataset = framework.prepare_dataset(
    dataset_name="my_supernova_data",
    raw_data_path=Path("data/raw/sn_data.csv"),
    metadata={"dataset_type": "sn", "version": "1.0"}
)

print(f"Processed {len(dataset.z)} supernovae")
```

## Basic Usage

### Processing a Dataset

The main workflow involves three steps:

1. **Initialize the Framework**
2. **Register Derivation Modules**
3. **Process Datasets**

#### Step 1: Initialize Framework

```python
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework

# Basic initialization
framework = DataPreparationFramework()

# With custom output directory
framework = DataPreparationFramework(
    output_directory=Path("my_output_dir")
)

# With registry integration (if available)
from pipelines.dataset_registry.core.registry_manager import RegistryManager
registry_manager = RegistryManager()
framework = DataPreparationFramework(registry_manager=registry_manager)
```

#### Step 2: Register Derivation Modules

```python
from pipelines.data_preparation.derivation.sn_derivation import SNDerivationModule
from pipelines.data_preparation.derivation.bao_derivation import BAODerivationModule
from pipelines.data_preparation.derivation.cmb_derivation import CMBDerivationModule

# Register modules for dataset types you'll use
framework.register_derivation_module(SNDerivationModule())
framework.register_derivation_module(BAODerivationModule())
framework.register_derivation_module(CMBDerivationModule())
```

#### Step 3: Process Datasets

```python
# Process with explicit file path and metadata
dataset = framework.prepare_dataset(
    dataset_name="pantheon_plus",
    raw_data_path=Path("data/raw/pantheon_plus.csv"),
    metadata={
        "dataset_type": "sn",
        "description": "Pantheon+ Supernova Sample",
        "citation": "Brout et al. 2022",
        "version": "1.0"
    }
)

# Process with registry integration (if configured)
dataset = framework.prepare_dataset("pantheon_plus")

# Force reprocessing (ignore cache)
dataset = framework.prepare_dataset(
    "pantheon_plus", 
    force_reprocess=True
)
```

### Working with StandardDataset

The processed dataset is returned as a `StandardDataset` object:

```python
# Access data arrays
redshifts = dataset.z
observables = dataset.observable
uncertainties = dataset.uncertainty
covariance_matrix = dataset.covariance  # May be None

# Access metadata
print(dataset.metadata["description"])
print(dataset.metadata["citation"])

# Validate dataset
if dataset.validate_schema():
    print("Dataset schema is valid")

if dataset.validate_numerical():
    print("Dataset numerical integrity is good")

# Save dataset
dataset.save(Path("output/processed_dataset.json"))
```

### Validation and Quality Assurance

```python
from pipelines.data_preparation.core.validation import ValidationEngine

validation_engine = ValidationEngine()
results = validation_engine.validate_dataset(dataset, "pantheon_plus")

if results['validation_passed']:
    print("✅ Dataset validation passed")
else:
    print("❌ Dataset validation failed")
    for error in results.get('errors', []):
        print(f"  Error: {error}")
    for warning in results.get('warnings', []):
        print(f"  Warning: {warning}")
```

## Adding New Dataset Types

### Creating a New Derivation Module

To add support for a new dataset type, create a derivation module that implements the `DerivationModule` interface:

```python
from pipelines.data_preparation.core.interfaces import DerivationModule
from pipelines.data_preparation.core.schema import StandardDataset
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

class MyCustomDerivationModule(DerivationModule):
    """Derivation module for my custom dataset type."""
    
    @property
    def dataset_type(self) -> str:
        return "my_custom_type"
    
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        """Validate raw data before processing."""
        try:
            # Check file exists and is readable
            if not raw_data_path.exists():
                return False
            
            # Check file format (example for CSV)
            if raw_data_path.suffix.lower() != '.csv':
                return False
            
            # Load and validate data structure
            data = pd.read_csv(raw_data_path)
            required_columns = ['z', 'observable', 'error']
            
            if not all(col in data.columns for col in required_columns):
                return False
            
            # Additional validation checks
            if len(data) == 0:
                return False
            
            if data['z'].min() < 0 or data['z'].max() > 10:
                return False
            
            return True
            
        except Exception:
            return False
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """Transform raw data to standard format."""
        # Load raw data
        data = pd.read_csv(raw_data_path)
        
        # Apply transformations
        z = data['z'].values
        observable = data['observable'].values
        uncertainty = data['error'].values
        
        # Apply any necessary corrections or transformations
        # Example: unit conversion, systematic corrections, etc.
        observable = self._apply_corrections(observable, z)
        
        # Create covariance matrix if needed
        covariance = None
        if 'covariance_file' in metadata:
            covariance = self._load_covariance_matrix(metadata['covariance_file'])
        
        # Prepare metadata
        processed_metadata = {
            'source_file': str(raw_data_path),
            'processing_date': datetime.now(timezone.utc).isoformat(),
            'dataset_type': self.dataset_type,
            'n_data_points': len(z),
            'redshift_range': [float(z.min()), float(z.max())],
            **metadata  # Include original metadata
        }
        
        return StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=covariance,
            metadata=processed_metadata
        )
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return summary of applied transformations."""
        return {
            'transformations_applied': [
                'Unit conversion to standard units',
                'Systematic correction application',
                'Error propagation'
            ],
            'formulas_used': [
                'corrected_obs = raw_obs * correction_factor',
                'final_error = sqrt(stat_error^2 + sys_error^2)'
            ],
            'references': [
                'Smith et al. 2023 - Correction methodology',
                'Jones et al. 2022 - Error analysis'
            ]
        }
    
    def _apply_corrections(self, observable: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Apply dataset-specific corrections."""
        # Implement your correction logic here
        correction_factor = 1.0 + 0.01 * z  # Example correction
        return observable * correction_factor
    
    def _load_covariance_matrix(self, covariance_file: str) -> np.ndarray:
        """Load covariance matrix from file."""
        # Implement covariance matrix loading
        return np.loadtxt(covariance_file)
```

### Registering Your Module

```python
# Register your custom module
custom_module = MyCustomDerivationModule()
framework.register_derivation_module(custom_module)

# Now you can process datasets of your custom type
dataset = framework.prepare_dataset(
    dataset_name="my_custom_dataset",
    raw_data_path=Path("data/raw/custom_data.csv"),
    metadata={"dataset_type": "my_custom_type"}
)
```

### Testing Your Module

Create unit tests for your derivation module:

```python
import unittest
from pathlib import Path
import tempfile
import pandas as pd

class TestMyCustomDerivationModule(unittest.TestCase):
    
    def setUp(self):
        self.module = MyCustomDerivationModule()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'z': [0.1, 0.2, 0.3, 0.4, 0.5],
            'observable': [1.0, 1.1, 1.2, 1.3, 1.4],
            'error': [0.1, 0.1, 0.1, 0.1, 0.1]
        })
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def test_validate_input_valid_data(self):
        """Test input validation with valid data."""
        metadata = {"dataset_type": "my_custom_type"}
        result = self.module.validate_input(Path(self.temp_file.name), metadata)
        self.assertTrue(result)
    
    def test_derive_produces_standard_dataset(self):
        """Test that derive produces a valid StandardDataset."""
        metadata = {"dataset_type": "my_custom_type"}
        dataset = self.module.derive(Path(self.temp_file.name), metadata)
        
        # Check dataset structure
        self.assertEqual(len(dataset.z), 5)
        self.assertEqual(len(dataset.observable), 5)
        self.assertEqual(len(dataset.uncertainty), 5)
        self.assertIsInstance(dataset.metadata, dict)
        
        # Check data validity
        self.assertTrue(dataset.validate_schema())
        self.assertTrue(dataset.validate_numerical())
    
    def tearDown(self):
        Path(self.temp_file.name).unlink()

if __name__ == '__main__':
    unittest.main()
```

## Integration with PBUF Infrastructure

### Registry Integration

The framework integrates with the existing PBUF dataset registry:

```python
from pipelines.dataset_registry.core.registry_manager import RegistryManager

# Initialize with registry
registry_manager = RegistryManager()
framework = DataPreparationFramework(registry_manager=registry_manager)

# Process datasets from registry
dataset = framework.prepare_dataset("cmb_planck2018")
```

### Fit Pipeline Integration

The framework integrates seamlessly with existing fit pipelines:

```python
# The existing load_dataset function automatically uses the framework
from pipelines.fit_core.datasets import load_dataset

# This will use the preparation framework if available
dataset_dict = load_dataset("pantheon_plus")

# Access data in the familiar format
observations = dataset_dict["observations"]
uncertainties = dataset_dict["uncertainties"]
redshifts = dataset_dict["redshifts"]
covariance = dataset_dict.get("covariance")
metadata = dataset_dict["metadata"]
```

### Provenance Tracking

All processed datasets are automatically registered with provenance information:

```python
# Provenance is automatically tracked
dataset = framework.prepare_dataset("pantheon_plus")

# Access provenance information
provenance = dataset.metadata.get("provenance", {})
print(f"Source hash: {provenance.get('source_hash')}")
print(f"Processing timestamp: {provenance.get('processing_timestamp')}")
print(f"Environment hash: {provenance.get('environment_hash')}")
```

## Advanced Features

### Caching and Performance

```python
# Enable/disable caching
framework = DataPreparationFramework()

# Force reprocessing (ignore cache)
dataset = framework.prepare_dataset("dataset_name", force_reprocess=True)

# Check if dataset is cached
cache_key = framework._generate_cache_key("dataset_name", raw_data_path, metadata)
is_cached = cache_key in framework._processing_cache
```

### Error Handling and Recovery

```python
from pipelines.data_preparation.core.interfaces import ProcessingError
from pipelines.data_preparation.core.error_handling import EnhancedProcessingError

try:
    dataset = framework.prepare_dataset("problematic_dataset")
except EnhancedProcessingError as e:
    print(f"Processing failed at stage: {e.stage}")
    print(f"Error type: {e.error_type}")
    print(f"Error message: {e.error_message}")
    print(f"Suggested actions:")
    for action in e.suggested_actions:
        print(f"  - {action}")
    
    # Check if recovery was attempted
    if e.recovery_attempted:
        print(f"Recovery successful: {e.recovery_successful}")
```

### Logging and Monitoring

```python
import logging

# Enable detailed logging
logging.getLogger('data_preparation').setLevel(logging.DEBUG)

# Access transformation logs
from pipelines.data_preparation.core.transformation_logging import TransformationLogger

logger = TransformationLogger("dataset_name", "sn", Path("logs"))
logger.start_processing(metadata)
logger.log_transformation_step(
    "magnitude_conversion",
    "Convert apparent magnitude to distance modulus",
    input_data={"m": apparent_mag},
    output_data={"mu": distance_modulus},
    formula="μ = m - M"
)
logger.end_processing({"final_dataset": dataset})
```

### System Health Monitoring

```python
from pipelines.data_preparation.core.transformation_logging import SystemHealthMonitor

monitor = SystemHealthMonitor()
monitor.start_monitoring()

# Process datasets...

monitor.stop_monitoring()
health_summary = monitor.get_health_summary()
print(f"Peak memory usage: {health_summary['peak_memory_mb']} MB")
print(f"Processing duration: {health_summary['total_duration_s']} seconds")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when importing framework components

**Solution**: 
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH=/path/to/PBUF:$PYTHONPATH

# Or add to Python script
import sys
sys.path.insert(0, '/path/to/PBUF')
```

#### 2. File Not Found Errors

**Problem**: Raw data files cannot be found

**Solution**:
```python
# Check file paths
raw_data_path = Path("data/raw/dataset.csv")
if not raw_data_path.exists():
    print(f"File not found: {raw_data_path}")
    print(f"Current directory: {Path.cwd()}")

# Use absolute paths
raw_data_path = Path("/absolute/path/to/data/raw/dataset.csv")
```

#### 3. Validation Failures

**Problem**: Dataset validation fails

**Solution**:
```python
# Get detailed validation results
results = validation_engine.validate_dataset(dataset, "dataset_name")
print("Validation details:")
for category, details in results.items():
    if isinstance(details, dict) and 'errors' in details:
        for error in details['errors']:
            print(f"  {category}: {error}")
```

#### 4. Memory Issues

**Problem**: Out of memory errors with large datasets

**Solution**:
```python
# Process in chunks or reduce dataset size
# Monitor memory usage
import psutil
print(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")

# Use streaming processing for large files
# (implement in your derivation module)
```

#### 5. Performance Issues

**Problem**: Processing takes too long

**Solution**:
```python
# Enable caching
framework = DataPreparationFramework()

# Use force_reprocess=False (default)
dataset = framework.prepare_dataset("dataset_name", force_reprocess=False)

# Monitor processing stages
import time
start_time = time.time()
dataset = framework.prepare_dataset("dataset_name")
print(f"Processing took {time.time() - start_time:.2f} seconds")
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('data_preparation')
logger.setLevel(logging.DEBUG)

# Add file handler for persistent logs
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

## Best Practices

### 1. Data Organization

```
data/
├── raw/                    # Original, unmodified datasets
│   ├── sn_pantheon_plus.csv
│   ├── bao_boss_dr12.csv
│   └── cmb_planck2018.json
├── derived/                # Processed, analysis-ready datasets
│   ├── sn_pantheon_plus_derived.json
│   └── bao_boss_dr12_derived.json
└── logs/                   # Processing logs and reports
    ├── transformation_logs/
    └── validation_reports/
```

### 2. Metadata Management

Always provide comprehensive metadata:

```python
metadata = {
    "dataset_type": "sn",
    "description": "Pantheon+ Type Ia Supernova Sample",
    "citation": "Brout et al. 2022, ApJ 938, 110",
    "version": "1.0",
    "download_date": "2023-01-15",
    "source_url": "https://github.com/PantheonPlusSH0ES/DataRelease",
    "processing_notes": "Applied host galaxy corrections",
    "systematic_uncertainties": "Included in covariance matrix"
}
```

### 3. Error Handling

Implement robust error handling:

```python
from pipelines.data_preparation.core.interfaces import ProcessingError

def safe_dataset_processing(framework, dataset_name, **kwargs):
    """Safely process a dataset with comprehensive error handling."""
    try:
        return framework.prepare_dataset(dataset_name, **kwargs)
    except ProcessingError as e:
        print(f"Processing error for {dataset_name}:")
        print(f"  Stage: {e.stage}")
        print(f"  Type: {e.error_type}")
        print(f"  Message: {e.error_message}")
        
        # Log error details
        with open(f"error_log_{dataset_name}.txt", "w") as f:
            f.write(f"Error processing {dataset_name}\n")
            f.write(f"Timestamp: {e.timestamp}\n")
            f.write(f"Stage: {e.stage}\n")
            f.write(f"Error: {e.error_message}\n")
            f.write(f"Context: {e.context}\n")
            f.write("Suggested actions:\n")
            for action in e.suggested_actions:
                f.write(f"  - {action}\n")
        
        return None
    except Exception as e:
        print(f"Unexpected error processing {dataset_name}: {e}")
        return None
```

### 4. Testing

Always test your derivation modules:

```python
def test_derivation_module(module, test_data_path, expected_results):
    """Test a derivation module with known data."""
    # Test input validation
    assert module.validate_input(test_data_path, {"dataset_type": module.dataset_type})
    
    # Test derivation
    dataset = module.derive(test_data_path, {"dataset_type": module.dataset_type})
    
    # Validate output
    assert dataset.validate_schema()
    assert dataset.validate_numerical()
    
    # Check expected results
    assert len(dataset.z) == expected_results["n_points"]
    assert abs(dataset.z.min() - expected_results["z_min"]) < 0.001
    assert abs(dataset.z.max() - expected_results["z_max"]) < 0.001
    
    print(f"✅ {module.dataset_type} module tests passed")
```

### 5. Documentation

Document your derivation modules thoroughly:

```python
class MyDerivationModule(DerivationModule):
    """
    Derivation module for processing my custom dataset type.
    
    This module processes datasets containing [description of data type]
    and applies the following transformations:
    1. [Transformation 1 description]
    2. [Transformation 2 description]
    
    Input Format:
        CSV file with columns: [list columns]
        
    Output Format:
        StandardDataset with:
        - z: [description]
        - observable: [description] 
        - uncertainty: [description]
        
    References:
        - Author et al. Year - Paper describing methodology
        - Author et al. Year - Paper describing corrections
    """
```

### 6. Version Control

Track changes to your derivation modules:

```python
class MyDerivationModule(DerivationModule):
    VERSION = "1.2.0"
    CHANGELOG = {
        "1.2.0": "Added systematic uncertainty handling",
        "1.1.0": "Improved error propagation",
        "1.0.0": "Initial implementation"
    }
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        summary = super().get_transformation_summary()
        summary["module_version"] = self.VERSION
        summary["changelog"] = self.CHANGELOG
        return summary
```

This user guide provides comprehensive information for using and extending the data preparation framework. For additional technical details, refer to the API Reference documentation.