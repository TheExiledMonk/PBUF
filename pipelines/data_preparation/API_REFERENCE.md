# Data Preparation Framework - API Reference

## Overview

The PBUF Data Preparation & Derivation Framework provides a unified interface for transforming raw cosmological datasets into standardized, analysis-ready formats. This document provides comprehensive API documentation for all framework components.

## Core Components

### StandardDataset

The `StandardDataset` class represents the unified format for all processed datasets.

```python
from pipelines.data_preparation.core.schema import StandardDataset

@dataclass
class StandardDataset:
    """Standardized dataset format for all cosmological data types"""
    z: np.ndarray              # Redshift array
    observable: np.ndarray     # Measured quantities (μ, D_M/r_d, H(z), etc.)
    uncertainty: np.ndarray    # One-sigma uncertainties
    covariance: Optional[np.ndarray]  # Full covariance matrix (N×N or None)
    metadata: Dict[str, Any]   # Source, citation, processing info
```

#### Methods

- `validate_schema() -> bool`: Validate dataset schema compliance
- `validate_numerical() -> bool`: Validate numerical integrity
- `validate_covariance() -> bool`: Validate covariance matrix properties
- `to_dict() -> Dict[str, Any]`: Convert to dictionary format
- `save(filepath: Path) -> None`: Save dataset to file

### DataPreparationFramework

The main orchestration engine for the data preparation workflow.

```python
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework

class DataPreparationFramework:
    def __init__(self, registry_manager=None, output_directory: Optional[Path] = None)
```

#### Methods

##### `prepare_dataset(dataset_name: str, raw_data_path: Optional[Path] = None, metadata: Optional[Dict[str, Any]] = None, force_reprocess: bool = False) -> StandardDataset`

Process a dataset through the complete preparation pipeline.

**Parameters:**
- `dataset_name`: Name of dataset to process
- `raw_data_path`: Optional path to raw data (if not using registry)
- `metadata`: Optional metadata (if not using registry)
- `force_reprocess`: Force reprocessing even if cached result exists

**Returns:**
- `StandardDataset`: Processed dataset in standard format

**Raises:**
- `EnhancedProcessingError`: If processing fails at any stage

**Example:**
```python
framework = DataPreparationFramework()
dataset = framework.prepare_dataset(
    dataset_name="sn_pantheon_plus",
    raw_data_path=Path("data/raw/pantheon_plus.csv"),
    metadata={"dataset_type": "sn", "version": "1.0"}
)
```

##### `register_derivation_module(module: DerivationModule) -> None`

Register a derivation module for a specific dataset type.

**Parameters:**
- `module`: DerivationModule instance to register

**Example:**
```python
sn_module = SNDerivationModule()
framework.register_derivation_module(sn_module)
```

##### `get_available_dataset_types() -> List[str]`

Get list of supported dataset types.

**Returns:**
- List of dataset type identifiers

### ValidationEngine

Comprehensive validation system for processed datasets.

```python
from pipelines.data_preparation.core.validation import ValidationEngine

class ValidationEngine:
    def validate_dataset(self, dataset: StandardDataset, dataset_name: str) -> Dict[str, Any]
```

#### Methods

##### `validate_dataset(dataset: StandardDataset, dataset_name: str) -> Dict[str, Any]`

Perform comprehensive validation of a processed dataset.

**Parameters:**
- `dataset`: StandardDataset to validate
- `dataset_name`: Name of dataset for error reporting

**Returns:**
- Dictionary containing validation results with keys:
  - `validation_passed`: Boolean indicating overall validation status
  - `schema_validation`: Schema compliance results
  - `numerical_validation`: Numerical integrity results
  - `covariance_validation`: Covariance matrix validation results
  - `physical_validation`: Physical consistency checks

**Example:**
```python
validation_engine = ValidationEngine()
results = validation_engine.validate_dataset(dataset, "sn_pantheon_plus")
if results['validation_passed']:
    print("Dataset validation passed")
```

## Derivation Modules

### DerivationModule Interface

Abstract base class for all dataset-specific derivation modules.

```python
from pipelines.data_preparation.core.interfaces import DerivationModule

class DerivationModule(ABC):
    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Return the dataset type this module handles"""
        pass
    
    @abstractmethod
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        """Validate raw data before processing"""
        pass
    
    @abstractmethod
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """Transform raw data to standard format"""
        pass
    
    @abstractmethod
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return summary of applied transformations"""
        pass
```

### SNDerivationModule

Supernova dataset processing module.

```python
from pipelines.data_preparation.derivation.sn_derivation import SNDerivationModule

class SNDerivationModule(DerivationModule):
    dataset_type = "sn"
```

#### Supported Input Formats
- CSV files with columns: `CID`, `zHD`, `zHDERR`, `zCMB`, `zCMBERR`, `MU`, `MUERR`
- FITS files with equivalent structure

#### Transformations Applied
1. Duplicate removal by coordinate matching
2. Calibration system homogenization (SALT2, MLCS2k2)
3. Magnitude to distance modulus conversion
4. Systematic covariance matrix application
5. Redshift-distance modulus-uncertainty extraction

#### Output Format
- `z`: Redshift array (zCMB preferred, zHD fallback)
- `observable`: Distance modulus μ(z)
- `uncertainty`: Distance modulus uncertainty σ_μ
- `covariance`: Systematic covariance matrix (if available)

### BAODerivationModule

Baryon Acoustic Oscillation dataset processing module.

```python
from pipelines.data_preparation.derivation.bao_derivation import BAODerivationModule

class BAODerivationModule(DerivationModule):
    dataset_type = "bao"
```

#### Supported Input Formats
- CSV files with columns for isotropic: `z_eff`, `DV_over_rd`, `DV_err`
- CSV files with columns for anisotropic: `z_eff`, `DM_over_rd`, `DM_err`, `DH_over_rd`, `DH_err`, `correlation`

#### Transformations Applied
1. Separation of isotropic (D_V/r_d) and anisotropic (D_M/r_d, D_H/r_d) measurements
2. Distance measure unit conversion to consistent Mpc units
3. Correlation matrix validation and reconstruction
4. Survey-specific systematic corrections

#### Output Format
- `z`: Effective redshift array
- `observable`: Distance ratios (D_V/r_d or [D_M/r_d, D_H/r_d])
- `uncertainty`: Distance ratio uncertainties
- `covariance`: Full correlation/covariance matrix

### CMBDerivationModule

Cosmic Microwave Background dataset processing module.

```python
from pipelines.data_preparation.derivation.cmb_derivation import CMBDerivationModule

class CMBDerivationModule(DerivationModule):
    dataset_type = "cmb"
```

#### Supported Input Formats
- JSON files with CMB distance priors: `R`, `l_A`, `theta_star`
- Planck chain files with compressed parameters

#### Transformations Applied
1. Distance prior extraction from Planck chains
2. Dimensionless consistency validation
3. Covariance matrix application
4. Cosmological constant verification

#### Output Format
- `z`: Recombination redshift (z_* ≈ 1089.8)
- `observable`: [R, l_A, θ_*] compressed parameters
- `uncertainty`: Parameter uncertainties
- `covariance`: Parameter covariance matrix

### CCDerivationModule

Cosmic Chronometers dataset processing module.

```python
from pipelines.data_preparation.derivation.cc_derivation import CCDerivationModule

class CCDerivationModule(DerivationModule):
    dataset_type = "cc"
```

#### Supported Input Formats
- CSV files with columns: `z`, `H_z`, `H_z_err`

#### Transformations Applied
1. Multi-compilation data merging
2. Overlapping redshift bin filtering
3. Systematic uncertainty propagation
4. H(z) sign convention validation

#### Output Format
- `z`: Redshift array
- `observable`: Hubble parameter H(z) in km/s/Mpc
- `uncertainty`: H(z) uncertainties
- `covariance`: Covariance matrix (if available)

### RSDDerivationModule

Redshift Space Distortion dataset processing module.

```python
from pipelines.data_preparation.derivation.rsd_derivation import RSDDerivationModule

class RSDDerivationModule(DerivationModule):
    dataset_type = "rsd"
```

#### Supported Input Formats
- CSV files with columns: `z`, `f_sigma8`, `f_sigma8_err`

#### Transformations Applied
1. Growth rate sign convention validation
2. Covariance homogenization from published sources
3. Survey-specific correction application
4. Error propagation

#### Output Format
- `z`: Redshift array
- `observable`: Growth rate fσ₈(z)
- `uncertainty`: fσ₈ uncertainties
- `covariance`: Covariance matrix

## Error Handling

### ProcessingError

Base exception class for data preparation errors.

```python
from pipelines.data_preparation.core.interfaces import ProcessingError

@dataclass
class ProcessingError(Exception):
    dataset_name: str
    stage: str
    error_type: str
    error_message: str
    context: Dict[str, Any]
    suggested_actions: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
```

### EnhancedProcessingError

Extended error class with additional context and recovery information.

```python
from pipelines.data_preparation.core.error_handling import EnhancedProcessingError

@dataclass
class EnhancedProcessingError(ProcessingError):
    severity: ErrorSeverity
    system_info: Dict[str, Any]
    stack_trace: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    processing_duration: Optional[float] = None
    original_exception: Optional[Exception] = None
```

## Registry Integration

### RegistryIntegration

Interface with the existing PBUF dataset registry system.

```python
from pipelines.data_preparation.core.registry_integration import RegistryIntegration

class RegistryIntegration:
    def __init__(self, registry_manager)
    
    def get_verified_dataset(self, dataset_name: str) -> Dict[str, Any]
    def register_derived_dataset(self, dataset_name: str, derived_dataset: StandardDataset, 
                               source_provenance: Any, transformation_summary: Dict[str, Any], 
                               output_file_path: Path) -> str
```

## Logging and Monitoring

### TransformationLogger

Detailed logging of transformation steps for reproducibility.

```python
from pipelines.data_preparation.core.transformation_logging import TransformationLogger

class TransformationLogger:
    def __init__(self, dataset_name: str, dataset_type: str, output_directory: Path)
    
    def start_processing(self, metadata: Dict[str, Any]) -> None
    def log_transformation_step(self, step_name: str, description: str, 
                              input_data: Any, output_data: Any, 
                              formula: Optional[str] = None) -> None
    def end_processing(self, final_results: Dict[str, Any]) -> None
    def save_processing_summary(self) -> Path
    def save_publication_summary(self) -> Path
```

### SystemHealthMonitor

System resource monitoring during processing.

```python
from pipelines.data_preparation.core.transformation_logging import SystemHealthMonitor

class SystemHealthMonitor:
    def start_monitoring(self) -> None
    def stop_monitoring(self) -> None
    def get_health_summary(self) -> Dict[str, Any]
    def save_health_report(self, filename: str) -> Path
```

## Usage Examples

### Basic Dataset Processing

```python
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
from pipelines.data_preparation.derivation.sn_derivation import SNDerivationModule
from pathlib import Path

# Initialize framework
framework = DataPreparationFramework(output_directory=Path("data/derived"))

# Register derivation modules
sn_module = SNDerivationModule()
framework.register_derivation_module(sn_module)

# Process dataset
dataset = framework.prepare_dataset(
    dataset_name="pantheon_plus",
    raw_data_path=Path("data/raw/pantheon_plus.csv"),
    metadata={
        "dataset_type": "sn",
        "description": "Pantheon+ Supernova Sample",
        "version": "1.0"
    }
)

print(f"Processed {len(dataset.z)} data points")
print(f"Redshift range: {dataset.z.min():.3f} - {dataset.z.max():.3f}")
```

### Validation and Quality Assurance

```python
from pipelines.data_preparation.core.validation import ValidationEngine

# Validate processed dataset
validation_engine = ValidationEngine()
results = validation_engine.validate_dataset(dataset, "pantheon_plus")

if results['validation_passed']:
    print("✅ Dataset validation passed")
    print(f"Schema compliance: {results['schema_validation']['passed']}")
    print(f"Numerical integrity: {results['numerical_validation']['passed']}")
else:
    print("❌ Dataset validation failed")
    for error in results.get('errors', []):
        print(f"  - {error}")
```

### Error Handling

```python
from pipelines.data_preparation.core.interfaces import ProcessingError

try:
    dataset = framework.prepare_dataset("invalid_dataset")
except ProcessingError as e:
    print(f"Processing failed: {e.error_message}")
    print(f"Stage: {e.stage}")
    print(f"Suggested actions:")
    for action in e.suggested_actions:
        print(f"  - {action}")
```

### Integration with Existing Fit Pipelines

```python
# The framework integrates seamlessly with existing fit pipelines
from pipelines.fit_core.datasets import load_dataset

# This will automatically use the preparation framework if available
dataset_dict = load_dataset("pantheon_plus")

# Access data in familiar format
observations = dataset_dict["observations"]
uncertainties = dataset_dict["uncertainties"]
redshifts = dataset_dict["redshifts"]
```

## Configuration

### Environment Variables

- `PBUF_DATA_PREPARATION_OUTPUT_DIR`: Default output directory for derived datasets
- `PBUF_DATA_PREPARATION_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `PBUF_DATA_PREPARATION_CACHE_ENABLED`: Enable/disable processing cache (true/false)

### Configuration Files

The framework can be configured via JSON configuration files:

```json
{
  "output_directory": "data/derived",
  "logging": {
    "level": "INFO",
    "file_logging": true,
    "console_logging": true
  },
  "processing": {
    "cache_enabled": true,
    "parallel_processing": false,
    "max_memory_usage_gb": 8
  },
  "validation": {
    "strict_mode": true,
    "covariance_validation": true,
    "physical_consistency_checks": true
  }
}
```

## Performance Considerations

### Memory Usage

- Large datasets are processed in chunks to manage memory usage
- Covariance matrices are handled efficiently using sparse representations when possible
- Streaming processing is used for datasets that don't fit in memory

### Processing Speed

- Caching prevents reprocessing of identical inputs
- Parallel processing is available for independent datasets (deterministic only)
- Optimized numerical operations using NumPy vectorization

### Scalability

- Modular architecture allows adding new dataset types without performance impact
- Configuration-driven processing parameters adapt to system resources
- Graceful degradation for resource-constrained environments

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes the PBUF root directory
2. **File Not Found**: Check raw data paths and file permissions
3. **Validation Failures**: Review input data format and completeness
4. **Memory Issues**: Reduce dataset size or increase available memory
5. **Processing Timeouts**: Check system resources and optimize processing parameters

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger('data_preparation').setLevel(logging.DEBUG)
```

### Support

For additional support:
1. Check the troubleshooting guide in `TROUBLESHOOTING.md`
2. Review error logs in the `data/logs` directory
3. Consult the user guide in `USER_GUIDE.md`
4. Report issues with detailed error context and system information