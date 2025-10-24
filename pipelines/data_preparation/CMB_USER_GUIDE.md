# CMB Raw Parameter Processing - User Guide

## Introduction

This guide provides step-by-step instructions for processing Planck-style cosmological parameter datasets using the enhanced CMB derivation module. The module can automatically detect and process raw parameter files, computing distance priors internally while maintaining full backward compatibility.

## Quick Start

### Basic Usage

```python
from pipelines.data_preparation.derivation.cmb_derivation import process_cmb_dataset
from pipelines.data_preparation.derivation.cmb_models import CMBConfig

# Registry entry with raw parameter files
registry_entry = {
    "metadata": {
        "dataset_type": "cmb",
        "description": "Planck 2018 base ΛCDM parameters",
        "citation": "Planck Collaboration 2020"
    },
    "sources": {
        "planck_params": {
            "url": "https://pla.esac.esa.int/pla/aio/planck2018_base_plikHM_TTTEEE_lowl_lowE.csv",
            "extraction": {
                "target_files": ["base_plikHM_TTTEEE_lowl_lowE_post_lensing.csv"]
            }
        }
    }
}

# Process with default configuration
dataset = process_cmb_dataset(registry_entry)

print(f"Processed CMB dataset:")
print(f"  Redshift: {dataset.z[0]:.1f}")
print(f"  Distance priors: R={dataset.observable[0]:.4f}, l_A={dataset.observable[1]:.2f}")
print(f"  Processing method: {dataset.metadata['processing']}")
```

### Custom Configuration

```python
# Configure processing options
config = CMBConfig(
    use_raw_parameters=True,
    z_recombination=1090.0,      # Custom recombination redshift
    jacobian_step_size=1e-5,     # Larger step for faster computation
    validation_tolerance=1e-7,    # Stricter covariance validation
    cache_computations=True       # Enable caching for repeated runs
)

# Process with custom configuration
dataset = process_cmb_dataset(registry_entry, config)
```

## Supported Input Formats

### 1. CSV Format

**Simple parameter file (planck_params.csv):**
```csv
H0,Omega_m,Omega_b_h2,n_s,tau,A_s
67.36,0.3153,0.02237,0.9649,0.0544,2.1e-9
```

**With parameter names in first column:**
```csv
parameter,value
H0,67.36
Omega_m,0.3153
Omega_b_h2,0.02237
n_s,0.9649
tau,0.0544
A_s,2.1e-9
```

**MCMC chain format (multiple rows):**
```csv
H0,Omega_m,Omega_b_h2,n_s,tau
67.32,0.3151,0.02235,0.9647,0.0542
67.38,0.3155,0.02239,0.9651,0.0546
67.34,0.3152,0.02236,0.9648,0.0543
# ... more samples (mean will be computed)
```

### 2. JSON Format

**Simple parameter dictionary:**
```json
{
  "H0": 67.36,
  "Omega_m": 0.3153,
  "Omega_b_h2": 0.02237,
  "n_s": 0.9649,
  "tau": 0.0544,
  "A_s": 2.1e-9
}
```

**Nested structure:**
```json
{
  "cosmological_parameters": {
    "H0": 67.36,
    "Omega_m": 0.3153,
    "Omega_b_h2": 0.02237,
    "n_s": 0.9649,
    "tau": 0.0544
  },
  "covariance_matrix": [
    [0.54, 0.12, 0.03, 0.01, 0.02],
    [0.12, 0.0025, 0.0008, 0.0003, 0.0001],
    [0.03, 0.0008, 0.000015, 0.000002, 0.000001],
    [0.01, 0.0003, 0.000002, 0.000042, 0.000005],
    [0.02, 0.0001, 0.000001, 0.000005, 0.000081]
  ]
}
```

### 3. NumPy Format

**Single parameter array (.npy):**
```python
import numpy as np

# Save parameters in standard order: [H0, Omega_m, Omega_b_h2, n_s, tau, A_s]
params = np.array([67.36, 0.3153, 0.02237, 0.9649, 0.0544, 2.1e-9])
np.save('planck_params.npy', params)
```

**Parameter dictionary (.npz):**
```python
import numpy as np

# Save as named arrays
np.savez('planck_params.npz',
         H0=67.36,
         Omega_m=0.3153,
         Omega_b_h2=0.02237,
         n_s=0.9649,
         tau=0.0544,
         A_s=2.1e-9)
```

### 4. Text Format

**Key=value format:**
```
# Planck 2018 base ΛCDM parameters
H0 = 67.36
Omega_m = 0.3153
Omega_b_h2 = 0.02237
n_s = 0.9649
tau = 0.0544
A_s = 2.1e-9
```

**Tab-separated format:**
```
H0      67.36
Omega_m 0.3153
Omega_b_h2      0.02237
n_s     0.9649
tau     0.0544
```

## Registry Entry Examples

### Example 1: Simple Parameter File

```json
{
  "name": "planck2018_base_lcdm",
  "metadata": {
    "dataset_type": "cmb",
    "description": "Planck 2018 base ΛCDM cosmological parameters",
    "citation": "Planck Collaboration VI. Cosmological parameters. A&A 641, A6 (2020)",
    "version": "1.0",
    "z_recombination": 1089.8
  },
  "sources": {
    "parameters": {
      "url": "https://pla.esac.esa.int/pla/aio/planck2018_base_plikHM_TTTEEE_lowl_lowE.csv",
      "extraction": {
        "target_files": ["base_plikHM_TTTEEE_lowl_lowE_post_lensing.csv"],
        "format": "csv"
      }
    }
  }
}
```

### Example 2: Parameters with Covariance Matrix

```json
{
  "name": "planck2018_base_lcdm_with_covariance",
  "metadata": {
    "dataset_type": "cmb",
    "description": "Planck 2018 base ΛCDM with full parameter covariance",
    "citation": "Planck Collaboration VI. Cosmological parameters. A&A 641, A6 (2020)",
    "version": "1.0"
  },
  "sources": {
    "parameters": {
      "url": "planck2018_params.json",
      "extraction": {
        "target_files": ["planck2018_params.json"],
        "format": "json"
      }
    },
    "covariance": {
      "url": "planck2018_covariance.csv",
      "extraction": {
        "target_files": ["planck2018_covariance.csv"],
        "format": "csv"
      }
    }
  }
}
```

### Example 3: MCMC Chain File

```json
{
  "name": "planck2018_mcmc_chain",
  "metadata": {
    "dataset_type": "cmb",
    "description": "Planck 2018 MCMC chain samples",
    "citation": "Planck Collaboration VI. Cosmological parameters. A&A 641, A6 (2020)",
    "version": "1.0",
    "parameter_file": "planck2018_chain.csv",
    "covariance_file": "planck2018_chain_cov.csv"
  },
  "sources": {
    "mcmc_chain": {
      "url": "https://pla.esac.esa.int/pla/aio/planck2018_base_plikHM_TTTEEE_lowl_lowE_chain.csv",
      "extraction": {
        "target_files": [
          "planck2018_chain.csv",
          "planck2018_chain_cov.csv"
        ]
      }
    }
  }
}
```

### Example 4: Legacy Distance Priors (Backward Compatibility)

```json
{
  "name": "planck2018_distance_priors",
  "metadata": {
    "dataset_type": "cmb",
    "description": "Planck 2018 pre-computed distance priors",
    "citation": "Planck Collaboration VI. Cosmological parameters. A&A 641, A6 (2020)",
    "version": "1.0",
    "processing_mode": "legacy_distance_priors"
  },
  "sources": {
    "distance_priors": {
      "url": "planck2018_distance_priors.json",
      "extraction": {
        "target_files": ["planck2018_distance_priors.json"]
      }
    }
  }
}
```

## Parameter Name Variations

The module automatically handles various parameter naming conventions:

| Standard Name | Accepted Variations |
|---------------|-------------------|
| `H0` | H0, h0, hubble, H_0, h_0, hubble_constant |
| `Omega_m` | Omega_m, Om0, omega_m, OmegaM, Ωm, omega_matter, Omega_matter |
| `Omega_b_h2` | Omega_b_h2, omegabh2, omega_b_h2, Ωbh², omega_baryon_h2, Omega_baryon_h2 |
| `n_s` | n_s, ns, n_scalar, spectral_index, scalar_spectral_index |
| `tau` | tau, τ, tau_reio, optical_depth, tau_optical |
| `A_s` | A_s, As, A_scalar, scalar_amplitude, amplitude_scalar |

## Configuration Options

### CMBConfig Parameters

```python
@dataclass
class CMBConfig:
    use_raw_parameters: bool = True          # Enable raw parameter processing
    z_recombination: float = 1089.8         # Recombination redshift
    jacobian_step_size: float = 1e-6       # Numerical differentiation step
    validation_tolerance: float = 1e-8      # Covariance validation tolerance
    fallback_to_legacy: bool = True         # Auto-fallback if raw params unavailable
    cache_computations: bool = True         # Cache expensive computations
```

### Environment Variables

```bash
# Enable/disable raw parameter processing
export PBUF_CMB_USE_RAW_PARAMETERS=true

# Set recombination redshift
export PBUF_CMB_Z_RECOMBINATION=1089.8

# Configure numerical differentiation
export PBUF_CMB_JACOBIAN_STEP_SIZE=1e-6

# Enable computation caching
export PBUF_CMB_CACHE_ENABLED=true
```

### Configuration File

Create `cmb_config.json`:
```json
{
  "cmb_processing": {
    "use_raw_parameters": true,
    "z_recombination": 1089.8,
    "jacobian_step_size": 1e-6,
    "validation_tolerance": 1e-8,
    "fallback_to_legacy": true,
    "cache_computations": true
  }
}
```

Load configuration:
```python
from pipelines.data_preparation.core.cmb_config_integration import load_config_from_file

config = load_config_from_file("cmb_config.json")
dataset = process_cmb_dataset(registry_entry, config)
```

## Advanced Usage Examples

### Processing Multiple Datasets

```python
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
from pipelines.data_preparation.derivation.cmb_derivation import CMBDerivationModule

# Initialize framework
framework = DataPreparationFramework()

# Register CMB module
cmb_module = CMBDerivationModule()
framework.register_derivation_module(cmb_module)

# Process multiple datasets
datasets = [
    "planck2018_base_lcdm",
    "planck2018_base_lcdm_lensing",
    "planck2018_base_lcdm_bao"
]

results = {}
for dataset_name in datasets:
    try:
        dataset = framework.prepare_dataset(dataset_name)
        results[dataset_name] = dataset
        print(f"✅ Processed {dataset_name}")
    except Exception as e:
        print(f"❌ Failed to process {dataset_name}: {e}")
```

### Custom Parameter Validation

```python
from pipelines.data_preparation.derivation.cmb_derivation import validate_parameter_ranges
from pipelines.data_preparation.derivation.cmb_models import ParameterSet

# Custom parameter bounds
custom_bounds = {
    'H0': (60.0, 75.0),      # Tighter H0 constraint
    'Omega_m': (0.25, 0.35), # Tighter matter density
    'tau': (0.04, 0.08)      # Tighter optical depth
}

# Validate parameters with custom bounds
params = ParameterSet(H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, 
                     n_s=0.9649, tau=0.0544)

validation_result = validate_parameter_ranges(params, custom_bounds)
if validation_result['valid']:
    print("Parameters pass custom validation")
else:
    print(f"Validation failed: {validation_result['errors']}")
```

### Covariance Matrix Analysis

```python
import numpy as np
from pipelines.data_preparation.derivation.cmb_derivation import (
    compute_jacobian, propagate_covariance, compute_correlation_matrix
)

# Compute Jacobian for sensitivity analysis
params = ParameterSet(H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, 
                     n_s=0.9649, tau=0.0544)
jacobian = compute_jacobian(params, z_recomb=1089.8)

print("Jacobian matrix (∂observables/∂parameters):")
print(f"Shape: {jacobian.shape}")
print(f"∂R/∂H0 = {jacobian[0, 0]:.6f}")
print(f"∂l_A/∂Omega_m = {jacobian[1, 1]:.6f}")

# Analyze parameter covariance propagation
param_cov = np.array([
    [0.54, 0.12, 0.03, 0.01, 0.02],
    [0.12, 0.0025, 0.0008, 0.0003, 0.0001],
    [0.03, 0.0008, 0.000015, 0.000002, 0.000001],
    [0.01, 0.0003, 0.000002, 0.000042, 0.000005],
    [0.02, 0.0001, 0.000001, 0.000005, 0.000081]
])

derived_cov = propagate_covariance(param_cov, jacobian)
correlation_matrix = compute_correlation_matrix(derived_cov)

print(f"Derived covariance matrix shape: {derived_cov.shape}")
print(f"R-l_A correlation: {correlation_matrix[0, 1]:.3f}")
```

### Performance Monitoring

```python
import time
from pipelines.data_preparation.derivation.cmb_derivation import process_cmb_dataset

# Monitor processing performance
start_time = time.time()
dataset = process_cmb_dataset(registry_entry)
processing_time = time.time() - start_time

print(f"Processing completed in {processing_time:.3f} seconds")
print(f"Performance metrics:")
print(f"  - Parameter detection: {dataset.metadata.get('detection_time', 'N/A')}")
print(f"  - Distance derivation: {dataset.metadata.get('derivation_time', 'N/A')}")
print(f"  - Covariance propagation: {dataset.metadata.get('covariance_time', 'N/A')}")
```

## Integration with Fitting Pipelines

### Direct Integration

```python
# The processed dataset works directly with existing fit pipelines
from pipelines.fit_cmb import fit_cmb_data

# Process CMB data
cmb_dataset = process_cmb_dataset(registry_entry)

# Use in fitting pipeline (no changes required)
fit_results = fit_cmb_data(
    z=cmb_dataset.z,
    observables=cmb_dataset.observable,
    uncertainties=cmb_dataset.uncertainty,
    covariance=cmb_dataset.covariance
)
```

### Framework Integration

```python
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework

# Initialize framework with CMB module
framework = DataPreparationFramework()
framework.register_derivation_module(CMBDerivationModule())

# Process dataset through framework
dataset = framework.prepare_dataset("planck2018_base_lcdm")

# Framework automatically handles:
# - Registry integration
# - Validation
# - Error handling
# - Provenance tracking
# - Output formatting
```

## Error Handling and Troubleshooting

### Common Error Scenarios

#### 1. Parameter Detection Failure

```python
from pipelines.data_preparation.derivation.cmb_exceptions import ParameterDetectionError

try:
    dataset = process_cmb_dataset(registry_entry)
except ParameterDetectionError as e:
    print(f"Parameter detection failed: {e.message}")
    print(f"File path: {e.file_path}")
    print(f"Suggested actions:")
    for action in e.suggested_actions:
        print(f"  - {action}")
    
    # Try with legacy mode
    config = CMBConfig(use_raw_parameters=False)
    dataset = process_cmb_dataset(registry_entry, config)
```

#### 2. Parameter Validation Failure

```python
from pipelines.data_preparation.derivation.cmb_exceptions import ParameterValidationError

try:
    dataset = process_cmb_dataset(registry_entry)
except ParameterValidationError as e:
    print(f"Parameter validation failed: {e.message}")
    print(f"Invalid parameters: {e.context.get('invalid_params', [])}")
    print(f"Valid ranges: {e.context.get('valid_ranges', {})}")
```

#### 3. Covariance Propagation Issues

```python
from pipelines.data_preparation.derivation.cmb_exceptions import CovarianceError

try:
    dataset = process_cmb_dataset(registry_entry)
except CovarianceError as e:
    print(f"Covariance propagation failed: {e.message}")
    
    # Process without covariance matrix
    config = CMBConfig(use_raw_parameters=True)
    # This will use diagonal uncertainties instead
    dataset = process_cmb_dataset(registry_entry, config)
```

### Debugging Tips

#### Enable Debug Logging

```python
import logging

# Enable detailed logging
logging.getLogger('data_preparation.cmb').setLevel(logging.DEBUG)

# Process with detailed logs
dataset = process_cmb_dataset(registry_entry)
```

#### Validate Input Data

```python
from pipelines.data_preparation.derivation.cmb_derivation import (
    detect_raw_parameters, validate_parameter_completeness
)

# Check parameter detection
raw_params = detect_raw_parameters(registry_entry)
if raw_params:
    print(f"Detected parameters in: {raw_params.file_path}")
    print(f"Format: {raw_params.format_type}")
else:
    print("No raw parameters detected - will use legacy mode")

# Validate parameter completeness
if raw_params:
    params = parse_parameter_file(raw_params.file_path, raw_params.format_type)
    validation = validate_parameter_completeness(params.to_dict())
    print(f"Required parameters found: {validation['found']}")
    print(f"Missing parameters: {validation['missing']}")
```

#### Check Numerical Stability

```python
from pipelines.data_preparation.derivation.cmb_derivation import check_numerical_stability

# Validate numerical properties
params = ParameterSet(H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, 
                     n_s=0.9649, tau=0.0544)
stability_check = check_numerical_stability(params)

if not stability_check['stable']:
    print(f"Numerical issues detected: {stability_check['issues']}")
```

## Best Practices

### 1. Parameter File Organization

- Use descriptive file names: `planck2018_base_lcdm_params.csv`
- Include metadata in file headers or companion files
- Store covariance matrices in matching format and location
- Use consistent parameter naming conventions

### 2. Registry Entry Design

- Always specify `dataset_type: "cmb"` in metadata
- Include citation and version information
- Reference both parameter and covariance files when available
- Use descriptive dataset names

### 3. Configuration Management

- Use environment variables for deployment settings
- Store configuration files in version control
- Document configuration changes and their impact
- Test configuration changes with known datasets

### 4. Error Handling

- Always wrap processing calls in try-catch blocks
- Log processing steps for debugging
- Implement graceful fallbacks for production systems
- Monitor processing performance and resource usage

### 5. Validation and Testing

- Validate processed results against published values
- Test with various input formats and parameter sets
- Monitor covariance matrix properties (symmetry, positive-definiteness)
- Compare raw parameter and legacy distance prior results

This user guide provides comprehensive instructions for effectively using the CMB raw parameter processing capabilities while maintaining compatibility with existing workflows.