# CMB Raw Parameter Processing - Troubleshooting Guide

## Common Issues and Solutions

### 1. Parameter Detection Issues

#### Issue: "No raw parameters detected"

**Symptoms:**
- Processing falls back to legacy mode unexpectedly
- Warning: "Raw parameter processing failed, falling back to legacy"

**Causes and Solutions:**

**A. Missing dataset_type in metadata**
```python
# ❌ Incorrect - missing dataset_type
registry_entry = {
    "metadata": {
        "description": "Planck parameters"
    }
}

# ✅ Correct - include dataset_type
registry_entry = {
    "metadata": {
        "dataset_type": "cmb",  # Required for CMB detection
        "description": "Planck parameters"
    }
}
```

**B. Parameter files not properly referenced**
```python
# ❌ Incorrect - no parameter file indicators
registry_entry = {
    "metadata": {"dataset_type": "cmb"},
    "sources": {
        "data": {
            "url": "some_file.txt"  # Generic name, no parameter indicators
        }
    }
}

# ✅ Correct - clear parameter file reference
registry_entry = {
    "metadata": {"dataset_type": "cmb"},
    "sources": {
        "planck_params": {  # Name indicates parameters
            "url": "planck2018_cosmological_parameters.csv"
        }
    }
}
```

**C. File format not recognized**
```python
# Check file format detection
from pipelines.data_preparation.derivation.cmb_derivation import classify_parameter_format

file_path = "my_parameters.dat"
format_type = classify_parameter_format(file_path)
print(f"Detected format: {format_type}")

# If format is UNKNOWN, rename file with clear extension
# .csv, .json, .npy, .txt are supported
```

#### Issue: "Parameter file format not supported"

**Solution:** Convert to supported format

```python
# Convert text file to CSV
import pandas as pd

# Read text file manually
params = {}
with open('params.txt', 'r') as f:
    for line in f:
        if '=' in line:
            key, value = line.strip().split('=')
            params[key.strip()] = float(value.strip())

# Save as CSV
df = pd.DataFrame([params])
df.to_csv('params.csv', index=False)
```

### 2. Parameter Parsing Issues

#### Issue: "Parameter name not recognized"

**Symptoms:**
- ParameterValidationError: "Missing required parameters"
- Parameters exist in file but not detected

**Solution:** Check parameter name variations

```python
# Debug parameter name mapping
from pipelines.data_preparation.derivation.cmb_derivation import normalize_parameter_names

raw_params = {
    "hubble_constant": 67.36,  # Non-standard name
    "matter_density": 0.3153,  # Non-standard name
    "baryon_density_h2": 0.02237
}

normalized = normalize_parameter_names(raw_params)
print(f"Normalized parameters: {normalized}")

# If parameters still not recognized, add custom aliases
CUSTOM_ALIASES = {
    'H0': ['H0', 'h0', 'hubble', 'hubble_constant', 'H_0'],
    'Omega_m': ['Omega_m', 'Om0', 'omega_m', 'matter_density', 'Omega_matter']
}
```

#### Issue: "Invalid parameter values"

**Symptoms:**
- ParameterValidationError with specific parameter bounds
- Parameters outside physical ranges

**Solution:** Check and correct parameter values

```python
# Check parameter bounds
PARAMETER_BOUNDS = {
    'H0': (50.0, 80.0),           # km/s/Mpc
    'Omega_m': (0.1, 0.5),        # Matter density
    'Omega_b_h2': (0.01, 0.05),   # Baryon density
    'n_s': (0.9, 1.1),            # Spectral index
    'tau': (0.01, 0.15),          # Optical depth
    'A_s': (1e-10, 5e-9)          # Scalar amplitude
}

# Validate your parameters
params = {"H0": 167.36}  # Invalid - too high
if not (50.0 <= params["H0"] <= 80.0):
    print(f"H0 = {params['H0']} is outside valid range (50-80)")
```

### 3. Distance Prior Derivation Issues

#### Issue: "Background integrator not available"

**Symptoms:**
- DerivationError: "PBUF background integrator not found"
- Import errors for cmb_background module

**Solution:** Check PBUF integration

```python
# Test background integrator availability
try:
    from pipelines.data_preparation.derivation.cmb_background import (
        BackgroundIntegrator, compute_sound_horizon
    )
    print("✅ Background integrator available")
except ImportError as e:
    print(f"❌ Background integrator not available: {e}")
    
    # Fallback: use legacy distance prior mode
    from pipelines.data_preparation.derivation.cmb_models import CMBConfig
    config = CMBConfig(use_raw_parameters=False)
```

#### Issue: "Numerical integration failed"

**Symptoms:**
- DerivationError during distance prior computation
- NaN or infinite values in results

**Solution:** Check parameter values and numerical stability

```python
from pipelines.data_preparation.derivation.cmb_derivation import check_numerical_stability

params = ParameterSet(H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, 
                     n_s=0.9649, tau=0.0544)

# Check for numerical issues
stability = check_numerical_stability(params)
if not stability['stable']:
    print(f"Numerical issues: {stability['issues']}")
    
    # Try with different parameter values or integration settings
    config = CMBConfig(jacobian_step_size=1e-5)  # Larger step size
```

### 4. Covariance Matrix Issues

#### Issue: "Covariance matrix not positive definite"

**Symptoms:**
- CovarianceError: "Matrix is not positive definite"
- Negative eigenvalues in covariance matrix

**Solution:** Validate and fix covariance matrix

```python
import numpy as np
from pipelines.data_preparation.derivation.cmb_derivation import validate_covariance_properties

# Load and check covariance matrix
cov_matrix = np.loadtxt('covariance.txt')

validation = validate_covariance_properties(cov_matrix)
if not validation['positive_definite']:
    print("Matrix is not positive definite")
    
    # Fix by adding small diagonal regularization
    regularized_cov = cov_matrix + 1e-10 * np.eye(cov_matrix.shape[0])
    
    # Or use diagonal approximation
    diagonal_cov = np.diag(np.diag(cov_matrix))
```

#### Issue: "Covariance matrix dimension mismatch"

**Symptoms:**
- CovarianceError: "Matrix size doesn't match parameter count"
- Shape errors during covariance propagation

**Solution:** Check matrix dimensions

```python
# Check covariance matrix dimensions
param_count = 5  # H0, Omega_m, Omega_b_h2, n_s, tau
cov_shape = cov_matrix.shape

if cov_shape != (param_count, param_count):
    print(f"Dimension mismatch: expected ({param_count}, {param_count}), got {cov_shape}")
    
    # Extract relevant submatrix if larger
    if cov_shape[0] > param_count:
        cov_matrix = cov_matrix[:param_count, :param_count]
```

### 5. Configuration Issues

#### Issue: "Configuration not loaded"

**Symptoms:**
- Default configuration used instead of custom settings
- Environment variables ignored

**Solution:** Check configuration loading

```python
# Debug configuration loading
from pipelines.data_preparation.core.cmb_config_integration import get_cmb_config
import os

# Check environment variables
print(f"PBUF_CMB_USE_RAW_PARAMETERS: {os.getenv('PBUF_CMB_USE_RAW_PARAMETERS')}")
print(f"PBUF_CMB_Z_RECOMBINATION: {os.getenv('PBUF_CMB_Z_RECOMBINATION')}")

# Check loaded configuration
config = get_cmb_config()
print(f"Loaded config: {config}")

# Explicitly set configuration if needed
from pipelines.data_preparation.derivation.cmb_models import CMBConfig
custom_config = CMBConfig(use_raw_parameters=True, z_recombination=1090.0)
```

### 6. Performance Issues

#### Issue: "Processing too slow"

**Symptoms:**
- Long processing times (> 30 seconds for typical datasets)
- High CPU usage during Jacobian computation

**Solutions:**

**A. Optimize Jacobian computation**
```python
# Use larger step size for faster computation (less accurate)
config = CMBConfig(jacobian_step_size=1e-4)  # Default: 1e-6

# Enable caching for repeated computations
config = CMBConfig(cache_computations=True)
```

**B. Skip covariance propagation if not needed**
```python
# Process without covariance matrix for faster results
# Remove covariance file reference from registry entry
registry_entry_fast = {
    "metadata": {"dataset_type": "cmb"},
    "sources": {
        "parameters": {"url": "params.csv"}
        # No covariance source
    }
}
```

#### Issue: "Memory usage too high"

**Symptoms:**
- Out of memory errors
- System slowdown during processing

**Solution:** Optimize memory usage

```python
# Use smaller datasets for testing
# Process parameters in batches if handling multiple datasets
# Clear caches periodically

import gc
from pipelines.data_preparation.derivation.cmb_derivation import clear_computation_cache

# Clear caches to free memory
clear_computation_cache()
gc.collect()
```

### 7. Integration Issues

#### Issue: "Registry integration not working"

**Symptoms:**
- Registry entries not found
- File download failures

**Solution:** Check registry integration

```python
# Test registry integration
from pipelines.data_preparation.core.registry_integration import RegistryIntegration

try:
    registry = RegistryIntegration()
    dataset_info = registry.get_verified_dataset("planck2018_base_lcdm")
    print(f"✅ Registry integration working: {dataset_info['name']}")
except Exception as e:
    print(f"❌ Registry integration failed: {e}")
    
    # Use direct file processing as fallback
    dataset = process_cmb_dataset(registry_entry)
```

#### Issue: "Fitting pipeline compatibility"

**Symptoms:**
- Errors when using processed dataset in fit_cmb.py
- Data format mismatches

**Solution:** Validate output format

```python
# Check StandardDataset format
print(f"Dataset format:")
print(f"  z shape: {dataset.z.shape}")
print(f"  observable shape: {dataset.observable.shape}")
print(f"  uncertainty shape: {dataset.uncertainty.shape}")
print(f"  covariance shape: {dataset.covariance.shape if dataset.covariance is not None else None}")

# Ensure compatibility with existing fit code
expected_format = {
    "z": "1D array with recombination redshift",
    "observable": "1D array with [R, l_A, Omega_b_h2, theta_star]",
    "uncertainty": "1D array with uncertainties",
    "covariance": "2D array (4x4) or None"
}
```

### 8. Debugging Workflow

#### Step 1: Enable Debug Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('data_preparation.cmb')
logger.setLevel(logging.DEBUG)

# Add file handler for persistent logs
handler = logging.FileHandler('cmb_debug.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
```

#### Step 2: Test Individual Components

```python
# Test parameter detection
raw_params = detect_raw_parameters(registry_entry)
print(f"Parameter detection result: {raw_params}")

# Test parameter parsing
if raw_params:
    params = parse_parameter_file(raw_params.file_path, raw_params.format_type)
    print(f"Parsed parameters: {params}")

# Test distance prior derivation
if params:
    priors = compute_distance_priors(params)
    print(f"Derived priors: {priors}")
```

#### Step 3: Validate Intermediate Results

```python
# Check parameter values
print(f"Parameter validation:")
for param, value in params.to_dict().items():
    bounds = PARAMETER_BOUNDS.get(param, (None, None))
    if bounds[0] is not None and bounds[1] is not None:
        valid = bounds[0] <= value <= bounds[1]
        print(f"  {param}: {value} {'✅' if valid else '❌'} (range: {bounds})")

# Check derived priors
print(f"Derived prior validation:")
print(f"  R: {priors.R:.4f} (typical range: 1.7-1.8)")
print(f"  l_A: {priors.l_A:.1f} (typical range: 300-305)")
print(f"  theta_star: {priors.theta_star:.5f} (typical range: 1.040-1.042)")
```

### 9. Error Recovery Strategies

#### Automatic Fallback

```python
def robust_cmb_processing(registry_entry, max_retries=3):
    """Process CMB dataset with automatic error recovery."""
    
    # Try raw parameter processing first
    try:
        config = CMBConfig(use_raw_parameters=True)
        return process_cmb_dataset(registry_entry, config)
    except ParameterDetectionError:
        print("Raw parameter processing failed, trying legacy mode...")
        
    # Fallback to legacy distance prior mode
    try:
        config = CMBConfig(use_raw_parameters=False)
        return process_cmb_dataset(registry_entry, config)
    except Exception as e:
        print(f"Legacy processing also failed: {e}")
        raise

# Use robust processing
dataset = robust_cmb_processing(registry_entry)
```

#### Graceful Degradation

```python
def process_with_degradation(registry_entry):
    """Process with graceful degradation of features."""
    
    try:
        # Try full processing with covariance
        return process_cmb_dataset(registry_entry)
    except CovarianceError:
        print("Covariance processing failed, using diagonal uncertainties...")
        
        # Remove covariance file reference and retry
        modified_entry = registry_entry.copy()
        if 'covariance' in modified_entry.get('sources', {}):
            del modified_entry['sources']['covariance']
        
        return process_cmb_dataset(modified_entry)

dataset = process_with_degradation(registry_entry)
```

### 10. Getting Help

#### Check System Status

```python
# System health check
from pipelines.data_preparation.derivation.cmb_derivation import system_health_check

health = system_health_check()
print(f"System health: {health}")

# Check dependencies
dependencies = [
    'numpy', 'scipy', 'pandas', 
    'pipelines.data_preparation.derivation.cmb_background'
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f"✅ {dep}")
    except ImportError:
        print(f"❌ {dep} - not available")
```

#### Collect Diagnostic Information

```python
def collect_diagnostics(registry_entry, error=None):
    """Collect diagnostic information for support."""
    
    import sys
    import platform
    
    diagnostics = {
        "system_info": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__
        },
        "registry_entry": registry_entry,
        "error_info": str(error) if error else None,
        "configuration": get_cmb_config().to_dict()
    }
    
    # Save diagnostics
    import json
    with open('cmb_diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    
    print("Diagnostics saved to cmb_diagnostics.json")
    return diagnostics

# Use when reporting issues
try:
    dataset = process_cmb_dataset(registry_entry)
except Exception as e:
    diagnostics = collect_diagnostics(registry_entry, e)
```

This troubleshooting guide covers the most common issues encountered when using the CMB raw parameter processing capabilities and provides practical solutions for each scenario.