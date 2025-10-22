# PBUF Cosmology Pipeline - Troubleshooting Guide

This guide provides solutions to common issues and frequently asked questions when using the PBUF cosmology pipeline.

## Common Issues and Solutions

### 1. Parameter and Model Issues

#### Issue: "ValueError: Model 'xyz' not found"
```
ValueError: Model 'xyz' not found in DEFAULTS
```

**Solution:**
```python
# Check available models
from pipelines.fit_core.parameter import DEFAULTS
print("Available models:", list(DEFAULTS.keys()))

# Add new model to DEFAULTS if needed
DEFAULTS["xyz"] = {
    "H0": 67.4,
    "Om0": 0.315,
    # ... other parameters
}
```

#### Issue: "KeyError: Parameter 'param_name' not found"
```
KeyError: Parameter 'param_name' not found for model 'pbuf'
```

**Solution:**
```python
# Check required parameters for model
params = build_params("pbuf")
print("Available parameters:", list(params.keys()))

# Add missing parameter to model defaults
DEFAULTS["pbuf"]["param_name"] = default_value
```

#### Issue: Parameter values outside physical bounds
```
ValueError: Parameter H0=120.0 outside physical bounds [40, 100]
```

**Solution:**
```python
# Check parameter bounds
from pipelines.fit_core.parameter import validate_params

params = {"H0": 120.0, "Om0": 0.3}
try:
    validate_params(params)
except ValueError as e:
    print(f"Validation error: {e}")
    # Adjust parameters to valid range
    params["H0"] = 70.0  # Within bounds
```

### 2. Dataset Issues

#### Issue: "KeyError: Dataset 'dataset_name' not found"
```
KeyError: Dataset 'dataset_name' not found in DATASET_LOADERS
```

**Solution:**
```python
# Check available datasets
from pipelines.fit_core.datasets import DATASET_LOADERS
print("Available datasets:", list(DATASET_LOADERS.keys()))

# Add new dataset loader if needed
def load_custom_dataset():
    return {"observations": data, "covariance": cov}

DATASET_LOADERS["dataset_name"] = load_custom_dataset
```

#### Issue: Covariance matrix not positive definite
```
LinAlgError: Covariance matrix is not positive definite
```

**Solution:**
```python
import numpy as np
from scipy.linalg import eigvals

# Check covariance matrix
cov_matrix = data["covariance"]
eigenvalues = eigvals(cov_matrix)

if np.any(eigenvalues <= 0):
    print("Negative eigenvalues found:", eigenvalues[eigenvalues <= 0])
    
    # Fix by adding small diagonal term
    regularization = 1e-10
    cov_matrix += regularization * np.eye(len(cov_matrix))
    
    # Or use Moore-Penrose pseudoinverse
    from scipy.linalg import pinv
    cov_inv = pinv(cov_matrix)
```

#### Issue: Dataset format mismatch
```
ValueError: Expected observations to be array-like, got dict
```

**Solution:**
```python
# Check dataset format
data = load_dataset("cmb")
print("Data structure:")
for key, value in data.items():
    print(f"  {key}: {type(value)}")

# Ensure correct format
expected_format = {
    "observations": np.ndarray,  # or dict with named observations
    "covariance": np.ndarray,
    "metadata": dict
}
```

### 3. Optimization Issues

#### Issue: Optimization fails to converge
```
OptimizationWarning: Optimization terminated unsuccessfully
```

**Solutions:**

**A. Adjust optimization parameters:**
```python
optimizer_config = {
    "method": "minimize",
    "options": {
        "maxiter": 2000,      # Increase iterations
        "ftol": 1e-8,         # Tighter tolerance
        "gtol": 1e-8,
        "method": "L-BFGS-B"  # Try different method
    }
}
```

**B. Use global optimization:**
```python
optimizer_config = {
    "method": "differential_evolution",
    "options": {
        "maxiter": 1000,
        "seed": 42,           # For reproducibility
        "popsize": 20         # Larger population
    },
    "bounds": {               # Provide bounds
        "H0": [60, 80],
        "Om0": [0.2, 0.4]
    }
}
```

**C. Check initial parameters:**
```python
# Ensure reasonable starting point
params = build_params("pbuf")
print("Initial parameters:", params)

# Adjust if needed
overrides = {"H0": 67.4, "Om0": 0.315}  # Known good values
```

#### Issue: "Degrees of freedom cannot be negative"
```
ValueError: Degrees of freedom cannot be negative: got -5
```

**Solution:**
```python
# Check data points vs parameters
from pipelines.fit_core.statistics import compute_dof

datasets = ["cmb", "bao", "sn"]
n_params = len(build_params("pbuf"))
dof = compute_dof(datasets, n_params)

print(f"Data points: {sum(get_dataset_info(d)['n_points'] for d in datasets)}")
print(f"Parameters: {n_params}")
print(f"DOF: {dof}")

# Solutions:
# 1. Use fewer parameters (fix some parameters)
# 2. Use more datasets
# 3. Reduce model complexity
```

### 4. Physics Consistency Issues

#### Issue: H(z) ratio warnings
```
Warning: H(z) ratios exceed tolerance at z=1.0: ratio=1.05e-3 > 1e-4
```

**Solution:**
```python
# Check PBUF parameters
params = build_params("pbuf")
print(f"k_sat = {params['k_sat']}")

# For ΛCDM limit, k_sat should be close to 1
if abs(params['k_sat'] - 1.0) > 0.1:
    print("k_sat far from ΛCDM limit")
    
# Adjust tolerance if physically reasonable
from pipelines.fit_core.integrity import verify_h_ratios
is_consistent = verify_h_ratios(params, tolerance=1e-3)
```

#### Issue: Recombination redshift mismatch
```
Warning: Recombination redshift z*=1095.2 differs from reference 1089.8
```

**Solution:**
```python
# Check recombination method
params = build_params("lcdm")
print(f"Recombination method: {params.get('recomb_method', 'default')}")

# Try different method
params["recomb_method"] = "PLANCK18"  # or "HS96", "EH98"

# Check if difference is within acceptable range
z_star_computed = 1095.2
z_star_reference = 1089.8
relative_diff = abs(z_star_computed - z_star_reference) / z_star_reference

if relative_diff < 0.01:  # 1% tolerance
    print("Difference within acceptable range")
```

### 5. Performance Issues

#### Issue: Slow optimization
```
# Optimization taking too long
```

**Solutions:**

**A. Use faster optimizer:**
```python
# Switch to L-BFGS-B for local optimization
optimizer_config = {
    "method": "minimize",
    "options": {
        "method": "L-BFGS-B",
        "maxiter": 1000
    }
}
```

**B. Reduce precision requirements:**
```python
optimizer_config = {
    "method": "minimize",
    "options": {
        "ftol": 1e-4,    # Less strict tolerance
        "gtol": 1e-4
    }
}
```

**C. Use caching:**
```python
# Enable likelihood caching (if implemented)
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_likelihood(param_hash):
    # Cached likelihood computation
    pass
```

#### Issue: Memory usage too high
```
MemoryError: Unable to allocate array
```

**Solutions:**

**A. Process datasets individually:**
```python
# Instead of joint fitting, fit individually
results = {}
for dataset in ["cmb", "bao", "sn"]:
    results[dataset] = run_fit("pbuf", [dataset])
```

**B. Reduce data precision:**
```python
# Use float32 instead of float64 if appropriate
data["covariance"] = data["covariance"].astype(np.float32)
```

### 6. Configuration Issues

#### Issue: Configuration file not loading
```
FileNotFoundError: Configuration file 'config.json' not found
```

**Solution:**
```python
import json
from pathlib import Path

config_file = "config.json"
if not Path(config_file).exists():
    print(f"Configuration file {config_file} not found")
    
    # Create default configuration
    default_config = {
        "model": "pbuf",
        "datasets": ["cmb", "bao"],
        "parameters": {"H0": 67.4, "Om0": 0.315}
    }
    
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Created default configuration: {config_file}")
```

#### Issue: Invalid JSON in configuration
```
JSONDecodeError: Expecting ',' delimiter: line 5 column 10
```

**Solution:**
```python
import json

try:
    with open("config.json") as f:
        config = json.load(f)
except json.JSONDecodeError as e:
    print(f"JSON error: {e}")
    print(f"Line {e.lineno}, Column {e.colno}")
    
    # Common fixes:
    # 1. Add missing commas
    # 2. Remove trailing commas
    # 3. Use double quotes for strings
    # 4. Escape backslashes in paths
```

## Debugging Techniques

### 1. Enable Debug Logging

```python
import logging

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run with debug output
result = run_fit("pbuf", ["cmb"])
```

### 2. Step-by-Step Debugging

```python
# Debug parameter building
try:
    params = build_params("pbuf")
    print("✓ Parameters built successfully")
except Exception as e:
    print(f"❌ Parameter building failed: {e}")
    return

# Debug dataset loading
try:
    data = load_dataset("cmb")
    print("✓ Dataset loaded successfully")
except Exception as e:
    print(f"❌ Dataset loading failed: {e}")
    return

# Debug likelihood computation
try:
    chi2, predictions = likelihood_cmb(params, data)
    print(f"✓ Likelihood computed: χ² = {chi2:.3f}")
except Exception as e:
    print(f"❌ Likelihood computation failed: {e}")
    return
```

### 3. Validate Intermediate Results

```python
# Check parameter values
params = build_params("pbuf")
for key, value in params.items():
    if isinstance(value, (int, float)):
        if not np.isfinite(value):
            print(f"Warning: {key} = {value} (not finite)")
        if value < 0 and key in ["H0", "Om0", "alpha"]:
            print(f"Warning: {key} = {value} (negative)")

# Check data integrity
data = load_dataset("cmb")
obs = data["observations"]
cov = data["covariance"]

print(f"Observations shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
print(f"Covariance shape: {cov.shape}")
print(f"Covariance condition number: {np.linalg.cond(cov):.2e}")
```

## Frequently Asked Questions

### Q1: How do I add a new cosmological model?

**A:** Add the model parameters to `DEFAULTS` in `parameter.py`:

```python
DEFAULTS["new_model"] = {
    "H0": 67.4,
    "Om0": 0.315,
    "new_param": 1.0,
    "model_class": "new_model"
}
```

The model is immediately available through the unified interface.

### Q2: Can I use custom datasets?

**A:** Yes, implement a loader function and register it:

```python
def load_my_dataset():
    return {
        "observations": my_data,
        "covariance": my_cov,
        "metadata": {"info": "value"}
    }

DATASET_LOADERS["my_dataset"] = load_my_dataset
```

### Q3: How do I ensure reproducible results?

**A:** Set random seeds and use consistent parameters:

```python
optimizer_config = {
    "method": "differential_evolution",
    "options": {"seed": 42}
}

# Or for numpy-based operations
np.random.seed(42)
```

### Q4: What's the difference between individual and joint fitting?

**A:** 
- **Individual fitting**: Fits each dataset separately
- **Joint fitting**: Fits all datasets simultaneously, sharing parameters

```python
# Individual fitting
cmb_result = run_fit("pbuf", ["cmb"])
bao_result = run_fit("pbuf", ["bao"])

# Joint fitting
joint_result = run_fit("pbuf", ["cmb", "bao"])
```

### Q5: How do I interpret the fit statistics?

**A:**
- **χ²**: Goodness of fit (lower is better)
- **AIC**: Akaike Information Criterion (lower is better, penalizes parameters)
- **BIC**: Bayesian Information Criterion (lower is better, stronger parameter penalty)
- **DOF**: Degrees of freedom (data points - parameters)
- **p-value**: Probability of observing χ² by chance

### Q6: Can I run fits in parallel?

**A:** Yes, for independent fits:

```python
from multiprocessing import Pool

def fit_single_dataset(dataset):
    return run_fit("pbuf", [dataset])

datasets = ["cmb", "bao", "sn"]
with Pool() as pool:
    results = pool.map(fit_single_dataset, datasets)
```

### Q7: How do I handle convergence failures?

**A:** Try these approaches:
1. Use global optimization (differential evolution)
2. Provide parameter bounds
3. Adjust initial parameters
4. Increase iteration limits
5. Use different optimization methods

### Q8: What should I do if physics consistency checks fail?

**A:** 
1. Check parameter values for physical reasonableness
2. Verify model implementation against documentation
3. Adjust tolerance if differences are small and understood
4. Investigate potential bugs in physics modules

## Getting Help

### 1. Check Documentation
- **API Reference**: `pipelines/fit_core/API_REFERENCE.md`
- **Usage Examples**: `pipelines/fit_core/USAGE_EXAMPLES.md`
- **Developer Guide**: `pipelines/fit_core/DEVELOPER_GUIDE.md`

### 2. Run Diagnostic Tests
```python
# Run integrity suite
from pipelines.fit_core.integrity import run_integrity_suite

params = build_params("pbuf")
results = run_integrity_suite(params, ["cmb", "bao", "sn"])
print("Integrity results:", results)
```

### 3. Enable Verbose Output
```python
# Use debug mode for detailed output
result = run_fit("pbuf", ["cmb"], mode="debug")
```

### 4. Check System Status
```python
# Run system validation
python pipelines/fit_core/test_end_to_end_integration.py
```

This troubleshooting guide covers the most common issues encountered when using the PBUF cosmology pipeline. For additional help, consult the comprehensive documentation or run the built-in diagnostic tools.