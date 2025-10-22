# PBUF Cosmology Pipeline - API Reference

## Core API Functions

### engine.py

#### `run_fit(model, datasets_list, mode="joint", overrides=None, optimizer_config=None)`

Central optimization function used by all fitters.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")
- `datasets_list` (List[str]): List of dataset names to include in fit
- `mode` (str, optional): Fitting mode ("joint" or "individual"). Default: "joint"
- `overrides` (Dict, optional): Parameter overrides. Default: None
- `optimizer_config` (Dict, optional): Optimizer configuration. Default: None

**Returns:**
- `ResultsDict`: Complete results dictionary with parameters, χ² breakdown, and metrics

**Example:**
```python
result = run_fit(
    model="pbuf",
    datasets_list=["cmb", "bao"],
    mode="joint",
    overrides={"H0": 70.0}
)
```

**Raises:**
- `ValueError`: If model is not "lcdm" or "pbuf"
- `ValueError`: If datasets_list is empty
- `KeyError`: If dataset not found

---

### parameter.py

#### `build_params(model, overrides=None)`

Build parameter dictionary for specified model with optional overrides.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")
- `overrides` (Dict, optional): Parameter overrides to apply. Default: None

**Returns:**
- `ParameterDict`: Complete parameter dictionary with all required parameters

**Example:**
```python
params = build_params("lcdm", overrides={"H0": 70.0, "Om0": 0.3})
```

#### `get_defaults(model)`

Get default parameter values for specified model.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")

**Returns:**
- `Dict`: Default parameter dictionary

**Example:**
```python
defaults = get_defaults("pbuf")
print(f"Default alpha: {defaults['alpha']}")
```

#### `validate_params(params)`

Validate parameter dictionary for physical consistency.

**Parameters:**
- `params` (Dict): Parameter dictionary to validate

**Returns:**
- `bool`: True if parameters are valid

**Raises:**
- `ValueError`: If parameters are outside physical bounds

---

### likelihoods.py

#### `likelihood_cmb(params, data)`

Compute CMB likelihood using distance priors.

**Parameters:**
- `params` (ParameterDict): Cosmological parameters
- `data` (Dict): CMB dataset dictionary

**Returns:**
- `Tuple[float, Dict]`: (χ² value, predictions dictionary)

**Example:**
```python
from pipelines.fit_core.datasets import load_dataset

params = build_params("lcdm")
cmb_data = load_dataset("cmb")
chi2, predictions = likelihood_cmb(params, cmb_data)
```

#### `likelihood_bao(params, data)`

Compute isotropic BAO likelihood.

**Parameters:**
- `params` (ParameterDict): Cosmological parameters
- `data` (Dict): BAO dataset dictionary

**Returns:**
- `Tuple[float, Dict]`: (χ² value, predictions dictionary)

#### `likelihood_bao_ani(params, data)`

Compute anisotropic BAO likelihood.

**Parameters:**
- `params` (ParameterDict): Cosmological parameters
- `data` (Dict): Anisotropic BAO dataset dictionary

**Returns:**
- `Tuple[float, Dict]`: (χ² value, predictions dictionary)

#### `likelihood_sn(params, data)`

Compute supernova likelihood.

**Parameters:**
- `params` (ParameterDict): Cosmological parameters
- `data` (Dict): Supernova dataset dictionary

**Returns:**
- `Tuple[float, Dict]`: (χ² value, predictions dictionary)

---

### datasets.py

#### `load_dataset(name)`

Load and validate observational dataset.

**Parameters:**
- `name` (str): Dataset name ("cmb", "bao", "bao_ani", "sn")

**Returns:**
- `Dict`: Dataset dictionary with observations, covariance, and metadata

**Example:**
```python
cmb_data = load_dataset("cmb")
print(f"Data points: {len(cmb_data['observations'])}")
```

#### `validate_dataset(data, expected_format)`

Validate dataset format and properties.

**Parameters:**
- `data` (Dict): Dataset dictionary to validate
- `expected_format` (str): Expected format specification

**Returns:**
- `bool`: True if dataset is valid

#### `get_dataset_info(name)`

Get metadata about dataset.

**Parameters:**
- `name` (str): Dataset name

**Returns:**
- `Dict`: Dataset metadata (redshift range, number of points, etc.)

---

### statistics.py

#### `chi2_generic(predictions, observations, covariance)`

Compute χ² using generic matrix operations.

**Parameters:**
- `predictions` (Dict): Theoretical predictions
- `observations` (Dict): Observational data
- `covariance` (np.ndarray): Covariance matrix

**Returns:**
- `float`: χ² value

**Example:**
```python
chi2 = chi2_generic(
    predictions={"R": 1.75, "l_A": 301.8},
    observations={"R": 1.74, "l_A": 301.9},
    covariance=cov_matrix
)
```

#### `compute_metrics(chi2, n_params, datasets)`

Compute fit statistics and information criteria.

**Parameters:**
- `chi2` (float): Total χ² value
- `n_params` (int): Number of free parameters
- `datasets` (List[str]): List of datasets used

**Returns:**
- `MetricsDict`: Dictionary with AIC, BIC, DOF, p-value

#### `compute_dof(datasets, n_params)`

Compute degrees of freedom.

**Parameters:**
- `datasets` (List[str]): List of datasets
- `n_params` (int): Number of free parameters

**Returns:**
- `int`: Degrees of freedom

#### `delta_aic(aic1, aic2)`

Compute ΔAIC for model comparison.

**Parameters:**
- `aic1` (float): AIC of first model
- `aic2` (float): AIC of second model

**Returns:**
- `float`: ΔAIC value

---

### integrity.py

#### `verify_h_ratios(params, redshifts=None, tolerance=None)`

Verify Hubble parameter consistency.

**Parameters:**
- `params` (ParameterDict): Parameter dictionary
- `redshifts` (List[float], optional): Test redshifts. Default: [0.1, 0.5, 1.0, 2.0]
- `tolerance` (float, optional): Tolerance for comparison. Default: 1e-4

**Returns:**
- `bool`: True if H(z) ratios are consistent

#### `verify_recombination(params, reference=1089.80)`

Verify recombination redshift calculation.

**Parameters:**
- `params` (ParameterDict): Parameter dictionary
- `reference` (float, optional): Reference z* value. Default: 1089.80

**Returns:**
- `bool`: True if recombination is consistent

#### `verify_covariance_matrices(datasets)`

Verify covariance matrix properties.

**Parameters:**
- `datasets` (List[str]): List of datasets to check

**Returns:**
- `bool`: True if all covariance matrices are valid

#### `run_integrity_suite(params, datasets)`

Run comprehensive integrity validation.

**Parameters:**
- `params` (ParameterDict): Parameter dictionary
- `datasets` (List[str]): List of datasets

**Returns:**
- `Dict`: Comprehensive validation results

---

### logging_utils.py

#### `log_run(model, mode, results, metrics)`

Log standardized run information.

**Parameters:**
- `model` (str): Model type
- `mode` (str): Fitting mode
- `results` (Dict): Fit results
- `metrics` (Dict): Fit metrics

**Returns:**
- `None`

#### `log_diagnostics(params, predictions)`

Log physics diagnostics.

**Parameters:**
- `params` (ParameterDict): Parameter dictionary
- `predictions` (Dict): Model predictions

**Returns:**
- `None`

#### `format_results_table(results)`

Format results as human-readable table.

**Parameters:**
- `results` (ResultsDict): Complete results dictionary

**Returns:**
- `str`: Formatted table string

---

## Data Structures

### ParameterDict

Dictionary containing cosmological parameters:

```python
{
    # ΛCDM parameters
    "H0": float,              # Hubble constant (km/s/Mpc)
    "Om0": float,             # Matter density fraction
    "Obh2": float,            # Physical baryon density
    "ns": float,              # Scalar spectral index
    "Neff": float,            # Effective neutrino species
    "Tcmb": float,            # CMB temperature (K)
    
    # PBUF parameters (if model="pbuf")
    "alpha": float,           # Elasticity amplitude
    "Rmax": float,            # Saturation length scale
    "eps0": float,            # Elasticity bias term
    "n_eps": float,           # Evolution exponent
    "k_sat": float,           # Saturation coefficient
    
    # Derived parameters
    "Omh2": float,            # Physical matter density
    "z_recomb": float,        # Recombination redshift
    "z_drag": float,          # Drag epoch redshift
    "r_s_drag": float,        # Sound horizon at drag epoch
    
    # Metadata
    "model_class": str,       # "lcdm" or "pbuf"
    "recomb_method": str      # Recombination method
}
```

### ResultsDict

Dictionary containing complete fit results:

```python
{
    "params": ParameterDict,  # Final optimized parameters
    "results": {              # Per-block results
        "cmb": {
            "chi2": float,
            "predictions": Dict,
            "residuals": np.ndarray
        },
        "bao": {...},
        "sn": {...}
    },
    "metrics": MetricsDict,   # Overall fit statistics
    "diagnostics": {          # Physics consistency checks
        "h_ratios": List[float],
        "recomb_check": bool,
        "covariance_status": str
    }
}
```

### MetricsDict

Dictionary containing fit statistics:

```python
{
    "total_chi2": float,      # Total χ² value
    "aic": float,             # Akaike Information Criterion
    "bic": float,             # Bayesian Information Criterion
    "dof": int,               # Degrees of freedom
    "p_value": float,         # p-value from χ² distribution
    "reduced_chi2": float     # χ²/DOF
}
```

## Error Handling

### Common Exceptions

#### `ValueError`
Raised for invalid parameter values or model specifications:
```python
try:
    params = build_params("invalid_model")
except ValueError as e:
    print(f"Invalid model: {e}")
```

#### `KeyError`
Raised for missing datasets or parameters:
```python
try:
    data = load_dataset("nonexistent_dataset")
except KeyError as e:
    print(f"Dataset not found: {e}")
```

#### `TypeError`
Raised for incorrect parameter types:
```python
try:
    params = build_params("lcdm", overrides={"H0": "invalid"})
except TypeError as e:
    print(f"Invalid parameter type: {e}")
```

### Validation Errors

The system performs extensive validation:

- **Parameter bounds checking**: Ensures physical parameter ranges
- **Covariance matrix validation**: Checks positive definiteness
- **Dataset format validation**: Verifies expected data structure
- **Physics consistency checks**: Validates model predictions

## Configuration

### Optimizer Configuration

The system supports multiple optimization methods through the `optimizer_config` parameter:

```python
# Local optimization (default)
optimizer_config = {
    "method": "minimize",           # scipy.optimize.minimize
    "options": {
        "maxiter": 1000,           # Maximum iterations
        "ftol": 1e-6,              # Function tolerance
        "xtol": 1e-6,              # Parameter tolerance
        "method": "L-BFGS-B"       # Specific scipy method
    },
    "bounds": {                    # Parameter bounds (optional)
        "H0": (50, 100),
        "Om0": (0.1, 0.5)
    }
}

# Global optimization
optimizer_config = {
    "method": "differential_evolution",
    "options": {
        "maxiter": 1000,
        "seed": 42,                # For reproducibility
        "popsize": 15,             # Population size multiplier
        "atol": 1e-6,              # Absolute tolerance
        "tol": 1e-6                # Relative tolerance
    },
    "bounds": {                    # Required for differential evolution
        "H0": (50, 100),
        "Om0": (0.1, 0.5),
        "alpha": (1e-6, 1e-2)      # PBUF parameter bounds
    }
}

# Custom optimization method (extensible)
optimizer_config = {
    "method": "custom_basinhopping",  # Must be registered in engine
    "options": {
        "niter": 100,
        "T": 1.0,
        "stepsize": 0.5
    }
}
```

### Integrity Check Configuration

```python
integrity_config = {
    "h_ratios": {
        "tolerance": 1e-4,
        "redshifts": [0.1, 0.5, 1.0, 2.0]
    },
    "recombination": {
        "reference": 1089.80,
        "tolerance": 1.0
    },
    "covariance": {
        "min_eigenvalue": 1e-10
    }
}
```

## Performance Considerations

### Optimization Tips

1. **Use appropriate optimizer**: `minimize` for local optimization, `differential_evolution` for global
2. **Set reasonable bounds**: Constrain parameters to physical ranges
3. **Cache results**: Reuse parameter dictionaries when possible
4. **Batch operations**: Use joint fitting for multiple datasets

### Memory Management

- Parameter dictionaries are lightweight and can be cached
- Covariance matrices are the largest memory consumers
- Dataset loading is optimized with lazy evaluation where possible

### Computational Complexity

- **Individual fits**: O(N) where N is number of data points
- **Joint fits**: O(N₁ + N₂ + ... + Nₖ) for k datasets
- **Optimization**: Depends on algorithm and convergence criteria

## Extensibility API

### Adding New Models (Requirement 7.1)

The system supports new cosmological models through parameter configuration only:

```python
# 1. Add model parameters to DEFAULTS in parameter.py
DEFAULTS["new_model"] = {
    "H0": 67.4,
    "Om0": 0.315,
    "new_param1": 1.0,
    "new_param2": 0.5,
    "model_class": "new_model"
}

# 2. Model is automatically available through unified interface
params = build_params("new_model")
result = run_fit("new_model", ["cmb", "bao"])
```

### Registering New Likelihood Functions (Requirement 7.2)

Add new likelihood functions without modifying core engine logic:

```python
# In likelihoods.py, add new likelihood function
def likelihood_new_dataset(params: Dict, data: Dict) -> Tuple[float, Dict]:
    """New dataset likelihood function."""
    # Compute theoretical predictions
    predictions = compute_new_theory(params)
    
    # Calculate chi-squared
    chi2 = chi2_generic(predictions, data["observations"], data["covariance"])
    
    return chi2, predictions

# Register in likelihood dispatcher
LIKELIHOOD_FUNCTIONS[("model_name", "new_dataset")] = likelihood_new_dataset
```

### Adding New Datasets (Requirement 7.3)

Integrate new datasets through the unified loader interface:

```python
# In datasets.py, add new dataset loader
def load_new_dataset():
    """Load new observational dataset."""
    return {
        "observations": data_vector,
        "covariance": covariance_matrix,
        "metadata": {
            "n_points": len(data_vector),
            "redshift_range": [z_min, z_max],
            "survey": "survey_name"
        }
    }

# Register in dataset loader registry
DATASET_LOADERS["new_dataset"] = load_new_dataset
```

### Adding New Optimization Methods (Requirement 7.5)

Support new optimization methods through engine configuration:

```python
# In engine.py, add new optimizer
def custom_optimizer(objective_func, initial_params, bounds=None, options=None):
    """Custom optimization algorithm."""
    # Implement custom optimization logic
    result = custom_algorithm(objective_func, initial_params, **options)
    return result

# Register in optimizer registry
OPTIMIZERS["custom_method"] = custom_optimizer

# Use in configuration
optimizer_config = {
    "method": "custom_method",
    "options": {"custom_param": 1.0}
}
```

### Physics Documentation References (Requirement 7.4)

The system references documented physics derivations in the `documents/` directory:

- **PBUF Model**: `documents/PBUF-Math-Supplement-v9.md`
- **Empirical Relations**: `documents/Empirical_Summary_v9.md`
- **Evolution Theory**: `documents/evolution_theory.md`
- **Equation Reference**: `documents/equations_reference.mc`

Model implementations should reference these documents for physics validation.

## Thread Safety

The core API functions are thread-safe for read operations:
- Parameter building is stateless
- Likelihood computations are pure functions
- Dataset loading uses immutable data structures

Optimization operations should not be run concurrently on the same parameter space due to scipy optimizer limitations.