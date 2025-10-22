# PBUF Cosmology Pipeline - API Reference

## Core API Functions

### engine.py

#### `run_fit(model, datasets_list, mode="joint", overrides=None, optimizer_config=None, optimization_params=None)`

Central optimization function used by all fitters with parameter optimization support.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")
- `datasets_list` (List[str]): List of dataset names to include in fit
- `mode` (str, optional): Fitting mode ("joint" or "individual"). Default: "joint"
- `overrides` (Dict, optional): Parameter overrides. Default: None
- `optimizer_config` (Dict, optional): Optimizer configuration. Default: None
- `optimization_params` (List[str], optional): Parameters to optimize. Default: None (use fixed values)

**Returns:**
- `ResultsDict`: Complete results dictionary with parameters, χ² breakdown, metrics, and optimization metadata

**Example:**
```python
# Standard fitting with fixed parameters
result = run_fit(
    model="pbuf",
    datasets_list=["cmb", "bao"],
    mode="joint",
    overrides={"H0": 70.0}
)

# Parameter optimization for CMB fitting
result = run_fit(
    model="pbuf",
    datasets_list=["cmb"],
    optimization_params=["k_sat", "alpha", "H0", "Om0"]
)

# ΛCDM parameter optimization
result = run_fit(
    model="lcdm",
    datasets_list=["cmb"],
    optimization_params=["H0", "Om0", "Obh2", "ns"]
)
```

**Raises:**
- `ValueError`: If model is not "lcdm" or "pbuf"
- `ValueError`: If datasets_list is empty
- `ValueError`: If optimization_params contains invalid parameters for the model
- `KeyError`: If dataset not found

---

### optimizer.py

#### `ParameterOptimizer`

Class for optimizing cosmological parameters using numerical minimization.

##### `optimize_parameters(model, datasets, optimize_params, starting_values=None, covariance_scaling=1.0)`

Optimize specified parameters for given model and datasets.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")
- `datasets` (List[str]): List of datasets for optimization
- `optimize_params` (List[str]): Parameters to optimize
- `starting_values` (Dict, optional): Starting parameter values. Default: None (use model defaults)
- `covariance_scaling` (float, optional): Covariance matrix scaling factor. Default: 1.0

**Returns:**
- `OptimizationResult`: Optimization results with final parameters, χ² improvement, and metadata

**Example:**
```python
from pipelines.fit_core.optimizer import ParameterOptimizer

optimizer = ParameterOptimizer()

# Optimize PBUF parameters for CMB
result = optimizer.optimize_parameters(
    model="pbuf",
    datasets=["cmb"],
    optimize_params=["k_sat", "alpha"],
    covariance_scaling=1.0
)

print(f"χ² improvement: {result.chi2_improvement}")
print(f"Optimized k_sat: {result.optimized_params['k_sat']}")
```

##### `get_optimization_bounds(model, param)`

Get physical bounds for parameter optimization.

**Parameters:**
- `model` (str): Model type
- `param` (str): Parameter name

**Returns:**
- `Tuple[float, float]`: (lower_bound, upper_bound)

##### `validate_optimization_request(model, optimize_params)`

Validate that parameters can be optimized for the given model.

**Parameters:**
- `model` (str): Model type
- `optimize_params` (List[str]): Parameters to validate

**Returns:**
- `bool`: True if all parameters are valid for optimization

**Raises:**
- `ValueError`: If any parameter is invalid for the model

#### `optimize_cmb_parameters(model, optimize_params, starting_params=None, covariance_scaling=1.0)`

Specialized CMB parameter optimization function.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")
- `optimize_params` (List[str]): Parameters to optimize
- `starting_params` (Dict, optional): Starting parameter values
- `covariance_scaling` (float, optional): Covariance scaling factor

**Returns:**
- `CMBOptimizationResult`: CMB-specific optimization results

**Example:**
```python
# Optimize ΛCDM parameters for CMB
result = optimize_cmb_parameters(
    model="lcdm",
    optimize_params=["H0", "Om0", "Obh2", "ns"],
    covariance_scaling=1.2
)
```

---

### parameter_store.py

#### `OptimizedParameterStore`

Class for managing optimized parameter storage and retrieval.

##### `get_model_defaults(model)`

Get current default parameters for model, including any optimized values.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")

**Returns:**
- `ParameterDict`: Current model defaults with optimization metadata

##### `update_model_defaults(model, optimized_params, optimization_metadata, dry_run=False)`

Update model defaults with optimized parameter values.

**Parameters:**
- `model` (str): Model type
- `optimized_params` (Dict[str, float]): Optimized parameter values
- `optimization_metadata` (Dict): Optimization metadata (timestamp, χ² improvement, etc.)
- `dry_run` (bool, optional): If True, don't persist changes. Default: False

**Returns:**
- `None`

##### `get_optimization_history(model)`

Get optimization history for model.

**Parameters:**
- `model` (str): Model type

**Returns:**
- `List[OptimizationRecord]`: List of optimization records

##### `validate_cross_model_consistency()`

Validate consistency of shared parameters across ΛCDM and PBUF models.

**Returns:**
- `Dict[str, float]`: Divergence values for shared parameters

##### `export_optimization_summary(output_path="reports/optimization_summary.json")`

Export optimization summary for reporting.

**Parameters:**
- `output_path` (str, optional): Output file path

**Returns:**
- `None`

**Example:**
```python
from pipelines.fit_core.parameter_store import OptimizedParameterStore

store = OptimizedParameterStore()

# Get current defaults (may include optimized values)
params = store.get_model_defaults("pbuf")

# Update with new optimization results
store.update_model_defaults(
    model="pbuf",
    optimized_params={"k_sat": 0.9762, "alpha": 5e-4},
    optimization_metadata={
        "timestamp": "2025-10-22T14:42:00Z",
        "chi2_improvement": 5.67,
        "convergence_status": "success"
    }
)

# Check cross-model consistency
divergence = store.validate_cross_model_consistency()
```

---

### parameter.py

#### `build_params(model, overrides=None, optimization_metadata=None)`

Build parameter dictionary for specified model with optional overrides and optimization metadata.

**Parameters:**
- `model` (str): Model type ("lcdm" or "pbuf")
- `overrides` (Dict, optional): Parameter overrides to apply. Default: None
- `optimization_metadata` (Dict, optional): Optimization metadata to include. Default: None

**Returns:**
- `ParameterDict`: Complete parameter dictionary with all required parameters and optimization metadata

**Example:**
```python
# Standard parameter building
params = build_params("lcdm", overrides={"H0": 70.0, "Om0": 0.3})

# With optimization metadata
params = build_params(
    "pbuf", 
    overrides={"k_sat": 0.9762},
    optimization_metadata={
        "optimized_params": ["k_sat", "alpha"],
        "optimization_source": "cmb",
        "timestamp": "2025-10-22T14:42:00Z"
    }
)
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

### OptimizationResult

Dictionary containing parameter optimization results:

```python
{
    "model": str,                           # Model type ("lcdm" or "pbuf")
    "optimized_params": Dict[str, float],   # Final optimized parameter values
    "starting_params": Dict[str, float],    # Initial parameter values
    "final_chi2": float,                    # Final χ² value
    "chi2_improvement": float,              # χ² reduction from optimization
    "convergence_status": str,              # "success", "failed", "max_iter"
    "n_function_evaluations": int,          # Number of likelihood evaluations
    "optimization_time": float,             # Time taken (seconds)
    "bounds_reached": List[str],            # Parameters that hit bounds
    "optimizer_info": {                     # Optimizer metadata
        "method": str,                      # Optimization method used
        "library": str,                     # Library name (e.g., "scipy")
        "version": str                      # Library version
    },
    "covariance_scaling": float,            # Covariance scaling factor used
    "metadata": Dict[str, Any]              # Additional metadata
}
```

### OptimizationRecord

Dictionary containing optimization history entry:

```python
{
    "timestamp": str,                       # ISO format timestamp
    "model": str,                           # Model type
    "dataset": str,                         # Dataset used for optimization
    "optimized_params": List[str],          # Parameters that were optimized
    "final_values": Dict[str, float],       # Final parameter values
    "chi2_improvement": float,              # χ² improvement achieved
    "convergence_status": str               # Convergence status
}
```

### CMBOptimizationResult

Specialized result for CMB parameter optimization:

```python
{
    "model": str,                           # Model type
    "optimized_params": Dict[str, float],   # Optimized parameter values
    "cmb_chi2": float,                      # CMB χ² value
    "chi2_improvement": float,              # Improvement over starting values
    "convergence_diagnostics": {            # Detailed convergence info
        "n_iterations": int,
        "final_gradient_norm": float,
        "function_tolerance_reached": bool
    },
    "parameter_bounds_info": {              # Bounds information
        "bounds_used": Dict[str, Tuple[float, float]],
        "bounds_reached": List[str]
    },
    "provenance": {                         # Full provenance information
        "timestamp": str,
        "optimizer_method": str,
        "library_version": str,
        "covariance_scaling": float,
        "starting_values": Dict[str, float]
    }
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

## Parameter Optimization

### Command-Line Flags

All fitter scripts support the following optimization flags:

#### `--optimize PARAM1,PARAM2,...`
Specify parameters to optimize during fitting.

**ΛCDM Parameters:**
- `H0`: Hubble constant (bounds: 20.0-150.0 km/s/Mpc)
- `Om0`: Matter density fraction (bounds: 0.01-0.99)
- `Obh2`: Physical baryon density (bounds: 0.005-0.1)
- `ns`: Scalar spectral index (bounds: 0.8-1.2)

**PBUF Parameters:**
- `k_sat`: Saturation coefficient (bounds: 0.1-2.0)
- `alpha`: Elasticity amplitude (bounds: 1e-6-1e-2)
- `H0`, `Om0`, `Obh2`, `ns`: Same as ΛCDM

**Examples:**
```bash
# Optimize PBUF parameters for CMB
python fit_cmb.py --model pbuf --optimize k_sat,alpha

# Optimize ΛCDM core parameters
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns

# Optimize specific parameters with overrides
python fit_cmb.py --model pbuf --optimize k_sat --H0 70.0
```

#### `--cov-scale FACTOR`
Scale covariance matrices by specified factor (default: 1.0).

**Example:**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0 --cov-scale 1.2
```

#### `--dry-run`
Perform optimization without updating stored defaults.

**Example:**
```bash
python fit_cmb.py --model pbuf --optimize k_sat --dry-run
```

#### `--warm-start`
Use recent optimization results as starting values (if available within 24 hours).

**Example:**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0 --warm-start
```

### Configuration File Support

Add optimization settings to configuration files:

```json
{
  "optimization": {
    "optimize_parameters": ["k_sat", "alpha"],
    "frozen_parameters": ["Tcmb", "Neff"],
    "use_precomputed": true,
    "save_results": true,
    "convergence_tolerance": 1e-6,
    "covariance_scaling": 1.0,
    "warm_start": true,
    "dry_run": false
  },
  "parameter_overrides": {
    "H0": 70.0
  }
}
```

**Configuration Options:**
- `optimize_parameters`: List of parameters to optimize
- `frozen_parameters`: List of parameters to keep fixed (overrides optimize_parameters)
- `use_precomputed`: Use stored optimized values as starting points
- `save_results`: Save optimization results for future use
- `convergence_tolerance`: Optimization convergence tolerance
- `covariance_scaling`: Scale factor for covariance matrices
- `warm_start`: Use recent optimization results as starting values
- `dry_run`: Perform optimization without persisting results

### Optimization Workflow

1. **CMB Optimization**: Optimize parameters using CMB data to find best-fit values
2. **Parameter Propagation**: Updated defaults are automatically used by all subsequent fitters
3. **Cross-Model Consistency**: System validates shared parameters remain consistent
4. **Result Storage**: Optimization results are stored with metadata for reproducibility

**Example Workflow:**
```bash
# Step 1: Optimize PBUF parameters using CMB
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0

# Step 2: BAO fitting automatically uses optimized parameters
python fit_bao.py --model pbuf

# Step 3: Joint fitting benefits from pre-optimized individual parameters
python fit_joint.py --model pbuf --datasets cmb,bao,sn
```

### Troubleshooting Optimization

#### Convergence Issues

**Problem**: Optimization fails to converge
**Solutions:**
- Reduce `convergence_tolerance` in configuration
- Use `--warm-start` to begin from previous results
- Check parameter bounds are reasonable for your data
- Try different `--cov-scale` values (0.8-1.5 range)

**Problem**: Optimization reaches parameter bounds
**Solutions:**
- Review physical bounds in `parameter.py`
- Check if data supports the parameter range
- Consider if model is appropriate for the dataset

**Problem**: χ² improvement is minimal
**Solutions:**
- Verify parameters are actually free in the model
- Check if starting values are already near optimal
- Ensure covariance matrices are properly scaled

#### Parameter Validation Errors

**Problem**: "Invalid parameter for model" error
**Solutions:**
- Check parameter names match exactly (case-sensitive)
- Verify parameter is supported for the model type
- Use `--help` to see valid parameters for each model

**Problem**: Cross-model consistency warnings
**Solutions:**
- Run optimization for both models with same datasets
- Check if shared parameters (H0, Om0) are diverging
- Review optimization convergence for both models

#### Storage and Persistence Issues

**Problem**: Optimized parameters not being used
**Solutions:**
- Verify optimization completed successfully
- Check file permissions for parameter storage
- Ensure fitters are calling `get_model_defaults()`
- Use `--dry-run` to test without persistence

**Problem**: Optimization history not available
**Solutions:**
- Check `optimization_results/` directory exists
- Verify JSON files are not corrupted
- Run optimization with `save_results: true`

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