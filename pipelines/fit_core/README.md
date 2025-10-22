# PBUF Cosmology Pipeline - Unified Architecture

## Overview

The PBUF cosmology pipeline provides a unified, modular architecture for fitting cosmological models to observational data. The system supports both ΛCDM and PBUF (Phenomenological Bulk Viscosity) models across multiple observational blocks: CMB distance priors, BAO measurements, and supernova data.

## Key Features

- **Unified Parameter Management**: Single source of truth for all cosmological parameters
- **Modular Architecture**: Clean separation between physics, data, and optimization layers
- **Extensible Design**: Easy addition of new cosmological models and datasets
- **Physics Consistency**: Built-in validation and integrity checks
- **Multiple Interfaces**: Individual fitters and joint fitting capabilities

## Architecture

```
pipelines/
├── fit_core/                   # Core unified architecture
│   ├── engine.py              # Central optimization engine
│   ├── parameter.py           # Centralized parameter management
│   ├── likelihoods.py         # Likelihood functions for all blocks
│   ├── datasets.py            # Unified dataset loading
│   ├── statistics.py          # Statistical computations (χ², AIC, BIC)
│   ├── logging_utils.py       # Standardized logging and diagnostics
│   └── integrity.py           # Physics validation and consistency checks
├── fit_cmb.py                 # CMB fitting wrapper
├── fit_bao.py                 # BAO fitting wrapper
├── fit_sn.py                  # Supernova fitting wrapper
└── fit_joint.py               # Joint fitting wrapper
```

## Core Modules

### engine.py - Central Optimization Engine

The heart of the unified system. Orchestrates parameter building, likelihood computation, and optimization.

**Key Function:**
```python
def run_fit(model: str, datasets_list: List[str], mode: str = "joint", 
           overrides: Optional[Dict] = None) -> Dict
```

### parameter.py - Parameter Management

Centralized parameter handling ensuring consistency across all fitters.

**Key Functions:**
```python
def build_params(model: str, overrides: Optional[Dict] = None) -> Dict
def get_defaults(model: str) -> Dict
```

**Supported Models:**
- `"lcdm"`: Standard ΛCDM cosmology
- `"pbuf"`: PBUF model with bulk viscosity

### likelihoods.py - Likelihood Functions

Pure functions computing χ² for each observational block.

**Available Likelihoods:**
```python
def likelihood_cmb(params: Dict, data: Dict) -> Tuple[float, Dict]
def likelihood_bao(params: Dict, data: Dict) -> Tuple[float, Dict]
def likelihood_bao_ani(params: Dict, data: Dict) -> Tuple[float, Dict]
def likelihood_sn(params: Dict, data: Dict) -> Tuple[float, Dict]
```

### datasets.py - Data Management

Unified interface for loading and validating observational datasets.

**Supported Datasets:**
- `"cmb"`: Planck 2018 distance priors
- `"bao"`: Isotropic BAO measurements
- `"bao_ani"`: Anisotropic BAO measurements  
- `"sn"`: Pantheon+ supernova compilation

### statistics.py - Statistical Analysis

Centralized computation of fit statistics and model comparison metrics.

**Key Functions:**
```python
def chi2_generic(predictions: Dict, observations: Dict, covariance: np.ndarray) -> float
def compute_metrics(chi2: float, n_params: int, datasets: List[str]) -> Dict
```

### integrity.py - Physics Validation

Optional consistency checks and physics validation.

**Validation Functions:**
```python
def verify_h_ratios(params: Dict, redshifts: List[float]) -> bool
def verify_recombination(params: Dict, reference: float = 1089.80) -> bool
def run_integrity_suite(params: Dict, datasets: List[str]) -> Dict
```

## Usage Examples

### Basic Individual Fitting

```python
from pipelines.fit_core.engine import run_fit

# Fit ΛCDM to CMB data
result = run_fit(
    model="lcdm",
    datasets_list=["cmb"],
    mode="individual"
)

print(f"χ² = {result['metrics']['total_chi2']:.3f}")
print(f"AIC = {result['metrics']['aic']:.3f}")
```

### Joint Fitting with Multiple Datasets

```python
# Joint fit PBUF model to CMB + BAO + SN
result = run_fit(
    model="pbuf",
    datasets_list=["cmb", "bao", "sn"],
    mode="joint"
)

# Access individual block results
cmb_chi2 = result['results']['cmb']['chi2']
bao_chi2 = result['results']['bao']['chi2']
sn_chi2 = result['results']['sn']['chi2']
```

### Parameter Overrides

```python
# Override default parameters
overrides = {
    "H0": 70.0,
    "Om0": 0.3,
    "alpha": 1e-3  # PBUF parameter
}

result = run_fit(
    model="pbuf",
    datasets_list=["cmb", "bao"],
    overrides=overrides
)
```

### Using Wrapper Scripts

```bash
# CMB fitting
python pipelines/fit_cmb.py --model lcdm

# Joint fitting
python pipelines/fit_joint.py --model pbuf --datasets cmb bao sn

# With parameter overrides
python pipelines/fit_cmb.py --model lcdm --H0 70.0 --Om0 0.3

# With integrity checks
python pipelines/fit_cmb.py --model pbuf --verify-integrity
```

## Configuration Options

### Command Line Arguments

All wrapper scripts support comprehensive configuration:

```bash
# Basic model selection
--model {lcdm,pbuf}              # Cosmological model

# Parameter overrides
--H0 FLOAT                       # Hubble constant (km/s/Mpc)
--Om0 FLOAT                      # Matter density fraction
--Obh2 FLOAT                     # Physical baryon density
--ns FLOAT                       # Scalar spectral index
--alpha FLOAT                    # PBUF elasticity amplitude
--eps0 FLOAT                     # PBUF elasticity bias
--k_sat FLOAT                    # PBUF saturation coefficient

# Analysis options
--verify-integrity               # Run physics consistency checks
--output-format {human,json,csv} # Output format
--save-results FILE              # Save results to file
--config FILE                    # Load configuration from file

# Optimization control
--optimizer {minimize,differential_evolution}
--maxiter INT                    # Maximum optimization iterations
--seed INT                       # Random seed for reproducibility
```

### Configuration Files

Create comprehensive configuration files for complex analysis workflows:

#### Basic Configuration
```json
{
    "model": "pbuf",
    "datasets": ["cmb", "bao", "sn"],
    "parameters": {
        "H0": 67.4,
        "Om0": 0.315,
        "alpha": 5e-4,
        "eps0": 0.7
    },
    "optimizer": {
        "method": "minimize",
        "options": {"maxiter": 1000}
    },
    "output": {
        "format": "json",
        "save_file": "results.json"
    }
}
```

#### Advanced Configuration with Extensions
```json
{
    "model": "pbuf",
    "datasets": ["cmb", "bao", "sn"],
    "parameters": {
        "H0": 67.4,
        "Om0": 0.315,
        "alpha": 5e-4
    },
    "optimizer": {
        "method": "differential_evolution",
        "options": {
            "maxiter": 1000,
            "seed": 42,
            "popsize": 15
        },
        "bounds": {
            "H0": [60, 80],
            "Om0": [0.2, 0.4],
            "alpha": [1e-6, 1e-2]
        }
    },
    "integrity_checks": {
        "enabled": true,
        "tolerance": 1e-4,
        "h_ratio_redshifts": [0.1, 0.5, 1.0, 2.0]
    },
    "extensions": {
        "models": ["wcdm"],
        "datasets": ["wl"],
        "config_file": "extensions.json"
    },
    "output": {
        "format": "json",
        "save_file": "pbuf_analysis.json",
        "include_diagnostics": true,
        "include_predictions": true
    }
}
```

#### Multi-Model Comparison Configuration
```json
{
    "analysis_type": "model_comparison",
    "models": ["lcdm", "pbuf", "wcdm"],
    "datasets": ["cmb", "bao", "sn"],
    "comparison_metrics": ["aic", "bic", "chi2"],
    "parameter_grids": {
        "pbuf": {
            "alpha": [1e-4, 5e-4, 1e-3],
            "eps0": [0.5, 0.7, 0.9]
        },
        "wcdm": {
            "w0": [-1.2, -1.0, -0.8],
            "wa": [-0.2, 0.0, 0.2]
        }
    },
    "output": {
        "comparison_table": "model_comparison.csv",
        "best_fit_summary": "best_fits.json"
    }
}
```

Use configurations with: `python pipelines/fit_joint.py --config analysis_config.json`

## Physics Consistency

The system includes built-in physics validation:

### H(z) Ratio Checks
Verifies PBUF reduces to ΛCDM when k_sat → 1:
```python
from pipelines.fit_core.integrity import verify_h_ratios

params = build_params("pbuf")
is_consistent = verify_h_ratios(params, redshifts=[0.5, 1.0, 2.0])
```

### Recombination Validation
Checks z* computation against Planck 2018 reference:
```python
from pipelines.fit_core.integrity import verify_recombination

params = build_params("lcdm")
is_valid = verify_recombination(params, reference=1089.80)
```

### Comprehensive Integrity Suite
```python
from pipelines.fit_core.integrity import run_integrity_suite

params = build_params("pbuf")
results = run_integrity_suite(params, ["cmb", "bao", "sn"])
```

## Extending the System

The unified architecture supports seamless extension through configuration-based approaches that minimize code changes.

### Adding New Models (Requirement 7.1)

New cosmological models can be added through parameter configuration only:

```python
# 1. Add model parameters to DEFAULTS in parameter.py
DEFAULTS["new_model"] = {
    "H0": 67.4,           # Standard cosmological parameters
    "Om0": 0.315,
    "Obh2": 0.02237,
    "ns": 0.9649,
    "new_param1": 1.0,    # Model-specific parameters
    "new_param2": 0.5,
    "model_class": "new_model"
}

# 2. Model is immediately available through unified interface
params = build_params("new_model")
result = run_fit("new_model", ["cmb", "bao", "sn"])
```

For models requiring custom physics, implement in separate modules:
```python
# pipelines/new_model_physics.py
def new_model_hubble(z, params):
    """Hubble parameter for new model."""
    # Physics Reference: documents/new_model_theory.md
    return h_z

def new_model_distances(z, params):
    """Distance calculations for new model."""
    # Implementation following documented equations
    return distances
```

### Adding New Datasets (Requirement 7.3)

Integrate new observational datasets through the unified loader interface:

```python
# 1. Implement data loader in datasets.py
def load_new_dataset():
    """Load new observational dataset."""
    return {
        "observations": data_vector,
        "covariance": cov_matrix,
        "metadata": {
            "n_points": len(data_vector),
            "redshift_range": [z_min, z_max],
            "survey": "survey_name",
            "data_type": "measurement_type"
        }
    }

# 2. Register in dataset loader registry
DATASET_LOADERS["new_dataset"] = load_new_dataset

# 3. Add likelihood function if needed
def likelihood_new_dataset(params: Dict, data: Dict) -> Tuple[float, Dict]:
    """Likelihood for new dataset."""
    predictions = compute_theory_predictions(params, data)
    chi2 = chi2_generic(predictions, data["observations"], data["covariance"])
    return chi2, predictions

# 4. Register likelihood for all relevant models
for model in ["lcdm", "pbuf", "new_model"]:
    LIKELIHOOD_FUNCTIONS[(model, "new_dataset")] = likelihood_new_dataset

# 5. Dataset is immediately available
result = run_fit("pbuf", ["cmb", "bao", "new_dataset"])
```

### Adding New Optimization Methods (Requirement 7.5)

Support new optimization algorithms through engine configuration:

```python
# 1. Implement optimizer in engine.py
def custom_optimizer(objective_func, initial_params, bounds=None, options=None):
    """Custom optimization algorithm."""
    # Implement optimization logic
    result = custom_algorithm(objective_func, initial_params, **options)
    return result

# 2. Register in optimizer registry
OPTIMIZERS["custom_method"] = custom_optimizer

# 3. Use in configuration
optimizer_config = {
    "method": "custom_method",
    "options": {
        "custom_param1": 1.0,
        "custom_param2": 0.5
    }
}

result = run_fit("pbuf", ["cmb"], optimizer_config=optimizer_config)
```

### Configuration-Based Extensions

For complex extensions, use configuration files:

```json
{
    "extensions": {
        "models": {
            "wcdm": {
                "parameters": {
                    "H0": 67.4, "Om0": 0.315,
                    "w0": -1.0, "wa": 0.0
                },
                "physics_module": "wcdm_models",
                "documentation": "documents/wcdm_theory.md"
            }
        },
        "datasets": {
            "weak_lensing": {
                "loader_module": "wl_data",
                "loader_function": "load_des_y3",
                "likelihood_function": "likelihood_wl"
            }
        },
        "optimizers": {
            "mcmc": {
                "module": "mcmc_optimizer",
                "function": "emcee_sampler"
            }
        }
    }
}
```

### Physics Documentation Integration (Requirement 7.4)

All extensions should reference documented physics in the `documents/` directory:

- **Theory Documentation**: `documents/model_theory.md`
- **Empirical Validation**: `documents/empirical_tests.md`
- **Equation Reference**: `documents/equations_reference.mc`
- **Implementation Notes**: `documents/implementation_guide.md`

```python
def new_physics_function(params):
    """
    New physics calculation.
    
    Physics Reference: documents/new_theory.md, Equation (15)
    Validation: documents/validation_tests.md, Section 3.2
    """
    # Implementation with explicit documentation references
    pass
```

## Performance and Optimization

### Optimization Methods

The system supports multiple optimization algorithms:
- `scipy.optimize.minimize` (default): Fast local optimization
- `scipy.optimize.differential_evolution`: Global optimization

Configure via:
```python
optimizer_config = {
    "method": "differential_evolution",
    "options": {"maxiter": 1000, "seed": 42}
}
```

### Caching and Performance

- Parameter dictionaries are cached to avoid recomputation
- Covariance matrix inversions are cached per dataset
- Likelihood computations are optimized for repeated calls

## Error Handling

The system provides robust error handling:

### Parameter Validation
```python
try:
    params = build_params("invalid_model")
except ValueError as e:
    print(f"Invalid model: {e}")
```

### Dataset Validation
```python
try:
    result = run_fit("lcdm", ["invalid_dataset"])
except KeyError as e:
    print(f"Dataset not found: {e}")
```

### Physics Consistency Warnings
```python
# Automatic warnings for physics inconsistencies
result = run_fit("pbuf", ["cmb"], verify_integrity=True)
# Logs warnings if H(z) ratios exceed tolerance
```

## Testing and Validation

### Unit Tests
Run individual module tests:
```bash
python -m pytest pipelines/fit_core/test_*.py
```

### Integration Tests
Run comprehensive system integration tests:
```bash
python integration_test_report.py
```

### Parity Testing
Validate against legacy system:
```bash
python pipelines/run_parity_tests.py
```

## Migration from Legacy System

### Backward Compatibility
- All wrapper scripts maintain identical interfaces to legacy versions
- Results are numerically equivalent within 1e-6 tolerance
- Existing analysis scripts continue to work unchanged

### Migration Steps
1. Replace individual fitter calls with unified wrapper scripts
2. Update configuration files to new format (optional)
3. Add integrity checks for enhanced validation (optional)
4. Migrate to direct engine API for advanced usage (optional)

## Troubleshooting

### Common Issues

**Degrees of Freedom Error:**
```
Error: Degrees of freedom cannot be negative
```
Solution: Ensure sufficient data points relative to free parameters.

**Convergence Issues:**
```
Optimization failed to converge
```
Solution: Try different optimizer or adjust initial parameters.

**Physics Consistency Warnings:**
```
Warning: H(z) ratios exceed tolerance
```
Solution: Check parameter values and model configuration.

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

result = run_fit("pbuf", ["cmb"], mode="individual")
```

### Support and Documentation

The system includes comprehensive documentation covering all aspects of usage and extension:

#### Core Documentation
- **[API Reference](API_REFERENCE.md)**: Complete API documentation with function signatures and examples
- **[Usage Examples](USAGE_EXAMPLES.md)**: Comprehensive usage examples from basic to advanced scenarios
- **[Developer Guide](DEVELOPER_GUIDE.md)**: Guide for extending the system with new models and datasets
- **[Configuration Examples](CONFIGURATION_EXAMPLES.md)**: Complete configuration file examples and patterns
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Solutions to common issues and debugging techniques

#### Physics Documentation
- **PBUF Theory**: `documents/PBUF-Math-Supplement-v9.md`
- **Empirical Summary**: `documents/Empirical_Summary_v9.md`
- **Evolution Theory**: `documents/evolution_theory.md`
- **Equation Reference**: `documents/equations_reference.mc`

#### System Validation
- **Integration Tests**: Run `python integration_test_report.py`
- **Parity Testing**: Run `python pipelines/run_parity_tests.py`
- **Physics Validation**: See `parity_results/validation_certification.md`

#### Quick Start Documentation Access
```python
# View documentation from Python
from pipelines.fit_core import help
help.show_api_reference()
help.show_usage_examples()
help.show_troubleshooting()
```

## References

- **Planck Collaboration 2020**: CMB distance priors (A&A 641, A6)
- **Eisenstein & Hu 1998**: BAO drag epoch calculations (ApJ 496, 605)
- **Pantheon+ Collaboration 2022**: Supernova distance measurements (ApJ 938, 110)
- **PBUF Model**: See `documents/PBUF-Unified-Manuscript-v9.pdf`
- **Numerical Methods**: scipy.optimize documentation
- **Statistical Analysis**: Akaike (1974), Schwarz (1978) for information criteria