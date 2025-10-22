# Design Document

## Overview

This design implements a parameter optimization system for both ΛCDM and PBUF cosmological models, with a focus on CMB-based optimization that produces verified best-fit parameters and updates global defaults for consistent use across all subsequent fits. The system provides a `--optimize` command-line flag and configuration file support to control which parameters are optimized versus held fixed, with automatic parameter propagation to ensure all fitters use the most recently optimized values.

## Architecture

### Core Components

1. **Parameter Optimization Engine** - Extends the existing unified engine to support selective parameter optimization
2. **Default Parameter Store** - Centralized storage and retrieval of model-specific optimized parameters  
3. **CMB Optimization Routine** - Specialized CMB χ² minimization for both ΛCDM and PBUF models
4. **Parameter Propagation System** - Ensures optimized values are used consistently across all fitters
5. **Configuration Management** - Handles optimization settings via CLI flags and config files

### Integration Points

- Extends `pipelines/fit_core/parameter.py` for optimization metadata
- Enhances `pipelines/fit_core/engine.py` for optimization dispatch
- Modifies `pipelines/fit_core/config.py` for optimization configuration
- Updates all fitter scripts (`fit_cmb.py`, `fit_bao.py`, etc.) to support optimization flags

## Components and Interfaces

### 1. Parameter Optimization Engine

```python
class ParameterOptimizer:
    def optimize_parameters(
        self, 
        model: str,
        datasets: List[str], 
        optimize_params: List[str],
        starting_values: Optional[Dict] = None
    ) -> OptimizationResult
    
    def get_optimization_bounds(self, model: str, param: str) -> Tuple[float, float]
    
    def validate_optimization_request(
        self, 
        model: str, 
        optimize_params: List[str]
    ) -> bool
```

**Responsibilities:**
- Validate optimization parameter lists against model capabilities
- Set up bounded optimization problems using existing physical bounds
- Execute χ² minimization with convergence diagnostics
- Return validated optimization results with metadata

### 2. Default Parameter Store

```python
class OptimizedParameterStore:
    def get_model_defaults(self, model: str) -> ParameterDict
    
    def update_model_defaults(
        self, 
        model: str, 
        optimized_params: Dict[str, float],
        optimization_metadata: Dict[str, Any],
        dry_run: bool = False
    ) -> None
    
    def get_optimization_history(self, model: str) -> List[OptimizationRecord]
    
    def is_optimized(self, model: str, dataset: str) -> bool
    
    def get_warm_start_params(self, model: str) -> Optional[ParameterDict]
    
    def validate_cross_model_consistency(self) -> Dict[str, float]
    
    def export_optimization_summary(self, output_path: str = "reports/optimization_summary.json") -> None
```

**Storage Format:**
```json
{
  "lcdm": {
    "defaults": {"H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649},
    "optimization_metadata": {
      "cmb_optimized": "2025-10-22T14:42:00Z",
      "source_dataset": "cmb",
      "optimized_params": ["H0", "Om0", "Obh2", "ns"],
      "chi2_improvement": 2.34,
      "convergence_status": "success",
      "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
      "covariance_scaling": 1.0
    }
  },
  "pbuf": {
    "defaults": {"H0": 67.4, "Om0": 0.315, "k_sat": 0.9762, "alpha": 5e-4},
    "optimization_metadata": {
      "cmb_optimized": "2025-10-22T14:42:00Z",
      "source_dataset": "cmb", 
      "optimized_params": ["H0", "Om0", "k_sat", "alpha"],
      "chi2_improvement": 5.67,
      "convergence_status": "success",
      "optimizer_info": {"method": "L-BFGS-B", "library": "scipy", "version": "1.13.0"},
      "covariance_scaling": 1.0
    }
  }
}
```

### 3. CMB Optimization Routine

```python
def optimize_cmb_parameters(
    model: str,
    optimize_params: List[str],
    starting_params: Optional[ParameterDict] = None
) -> CMBOptimizationResult:
    """
    Perform full numerical minimization of χ²_CMB for specified parameters.
    
    Process:
    1. Load current model defaults as starting point
    2. Define optimization objective: chi2_cmb(params) = residuals^T · C^-1 · residuals
    3. Set parameter bounds from existing physical constraints
    4. Execute bounded optimization with convergence diagnostics
    5. Validate and return best-fit parameters
    """
```

**Optimization Parameters by Model:**
- **ΛCDM**: `[H0, Om0, Obh2, ns]` - Core cosmological parameters
- **PBUF**: `[H0, Om0, Obh2, ns, alpha, Rmax, eps0, n_eps, k_sat]` - All PBUF parameters

### 4. Parameter Propagation System

```python
def propagate_optimized_parameters(
    model: str,
    optimized_params: Dict[str, float]
) -> None:
    """
    Update global defaults and ensure consistent use across all fitters.
    
    Process:
    1. Retrieve current model defaults
    2. Perform non-destructive merge (preserve non-optimized parameters)
    3. Store updated defaults with timestamp and metadata
    4. Validate parameter consistency across system
    """
```

**Injection Rule:** All fitters must call `get_model_defaults(model)` and never use hardcoded values, ensuring automatic use of optimized parameters.

### 5. Configuration Management

**Command Line Interface:**
```bash
# Optimize specific parameters
python fit_cmb.py --model pbuf --optimize k_sat,alpha

# Optimize core ΛCDM parameters  
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns

# Use configuration file with covariance scaling
python fit_cmb.py --config optimization_config.json --cov-scale 1.2

# Dry run without updating stored defaults
python fit_cmb.py --model pbuf --optimize k_sat --dry-run

# Use configuration file
python fit_cmb.py --config optimization_config.json
```

**Configuration File Format:**
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

## Data Models

### OptimizationResult

```python
@dataclass
class OptimizationResult:
    model: str
    optimized_params: Dict[str, float]
    starting_params: Dict[str, float]
    final_chi2: float
    chi2_improvement: float
    convergence_status: str
    n_function_evaluations: int
    optimization_time: float
    bounds_reached: List[str]
    optimizer_info: Dict[str, str]  # method, library, version
    covariance_scaling: float
    metadata: Dict[str, Any]
```

### OptimizationRecord

```python
@dataclass  
class OptimizationRecord:
    timestamp: str
    model: str
    dataset: str
    optimized_params: List[str]
    final_values: Dict[str, float]
    chi2_improvement: float
    convergence_status: str
```

## Error Handling

### Validation Errors
- **Invalid Parameter Names**: Clear error messages for parameters not valid for the selected model
- **Bound Violations**: Automatic constraint enforcement with warning logs when bounds are reached
- **Convergence Failures**: Graceful fallback to default values with detailed diagnostic reporting

### Optimization Failures
- **Non-convergence**: Return diagnostic information and use starting values as fallback
- **Numerical Issues**: Detect and report ill-conditioned optimization problems
- **Physics Violations**: Validate optimized parameters against physical consistency checks

### Storage Errors
- **File System Issues**: Handle read/write failures for parameter storage gracefully
- **Corruption Detection**: Validate stored parameter files and rebuild from defaults if needed
- **Concurrent Access**: Simple file locking to prevent simultaneous optimization runs

## Testing Strategy

### Unit Tests
- Parameter validation for both ΛCDM and PBUF models
- Optimization bounds enforcement and constraint handling
- Parameter store operations (get, update, merge)
- Configuration file parsing and validation
- **Round-trip persistence test**: Verify optimized parameters survive store/reload cycle
- Provenance logging and metadata validation
- Cross-model consistency checking

### Integration Tests  
- End-to-end CMB optimization for both models
- Parameter propagation across multiple fitters
- Configuration file and command-line integration
- Optimization result persistence and retrieval

### Validation Tests
- χ² reproducibility after parameter updates
- Cross-fitter consistency using optimized parameters
- Backward compatibility with existing workflows
- Performance benchmarks for optimization routines

### Acceptance Tests
- Complete optimization workflow for ΛCDM model
- Complete optimization workflow for PBUF model  
- Multi-dataset fitting using optimized parameters
- Configuration-driven optimization scenarios
- Dry-run mode validation (no persistence side effects)
- Warm-start functionality with recent optimization results
- Cross-model consistency validation after sequential optimizations

### Critical Test Case
```python
def test_propagation_roundtrip():
    """Ensure round-trip persistence of optimized parameters."""
    orig = store.get_model_defaults("pbuf")
    res = optimizer.optimize_parameters("pbuf", ["cmb"])
    store.update_model_defaults("pbuf", res.optimized_params, res.metadata)
    reloaded = store.get_model_defaults("pbuf")
    for k, v in res.optimized_params.items():
        assert np.isclose(reloaded[k], v, rtol=1e-12)
```

## Advanced Features

### Provenance Logging
All optimization results include comprehensive metadata for long-term reproducibility:
- Optimizer method and library version information
- Covariance scaling factors used
- Timestamp and convergence diagnostics
- Starting parameter values and bounds reached

### Warm Start Support
- If optimization run exists with timestamp < 24 hours, reuse last optimized values as starting point
- Configurable via `--warm-start` flag or `"warm_start": true` in configuration
- Improves convergence for iterative optimization workflows

### Cross-Model Consistency Validation
After updating defaults for both ΛCDM and PBUF models:
- Automatically validate that shared parameters (H0, Om0, Obh2, ns) remain numerically close
- Log warnings if divergence exceeds tolerance (default: 1e-3)
- Guards against parameter drift from sequential optimization runs

### Covariance Scaling Control
- Optional `--cov-scale` flag allows testing with scaled Planck covariances
- Useful for sensitivity analysis and robustness testing
- Scaling factor recorded in optimization metadata for reproducibility

### Dry Run Mode
- `--dry-run` flag performs all computations without updating stored defaults
- Useful for CI validation and testing optimization configurations
- Returns full optimization results without persistence

### HTML Report Integration
Optimization summaries are automatically included in unified HTML reports:
```json
"cmb_optimization_summary": {
  "lcdm": {
    "chi2": 12.34, 
    "improvement": 2.1, 
    "date": "2025-10-22T14:42:00Z",
    "optimized_params": ["H0", "Om0", "Obh2", "ns"]
  },
  "pbuf": {
    "chi2": 8.76, 
    "improvement": 5.2, 
    "date": "2025-10-22T14:42:00Z",
    "optimized_params": ["H0", "Om0", "k_sat", "alpha"]
  }
}
```

## Implementation Notes

### Backward Compatibility
- All existing functionality remains unchanged when no optimization flags are used
- Legacy parameter overrides continue to work exactly as before
- Existing configuration files work without modification
- Default behavior is identical to current implementation

### Performance Considerations
- Optimization is opt-in and doesn't affect standard fitting performance
- Parameter store uses efficient caching to minimize I/O overhead
- Optimization results are cached to avoid redundant computations
- Bounds checking is optimized for repeated evaluations

### Extensibility
- New models can be added by extending parameter definitions
- Additional optimization algorithms can be plugged in via configuration
- New datasets can be supported through the existing unified engine
- Custom optimization objectives can be registered for specialized use cases

### Concurrency and Safety Features
- **Per-Model File Locking**: Separate locks (pbuf.lock, lcdm.lock) allow concurrent optimization of different models
- **Dataset Integrity Validation**: CMB optimization aborts gracefully if dataset validation fails
- **Regression Testing**: Quick mode to verify optimization reproducibility within 1e-8 tolerance
- **Frozen Parameter Support**: Explicit parameter locking (e.g., Planck H0) via configuration

### Future Enhancements (Phase B)
- **Jackknife/Bootstrap Evaluation**: Quantify parameter stability after optimization
- **Multi-objective Optimization**: Simultaneous optimization across multiple datasets
- **Bayesian Parameter Estimation**: MCMC-based parameter uncertainty quantification
- **Automated Hyperparameter Tuning**: Optimize convergence tolerances and bounds