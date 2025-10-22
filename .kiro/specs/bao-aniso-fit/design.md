# Design Document

## Overview

The BAO anisotropic fit feature provides a dedicated pipeline for analyzing anisotropic Baryon Acoustic Oscillations (BAO) data independently from isotropic BAO measurements. This design leverages the existing unified fitting infrastructure while ensuring proper separation between isotropic and anisotropic BAO analyses, following cosmological best practices where both types are not typically combined in joint fits.

The implementation builds upon the existing `fit_aniso.py` script but enhances it with optimized parameter integration, improved validation, and better separation from joint fitting configurations.

## Architecture

### Component Integration

The BAO anisotropic fit integrates with the existing pipeline architecture:

```
fit_bao_aniso.py (new)
    ├── fit_core.engine (existing)
    ├── fit_core.parameter_store (existing) 
    ├── fit_core.datasets (existing)
    ├── fit_core.likelihoods (existing)
    └── fit_core.integrity (existing)
```

### Data Flow

1. **Parameter Initialization**: Load optimized parameters from `OptimizedParameterStore` or use defaults
2. **Dataset Loading**: Load anisotropic BAO dataset via `datasets.load_dataset("bao_ani")`
3. **Likelihood Computation**: Use `likelihoods.likelihood_bao_ani()` for χ² calculation
4. **Optimization**: Execute fitting via `engine.run_fit()` with `datasets_list=["bao_ani"]`
5. **Results Output**: Format and display results with anisotropic-specific metrics

### Separation from Joint Fits

The design ensures proper separation by:
- Using dedicated dataset identifier `"bao_ani"` (distinct from `"bao"`)
- Preventing simultaneous inclusion of both `"bao"` and `"bao_ani"` in joint fits
- Maintaining separate validation and integrity checks
- Using distinct parameter optimization paths

## Components and Interfaces

### Enhanced fit_bao_aniso.py Script

**Purpose**: Dedicated wrapper for anisotropic BAO fitting with optimized parameter integration

**Key Features**:
- Integration with `OptimizedParameterStore` for CMB-optimized starting parameters
- Enhanced command-line interface matching other pipeline scripts
- Comprehensive integrity checking with anisotropic-specific validations
- Detailed result reporting with anisotropic BAO metrics

**Interface**:
```python
def run_bao_aniso_fit(
    model: str,
    overrides: Optional[ParameterDict] = None,
    verify_integrity: bool = False,
    integrity_tolerance: float = 1e-4
) -> Dict[str, Any]
```

### Dataset Configuration

**Anisotropic BAO Dataset Structure**:
```python
{
    "observations": {
        "redshift": np.array([z1, z2, ...]),
        "DM_over_rs": np.array([dm1, dm2, ...]),  # Transverse BAO
        "H_times_rs": np.array([h1, h2, ...])    # Radial BAO
    },
    "covariance": np.ndarray,  # 2N x 2N matrix for N redshift bins
    "metadata": {...},
    "dataset_type": "bao_ani"
}
```

### Likelihood Function Enhancement

The existing `likelihood_bao_ani()` function handles:
- Separate transverse (D_M/r_s) and radial (H*r_s) measurements
- Proper covariance structure for correlated measurements
- Redshift-dependent predictions using background cosmology

### Parameter Optimization Integration

**Optimized Parameter Usage**:
- Prioritize CMB-optimized parameters when available
- Fall back to hardcoded defaults if no optimization exists
- Apply user overrides on top of optimized/default values
- Track parameter source for result metadata

## Data Models

### Parameter Dictionary Structure

```python
ParameterDict = {
    # Core cosmological parameters
    "H0": float,      # Hubble constant (km/s/Mpc)
    "Om0": float,     # Matter density fraction
    "Obh2": float,    # Physical baryon density
    "ns": float,      # Scalar spectral index
    
    # PBUF-specific parameters (if model="pbuf")
    "alpha": float,   # Elasticity amplitude
    "Rmax": float,    # Saturation length scale
    "eps0": float,    # Elasticity bias term
    "n_eps": float,   # Evolution exponent
    "k_sat": float,   # Saturation coefficient
    
    # Derived parameters (computed automatically)
    "z_recomb": float,    # Recombination redshift
    "r_s_drag": float,    # Sound horizon at drag epoch
    "model_class": str    # Model identifier
}
```

### Results Dictionary Structure

```python
ResultsDict = {
    "params": ParameterDict,
    "metrics": {
        "total_chi2": float,
        "aic": float,
        "bic": float,
        "dof": int,
        "p_value": float
    },
    "results": {
        "bao_ani": {
            "chi2": float,
            "predictions": {
                "DM_over_rs": np.ndarray,
                "H_times_rs": np.ndarray
            },
            "residuals": np.ndarray,
            "covariance_analysis": Dict[str, Any]
        }
    },
    "parameter_source": {
        "source": str,  # "cmb_optimized" or "defaults"
        "cmb_optimized": bool,
        "overrides_applied": int,
        "override_params": List[str]
    }
}
```

## Error Handling

### Validation Checks

1. **Dataset Validation**: Ensure anisotropic BAO dataset has proper structure
2. **Parameter Validation**: Check parameter ranges and physical consistency
3. **Covariance Validation**: Verify positive definiteness and conditioning
4. **Integrity Validation**: Run physics consistency checks before fitting

### Error Recovery

- **Missing Optimization**: Gracefully fall back to default parameters
- **Invalid Parameters**: Provide clear error messages with valid ranges
- **Numerical Issues**: Handle singular covariance matrices and optimization failures
- **Data Issues**: Validate dataset completeness and format

### Error Reporting

```python
# Error structure for failed fits
{
    "error": str,                    # Error description
    "error_type": str,              # Category of error
    "validation_results": Dict,      # Detailed validation info
    "suggested_fixes": List[str]     # Actionable recommendations
}
```

## Testing Strategy

### Unit Tests

1. **Parameter Loading Tests**: Verify optimized parameter retrieval and fallback
2. **Dataset Loading Tests**: Validate anisotropic BAO dataset structure
3. **Likelihood Tests**: Check χ² calculation accuracy
4. **Integration Tests**: Test full fitting pipeline

### Validation Tests

1. **Parity Tests**: Compare results against existing `fit_aniso.py` implementation
2. **Physics Tests**: Verify consistency with cosmological expectations
3. **Numerical Tests**: Check optimization convergence and stability

### Integration Tests

1. **Pipeline Integration**: Test with existing fit_core infrastructure
2. **Parameter Store Integration**: Verify optimized parameter usage
3. **Joint Fit Separation**: Ensure proper isolation from joint fitting

### Performance Tests

1. **Execution Time**: Benchmark fitting performance
2. **Memory Usage**: Monitor resource consumption
3. **Convergence**: Test optimization reliability

## Implementation Notes

### Existing Code Reuse

The design maximizes reuse of existing infrastructure:
- `fit_core.engine.run_fit()` for optimization
- `fit_core.likelihoods.likelihood_bao_ani()` for χ² computation
- `fit_core.datasets.load_dataset()` for data loading
- `fit_core.parameter_store.OptimizedParameterStore` for parameter management

### Key Enhancements

1. **Optimized Parameter Integration**: Unlike the current `fit_aniso.py`, the new implementation will integrate with the parameter optimization system
2. **Enhanced Validation**: More comprehensive integrity checking specific to anisotropic BAO
3. **Better Separation**: Explicit prevention of mixing with isotropic BAO in joint fits
4. **Improved Reporting**: Detailed parameter source tracking and validation metadata

### Compatibility

The implementation maintains backward compatibility with:
- Existing command-line interfaces
- Current dataset formats
- Established result structures
- Integration with other pipeline components