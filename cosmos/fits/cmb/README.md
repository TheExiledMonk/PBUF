# CMB Fitting Package

This package provides a model-agnostic interface for fitting cosmological models to CMB distance priors using Planck 2018 data.

## Features

- **Model-agnostic**: Works with any cosmological model (LCDM, PBUF, or future models)
- **Planck 2018 data**: Uses official compressed distance priors (R, l_A, θ*)
- **SciPy optimization**: Robust parameter fitting with bounds and constraints
- **Minimal dependencies**: Only numpy, scipy, and cosmos models required

## Quick Start

```python
from cosmos.fits import fit_cmb

# Fit LCDM model
result = fit_cmb(model_type="lcdm")
print(f"Best-fit χ²: {result['chi2']}")
print(f"Best-fit params: {result['params']}")

# Fit PBUF model
result = fit_cmb(model_type="pbuf")
print(f"Best-fit χ²: {result['chi2']}")
print(f"Best-fit params: {result['params']}")
```

## API Reference

### `fit_cmb(model_type, initial_params=None, bounds=None)`

Fit cosmological model parameters to Planck CMB distance priors.

**Parameters:**
- `model_type` (str): "lcdm" or "pbuf"
- `initial_params` (dict, optional): Starting parameter values
- `bounds` (dict, optional): Parameter bounds for optimization

**Returns:**
```python
{
    "status": "success",      # "success" or "fail"
    "chi2": 2765.77,         # Best-fit χ²
    "params": {...},         # Best-fit parameters
    "observables": {...},    # Derived CMB observables
}
```

### `compute_cmb_observables(model)`

Compute CMB observables (R, l_A, θ*) for a given model.

**Parameters:**
- `model`: LCDM or PBUF model instance

**Returns:**
```python
{
    "R": 1.749,           # Shift parameter
    "la": 301.73,         # Acoustic scale
    "theta_star": 0.0104  # Angular scale
}
```

### `chi_squared_cmb(model, priors=None)`

Compute χ² between model predictions and Planck data.

**Parameters:**
- `model`: Cosmological model instance
- `priors`: Planck priors dict (loads defaults if None)

## Model Parameters

### LCDM
- `H0`: Hubble constant [km/s/Mpc]
- `Om0`: Matter density parameter
- `Ol0`: Dark energy density parameter

### PBUF
- `H0`: Hubble constant [km/s/Mpc]
- `Om0`: Matter density parameter
- `alpha`: Elastic amplitude parameter
- `Rmax`: Saturation scale factor
- `k_sat`: Rigidity fraction (k_sat > 0; values > 1 delay saturation)

## Data

Planck 2018 distance priors are loaded from `data/priors/planck2018_distance_priors.json`:

- R = 1.7492 ± 0.0188
- l_A = 301.729 ± 0.077
- θ* = 0.0104086 ± 0.0000041

## Examples

See the source code for complete examples of usage with both model types.
