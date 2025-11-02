# RSD Fitting Module

The Redshift-Space Distortion (RSD) fitting module provides tools to fit cosmological models to fσ8(z) measurements, which probe the growth rate of cosmic structure.

## Overview

RSD measures the anisotropy in the galaxy clustering pattern caused by peculiar velocities along the line of sight. The key observable is fσ8(z), the product of the growth rate f(z) and the amplitude of matter fluctuations σ8(z).

## Module Structure

```
cosmos/fits/rsd/
├── __init__.py       # Module interface
├── data_loader.py    # Load RSD datasets
├── observables.py    # Compute theoretical fσ8(z)
├── chi2.py          # χ² calculation
└── optimizer.py     # Parameter fitting
```

## Usage

### Basic Example

```python
from cosmos.fits.rsd import fit_rsd

# Fit LCDM model to RSD data
result_lcdm = fit_rsd(model_type="lcdm")
print(f"LCDM χ²: {result_lcdm['chi2']:.3f}")
print(f"Best-fit parameters: {result_lcdm['params']}")

# Fit PBUF model to RSD data
result_pbuf = fit_rsd(model_type="pbuf")
print(f"PBUF χ²: {result_pbuf['chi2']:.3f}")
print(f"Best-fit parameters: {result_pbuf['params']}")
```

### Custom Parameters

```python
# Fit with custom initial parameters and bounds
initial_params = {"H0": 70.0, "Om0": 0.3, "Ol0": 0.7}
bounds = {"H0": (60, 80), "Om0": (0.2, 0.4), "Ol0": (0.6, 0.8)}

result = fit_rsd(
    model_type="lcdm",
    initial_params=initial_params,
    bounds=bounds
)
```

### Manual Computation

```python
from cosmos.lcdm.model import LCDM
from cosmos.fits.rsd import compute_rsd_observable, chi_squared_rsd

# Create a model
model = LCDM(omega_m=0.315, omega_lambda=0.685, h=0.675)

# Compute theoretical predictions
z_values = [0.15, 0.32, 0.57]
fs8_pred = compute_rsd_observable(model, z_values, sigma8_0=0.8)

# Compute χ²
chi2 = chi_squared_rsd(model)
```

## Data Format

The module expects RSD data in CSV format with columns:
- `z`: redshift
- `fsigma8`: measured fσ8 value
- `sigma_fsigma8`: 1σ uncertainty

Example data file (`data/rsd/rsd_data.csv`):
```csv
z,fsigma8,sigma_fsigma8,survey
0.15,0.49,0.14,SDSS
0.22,0.42,0.07,BOSS
0.25,0.35,0.06,SDSS
...
```

## Theory

The module evaluates the linear growth factor by integrating:

    d²D/d(ln a)² + [2 + dlnH/dln a] dD/dln a - 3Ω_m(a)D/2 = 0

and computes fσ₈(z) = (d ln D / d ln a) × σ₈(0) × D(z) without assuming
an LCDM-specific growth index. This keeps the predictions consistent with
both LCDM and PBUF expansion histories.

The observable fσ8(z) is then:
fσ8(z) = f(z) × σ8(z) = f(z) × σ8,0 × D(z)/D(0)

where D(z) is the growth factor normalized to D(0) = 1.

## Notes

- The module uses Planck 2018 default σ8,0 = 0.811
- Parameter bounds are set to physically reasonable ranges
- The optimization uses L-BFGS-B method from scipy.optimize
- Both LCDM and PBUF models are supported
