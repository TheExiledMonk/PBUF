# SN Ia Fitting Module

This module provides tools for fitting cosmological models to Type Ia supernova distance modulus data.

## Installation Requirements

```bash
pip install pandas numpy scipy
```

## Usage

```python
from cosmos.fits.sn import fit_sn, chi_squared_sn, compute_sn_mu_model
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF

# Create a model
model = LCDM(omega_m=0.3, omega_lambda=0.7, h=0.7)

# Compute predicted distance modulus
z = [0.1, 0.2, 0.3]
mu_pred = compute_sn_mu_model(model, z)

# Fit model parameters to SN data
result = fit_sn(model_type="lcdm")
print(f"Best-fit parameters: {result['params']}")
print(f"χ² = {result['chi2']}")

# Or fit PBUF model
result_pbuf = fit_sn(model_type="pbuf")
print(f"PBUF best-fit parameters: {result_pbuf['params']}")
```

## API Reference

### `fit_sn(model_type="lcdm", initial_params=None, bounds=None, fit_M=False, M_init=0.0)`

Fit cosmological model parameters to supernova data.

**Parameters:**
- `model_type`: "lcdm" or "pbuf"
- `initial_params`: Starting parameter values (optional)
- `bounds`: Parameter bounds for optimization (optional)
- `fit_M`: Whether to fit absolute magnitude offset as nuisance parameter
- `M_init`: Initial value for M if fit_M=True

**Returns:** Dictionary with fit results including status, χ², best-fit parameters, etc.

### `chi_squared_sn(model, M=0.0, data=None)`

Compute χ² between model predictions and supernova data.

### `compute_sn_mu_model(model, z, M=0.0)`

Compute theoretical distance modulus predictions μ(z) for a cosmological model.

## Data Format

The module expects supernova data in CSV format with columns:
- `redshift`: Redshift values
- `mu`: Distance modulus measurements

Optional covariance matrix should be provided as a separate `.cov` file with the same name as the data file.

Default search paths:
1. `data/supernovae/derived/supernova_index.csv`
2. `data/supernovae/derived/supernova_index.cov` (covariance)
