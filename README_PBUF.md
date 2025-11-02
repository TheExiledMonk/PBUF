# PBUF (Planck-Bound Unified Framework) Implementation

This directory contains a complete implementation of the PBUF cosmology model as specified.

## Overview

PBUF implements a cosmology where late-time acceleration is sourced by an elastic spacetime response instead of a cosmological constant. The key features are:

- **No Λ term**: Pure Friedmann equation with elastic sector
- **Elastic sector**: Ω_σ(a) = σ_eff(z)·S(z; k_sat) + Δ_rad(z; k_sat)
- **Rigidity control**: k_sat parameter (k_sat > 0, values > 1 allowed)
- **Standard components**: Matter, radiation, curvature terms

## File Structure

```
cosmos/
├── __init__.py          # Package initialization
├── pbuf/                # PBUF cosmology package
│   ├── __init__.py      # PBUF package exports
│   ├── model.py         # Main PBUF class
│   ├── equations.py     # Core background equations
│   ├── validators.py    # Parameter validation
│   └── utils.py         # Numerical utilities
└── helper/              # Shared utilities
    ├── constants.py     # Physical constants
    ├── units.py         # Unit conversions
    └── guards.py        # Validation functions
```

## Key Components

### PBUF Class (model.py)
Main cosmology class with methods for:
- Expansion history: `H(z)`, `hubble_function(a)`
- Elastic sector: `omega_sigma(a)`, `elastic_energy_density(z)`
- Density parameters: `density_parameters_at_z(z)`
- Parameter validation and inference

### Core Equations (equations.py)
- `omega_sigma_raw(a, params)`: Positive elastic reservoir σ_eff·S
- `omega_sigma_total(a, params)`: Full Ω_σ(a) contribution
- `H_pbuf_a(a, ...)`: Hubble parameter vs scale factor
- `H_pbuf_z(z, ...)`: Hubble parameter vs redshift
- `elastic_fraction(a, ...)`: Fraction of expansion from elasticity

### Validation (validators.py)
- Parameter bounds checking
- Physical constraint enforcement (k_sat > 0, etc.)
- Scale factor and expansion rate validation

## Usage Example

```python
from cosmos.pbuf import PBUF

# Create PBUF instance
pbuf = PBUF(
    omega_m=0.3,    # Matter density
    h=0.7,          # Hubble parameter
    alpha=0.7,      # Elastic amplitude
    Rmax=0.5,       # Saturation scale
    k_sat=1.5,      # Rigidity fraction
    omega_k=0.0,    # Curvature
    omega_r=None    # Inferred from T_cmb
)

# Use cosmology
H0 = pbuf.H(0)  # Hubble parameter today
omega_sigma_today = pbuf.omega_sigma(1.0)  # Elastic contribution today
```

## Key Features Implemented

✅ **Pure PBUF equations** (no Λ term)
✅ **k_sat > 0 freedom** with validation
✅ **Identical naming** to specification
✅ **Parameter validation** with physical bounds
✅ **Standalone equation functions** for direct use
✅ **Complete separation** from LCDM (no inheritance)
✅ **Radiation density inference** from T_cmb

The implementation is ready for use in cosmological calculations, distance computations, and CMB observables.

## Grid-Based Optimization

For joint dataset scoring and LCDM/PBUF comparisons, see `README_GRID_OPTIMIZER.md`.
It documents the deterministic grid evaluator (with binary physics validation)
that replaces the legacy multi-stage survivor pipeline and explains how to run
it from the CLI.
