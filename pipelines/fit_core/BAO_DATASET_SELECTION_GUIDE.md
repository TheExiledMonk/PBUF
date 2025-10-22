# BAO Dataset Selection Guide

This guide documents best practices for selecting between isotropic and anisotropic BAO datasets in cosmological fits, implementing the validation requirements from task 5.

## Overview

The PBUF cosmology pipeline supports two types of BAO (Baryon Acoustic Oscillations) measurements:

- **Isotropic BAO** (`bao`): Spherically averaged BAO measurements (D_V/r_s ratios)
- **Anisotropic BAO** (`bao_ani`): Directional BAO measurements (D_M/r_s and D_H/r_s)

**Critical Rule**: These datasets should **never** be used together in the same fit, as they measure the same physical scale and would constitute double-counting.

## Dataset Selection Rules

### ✅ Valid Configurations

#### Standard Joint Fits (Recommended)
```bash
# Full cosmological constraints
python pipelines/fit_joint.py --datasets cmb bao sn

# Distance + early universe constraints  
python pipelines/fit_joint.py --datasets cmb bao

# Distance ladder consistency
python pipelines/fit_joint.py --datasets bao sn
```

#### Dedicated Anisotropic Analysis
```bash
# Standalone anisotropic BAO analysis
python pipelines/fit_bao_aniso.py --model pbuf
```

#### Single Dataset Analysis (Specialized)
```bash
# CMB-only analysis
python pipelines/fit_cmb.py --model lcdm

# Supernova-only analysis  
python pipelines/fit_sn.py --model lcdm
```

### ❌ Invalid Configurations

#### Mixed BAO Types (Forbidden)
```bash
# ❌ NEVER DO THIS - Double counts BAO information
python pipelines/fit_joint.py --datasets bao bao_ani

# ❌ NEVER DO THIS - Mixes analysis types
python pipelines/fit_joint.py --datasets cmb bao bao_ani sn
```

## When to Use Each Dataset Type

### Isotropic BAO (`bao`)

**Use for:**
- Joint fits with CMB and/or supernova data
- Standard cosmological parameter constraints
- Robust distance scale measurements
- When you need spherically averaged BAO information
- Most cosmological analyses

**Advantages:**
- Well-established in joint fits
- Robust against systematic effects
- Smaller covariance matrices
- Standard in cosmological literature

**Example Use Cases:**
- Constraining H₀ and Ωₘ jointly with CMB
- Testing ΛCDM vs alternative models
- Precision cosmology with multiple probes

### Anisotropic BAO (`bao_ani`)

**Use for:**
- Dedicated anisotropic BAO analysis
- Separate transverse/radial BAO constraints
- Testing for anisotropic signatures in the universe
- When you need directional BAO information
- Specialized BAO studies

**Advantages:**
- Provides directional information
- Can test isotropy assumptions
- More detailed BAO physics
- Separate H(z) and D_A(z) constraints

**Example Use Cases:**
- Testing cosmic isotropy
- Detailed BAO physics studies
- Separate Hubble parameter constraints
- Anisotropic cosmology models

## Validation System

The pipeline automatically validates dataset configurations:

### Automatic Checks

1. **Separation Validation**: Prevents simultaneous use of `bao` and `bao_ani`
2. **Configuration Warnings**: Alerts when anisotropic BAO is used in joint fits
3. **Best Practice Recommendations**: Suggests optimal dataset combinations

### Error Messages

When invalid configurations are detected:

```
CRITICAL: Cannot use both isotropic ('bao') and anisotropic ('bao_ani') 
BAO datasets in the same fit.
This violates standard cosmological practice as both forms measure 
the same physical scale.
Choose either:
  - 'bao' for isotropic BAO analysis
  - 'bao_ani' for anisotropic BAO analysis
```

### Warnings

For suboptimal but valid configurations:

```
⚠️  Joint Fit Configuration Warnings:
   • Anisotropic BAO ('bao_ani') is typically analyzed independently. 
     Consider running separate anisotropic BAO fits for cleaner analysis.

💡 Recommendations:
   • Use dedicated anisotropic BAO script: python pipelines/fit_bao_aniso.py
```

## Implementation Details

### Validation Functions

The validation system provides several functions:

```python
from fit_core.bao_aniso_validation import (
    validate_bao_dataset_separation,      # Core validation
    validate_joint_fit_configuration,     # Joint fit guidance
    get_bao_dataset_selection_guide,      # Best practices
    print_dataset_separation_warning      # User warnings
)
```

### Integration Points

Validation is integrated at multiple levels:

1. **Engine Level** (`fit_core/engine.py`): Core validation in `run_fit()`
2. **Joint Fit Script** (`fit_joint.py`): Configuration guidance
3. **Individual Scripts**: Appropriate dataset usage

### Configuration Validation

```python
# Example validation usage
datasets = ["cmb", "bao", "sn"]
try:
    validate_bao_dataset_separation(datasets)
    config = validate_joint_fit_configuration(datasets)
    print("Configuration valid:", config["status"])
except BAOValidationError as e:
    print("Invalid configuration:", e)
```

## Troubleshooting

### Common Issues

#### "Cannot use both isotropic and anisotropic BAO"
**Problem**: Trying to use both `bao` and `bao_ani` datasets
**Solution**: Choose one BAO type based on your analysis goals

#### "Anisotropic BAO in joint fit detected"
**Problem**: Using `bao_ani` with other datasets
**Solution**: Consider dedicated anisotropic analysis with `fit_bao_aniso.py`

#### "Joint fitting requires at least 2 datasets"
**Problem**: Single dataset specified for joint fit
**Solution**: Add complementary datasets or use individual fit scripts

### Best Practice Checklist

- [ ] Use `bao` for standard joint fits
- [ ] Use `bao_ani` for dedicated anisotropic analysis
- [ ] Never mix `bao` and `bao_ani` in the same fit
- [ ] Include at least 2-3 datasets for robust joint fits
- [ ] Consider CMB + BAO + SN for full cosmological constraints
- [ ] Use dedicated scripts for specialized analysis

## References

### Cosmological Motivation

The separation between isotropic and anisotropic BAO is motivated by:

1. **Physical Consistency**: Both measure the same BAO scale
2. **Statistical Independence**: Avoid double-counting information
3. **Analysis Clarity**: Different analysis methods and systematics
4. **Literature Standards**: Established practice in cosmology

### Technical Implementation

- **Requirements**: 1.1, 1.2 from BAO anisotropic fit specification
- **Validation**: Comprehensive error checking and user guidance
- **Integration**: Seamless integration with existing pipeline
- **Testing**: Full test suite with 21 test cases

## Examples

### Recommended Workflow

```bash
# Step 1: Standard cosmological constraints
python pipelines/fit_joint.py --model pbuf --datasets cmb bao sn

# Step 2: Dedicated anisotropic BAO analysis (if needed)
python pipelines/fit_bao_aniso.py --model pbuf

# Step 3: Compare results between isotropic and anisotropic
# (but never combine in same fit)
```

### Advanced Usage

```bash
# Test different dataset combinations
python pipelines/fit_joint.py --datasets cmb bao    # Distance + early universe
python pipelines/fit_joint.py --datasets bao sn     # Distance ladder
python pipelines/fit_joint.py --datasets cmb sn     # No BAO

# Specialized anisotropic studies
python pipelines/fit_bao_aniso.py --model lcdm      # ΛCDM baseline
python pipelines/fit_bao_aniso.py --model pbuf      # PBUF comparison
```

This guide ensures proper dataset selection and prevents common configuration errors while maintaining the scientific integrity of cosmological analyses.