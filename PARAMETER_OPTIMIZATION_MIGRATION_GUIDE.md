# Parameter Optimization Migration Guide

## Overview

This guide helps users transition from the fixed parameter system to the new parameter optimization capabilities in the PBUF cosmology pipeline. The optimization system allows you to find best-fit parameter values rather than using fixed defaults, improving fitting precision and scientific accuracy.

## Key Changes

### Before: Fixed Parameters
```bash
# Old approach - all parameters fixed to defaults
python fit_cmb.py --model pbuf
python fit_bao.py --model pbuf  
python fit_sn.py --model pbuf
```

### After: Optimized Parameters
```bash
# New approach - optimize key parameters first
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0
python fit_bao.py --model pbuf  # Automatically uses optimized values
python fit_sn.py --model pbuf   # Automatically uses optimized values
```

## Migration Steps

### Step 1: Identify Your Current Workflow

**Scenario A: Basic Individual Fits**
```bash
# Current workflow
python fit_cmb.py --model lcdm
python fit_bao.py --model lcdm
python fit_sn.py --model lcdm
```

**Migration:**
```bash
# Optimized workflow
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns
python fit_bao.py --model lcdm  # Uses optimized parameters
python fit_sn.py --model lcdm   # Uses optimized parameters
```

**Scenario B: Joint Fitting**
```bash
# Current workflow
python fit_joint.py --model pbuf --datasets cmb,bao,sn
```

**Migration:**
```bash
# Optimized workflow - optimize CMB first, then joint fit
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0
python fit_joint.py --model pbuf --datasets cmb,bao,sn
```

**Scenario C: Parameter Overrides**
```bash
# Current workflow with overrides
python fit_cmb.py --model lcdm --H0 70.0 --Om0 0.3
```

**Migration:**
```bash
# Optimized workflow with starting values
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns --H0 70.0 --Om0 0.3
```

### Step 2: Choose Parameters to Optimize

#### ΛCDM Model Recommendations

**Conservative (2-3 parameters):**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0
```

**Standard (4 parameters):**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns
```

**Comprehensive (all free parameters):**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns,Neff
```

#### PBUF Model Recommendations

**Conservative (PBUF-specific only):**
```bash
python fit_cmb.py --model pbuf --optimize k_sat,alpha
```

**Standard (PBUF + key cosmological):**
```bash
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0
```

**Comprehensive (all parameters):**
```bash
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0,Obh2,ns
```

### Step 3: Update Configuration Files

**Before (config.json):**
```json
{
  "model": "pbuf",
  "datasets": ["cmb", "bao", "sn"],
  "parameters": {
    "H0": 67.4,
    "Om0": 0.315,
    "alpha": 5e-4
  }
}
```

**After (config.json):**
```json
{
  "model": "pbuf",
  "datasets": ["cmb", "bao", "sn"],
  "optimization": {
    "optimize_parameters": ["k_sat", "alpha", "H0", "Om0"],
    "covariance_scaling": 1.0,
    "warm_start": true,
    "save_results": true
  },
  "parameters": {
    "alpha": 5e-4
  }
}
```

### Step 4: Validate Migration

**Test Backward Compatibility:**
```bash
# Ensure old workflows still work
python fit_cmb.py --model lcdm  # Should work exactly as before
```

**Compare Results:**
```bash
# Run both approaches and compare
python fit_cmb.py --model lcdm --output old_results.json
python fit_cmb.py --model lcdm --optimize H0,Om0 --output new_results.json

# Compare χ² values
python -c "
import json
with open('old_results.json') as f: old = json.load(f)
with open('new_results.json') as f: new = json.load(f)
print(f'Fixed parameters χ²: {old[\"metrics\"][\"total_chi2\"]:.3f}')
print(f'Optimized parameters χ²: {new[\"metrics\"][\"total_chi2\"]:.3f}')
print(f'Improvement: {old[\"metrics\"][\"total_chi2\"] - new[\"metrics\"][\"total_chi2\"]:.3f}')
"
```

## Best Practices

### 1. Parameter Selection Strategy

**Start Conservative:**
- Begin with 1-2 key parameters
- Verify optimization converges reliably
- Gradually add more parameters

**Model-Specific Guidelines:**

**ΛCDM Priority Order:**
1. `H0, Om0` - Core cosmological parameters
2. `Obh2, ns` - CMB-sensitive parameters  
3. `Neff` - Advanced parameter (use with caution)

**PBUF Priority Order:**
1. `k_sat, alpha` - PBUF-specific parameters
2. `H0, Om0` - Shared cosmological parameters
3. `Obh2, ns` - CMB parameters
4. `Rmax, eps0, n_eps` - Advanced PBUF parameters

### 2. Optimization Workflow Best Practices

**Sequential Optimization:**
```bash
# 1. Optimize CMB parameters first (most constraining)
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0

# 2. Other datasets automatically use optimized values
python fit_bao.py --model pbuf
python fit_sn.py --model pbuf

# 3. Joint fit benefits from pre-optimized individual parameters
python fit_joint.py --model pbuf --datasets cmb,bao,sn
```

**Use Warm Start for Iterative Work:**
```bash
# First optimization
python fit_cmb.py --model pbuf --optimize k_sat,alpha

# Later refinements use previous results as starting point
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0 --warm-start
```

**Test with Dry Run:**
```bash
# Test optimization without saving results
python fit_cmb.py --model pbuf --optimize k_sat,alpha --dry-run
```

### 3. Covariance Scaling Guidelines

**Conservative Analysis (tighter constraints):**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0 --cov-scale 0.8
```

**Standard Analysis:**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0  # Default scaling = 1.0
```

**Sensitivity Testing (looser constraints):**
```bash
python fit_cmb.py --model lcdm --optimize H0,Om0 --cov-scale 1.2
```

### 4. Quality Assurance Checklist

**Before Optimization:**
- [ ] Verify parameter names are correct for the model
- [ ] Check that starting values are reasonable
- [ ] Ensure datasets are properly loaded and validated

**During Optimization:**
- [ ] Monitor convergence status
- [ ] Check for parameters hitting bounds
- [ ] Verify χ² improvement is significant

**After Optimization:**
- [ ] Validate optimized parameters are physically reasonable
- [ ] Check cross-model consistency (for shared parameters)
- [ ] Verify subsequent fits use optimized values

## Common Migration Scenarios

### Scenario 1: Research Group with Existing Scripts

**Challenge:** Multiple scripts and workflows using fixed parameters

**Solution:**
```bash
# Create optimization wrapper script
cat > optimize_and_fit.sh << 'EOF'
#!/bin/bash
MODEL=$1
DATASETS=$2

# Optimize CMB parameters first
if [ "$MODEL" = "lcdm" ]; then
    python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns
elif [ "$MODEL" = "pbuf" ]; then
    python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0
fi

# Run original workflow with optimized parameters
python fit_joint.py --model $MODEL --datasets $DATASETS
EOF

chmod +x optimize_and_fit.sh

# Use wrapper in existing workflows
./optimize_and_fit.sh pbuf "cmb,bao,sn"
```

### Scenario 2: Automated Analysis Pipeline

**Challenge:** Automated pipeline needs to handle optimization

**Solution:**
```python
# pipeline.py - Updated automated pipeline
import subprocess
import json
from pathlib import Path

def run_optimized_analysis(model, datasets):
    """Run analysis with automatic optimization."""
    
    # Step 1: Optimize CMB parameters
    if model == "lcdm":
        optimize_params = ["H0", "Om0", "Obh2", "ns"]
    elif model == "pbuf":
        optimize_params = ["k_sat", "alpha", "H0", "Om0"]
    
    cmd = [
        "python", "fit_cmb.py",
        "--model", model,
        "--optimize", ",".join(optimize_params)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Optimization failed: {result.stderr}")
        return None
    
    # Step 2: Run full analysis with optimized parameters
    cmd = [
        "python", "fit_joint.py",
        "--model", model,
        "--datasets", ",".join(datasets),
        "--output", f"{model}_optimized_results.json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return f"{model}_optimized_results.json"
    else:
        print(f"Analysis failed: {result.stderr}")
        return None

# Usage
results_file = run_optimized_analysis("pbuf", ["cmb", "bao", "sn"])
if results_file:
    print(f"Analysis complete: {results_file}")
```

### Scenario 3: Comparative Model Studies

**Challenge:** Comparing multiple models fairly

**Solution:**
```bash
# comparative_study.sh
#!/bin/bash

echo "Running comparative model study with optimization..."

# Optimize all models with same datasets
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0

# Run joint fits with optimized parameters
python fit_joint.py --model lcdm --datasets cmb,bao,sn --output lcdm_optimized.json
python fit_joint.py --model pbuf --datasets cmb,bao,sn --output pbuf_optimized.json

# Compare results
python -c "
import json

with open('lcdm_optimized.json') as f: lcdm = json.load(f)
with open('pbuf_optimized.json') as f: pbuf = json.load(f)

print('Model Comparison (Optimized Parameters):')
print(f'ΛCDM χ²: {lcdm[\"metrics\"][\"total_chi2\"]:.3f}')
print(f'PBUF χ²: {pbuf[\"metrics\"][\"total_chi2\"]:.3f}')
print(f'ΔAIC: {pbuf[\"metrics\"][\"aic\"] - lcdm[\"metrics\"][\"aic\"]:.3f}')
"
```

## Reproducibility and Provenance

### Version Control Integration

**Track Optimization Settings:**
```bash
# Add optimization configuration to version control
git add optimization_config.json
git commit -m "Add CMB parameter optimization configuration"

# Tag optimization results
git tag -a v1.0-optimized -m "Results with optimized parameters"
```

**Document Optimization Metadata:**
```python
# Check optimization provenance
from pipelines.fit_core.parameter_store import OptimizedParameterStore

store = OptimizedParameterStore()
params = store.get_model_defaults("pbuf")

if 'optimization_metadata' in params:
    metadata = params['optimization_metadata']
    print("Optimization Provenance:")
    print(f"  Timestamp: {metadata.get('cmb_optimized')}")
    print(f"  Method: {metadata.get('optimizer_info', {}).get('method')}")
    print(f"  Library: {metadata.get('optimizer_info', {}).get('library')}")
    print(f"  Version: {metadata.get('optimizer_info', {}).get('version')}")
    print(f"  χ² improvement: {metadata.get('chi2_improvement')}")
```

### Checksums and Validation

**Generate Analysis Checksums:**
```python
import hashlib
import json

def generate_analysis_checksum(results_file):
    """Generate checksum for analysis results."""
    
    with open(results_file) as f:
        data = json.load(f)
    
    # Extract key results for checksum
    checksum_data = {
        "model": data["params"]["model_class"],
        "total_chi2": round(data["metrics"]["total_chi2"], 6),
        "parameters": {k: round(v, 6) for k, v in data["params"].items() 
                     if isinstance(v, (int, float))}
    }
    
    # Generate checksum
    checksum_str = json.dumps(checksum_data, sort_keys=True)
    checksum = hashlib.sha256(checksum_str.encode()).hexdigest()[:16]
    
    return checksum

# Usage
checksum = generate_analysis_checksum("pbuf_results.json")
print(f"Analysis checksum: {checksum}")

# Store checksum with results
with open("pbuf_results.json") as f:
    results = json.load(f)

results["provenance"] = {
    "checksum": checksum,
    "timestamp": "2025-10-22T14:42:00Z",
    "optimization_used": True
}

with open("pbuf_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Reproducibility Validation

**Create Reproducibility Test:**
```python
# reproducibility_test.py
import subprocess
import json
import numpy as np

def test_optimization_reproducibility(model, optimize_params, tolerance=1e-8):
    """Test that optimization produces reproducible results."""
    
    results = []
    
    for run in range(3):  # Run optimization 3 times
        cmd = [
            "python", "fit_cmb.py",
            "--model", model,
            "--optimize", ",".join(optimize_params),
            "--output", f"test_run_{run}.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            with open(f"test_run_{run}.json") as f:
                data = json.load(f)
                results.append(data["metrics"]["total_chi2"])
        else:
            print(f"Run {run} failed: {result.stderr}")
            return False
    
    # Check reproducibility
    if len(results) == 3:
        chi2_std = np.std(results)
        if chi2_std < tolerance:
            print(f"✓ Optimization reproducible (σ_χ² = {chi2_std:.2e})")
            return True
        else:
            print(f"❌ Optimization not reproducible (σ_χ² = {chi2_std:.2e})")
            return False
    
    return False

# Test ΛCDM reproducibility
test_optimization_reproducibility("lcdm", ["H0", "Om0"])

# Test PBUF reproducibility  
test_optimization_reproducibility("pbuf", ["k_sat", "alpha"])
```

## Troubleshooting Migration Issues

### Issue: Results Different from Fixed Parameters

**Diagnosis:**
```python
# Compare fixed vs optimized results
fixed_result = run_fit("pbuf", ["cmb"])  # Uses fixed defaults
optimized_result = run_fit("pbuf", ["cmb"], optimization_params=["k_sat", "alpha"])

print("Parameter Comparison:")
for param in ["k_sat", "alpha", "H0", "Om0"]:
    fixed_val = fixed_result["params"][param]
    opt_val = optimized_result["params"][param]
    change = abs(opt_val - fixed_val) / fixed_val * 100
    print(f"{param}: {fixed_val:.6f} → {opt_val:.6f} ({change:.2f}% change)")

print(f"χ² improvement: {fixed_result['metrics']['total_chi2'] - optimized_result['metrics']['total_chi2']:.3f}")
```

**Expected:** Optimization should improve χ² and may change parameter values significantly.

### Issue: Optimization Not Converging

**Solution:**
```bash
# Use warm start and covariance scaling
python fit_cmb.py --model pbuf --optimize k_sat --warm-start --cov-scale 1.2

# Or try fewer parameters initially
python fit_cmb.py --model pbuf --optimize k_sat
python fit_cmb.py --model pbuf --optimize k_sat,alpha --warm-start
```

### Issue: Cross-Model Inconsistency

**Diagnosis:**
```python
from pipelines.fit_core.parameter_store import OptimizedParameterStore

store = OptimizedParameterStore()
divergence = store.validate_cross_model_consistency()

for param, div in divergence.items():
    if div > 1e-3:
        print(f"Large divergence in {param}: {div:.6f}")
```

**Solution:**
```bash
# Re-optimize both models with same datasets
python fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns
python fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0
```

## Advanced Migration Topics

### Custom Optimization Bounds

**Scenario:** Need tighter parameter constraints

**Solution:**
```python
# Custom bounds in configuration
config = {
    "optimization": {
        "optimize_parameters": ["H0", "Om0"],
        "parameter_bounds": {
            "H0": [65.0, 75.0],  # Tighter than default [20.0, 150.0]
            "Om0": [0.25, 0.35]   # Tighter than default [0.01, 0.99]
        }
    }
}
```

### Multi-Dataset Optimization Strategy

**Scenario:** Different datasets prefer different parameters

**Strategy:**
```bash
# Sequential optimization by dataset sensitivity
python fit_cmb.py --model pbuf --optimize H0,Om0,Obh2,ns  # CMB-sensitive
python fit_bao.py --model pbuf --optimize k_sat           # BAO-sensitive  
python fit_sn.py --model pbuf                             # Use all optimized
```

### Integration with Existing Analysis Tools

**Scenario:** Integration with external analysis scripts

**Solution:**
```python
# analysis_wrapper.py
def run_legacy_analysis_with_optimization(model, datasets):
    """Wrapper to add optimization to existing analysis."""
    
    # Step 1: Run optimization
    if model == "pbuf":
        optimize_params = ["k_sat", "alpha"]
    else:
        optimize_params = ["H0", "Om0"]
    
    run_fit(model, ["cmb"], optimization_params=optimize_params)
    
    # Step 2: Run existing analysis (now uses optimized parameters)
    return run_legacy_analysis(model, datasets)
```

This migration guide provides a comprehensive path for transitioning to the new parameter optimization system while maintaining backward compatibility and ensuring reproducible, scientifically sound results.