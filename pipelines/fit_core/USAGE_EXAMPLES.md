# PBUF Cosmology Pipeline - Usage Examples

## Basic Usage Examples

### 1. Parameter Optimization Examples

#### CMB Parameter Optimization for ΛCDM
```python
from pipelines.fit_core.engine import run_fit

# Optimize core ΛCDM parameters using CMB data
result = run_fit(
    model="lcdm",
    datasets_list=["cmb"],
    optimization_params=["H0", "Om0", "Obh2", "ns"]
)

print("ΛCDM CMB Optimization Results:")
print(f"Optimized H₀: {result['params']['H0']:.2f} km/s/Mpc")
print(f"Optimized Ωₘ: {result['params']['Om0']:.4f}")
print(f"χ² improvement: {result['optimization']['chi2_improvement']:.3f}")
print(f"Convergence: {result['optimization']['convergence_status']}")
```

#### PBUF Parameter Optimization
```python
# Optimize PBUF-specific parameters
result = run_fit(
    model="pbuf",
    datasets_list=["cmb"],
    optimization_params=["k_sat", "alpha", "H0", "Om0"]
)

print("PBUF CMB Optimization Results:")
print(f"Optimized k_sat: {result['params']['k_sat']:.4f}")
print(f"Optimized α: {result['params']['alpha']:.2e}")
print(f"χ² improvement: {result['optimization']['chi2_improvement']:.3f}")

# Check if parameters hit bounds
if result['optimization']['bounds_reached']:
    print(f"Parameters at bounds: {result['optimization']['bounds_reached']}")
```

#### Using Optimized Parameters in Subsequent Fits
```python
# Step 1: Optimize parameters using CMB
cmb_result = run_fit(
    model="pbuf",
    datasets_list=["cmb"],
    optimization_params=["k_sat", "alpha"]
)

# Step 2: BAO fitting automatically uses optimized parameters
bao_result = run_fit(
    model="pbuf",
    datasets_list=["bao"]
)

# Step 3: Joint fitting benefits from pre-optimized parameters
joint_result = run_fit(
    model="pbuf",
    datasets_list=["cmb", "bao", "sn"]
)

print(f"CMB χ²: {cmb_result['metrics']['total_chi2']:.3f}")
print(f"BAO χ² (with optimized params): {bao_result['metrics']['total_chi2']:.3f}")
print(f"Joint χ²: {joint_result['metrics']['total_chi2']:.3f}")
```

#### Optimization with Covariance Scaling
```python
# Test sensitivity to covariance matrix scaling
scaling_factors = [0.8, 1.0, 1.2, 1.5]
results = []

for scale in scaling_factors:
    result = run_fit(
        model="lcdm",
        datasets_list=["cmb"],
        optimization_params=["H0", "Om0"],
        covariance_scaling=scale
    )
    
    results.append({
        "scaling": scale,
        "H0": result['params']['H0'],
        "Om0": result['params']['Om0'],
        "chi2": result['metrics']['total_chi2']
    })

# Analyze scaling sensitivity
import pandas as pd
df = pd.DataFrame(results)
print("Covariance Scaling Analysis:")
print(df)
```

#### Dry Run Mode for Testing
```python
# Test optimization without updating stored defaults
result = run_fit(
    model="pbuf",
    datasets_list=["cmb"],
    optimization_params=["k_sat", "alpha"],
    dry_run=True
)

print("Dry Run Results (not saved):")
print(f"Would optimize k_sat to: {result['params']['k_sat']:.4f}")
print(f"Would optimize α to: {result['params']['alpha']:.2e}")
print(f"χ² improvement: {result['optimization']['chi2_improvement']:.3f}")
```

### 2. Simple Individual Fitting

#### CMB-only ΛCDM Fit
```python
from pipelines.fit_core.engine import run_fit

# Basic CMB fitting with ΛCDM
result = run_fit(
    model="lcdm",
    datasets_list=["cmb"],
    mode="individual"
)

print(f"Best-fit H₀: {result['params']['H0']:.2f} km/s/Mpc")
print(f"χ²: {result['metrics']['total_chi2']:.3f}")
print(f"AIC: {result['metrics']['aic']:.3f}")
```

#### BAO-only PBUF Fit
```python
# BAO fitting with PBUF model
result = run_fit(
    model="pbuf",
    datasets_list=["bao"],
    mode="individual"
)

print(f"Best-fit α: {result['params']['alpha']:.2e}")
print(f"Best-fit k_sat: {result['params']['k_sat']:.4f}")
```

### 2. Joint Fitting Examples

#### ΛCDM Joint Fit (CMB + BAO + SN)
```python
# Comprehensive ΛCDM analysis
result = run_fit(
    model="lcdm",
    datasets_list=["cmb", "bao", "sn"],
    mode="joint"
)

# Print results breakdown
print("ΛCDM Joint Fit Results:")
print(f"Total χ²: {result['metrics']['total_chi2']:.3f}")
print(f"DOF: {result['metrics']['dof']}")
print(f"Reduced χ²: {result['metrics']['reduced_chi2']:.3f}")

# Individual contributions
for dataset in ["cmb", "bao", "sn"]:
    chi2 = result['results'][dataset]['chi2']
    print(f"{dataset.upper()} χ²: {chi2:.3f}")
```

#### PBUF vs ΛCDM Comparison
```python
# Compare PBUF and ΛCDM models
lcdm_result = run_fit("lcdm", ["cmb", "bao", "sn"], "joint")
pbuf_result = run_fit("pbuf", ["cmb", "bao", "sn"], "joint")

# Model comparison
from pipelines.fit_core.statistics import delta_aic

delta_aic_value = delta_aic(
    pbuf_result['metrics']['aic'],
    lcdm_result['metrics']['aic']
)

print(f"ΛCDM AIC: {lcdm_result['metrics']['aic']:.3f}")
print(f"PBUF AIC: {pbuf_result['metrics']['aic']:.3f}")
print(f"ΔAIC (PBUF - ΛCDM): {delta_aic_value:.3f}")

if delta_aic_value < -2:
    print("PBUF is strongly preferred")
elif delta_aic_value < 2:
    print("Models are comparable")
else:
    print("ΛCDM is preferred")
```

### 3. Parameter Override Examples

#### Custom Parameter Values
```python
# Fit with specific parameter values
custom_params = {
    "H0": 70.0,        # Hubble constant
    "Om0": 0.3,        # Matter density
    "Obh2": 0.022,     # Baryon density
    "ns": 0.96         # Spectral index
}

result = run_fit(
    model="lcdm",
    datasets_list=["cmb"],
    overrides=custom_params
)
```

#### PBUF Parameter Exploration
```python
# Explore different PBUF parameter values
pbuf_configs = [
    {"alpha": 1e-4, "eps0": 0.5},
    {"alpha": 5e-4, "eps0": 0.7},
    {"alpha": 1e-3, "eps0": 0.9}
]

results = []
for config in pbuf_configs:
    result = run_fit(
        model="pbuf",
        datasets_list=["cmb", "bao"],
        overrides=config
    )
    results.append({
        "config": config,
        "chi2": result['metrics']['total_chi2'],
        "aic": result['metrics']['aic']
    })

# Find best configuration
best = min(results, key=lambda x: x['aic'])
print(f"Best configuration: {best['config']}")
print(f"Best AIC: {best['aic']:.3f}")
```

## Advanced Usage Examples

### 4. Custom Optimization Configuration

#### Using Differential Evolution
```python
# Global optimization with differential evolution
optimizer_config = {
    "method": "differential_evolution",
    "options": {
        "maxiter": 1000,
        "seed": 42,
        "popsize": 15
    }
}

result = run_fit(
    model="pbuf",
    datasets_list=["cmb", "bao", "sn"],
    optimizer_config=optimizer_config
)
```

#### Parameter Bounds
```python
# Constrained optimization with bounds
optimizer_config = {
    "method": "minimize",
    "bounds": {
        "H0": (60, 80),
        "Om0": (0.2, 0.4),
        "alpha": (1e-5, 1e-2)
    }
}

result = run_fit(
    model="pbuf",
    datasets_list=["cmb"],
    optimizer_config=optimizer_config
)
```

### 5. Physics Validation Examples

#### Comprehensive Integrity Checks
```python
from pipelines.fit_core.integrity import run_integrity_suite
from pipelines.fit_core.parameter import build_params

# Build parameters and run full validation
params = build_params("pbuf")
integrity_results = run_integrity_suite(params, ["cmb", "bao", "sn"])

print("Integrity Check Results:")
print(f"H(z) ratios: {integrity_results['h_ratios']['status']}")
print(f"Recombination: {integrity_results['recombination']['status']}")
print(f"Covariance matrices: {integrity_results['covariance']['status']}")
print(f"Overall status: {integrity_results['overall_status']}")
```

#### Custom H(z) Ratio Testing
```python
from pipelines.fit_core.integrity import verify_h_ratios

# Test H(z) ratios at specific redshifts
test_redshifts = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
params = build_params("pbuf", overrides={"k_sat": 0.99})

is_consistent = verify_h_ratios(
    params, 
    redshifts=test_redshifts,
    tolerance=1e-3
)

print(f"H(z) consistency (k_sat=0.99): {is_consistent}")
```

### 6. Data Analysis Examples

#### Residual Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

# Fit model and analyze residuals
result = run_fit("lcdm", ["cmb"], "individual")

cmb_results = result['results']['cmb']
residuals = cmb_results['residuals']
predictions = cmb_results['predictions']

# Plot residuals
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(residuals, bins=20, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.subplot(1, 2, 2)
plt.scatter(range(len(residuals)), residuals)
plt.xlabel('Data Point')
plt.ylabel('Residual')
plt.title('Residuals vs Data Point')
plt.tight_layout()
plt.show()
```

#### Parameter Correlation Analysis
```python
# Run multiple fits with parameter variations
import numpy as np

h0_values = np.linspace(65, 75, 11)
om0_values = []
chi2_values = []

for h0 in h0_values:
    result = run_fit(
        model="lcdm",
        datasets_list=["cmb", "bao"],
        overrides={"H0": h0}
    )
    om0_values.append(result['params']['Om0'])
    chi2_values.append(result['metrics']['total_chi2'])

# Plot correlation
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(h0_values, om0_values, 'o-')
plt.xlabel('H₀ (km/s/Mpc)')
plt.ylabel('Ωₘ')
plt.title('H₀ - Ωₘ Correlation')

plt.subplot(1, 2, 2)
plt.plot(h0_values, chi2_values, 'o-')
plt.xlabel('H₀ (km/s/Mpc)')
plt.ylabel('χ²')
plt.title('χ² vs H₀')
plt.tight_layout()
plt.show()
```

## Parameter Optimization Command Line Examples

### 7. Optimization Command Line Usage

#### Basic Parameter Optimization
```bash
# Optimize PBUF parameters for CMB
python pipelines/fit_cmb.py --model pbuf --optimize k_sat,alpha

# Optimize ΛCDM core parameters
python pipelines/fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns

# Optimize with covariance scaling
python pipelines/fit_cmb.py --model pbuf --optimize k_sat --cov-scale 1.2

# Dry run (don't save results)
python pipelines/fit_cmb.py --model lcdm --optimize H0,Om0 --dry-run

# Use warm start from recent optimization
python pipelines/fit_cmb.py --model pbuf --optimize k_sat,alpha --warm-start
```

#### Optimization Workflow Examples
```bash
# Complete optimization workflow
# Step 1: Optimize CMB parameters
python pipelines/fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0

# Step 2: BAO automatically uses optimized parameters
python pipelines/fit_bao.py --model pbuf

# Step 3: SN automatically uses optimized parameters  
python pipelines/fit_sn.py --model pbuf

# Step 4: Joint fit with all optimized parameters
python pipelines/fit_joint.py --model pbuf --datasets cmb,bao,sn
```

#### Configuration File with Optimization
```bash
# Create optimization configuration
cat > optimization_config.json << EOF
{
  "optimization": {
    "optimize_parameters": ["k_sat", "alpha"],
    "covariance_scaling": 1.0,
    "warm_start": true,
    "save_results": true
  },
  "parameter_overrides": {
    "H0": 70.0
  }
}
EOF

# Use configuration file
python pipelines/fit_cmb.py --model pbuf --config optimization_config.json
```

#### Cross-Model Optimization Comparison
```bash
# Optimize both models for comparison
python pipelines/fit_cmb.py --model lcdm --optimize H0,Om0,Obh2,ns
python pipelines/fit_cmb.py --model pbuf --optimize k_sat,alpha,H0,Om0

# Check optimization summary
python -c "
from pipelines.fit_core.parameter_store import OptimizedParameterStore
store = OptimizedParameterStore()
store.export_optimization_summary('optimization_comparison.json')
print('Optimization summary exported to optimization_comparison.json')
"
```

### 8. Wrapper Script Examples

#### Basic Command Line Usage
```bash
# Simple CMB fit
python pipelines/fit_cmb.py --model lcdm

# BAO fit with PBUF
python pipelines/fit_bao.py --model pbuf

# Supernova fit with parameter override
python pipelines/fit_sn.py --model lcdm --H0 70.0 --Om0 0.3

# Joint fit with multiple datasets
python pipelines/fit_joint.py --model pbuf --datasets cmb bao sn
```

#### Advanced Command Line Options
```bash
# Fit with integrity checks
python pipelines/fit_cmb.py --model pbuf --verify-integrity

# Save results to JSON file
python pipelines/fit_joint.py --model lcdm --datasets cmb bao \
    --output-format json --save-results results.json

# Use configuration file
python pipelines/fit_joint.py --config my_config.json

# Create example configuration
python pipelines/fit_joint.py --create-config example_config.json
```

### 8. Configuration File Examples

#### JSON Configuration
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
        "options": {
            "maxiter": 1000,
            "ftol": 1e-6
        }
    },
    "integrity_checks": {
        "enabled": true,
        "tolerance": 1e-4
    },
    "output": {
        "format": "json",
        "save_file": "pbuf_joint_results.json"
    }
}
```

#### YAML Configuration
```yaml
model: lcdm
datasets:
  - cmb
  - bao
parameters:
  H0: 70.0
  Om0: 0.3
  Obh2: 0.022
optimizer:
  method: differential_evolution
  options:
    maxiter: 500
    seed: 42
output:
  format: human
  save_file: lcdm_results.txt
```

## Batch Processing Examples

### 9. Parameter Grid Search
```python
import itertools
import pandas as pd

# Define parameter grid
h0_grid = [65, 67, 70, 73, 75]
om0_grid = [0.25, 0.30, 0.35]
alpha_grid = [1e-4, 5e-4, 1e-3]

results_list = []

# Grid search for PBUF model
for h0, om0, alpha in itertools.product(h0_grid, om0_grid, alpha_grid):
    overrides = {"H0": h0, "Om0": om0, "alpha": alpha}
    
    try:
        result = run_fit(
            model="pbuf",
            datasets_list=["cmb", "bao"],
            overrides=overrides
        )
        
        results_list.append({
            "H0": h0,
            "Om0": om0,
            "alpha": alpha,
            "chi2": result['metrics']['total_chi2'],
            "aic": result['metrics']['aic'],
            "bic": result['metrics']['bic']
        })
        
    except Exception as e:
        print(f"Failed for H0={h0}, Om0={om0}, alpha={alpha}: {e}")

# Convert to DataFrame for analysis
df = pd.DataFrame(results_list)
best_fit = df.loc[df['aic'].idxmin()]
print("Best fit parameters:")
print(best_fit)
```

### 10. Monte Carlo Analysis
```python
import numpy as np

# Monte Carlo parameter sampling
n_samples = 100
results = []

for i in range(n_samples):
    # Sample parameters from reasonable ranges
    h0_sample = np.random.normal(67.4, 2.0)
    om0_sample = np.random.normal(0.315, 0.02)
    alpha_sample = np.random.lognormal(np.log(5e-4), 0.5)
    
    overrides = {
        "H0": h0_sample,
        "Om0": om0_sample,
        "alpha": alpha_sample
    }
    
    try:
        result = run_fit(
            model="pbuf",
            datasets_list=["cmb"],
            overrides=overrides
        )
        
        results.append({
            "sample": i,
            "H0_in": h0_sample,
            "Om0_in": om0_sample,
            "alpha_in": alpha_sample,
            "H0_out": result['params']['H0'],
            "Om0_out": result['params']['Om0'],
            "alpha_out": result['params']['alpha'],
            "chi2": result['metrics']['total_chi2']
        })
        
    except Exception as e:
        print(f"Sample {i} failed: {e}")

# Analyze convergence
df_mc = pd.DataFrame(results)
print(f"Successful samples: {len(df_mc)}")
print(f"Mean output H0: {df_mc['H0_out'].mean():.2f} ± {df_mc['H0_out'].std():.2f}")
```

## Error Handling Examples

### 11. Robust Error Handling
```python
def safe_fit(model, datasets, overrides=None, max_retries=3):
    """Robust fitting with error handling and retries."""
    
    for attempt in range(max_retries):
        try:
            result = run_fit(
                model=model,
                datasets_list=datasets,
                overrides=overrides
            )
            return result
            
        except ValueError as e:
            print(f"Attempt {attempt + 1} failed with ValueError: {e}")
            if "negative" in str(e) and "degrees of freedom" in str(e):
                # Try with fewer parameters
                if overrides is None:
                    overrides = {}
                # Fix some parameters to reduce DOF
                overrides.update({"ns": 0.9649, "Neff": 3.046})
                
        except KeyError as e:
            print(f"Dataset error: {e}")
            # Remove problematic dataset
            if len(datasets) > 1:
                datasets = datasets[:-1]
                print(f"Retrying with datasets: {datasets}")
            else:
                raise
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise
    
    raise RuntimeError(f"Failed after {max_retries} attempts")

# Usage
try:
    result = safe_fit("pbuf", ["cmb", "bao", "sn"])
    print("Fit successful!")
except Exception as e:
    print(f"All attempts failed: {e}")
```

### 12. Validation and Diagnostics
```python
def comprehensive_analysis(model, datasets):
    """Perform comprehensive analysis with validation."""
    
    print(f"Analyzing {model} model with {datasets}")
    
    # 1. Parameter validation
    try:
        params = build_params(model)
        print("✓ Parameter building successful")
    except Exception as e:
        print(f"❌ Parameter building failed: {e}")
        return None
    
    # 2. Dataset validation
    try:
        for dataset in datasets:
            data = load_dataset(dataset)
            print(f"✓ Dataset {dataset} loaded successfully")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None
    
    # 3. Integrity checks
    try:
        integrity_results = run_integrity_suite(params, datasets)
        if integrity_results['overall_status'] == 'PASS':
            print("✓ Integrity checks passed")
        else:
            print("⚠️ Integrity checks failed")
            for check, result in integrity_results.items():
                if isinstance(result, dict) and result.get('status') == 'FAIL':
                    print(f"  - {check}: {result.get('message', 'Failed')}")
    except Exception as e:
        print(f"❌ Integrity checks failed: {e}")
    
    # 4. Fitting
    try:
        result = run_fit(model, datasets, "joint")
        print("✓ Fitting successful")
        
        # 5. Result validation
        chi2 = result['metrics']['total_chi2']
        dof = result['metrics']['dof']
        reduced_chi2 = chi2 / dof if dof > 0 else float('inf')
        
        print(f"  χ²/DOF = {reduced_chi2:.3f}")
        
        if reduced_chi2 < 0.5:
            print("  ⚠️ Suspiciously low χ²/DOF - possible overfitting")
        elif reduced_chi2 > 3.0:
            print("  ⚠️ High χ²/DOF - poor fit or underestimated errors")
        else:
            print("  ✓ Reasonable χ²/DOF")
            
        return result
        
    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        return None

# Usage
result = comprehensive_analysis("pbuf", ["cmb", "bao"])
```

## Extensibility Examples

### 13. Adding a New Cosmological Model

This example demonstrates how to add a w-CDM (dark energy equation of state) model:

#### Step 1: Define Model Parameters
```python
# Add to pipelines/fit_core/parameter.py
DEFAULTS["wcdm"] = {
    "H0": 67.4,
    "Om0": 0.315,
    "Obh2": 0.02237,
    "ns": 0.9649,
    "w0": -1.0,        # Dark energy equation of state
    "wa": 0.0,         # Dark energy evolution
    "model_class": "wcdm"
}
```

#### Step 2: Implement Model Physics
```python
# Create pipelines/wcdm_models.py
def hubble_wcdm(z, params):
    """Compute H(z) for w-CDM model."""
    Om0 = params["Om0"]
    w0 = params["w0"]
    wa = params["wa"]
    
    # Dark energy evolution: w(z) = w0 + wa * z/(1+z)
    z1 = 1 + z
    w_eff = 3 * (1 + w0 + wa)
    de_factor = z1**w_eff * np.exp(-3 * wa * z / z1)
    
    h_squared = Om0 * z1**3 + (1 - Om0) * de_factor
    return np.sqrt(h_squared)

def distance_modulus_wcdm(z_array, params):
    """Compute distance modulus for w-CDM."""
    # Integration and distance calculation
    # (Implementation details...)
    return mu_theory
```

#### Step 3: Register Custom Likelihood (if needed)
```python
# In pipelines/fit_core/likelihoods.py
def likelihood_sn_wcdm(params, data):
    """Custom supernova likelihood for w-CDM."""
    z_sn = data["redshifts"]
    mu_theory = distance_modulus_wcdm(z_sn, params)
    
    predictions = {"distance_moduli": mu_theory}
    chi2 = chi2_generic(predictions, 
                       {"distance_moduli": data["distance_moduli"]}, 
                       data["covariance"])
    
    return chi2, predictions

# Register the likelihood
LIKELIHOOD_FUNCTIONS[("wcdm", "sn")] = likelihood_sn_wcdm
```

#### Step 4: Use the New Model
```python
# The new model is automatically available
result = run_fit("wcdm", ["cmb", "bao", "sn"], 
                overrides={"w0": -0.9, "wa": 0.1})

print(f"w-CDM fit: w0 = {result['params']['w0']:.3f}")
print(f"w-CDM fit: wa = {result['params']['wa']:.3f}")
print(f"χ² = {result['metrics']['total_chi2']:.3f}")
```

### 14. Adding a New Dataset

Example of integrating weak lensing shear correlation data:

#### Step 1: Implement Data Loader
```python
# In pipelines/fit_core/datasets.py
def load_weak_lensing_dataset():
    """Load weak lensing correlation function data."""
    # Load from file or database
    ell_bins = np.logspace(1, 4, 20)
    xi_plus_obs = load_xi_plus_data()  # Your data loading
    xi_minus_obs = load_xi_minus_data()
    
    data_vector = np.concatenate([xi_plus_obs, xi_minus_obs])
    cov_matrix = load_wl_covariance()
    
    return {
        "observations": data_vector,
        "covariance": cov_matrix,
        "metadata": {
            "ell_bins": ell_bins,
            "n_plus": len(xi_plus_obs),
            "n_minus": len(xi_minus_obs),
            "survey": "DES_Y3"
        }
    }

# Register the dataset
DATASET_LOADERS["wl"] = load_weak_lensing_dataset
```

#### Step 2: Implement Likelihood Function
```python
def likelihood_weak_lensing(params, data):
    """Weak lensing likelihood function."""
    # Compute theoretical ξ± using cosmological code
    xi_theory = compute_wl_theory(params, data["metadata"])
    
    predictions = {"xi_combined": xi_theory}
    chi2 = chi2_generic(predictions,
                       {"xi_combined": data["observations"]},
                       data["covariance"])
    
    return chi2, predictions

# Register for all models
for model in ["lcdm", "pbuf", "wcdm"]:
    LIKELIHOOD_FUNCTIONS[(model, "wl")] = likelihood_weak_lensing
```

#### Step 3: Use the New Dataset
```python
# Weak lensing is now available for fitting
wl_result = run_fit("lcdm", ["wl"])
joint_result = run_fit("pbuf", ["cmb", "bao", "sn", "wl"])

print(f"WL-only χ²: {wl_result['metrics']['total_chi2']:.3f}")
print(f"Joint+WL χ²: {joint_result['metrics']['total_chi2']:.3f}")
```

### 15. Custom Optimization Methods

Adding a custom global optimization algorithm:

#### Step 1: Implement Custom Optimizer
```python
# In pipelines/fit_core/engine.py
from scipy.optimize import basinhopping

def basin_hopping_optimizer(objective_func, initial_params, bounds=None, options=None):
    """Basin hopping global optimization."""
    
    def objective_wrapper(x):
        return objective_func(x)
    
    # Default options
    default_options = {
        "niter": 100,
        "T": 1.0,
        "stepsize": 0.5
    }
    if options:
        default_options.update(options)
    
    result = basinhopping(
        objective_wrapper,
        initial_params,
        niter=default_options["niter"],
        T=default_options["T"],
        stepsize=default_options["stepsize"],
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "bounds": bounds
        }
    )
    
    return result

# Register the optimizer
OPTIMIZERS["basin_hopping"] = basin_hopping_optimizer
```

#### Step 2: Use Custom Optimizer
```python
# Configure custom optimization
optimizer_config = {
    "method": "basin_hopping",
    "options": {
        "niter": 200,
        "T": 2.0,
        "stepsize": 1.0
    },
    "bounds": {
        "H0": (60, 80),
        "Om0": (0.2, 0.4),
        "alpha": (1e-5, 1e-2)
    }
}

# Use in fitting
result = run_fit("pbuf", ["cmb", "bao"], 
                optimizer_config=optimizer_config)

print(f"Basin hopping result: χ² = {result['metrics']['total_chi2']:.3f}")
```

### 16. Configuration-Driven Analysis

Using configuration files for complex analysis workflows:

#### Configuration File (analysis_config.json)
```json
{
    "models": ["lcdm", "pbuf", "wcdm"],
    "datasets": ["cmb", "bao", "sn"],
    "parameter_grids": {
        "lcdm": {
            "H0": [65, 67, 70, 73, 75],
            "Om0": [0.25, 0.30, 0.35]
        },
        "pbuf": {
            "alpha": [1e-4, 5e-4, 1e-3],
            "eps0": [0.5, 0.7, 0.9]
        },
        "wcdm": {
            "w0": [-1.2, -1.0, -0.8],
            "wa": [-0.2, 0.0, 0.2]
        }
    },
    "optimization": {
        "method": "differential_evolution",
        "options": {"maxiter": 500, "seed": 42}
    },
    "output": {
        "save_results": true,
        "format": "json",
        "directory": "analysis_results"
    }
}
```

#### Analysis Script
```python
import json
import itertools
from pathlib import Path

def run_configuration_analysis(config_file):
    """Run analysis based on configuration file."""
    
    with open(config_file) as f:
        config = json.load(f)
    
    results = {}
    
    for model in config["models"]:
        print(f"Analyzing {model} model...")
        
        # Get parameter grid for this model
        param_grid = config["parameter_grids"].get(model, {})
        
        if param_grid:
            # Grid search
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            for param_combo in itertools.product(*param_values):
                overrides = dict(zip(param_names, param_combo))
                
                try:
                    result = run_fit(
                        model=model,
                        datasets_list=config["datasets"],
                        overrides=overrides,
                        optimizer_config=config["optimization"]
                    )
                    
                    # Store result
                    key = f"{model}_{param_combo}"
                    results[key] = {
                        "model": model,
                        "parameters": overrides,
                        "chi2": result["metrics"]["total_chi2"],
                        "aic": result["metrics"]["aic"],
                        "bic": result["metrics"]["bic"]
                    }
                    
                except Exception as e:
                    print(f"Failed for {model} with {overrides}: {e}")
        
        else:
            # Single fit with defaults
            result = run_fit(
                model=model,
                datasets_list=config["datasets"],
                optimizer_config=config["optimization"]
            )
            
            results[model] = {
                "model": model,
                "parameters": result["params"],
                "chi2": result["metrics"]["total_chi2"],
                "aic": result["metrics"]["aic"],
                "bic": result["metrics"]["bic"]
            }
    
    # Save results if requested
    if config["output"]["save_results"]:
        output_dir = Path(config["output"]["directory"])
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    return results

# Usage
results = run_configuration_analysis("analysis_config.json")

# Find best model
best_model = min(results.items(), key=lambda x: x[1]["aic"])
print(f"Best model: {best_model[0]} with AIC = {best_model[1]['aic']:.3f}")
```

These examples demonstrate the full range of capabilities of the PBUF cosmology pipeline, from basic usage to advanced analysis techniques and extensibility features. The system is designed to be both powerful for expert users and accessible for newcomers to cosmological parameter estimation, with a focus on extensibility through configuration rather than code modification.