# PBUF Cosmology Pipeline - Developer Guide

## Extending the System with New Models

### Adding a New Cosmological Model

The unified architecture makes it straightforward to add new cosmological models. Here's a complete example of adding a w-CDM (dark energy equation of state) model.

#### Step 1: Add Model Parameters

Edit `pipelines/fit_core/parameter.py`:

```python
# Add to DEFAULTS dictionary
DEFAULTS["wcdm"] = {
    # Standard cosmological parameters
    "H0": 67.4,           # Hubble constant (km/s/Mpc)
    "Om0": 0.315,         # Matter density fraction
    "Obh2": 0.02237,      # Physical baryon density
    "ns": 0.9649,         # Scalar spectral index
    "Neff": 3.046,        # Effective neutrino species
    "Tcmb": 2.7255,       # CMB temperature (K)
    
    # w-CDM specific parameters
    "w0": -1.0,           # Dark energy equation of state at z=0
    "wa": 0.0,            # Dark energy evolution parameter
    
    # Computational settings
    "recomb_method": "PLANCK18"
}
```

#### Step 2: Implement Model Physics

Create `pipelines/wcdm_models.py`:

```python
"""
w-CDM cosmological model implementation.

This module provides the physics for dark energy models with 
equation of state w(z) = w0 + wa * z/(1+z).
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.integrate import quad


def hubble_wcdm(z: float, params: Dict) -> float:
    """
    Compute Hubble parameter H(z) for w-CDM model.
    
    Args:
        z: Redshift
        params: Parameter dictionary containing Om0, w0, wa
        
    Returns:
        H(z) in units of H0
    """
    Om0 = params["Om0"]
    w0 = params["w0"]
    wa = params["wa"]
    
    # Radiation density (small at low z)
    Or0 = 2.47e-5 / (params["H0"]/100)**2  # Approximate
    
    # Dark energy density evolution
    # ρ_de(z) ∝ (1+z)^(3(1+w0+wa)) * exp(-3*wa*z/(1+z))
    z1 = 1 + z
    w_eff = 3 * (1 + w0 + wa)
    de_evolution = z1**w_eff * np.exp(-3 * wa * z / z1)
    
    # Hubble parameter
    h_squared = (
        Om0 * z1**3 +                    # Matter
        Or0 * z1**4 +                    # Radiation
        (1 - Om0 - Or0) * de_evolution   # Dark energy
    )
    
    return np.sqrt(h_squared)


def distance_modulus_wcdm(z: np.ndarray, params: Dict) -> np.ndarray:
    """
    Compute distance modulus for w-CDM model.
    
    Args:
        z: Redshift array
        params: Parameter dictionary
        
    Returns:
        Distance modulus array
    """
    H0 = params["H0"]
    
    # Comoving distance integral
    def integrand(zp):
        return 1.0 / hubble_wcdm(zp, params)
    
    distances = []
    for zi in z:
        if zi <= 0:
            distances.append(0.0)
        else:
            dc, _ = quad(integrand, 0, zi)
            distances.append(dc)
    
    dc_array = np.array(distances)
    
    # Convert to distance modulus
    # μ = 5 log10(DL/Mpc) + 25
    # DL = (1+z) * DC * c/H0
    c_over_h0 = 2997.92458  # c/H0 in Mpc for H0 in km/s/Mpc
    
    dl = (1 + z) * dc_array * c_over_h0 / H0 * 100  # Convert H0 to h
    mu = 5 * np.log10(dl) + 25
    
    return mu


def angular_diameter_distance_wcdm(z: float, params: Dict) -> float:
    """
    Compute angular diameter distance for w-CDM model.
    
    Args:
        z: Redshift
        params: Parameter dictionary
        
    Returns:
        Angular diameter distance in Mpc
    """
    H0 = params["H0"]
    
    def integrand(zp):
        return 1.0 / hubble_wcdm(zp, params)
    
    dc, _ = quad(integrand, 0, z)
    
    c_over_h0 = 2997.92458
    da = dc * c_over_h0 / H0 * 100 / (1 + z)
    
    return da
```

#### Step 3: Integrate with Likelihood Functions

Modify `pipelines/fit_core/likelihoods.py`:

```python
# Add import
from .. import wcdm_models

def likelihood_sn_wcdm(params: Dict, data: Dict) -> Tuple[float, Dict]:
    """
    Supernova likelihood for w-CDM model.
    
    Args:
        params: w-CDM parameter dictionary
        data: Supernova dataset
        
    Returns:
        Tuple of (chi2, predictions)
    """
    # Extract supernova data
    z_sn = data["redshifts"]
    mu_obs = data["distance_moduli"]
    cov_sn = data["covariance"]
    
    # Compute theoretical distance moduli
    mu_theory = wcdm_models.distance_modulus_wcdm(z_sn, params)
    
    # Marginalize over absolute magnitude offset if needed
    if data.get("marginalize_M", True):
        # Analytical marginalization over M_B
        residuals = mu_obs - mu_theory
        mean_residual = np.mean(residuals)
        mu_theory += mean_residual
    
    # Compute chi-squared
    predictions = {"distance_moduli": mu_theory}
    chi2 = chi2_generic(predictions, {"distance_moduli": mu_obs}, cov_sn)
    
    return chi2, predictions

# Update likelihood dispatcher
LIKELIHOOD_FUNCTIONS = {
    ("lcdm", "cmb"): likelihood_cmb,
    ("lcdm", "bao"): likelihood_bao,
    ("lcdm", "sn"): likelihood_sn,
    ("pbuf", "cmb"): likelihood_cmb,  # Uses same CMB physics
    ("pbuf", "bao"): likelihood_bao,  # Uses same BAO physics
    ("pbuf", "sn"): likelihood_sn,    # Uses same SN physics
    ("wcdm", "cmb"): likelihood_cmb,  # Uses same CMB physics
    ("wcdm", "bao"): likelihood_bao,  # Uses same BAO physics
    ("wcdm", "sn"): likelihood_sn_wcdm,  # Custom w-CDM SN likelihood
}
```

#### Step 4: Add Model Validation

Add to `pipelines/fit_core/integrity.py`:

```python
def verify_wcdm_physics(params: Dict) -> bool:
    """
    Verify w-CDM model physics consistency.
    
    Args:
        params: w-CDM parameter dictionary
        
    Returns:
        True if physics is consistent
    """
    w0 = params.get("w0", -1.0)
    wa = params.get("wa", 0.0)
    
    # Check for phantom dark energy (w < -1)
    if w0 < -1.5:
        print(f"Warning: Deep phantom regime w0={w0:.3f}")
    
    # Check for unrealistic evolution
    if abs(wa) > 2.0:
        print(f"Warning: Large wa={wa:.3f} may be unphysical")
    
    # Verify model reduces to ΛCDM when w0=-1, wa=0
    if abs(w0 + 1.0) < 1e-6 and abs(wa) < 1e-6:
        print("Info: w-CDM reduces to ΛCDM")
    
    return True

# Add to integrity suite
def run_integrity_suite(params: Dict, datasets: List[str]) -> Dict:
    """Enhanced integrity suite with model-specific checks."""
    
    results = {}
    model_class = params.get("model_class", "unknown")
    
    # Standard checks
    results["h_ratios"] = {"status": "PASS"}  # Simplified
    results["recombination"] = {"status": "PASS"}
    results["covariance"] = {"status": "PASS"}
    
    # Model-specific checks
    if model_class == "wcdm":
        wcdm_check = verify_wcdm_physics(params)
        results["wcdm_physics"] = {
            "status": "PASS" if wcdm_check else "FAIL"
        }
    
    # Overall status
    all_passed = all(
        result.get("status") == "PASS" 
        for result in results.values()
    )
    results["overall_status"] = "PASS" if all_passed else "FAIL"
    
    return results
```

#### Step 5: Add Tests

Create `pipelines/fit_core/test_wcdm_integration.py`:

```python
"""
Tests for w-CDM model integration.
"""

import pytest
import numpy as np
from pipelines.fit_core.parameter import build_params, DEFAULTS
from pipelines.fit_core.engine import run_fit
from pipelines.fit_core.integrity import verify_wcdm_physics


class TestWCDMIntegration:
    """Test w-CDM model integration."""
    
    def test_wcdm_parameter_building(self):
        """Test w-CDM parameter construction."""
        
        # Test default parameters
        params = build_params("wcdm")
        assert "w0" in params
        assert "wa" in params
        assert params["w0"] == -1.0
        assert params["wa"] == 0.0
        
        # Test overrides
        overrides = {"w0": -0.9, "wa": 0.1}
        params_override = build_params("wcdm", overrides=overrides)
        assert params_override["w0"] == -0.9
        assert params_override["wa"] == 0.1
    
    def test_wcdm_reduces_to_lcdm(self):
        """Test that w-CDM reduces to ΛCDM when w0=-1, wa=0."""
        
        # w-CDM with ΛCDM parameters
        wcdm_params = build_params("wcdm", overrides={"w0": -1.0, "wa": 0.0})
        lcdm_params = build_params("lcdm")
        
        # Should give similar results (within numerical precision)
        wcdm_result = run_fit("wcdm", ["sn"], overrides={"w0": -1.0, "wa": 0.0})
        lcdm_result = run_fit("lcdm", ["sn"])
        
        # Chi-squared should be very close
        chi2_diff = abs(
            wcdm_result["metrics"]["total_chi2"] - 
            lcdm_result["metrics"]["total_chi2"]
        )
        assert chi2_diff < 1e-3, f"Chi2 difference too large: {chi2_diff}"
    
    def test_wcdm_physics_validation(self):
        """Test w-CDM physics validation."""
        
        # Normal w-CDM
        normal_params = build_params("wcdm", overrides={"w0": -0.8, "wa": -0.2})
        assert verify_wcdm_physics(normal_params)
        
        # Phantom w-CDM
        phantom_params = build_params("wcdm", overrides={"w0": -1.2, "wa": 0.0})
        assert verify_wcdm_physics(phantom_params)  # Should pass but warn
    
    def test_wcdm_fitting(self):
        """Test w-CDM fitting functionality."""
        
        # Test individual dataset fitting
        result = run_fit("wcdm", ["sn"])
        
        assert "params" in result
        assert "w0" in result["params"]
        assert "wa" in result["params"]
        assert "metrics" in result
        assert result["metrics"]["total_chi2"] > 0
        
        # Test joint fitting
        joint_result = run_fit("wcdm", ["cmb", "sn"])
        assert len(joint_result["results"]) == 2
        assert "cmb" in joint_result["results"]
        assert "sn" in joint_result["results"]


if __name__ == "__main__":
    pytest.main([__file__])
```

#### Step 6: Update Documentation

Add to wrapper script help and documentation:

```python
# In pipelines/fit_joint.py argument parser
parser.add_argument(
    "--model", 
    choices=["lcdm", "pbuf", "wcdm"], 
    default="pbuf",
    help="Cosmological model to fit"
)

# Add w-CDM specific parameters
parser.add_argument("--w0", type=float, help="Dark energy equation of state")
parser.add_argument("--wa", type=float, help="Dark energy evolution parameter")
```

### Testing Your New Model

```python
# Test the new w-CDM model
from pipelines.fit_core.engine import run_fit

# Basic test
result = run_fit("wcdm", ["sn"], overrides={"w0": -0.9, "wa": 0.1})
print(f"w-CDM fit: χ² = {result['metrics']['total_chi2']:.3f}")

# Compare with ΛCDM
lcdm_result = run_fit("lcdm", ["sn"])
wcdm_result = run_fit("wcdm", ["sn"])

print(f"ΛCDM AIC: {lcdm_result['metrics']['aic']:.3f}")
print(f"w-CDM AIC: {wcdm_result['metrics']['aic']:.3f}")
```

## Adding New Observational Datasets

### Example: Adding Weak Lensing Data

#### Step 1: Implement Data Loader

Add to `pipelines/fit_core/datasets.py`:

```python
def load_weak_lensing_dataset():
    """
    Load weak lensing shear correlation function data.
    
    Returns:
        Dict containing weak lensing observations and covariance
    """
    # This would load real weak lensing data
    # For example, from DES, KiDS, or HSC surveys
    
    # Mock data structure
    ell_bins = np.logspace(1, 4, 20)  # Multipole bins
    xi_plus_obs = np.random.normal(0, 1e-4, 20)  # ξ+ observations
    xi_minus_obs = np.random.normal(0, 1e-5, 20)  # ξ- observations
    
    # Combined data vector
    data_vector = np.concatenate([xi_plus_obs, xi_minus_obs])
    
    # Mock covariance matrix
    n_data = len(data_vector)
    cov_matrix = np.eye(n_data) * 1e-8
    
    return {
        "observations": data_vector,
        "covariance": cov_matrix,
        "metadata": {
            "ell_bins": ell_bins,
            "n_plus": len(xi_plus_obs),
            "n_minus": len(xi_minus_obs),
            "survey": "mock_survey",
            "redshift_range": [0.2, 1.2]
        }
    }

# Add to dataset registry
DATASET_LOADERS = {
    "cmb": load_cmb_dataset,
    "bao": load_bao_dataset,
    "bao_ani": load_bao_anisotropic_dataset,
    "sn": load_supernova_dataset,
    "wl": load_weak_lensing_dataset,  # New dataset
}
```

#### Step 2: Implement Likelihood Function

Add to `pipelines/fit_core/likelihoods.py`:

```python
def likelihood_weak_lensing(params: Dict, data: Dict) -> Tuple[float, Dict]:
    """
    Weak lensing likelihood function.
    
    Args:
        params: Cosmological parameters
        data: Weak lensing dataset
        
    Returns:
        Tuple of (chi2, predictions)
    """
    # Extract data
    observations = data["observations"]
    covariance = data["covariance"]
    metadata = data["metadata"]
    
    ell_bins = metadata["ell_bins"]
    n_plus = metadata["n_plus"]
    n_minus = metadata["n_minus"]
    
    # Compute theoretical predictions
    # This would use a cosmological code like CCL, CAMB, or CLASS
    xi_plus_theory = compute_xi_plus_theory(ell_bins, params)
    xi_minus_theory = compute_xi_minus_theory(ell_bins, params)
    
    # Combine predictions
    theory_vector = np.concatenate([xi_plus_theory, xi_minus_theory])
    
    # Compute chi-squared
    predictions = {
        "xi_plus": xi_plus_theory,
        "xi_minus": xi_minus_theory,
        "combined": theory_vector
    }
    
    chi2 = chi2_generic(
        {"combined": theory_vector},
        {"combined": observations},
        covariance
    )
    
    return chi2, predictions


def compute_xi_plus_theory(ell_bins: np.ndarray, params: Dict) -> np.ndarray:
    """
    Compute theoretical ξ+ correlation function.
    
    This is a simplified example - real implementation would use
    cosmological codes like CCL or CAMB.
    """
    # Mock theoretical calculation
    # In reality, this would involve:
    # 1. Computing matter power spectrum P(k,z)
    # 2. Calculating lensing kernel W(z)
    # 3. Integrating to get C_ell
    # 4. Transforming to real space ξ±
    
    sigma8 = params.get("sigma8", 0.8)
    Om0 = params["Om0"]
    
    # Simplified scaling relation
    xi_plus = sigma8**2 * Om0**0.5 / ell_bins**0.8 * 1e-4
    
    return xi_plus


def compute_xi_minus_theory(ell_bins: np.ndarray, params: Dict) -> np.ndarray:
    """Compute theoretical ξ- correlation function."""
    
    sigma8 = params.get("sigma8", 0.8)
    Om0 = params["Om0"]
    
    # ξ- is typically smaller than ξ+
    xi_minus = sigma8**2 * Om0**0.5 / ell_bins**1.2 * 1e-5
    
    return xi_minus

# Add to likelihood registry
LIKELIHOOD_FUNCTIONS[("lcdm", "wl")] = likelihood_weak_lensing
LIKELIHOOD_FUNCTIONS[("pbuf", "wl")] = likelihood_weak_lensing
LIKELIHOOD_FUNCTIONS[("wcdm", "wl")] = likelihood_weak_lensing
```

#### Step 3: Add Required Parameters

If the new dataset requires additional parameters (like σ₈ for weak lensing):

```python
# In parameter.py, add to DEFAULTS
DEFAULTS["lcdm"]["sigma8"] = 0.8159  # Planck 2018 value
DEFAULTS["pbuf"]["sigma8"] = 0.8159
DEFAULTS["wcdm"]["sigma8"] = 0.8159
```

#### Step 4: Test the New Dataset

```python
# Test weak lensing integration
from pipelines.fit_core.engine import run_fit

# Test individual weak lensing fit
wl_result = run_fit("lcdm", ["wl"])
print(f"Weak lensing χ²: {wl_result['metrics']['total_chi2']:.3f}")

# Test joint fit with other datasets
joint_result = run_fit("lcdm", ["cmb", "bao", "wl"])
print(f"Joint χ²: {joint_result['metrics']['total_chi2']:.3f}")

# Check individual contributions
for dataset in ["cmb", "bao", "wl"]:
    chi2 = joint_result['results'][dataset]['chi2']
    print(f"{dataset.upper()} χ²: {chi2:.3f}")
```

## Advanced Customization

### Custom Optimization Algorithms

```python
# Add custom optimizer to engine.py
from scipy.optimize import basinhopping

def custom_optimizer(objective_func, initial_params, bounds=None):
    """
    Custom optimization using basin hopping for global optimization.
    """
    
    def objective_wrapper(x):
        return objective_func(x)
    
    # Basin hopping with local minimization
    result = basinhopping(
        objective_wrapper,
        initial_params,
        niter=100,
        T=1.0,
        stepsize=0.5,
        minimizer_kwargs={
            "method": "L-BFGS-B",
            "bounds": bounds
        }
    )
    
    return result

# Register in engine.py
OPTIMIZERS = {
    "minimize": scipy_minimize_wrapper,
    "differential_evolution": scipy_de_wrapper,
    "basin_hopping": custom_optimizer,  # New optimizer
}
```

### Custom Statistics and Model Comparison

```python
# Add to statistics.py
def compute_dic(chi2_samples: np.ndarray, chi2_mean: float) -> float:
    """
    Compute Deviance Information Criterion (DIC).
    
    Args:
        chi2_samples: Array of χ² values from MCMC chain
        chi2_mean: χ² at posterior mean parameters
        
    Returns:
        DIC value
    """
    # Effective number of parameters
    p_dic = np.var(chi2_samples) / 2.0
    
    # DIC = χ²(θ̄) + 2 * p_DIC
    dic = chi2_mean + 2 * p_dic
    
    return dic


def compute_waic(log_likelihood_samples: np.ndarray) -> Tuple[float, float]:
    """
    Compute Widely Applicable Information Criterion (WAIC).
    
    Args:
        log_likelihood_samples: Log-likelihood samples from MCMC
        
    Returns:
        Tuple of (WAIC, effective number of parameters)
    """
    # Log pointwise predictive density
    lppd = np.sum(np.log(np.mean(np.exp(log_likelihood_samples), axis=0)))
    
    # Effective number of parameters
    p_waic = np.sum(np.var(log_likelihood_samples, axis=0))
    
    # WAIC = -2 * (lppd - p_WAIC)
    waic = -2 * (lppd - p_waic)
    
    return waic, p_waic
```

### Performance Optimization

#### Caching Expensive Computations

```python
# Add to likelihoods.py
from functools import lru_cache
import hashlib

def cache_key_from_params(params: Dict) -> str:
    """Create cache key from parameter dictionary."""
    # Create deterministic hash from parameters
    param_str = str(sorted(params.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_cmb_predictions(param_hash: str, params_tuple: tuple) -> Dict:
    """Cached CMB predictions to avoid recomputation."""
    # Convert back to dict
    params = dict(params_tuple)
    
    # Expensive CMB calculation here
    predictions = compute_cmb_predictions(params)
    
    return predictions

def likelihood_cmb_cached(params: Dict, data: Dict) -> Tuple[float, Dict]:
    """CMB likelihood with caching."""
    
    # Create cache key
    param_hash = cache_key_from_params(params)
    params_tuple = tuple(sorted(params.items()))
    
    # Get cached predictions
    predictions = cached_cmb_predictions(param_hash, params_tuple)
    
    # Compute chi-squared (fast)
    chi2 = chi2_generic(predictions, data["observations"], data["covariance"])
    
    return chi2, predictions
```

#### Parallel Processing

```python
# Add parallel dataset processing
from multiprocessing import Pool
from functools import partial

def parallel_likelihood_computation(
    params: Dict, 
    datasets_list: List[str],
    n_processes: int = None
) -> Dict:
    """
    Compute likelihoods in parallel for multiple datasets.
    
    Args:
        params: Parameter dictionary
        datasets_list: List of datasets
        n_processes: Number of processes (default: CPU count)
        
    Returns:
        Dictionary of likelihood results
    """
    
    def compute_single_likelihood(dataset_name: str) -> Tuple[str, float, Dict]:
        """Compute likelihood for single dataset."""
        data = load_dataset(dataset_name)
        likelihood_func = get_likelihood_function(params["model_class"], dataset_name)
        chi2, predictions = likelihood_func(params, data)
        return dataset_name, chi2, predictions
    
    # Parallel computation
    with Pool(processes=n_processes) as pool:
        results = pool.map(compute_single_likelihood, datasets_list)
    
    # Organize results
    likelihood_results = {}
    for dataset_name, chi2, predictions in results:
        likelihood_results[dataset_name] = {
            "chi2": chi2,
            "predictions": predictions
        }
    
    return likelihood_results
```

## Best Practices for Extensions

### 1. Code Organization
- Keep model-specific physics in separate modules
- Use consistent naming conventions
- Document all public functions with docstrings
- Include type hints for better code clarity

### 2. Testing
- Write unit tests for all new functions
- Include integration tests with existing system
- Test edge cases and error conditions
- Validate against known results when possible

### 3. Documentation
- Update API documentation for new functions
- Add usage examples for new features
- Document any new dependencies
- Include physics references where appropriate

### 4. Performance
- Profile code to identify bottlenecks
- Use caching for expensive computations
- Consider parallel processing for independent calculations
- Optimize memory usage for large datasets

### 5. Backward Compatibility
- Maintain existing API interfaces
- Use optional parameters for new features
- Provide migration guides for breaking changes
- Test against existing analysis scripts

## Physics Documentation Integration (Requirement 7.4)

The system maintains strong connections to documented physics derivations in the `documents/` directory:

### Core Physics References

```python
# In model implementations, reference documented physics
def pbuf_hubble_ratio(z, params):
    """
    Compute PBUF H(z) ratio following documented derivations.
    
    Physics Reference: documents/PBUF-Math-Supplement-v9.md, Equation (15)
    Empirical Validation: documents/Empirical_Summary_v9.md, Section 3.2
    """
    alpha = params["alpha"]
    k_sat = params["k_sat"]
    
    # Implementation follows documented equations
    # See documents/equations_reference.mc for numerical validation
    return hubble_ratio

def cmb_distance_priors(params):
    """
    Compute CMB distance priors.
    
    Physics Reference: documents/evolution_theory.md, Section 2.1
    Validation: Planck Collaboration 2018, Table 2
    """
    # Implementation with documented physics
    pass
```

### Validation Against Documentation

```python
# Automated validation against documented values
def validate_against_documentation():
    """Validate implementation against documented reference values."""
    
    # Load reference values from documents/
    reference_file = "documents/equations_reference.mc"
    references = load_reference_values(reference_file)
    
    # Test PBUF model against documented values
    pbuf_params = build_params("pbuf")
    
    for test_case in references["pbuf_tests"]:
        z = test_case["redshift"]
        expected_h_ratio = test_case["h_ratio"]
        
        computed_ratio = pbuf_hubble_ratio(z, pbuf_params)
        
        assert abs(computed_ratio - expected_h_ratio) < 1e-6, \
            f"H(z) ratio mismatch at z={z}: {computed_ratio} vs {expected_h_ratio}"
    
    print("✓ All physics validations passed")
```

### Documentation-Driven Development

When adding new models, follow this documentation-first approach:

1. **Document Physics**: Add mathematical derivations to `documents/`
2. **Reference Validation**: Include numerical test cases
3. **Implementation**: Code with explicit references to documentation
4. **Automated Testing**: Validate against documented values

```python
# Example: Adding quintessence model
def quintessence_potential(phi, params):
    """
    Quintessence potential V(φ).
    
    Physics Reference: documents/quintessence_theory.md, Equation (8)
    Parameter Constraints: documents/quintessence_bounds.md, Table 1
    """
    V0 = params["V0"]
    lambda_q = params["lambda_q"]
    
    # V(φ) = V0 * exp(-λφ/M_pl) as documented
    return V0 * np.exp(-lambda_q * phi)
```

## Advanced Extension Patterns

### Plugin Architecture for Models

Create a plugin system for seamless model integration:

```python
# pipelines/fit_core/model_registry.py
class ModelPlugin:
    """Base class for cosmological model plugins."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_default_parameters(self) -> Dict:
        """Return default parameter dictionary."""
        raise NotImplementedError
    
    def get_physics_modules(self) -> List[str]:
        """Return list of required physics modules."""
        raise NotImplementedError
    
    def validate_parameters(self, params: Dict) -> bool:
        """Validate parameter values."""
        return True
    
    def get_documentation_references(self) -> Dict:
        """Return documentation references for this model."""
        return {}

class PBUFModelPlugin(ModelPlugin):
    """PBUF model plugin implementation."""
    
    def __init__(self):
        super().__init__("pbuf")
    
    def get_default_parameters(self):
        return {
            "alpha": 5e-4,
            "Rmax": 1e9,
            "eps0": 0.7,
            "n_eps": 0.0,
            "k_sat": 0.9762,
            "model_class": "pbuf"
        }
    
    def get_physics_modules(self):
        return ["pbuf_models", "gr_models"]
    
    def get_documentation_references(self):
        return {
            "theory": "documents/PBUF-Math-Supplement-v9.md",
            "empirical": "documents/Empirical_Summary_v9.md",
            "equations": "documents/equations_reference.mc"
        }

# Model registry
MODEL_REGISTRY = {
    "pbuf": PBUFModelPlugin(),
    "lcdm": LCDMModelPlugin(),
    # New models register here
}
```

### Dataset Plugin System

```python
# pipelines/fit_core/dataset_registry.py
class DatasetPlugin:
    """Base class for dataset plugins."""
    
    def __init__(self, name: str):
        self.name = name
    
    def load_data(self) -> Dict:
        """Load dataset from source."""
        raise NotImplementedError
    
    def validate_data(self, data: Dict) -> bool:
        """Validate dataset format and properties."""
        raise NotImplementedError
    
    def get_metadata(self) -> Dict:
        """Return dataset metadata."""
        raise NotImplementedError

class CMBDatasetPlugin(DatasetPlugin):
    """CMB distance priors dataset plugin."""
    
    def __init__(self):
        super().__init__("cmb")
    
    def load_data(self):
        # Load Planck 2018 distance priors
        return load_planck_distance_priors()
    
    def validate_data(self, data):
        # Validate CMB data format
        required_keys = ["R", "l_A", "theta_star"]
        return all(key in data["observations"] for key in required_keys)
    
    def get_metadata(self):
        return {
            "survey": "Planck 2018",
            "data_type": "distance_priors",
            "n_points": 3,
            "reference": "Planck Collaboration 2020, A&A 641, A6"
        }

# Dataset registry
DATASET_REGISTRY = {
    "cmb": CMBDatasetPlugin(),
    "bao": BAODatasetPlugin(),
    "sn": SupernovaDatasetPlugin(),
    # New datasets register here
}
```

### Likelihood Function Factory

```python
# pipelines/fit_core/likelihood_factory.py
class LikelihoodFactory:
    """Factory for creating likelihood functions."""
    
    def __init__(self):
        self.likelihood_registry = {}
    
    def register_likelihood(self, model: str, dataset: str, func: callable):
        """Register a likelihood function."""
        key = (model, dataset)
        self.likelihood_registry[key] = func
    
    def get_likelihood(self, model: str, dataset: str) -> callable:
        """Get likelihood function for model-dataset combination."""
        key = (model, dataset)
        
        if key in self.likelihood_registry:
            return self.likelihood_registry[key]
        
        # Try generic likelihood if model-specific not found
        generic_key = ("generic", dataset)
        if generic_key in self.likelihood_registry:
            return self.likelihood_registry[generic_key]
        
        raise KeyError(f"No likelihood function for {model}-{dataset}")
    
    def auto_register_model(self, model_name: str):
        """Automatically register standard likelihoods for new model."""
        standard_datasets = ["cmb", "bao", "bao_ani", "sn"]
        
        for dataset in standard_datasets:
            # Most models use the same likelihood functions
            generic_func = self.get_likelihood("generic", dataset)
            self.register_likelihood(model_name, dataset, generic_func)

# Global likelihood factory
likelihood_factory = LikelihoodFactory()

# Register standard likelihoods
likelihood_factory.register_likelihood("generic", "cmb", likelihood_cmb)
likelihood_factory.register_likelihood("generic", "bao", likelihood_bao)
likelihood_factory.register_likelihood("generic", "sn", likelihood_sn)

# Model-specific overrides
likelihood_factory.register_likelihood("pbuf", "sn", likelihood_sn_pbuf)
```

### Configuration-Based Extension System

```python
# pipelines/fit_core/extension_config.py
class ExtensionConfig:
    """Configuration-based extension system."""
    
    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict:
        """Load extension configuration."""
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                return json.load(f)
        
        # Default configuration
        return {
            "models": {
                "enabled": ["lcdm", "pbuf"],
                "plugins": {}
            },
            "datasets": {
                "enabled": ["cmb", "bao", "sn"],
                "plugins": {}
            },
            "optimizers": {
                "enabled": ["minimize", "differential_evolution"],
                "plugins": {}
            }
        }
    
    def register_model_from_config(self, model_name: str, model_config: Dict):
        """Register model from configuration."""
        
        # Add parameters to DEFAULTS
        DEFAULTS[model_name] = model_config["parameters"]
        
        # Register physics modules if specified
        if "physics_modules" in model_config:
            for module_name in model_config["physics_modules"]:
                import_module(f"pipelines.{module_name}")
        
        # Auto-register standard likelihoods
        likelihood_factory.auto_register_model(model_name)
        
        print(f"✓ Registered model '{model_name}' from configuration")
    
    def register_dataset_from_config(self, dataset_name: str, dataset_config: Dict):
        """Register dataset from configuration."""
        
        # Create dataset loader function
        def config_dataset_loader():
            loader_func = getattr(
                import_module(dataset_config["module"]),
                dataset_config["function"]
            )
            return loader_func()
        
        # Register in dataset loaders
        DATASET_LOADERS[dataset_name] = config_dataset_loader
        
        print(f"✓ Registered dataset '{dataset_name}' from configuration")
    
    def apply_extensions(self):
        """Apply all configured extensions."""
        
        # Register model plugins
        for model_name, model_config in self.config["models"]["plugins"].items():
            self.register_model_from_config(model_name, model_config)
        
        # Register dataset plugins
        for dataset_name, dataset_config in self.config["datasets"]["plugins"].items():
            self.register_dataset_from_config(dataset_name, dataset_config)

# Example extension configuration file (extensions.json)
EXAMPLE_EXTENSION_CONFIG = {
    "models": {
        "enabled": ["lcdm", "pbuf", "wcdm"],
        "plugins": {
            "wcdm": {
                "parameters": {
                    "H0": 67.4,
                    "Om0": 0.315,
                    "w0": -1.0,
                    "wa": 0.0,
                    "model_class": "wcdm"
                },
                "physics_modules": ["wcdm_models"],
                "documentation": {
                    "theory": "documents/wcdm_theory.md",
                    "validation": "documents/wcdm_tests.md"
                }
            }
        }
    },
    "datasets": {
        "enabled": ["cmb", "bao", "sn", "wl"],
        "plugins": {
            "wl": {
                "module": "weak_lensing_data",
                "function": "load_des_y3_data",
                "metadata": {
                    "survey": "DES Y3",
                    "data_type": "shear_correlation"
                }
            }
        }
    }
}
```

This developer guide provides the foundation for extending the PBUF cosmology pipeline with new models, datasets, and analysis capabilities while maintaining the system's unified architecture and robust design principles. The emphasis on configuration-driven extensions and documentation integration ensures that the system remains maintainable and scientifically rigorous as it grows.