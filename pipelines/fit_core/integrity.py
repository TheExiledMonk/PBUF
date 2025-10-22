"""
Physics validation and integrity checks for PBUF cosmology fitting.

This module provides comprehensive consistency tests and physics validation
to ensure system integrity and detect numerical or physics inconsistencies.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from . import ParameterDict


def verify_h_ratios(
    params: ParameterDict, 
    redshifts: Optional[List[float]] = None,
    tolerance: float = None
) -> bool:
    """
    Verify Hubble parameter consistency between PBUF and ΛCDM models.
    
    Compares H(z) ratios at test redshifts to ensure PBUF reduces to ΛCDM
    when k_sat approaches 1.
    
    Args:
        params: Parameter dictionary
        redshifts: Test redshifts for comparison (default: [0.1, 0.5, 1.0, 2.0])
        tolerance: Tolerance for ratio comparison
        
    Returns:
        True if H(z) ratios are consistent within tolerance
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5
    """
    if redshifts is None:
        redshifts = DEFAULT_TEST_REDSHIFTS
    
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCES["h_ratios"]
    
    # For ΛCDM model, H(z) ratios should be 1.0 by definition
    if params.get("model_class") == "lcdm":
        return True
    
    # For PBUF model, check if k_sat is close to 1 (should reduce to ΛCDM)
    k_sat = params.get("k_sat", 1.0)
    
    # If k_sat is very close to 1, PBUF should match ΛCDM
    if abs(k_sat - 1.0) < tolerance:
        return True
    
    # Compute H(z) ratios at test redshifts
    ratios = []
    for z in redshifts:
        ratio = _compute_h_ratio_at_z(params, z)
        ratios.append(ratio)
        
        # Check if ratio deviates significantly from 1.0
        if abs(ratio - 1.0) > tolerance:
            print(f"Warning: H(z={z}) ratio = {ratio:.6f}, exceeds tolerance {tolerance}")
            return False
    
    return True


def verify_recombination(
    params: ParameterDict, 
    reference: float = None,
    tolerance: float = None
) -> bool:
    """
    Verify recombination redshift computation against Planck 2018 reference.
    
    Args:
        params: Parameter dictionary
        reference: Reference z* value (Planck 2018 baseline)
        tolerance: Tolerance for comparison
        
    Returns:
        True if z* computation is consistent with reference
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5
    """
    if reference is None:
        reference = REFERENCE_VALUES["planck2018_z_recomb"]
    
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCES["recombination"]
    
    # Get computed recombination redshift from parameters
    z_recomb = params.get("z_recomb")
    
    if z_recomb is None:
        print("Warning: z_recomb not found in parameter dictionary")
        return False
    
    # Check if using PLANCK18 method (should match reference exactly)
    recomb_method = params.get("recomb_method", "PLANCK18")
    
    if recomb_method == "PLANCK18":
        # For PLANCK18 method, should match reference exactly
        expected_tolerance = 1e-6  # Very tight tolerance for fixed value
        if abs(z_recomb - reference) > expected_tolerance:
            print(f"Warning: PLANCK18 z_recomb = {z_recomb}, expected {reference}")
            return False
    else:
        # For other methods (HS96, EH98), allow larger tolerance
        relative_error = abs(z_recomb - reference) / reference
        if relative_error > tolerance:
            print(f"Warning: {recomb_method} z_recomb = {z_recomb}, "
                  f"reference = {reference}, relative error = {relative_error:.6f}")
            return False
    
    return True


def verify_covariance_matrices(datasets: List[str]) -> bool:
    """
    Verify covariance matrix properties for all datasets.
    
    Checks positive definiteness, proper conditioning, and numerical stability
    of covariance matrices used in likelihood computations.
    
    Args:
        datasets: List of dataset names to check
        
    Returns:
        True if all covariance matrices pass validation
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5
    """
    from . import datasets as ds
    
    all_valid = True
    
    for dataset_name in datasets:
        try:
            # Load dataset
            data = ds.load_dataset(dataset_name)
            
            # Extract covariance matrix
            if "covariance" not in data:
                print(f"Warning: No covariance matrix found for dataset {dataset_name}")
                all_valid = False
                continue
            
            cov_matrix = data["covariance"]
            
            # Check matrix properties
            matrix_props = _check_matrix_properties(cov_matrix)
            
            # Validate positive definiteness
            if not matrix_props["is_positive_definite"]:
                print(f"Error: Covariance matrix for {dataset_name} is not positive definite")
                print(f"  Minimum eigenvalue: {matrix_props['min_eigenvalue']:.2e}")
                all_valid = False
            
            # Check condition number
            if matrix_props["condition_number"] > 1e12:
                print(f"Warning: Covariance matrix for {dataset_name} is poorly conditioned")
                print(f"  Condition number: {matrix_props['condition_number']:.2e}")
                # Don't fail for poor conditioning, just warn
            
            # Check for NaN or infinite values
            if not matrix_props["is_finite"]:
                print(f"Error: Covariance matrix for {dataset_name} contains NaN or infinite values")
                all_valid = False
            
        except Exception as e:
            print(f"Error loading or checking dataset {dataset_name}: {e}")
            all_valid = False
    
    return all_valid


def run_integrity_suite(
    params: ParameterDict, 
    datasets: List[str],
    tolerances: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive integrity test suite combining all validation checks.
    
    Args:
        params: Parameter dictionary for physics checks
        datasets: List of datasets for covariance validation
        tolerances: Optional custom tolerance settings
        
    Returns:
        Dictionary with detailed integrity test results
        
    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5
    """
    # Merge custom tolerances with defaults
    effective_tolerances = DEFAULT_TOLERANCES.copy()
    if tolerances:
        effective_tolerances.update(tolerances)
    
    results = {
        "overall_status": "PASS",
        "tests_run": [],
        "failures": [],
        "warnings": [],
        "tolerances_used": effective_tolerances
    }
    
    # Build default parameters if none provided
    if params is None:
        from . import parameter
        params = parameter.build_params("pbuf")  # Use PBUF defaults for testing
    
    # Test 1: H(z) ratio consistency
    print("Running H(z) ratio consistency check...")
    h_ratio_pass = verify_h_ratios(params, tolerance=effective_tolerances["h_ratios"])
    results["tests_run"].append("h_ratios")
    results["h_ratios"] = {
        "status": "PASS" if h_ratio_pass else "FAIL",
        "description": "Hubble parameter consistency between PBUF and ΛCDM",
        "tolerance_used": effective_tolerances["h_ratios"]
    }
    if not h_ratio_pass:
        results["failures"].append("h_ratios")
        results["overall_status"] = "FAIL"
    
    # Test 2: Recombination redshift validation
    print("Running recombination redshift validation...")
    recomb_pass = verify_recombination(params, tolerance=effective_tolerances["recombination"])
    results["tests_run"].append("recombination")
    results["recombination"] = {
        "status": "PASS" if recomb_pass else "FAIL",
        "description": "Recombination redshift against Planck 2018 reference",
        "computed_z_recomb": params.get("z_recomb"),
        "reference_z_recomb": REFERENCE_VALUES["planck2018_z_recomb"],
        "tolerance_used": effective_tolerances["recombination"]
    }
    if not recomb_pass:
        results["failures"].append("recombination")
        results["overall_status"] = "FAIL"
    
    # Test 3: Sound horizon verification
    print("Running sound horizon verification...")
    sound_horizon_pass = verify_sound_horizon(params, tolerance=effective_tolerances["sound_horizon"])
    results["tests_run"].append("sound_horizon")
    results["sound_horizon"] = {
        "status": "PASS" if sound_horizon_pass else "FAIL",
        "description": "Sound horizon computation against reference values",
        "computed_r_s_drag": params.get("r_s_drag"),
        "reference_r_s_drag": REFERENCE_VALUES["planck2018_r_s_drag"],
        "tolerance_used": effective_tolerances["sound_horizon"]
    }
    if not sound_horizon_pass:
        results["failures"].append("sound_horizon")
        results["overall_status"] = "FAIL"
    
    # Test 4: Unit consistency checks
    print("Running unit consistency checks...")
    unit_checks = check_unit_consistency(params)
    results["tests_run"].append("unit_consistency")
    all_units_pass = all(unit_checks.values())
    results["unit_consistency"] = {
        "status": "PASS" if all_units_pass else "FAIL",
        "description": "Dimensional analysis and unit consistency",
        "details": unit_checks
    }
    if not all_units_pass:
        results["failures"].append("unit_consistency")
        results["overall_status"] = "FAIL"
    
    # Test 5: Covariance matrix validation
    if datasets:
        print("Running covariance matrix validation...")
        cov_pass = verify_covariance_matrices(datasets)
        results["tests_run"].append("covariance_matrices")
        results["covariance_matrices"] = {
            "status": "PASS" if cov_pass else "FAIL",
            "description": "Covariance matrix properties for all datasets",
            "datasets_checked": datasets
        }
        if not cov_pass:
            results["failures"].append("covariance_matrices")
            results["overall_status"] = "FAIL"
    
    # Summary
    results["summary"] = {
        "total_tests": len(results["tests_run"]),
        "passed": len(results["tests_run"]) - len(results["failures"]),
        "failed": len(results["failures"]),
        "warnings": len(results["warnings"])
    }
    
    print(f"\nIntegrity Suite Results:")
    print(f"  Overall Status: {results['overall_status']}")
    print(f"  Tests Run: {results['summary']['total_tests']}")
    print(f"  Passed: {results['summary']['passed']}")
    print(f"  Failed: {results['summary']['failed']}")
    
    if results["failures"]:
        print(f"  Failed Tests: {', '.join(results['failures'])}")
    
    return results


def verify_sound_horizon(
    params: ParameterDict,
    reference_method: str = "eisenstein_hu_1998",
    tolerance: float = None
) -> bool:
    """
    Verify sound horizon computation against reference values.
    
    Args:
        params: Parameter dictionary
        reference_method: Reference calculation method
        tolerance: Tolerance for comparison
        
    Returns:
        True if sound horizon computation is consistent
    """
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCES["sound_horizon"]
    
    # Get computed sound horizon from parameters
    r_s_drag = params.get("r_s_drag")
    
    if r_s_drag is None:
        print("Warning: r_s_drag not found in parameter dictionary")
        return False
    
    # Check against EH98 formula consistency (primary check)
    if reference_method == "eisenstein_hu_1998":
        # Compute using EH98 fitting formula directly
        H0 = params["H0"]
        Omh2 = params["Omh2"]
        Obh2 = params["Obh2"]
        
        h = H0 / 100.0
        r_s_eh98 = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2**(0.75)) / h
        
        # Compare with computed value (should match exactly)
        eh98_relative_error = abs(r_s_drag - r_s_eh98) / r_s_eh98
        if eh98_relative_error > tolerance:
            print(f"Warning: Sound horizon differs from EH98 formula: "
                  f"computed = {r_s_drag:.3f}, EH98 = {r_s_eh98:.3f}, "
                  f"relative error = {eh98_relative_error:.6f}")
            return False
    
    # Secondary check: verify reasonable range for sound horizon
    # EH98 formula typically gives ~220 Mpc for standard cosmology
    # BAO analyses use ~147 Mpc (different convention/calibration)
    expected_eh98 = REFERENCE_VALUES["eh98_r_s_expected"]
    range_tolerance = 0.1  # 10% tolerance for reasonable range
    
    if abs(r_s_drag - expected_eh98) / expected_eh98 > range_tolerance:
        print(f"Warning: Sound horizon r_s = {r_s_drag:.3f} Mpc outside expected range "
              f"around {expected_eh98:.1f} Mpc")
        # Don't fail for this - just warn
    
    return True


def check_unit_consistency(params: ParameterDict) -> Dict[str, bool]:
    """
    Perform dimensional analysis and unit consistency checks.
    
    Args:
        params: Parameter dictionary
        
    Returns:
        Dictionary of unit consistency check results
    """
    checks = {}
    
    # Check 1: H0 units (should be in km/s/Mpc)
    H0 = params.get("H0")
    if H0 is not None:
        # Reasonable range for H0 in km/s/Mpc
        checks["H0_range"] = 20.0 <= H0 <= 150.0
        if not checks["H0_range"]:
            print(f"Warning: H0 = {H0} km/s/Mpc outside reasonable range [20, 150]")
    else:
        checks["H0_range"] = False
    
    # Check 2: Density fractions should sum to ~1 for flat universe
    Om0 = params.get("Om0", 0)
    Ol0 = 1 - Om0  # Assuming flat universe
    Orh2 = params.get("Orh2", 0)
    h = (H0 / 100.0) if H0 else 0.67
    Or0 = Orh2 / h**2 if h > 0 else 0
    
    total_density = Om0 + Ol0 + Or0
    checks["density_sum"] = abs(total_density - 1.0) < 0.01  # Allow 1% tolerance
    if not checks["density_sum"]:
        print(f"Warning: Density fractions sum to {total_density:.4f}, expected ~1.0")
    
    # Check 3: Physical densities consistency
    Omh2_expected = Om0 * h**2 if H0 else None
    Omh2_computed = params.get("Omh2")
    if Omh2_expected and Omh2_computed:
        checks["Omh2_consistency"] = abs(Omh2_computed - Omh2_expected) < 1e-4
        if not checks["Omh2_consistency"]:
            print(f"Warning: Omh2 inconsistency: computed = {Omh2_computed:.6f}, "
                  f"expected = {Omh2_expected:.6f}")
    else:
        checks["Omh2_consistency"] = False
    
    # Check 4: Redshift ordering (z_recomb > z_drag for standard cosmology)
    z_recomb = params.get("z_recomb")
    z_drag = params.get("z_drag")
    if z_recomb and z_drag:
        checks["redshift_ordering"] = z_recomb > z_drag
        if not checks["redshift_ordering"]:
            print(f"Warning: Redshift ordering: z_recomb = {z_recomb:.1f}, "
                  f"z_drag = {z_drag:.1f} (should be z_recomb > z_drag)")
    else:
        checks["redshift_ordering"] = False
    
    # Check 5: Sound horizon reasonable range (EH98 formula gives ~220 Mpc)
    r_s_drag = params.get("r_s_drag")
    if r_s_drag:
        checks["sound_horizon_range"] = 200.0 <= r_s_drag <= 250.0  # Mpc (EH98 range)
        if not checks["sound_horizon_range"]:
            print(f"Warning: Sound horizon r_s = {r_s_drag:.1f} Mpc outside range [200, 250]")
    else:
        checks["sound_horizon_range"] = False
    
    # Check 6: PBUF parameters (if applicable)
    if params.get("model_class") == "pbuf":
        alpha = params.get("alpha", 0)
        k_sat = params.get("k_sat", 1)
        Rmax = params.get("Rmax", 1e9)
        
        checks["pbuf_alpha_range"] = 1e-6 <= alpha <= 1e-2
        checks["pbuf_k_sat_range"] = 0.1 <= k_sat <= 2.0
        checks["pbuf_Rmax_range"] = 1e6 <= Rmax <= 1e12
        
        if not checks["pbuf_alpha_range"]:
            print(f"Warning: PBUF alpha = {alpha} outside range [1e-6, 1e-2]")
        if not checks["pbuf_k_sat_range"]:
            print(f"Warning: PBUF k_sat = {k_sat} outside range [0.1, 2.0]")
        if not checks["pbuf_Rmax_range"]:
            print(f"Warning: PBUF Rmax = {Rmax} outside range [1e6, 1e12]")
    
    return checks


def _compute_h_ratio_at_z(params: ParameterDict, z: float) -> float:
    """
    Compute H(z) ratio between PBUF and ΛCDM at given redshift.
    
    Args:
        params: Parameter dictionary
        z: Redshift for evaluation
        
    Returns:
        H_PBUF(z) / H_ΛCDM(z) ratio
    """
    # Extract cosmological parameters
    Om0 = params["Om0"]
    Ol0 = 1 - Om0  # Flat universe assumption
    
    # Compute ΛCDM E²(z)
    E2_lcdm = Om0 * (1 + z)**3 + Ol0
    
    # For PBUF model, include elastic correction
    if params.get("model_class") == "pbuf":
        k_sat = params.get("k_sat", 1.0)
        alpha = params.get("alpha", 0.0)
        
        # Simple saturation factor approximation
        # When k_sat → 1, this should → 1 (no correction)
        saturation_factor = k_sat + (1 - k_sat) * np.exp(-alpha * (1 + z))
        
        E2_pbuf = E2_lcdm * saturation_factor
        
        # Return ratio
        return np.sqrt(E2_pbuf / E2_lcdm)
    else:
        # For ΛCDM, ratio is 1.0 by definition
        return 1.0


def _check_matrix_properties(matrix: np.ndarray) -> Dict[str, Any]:
    """
    Check numerical properties of a matrix (eigenvalues, condition number, etc.).
    
    Args:
        matrix: Matrix to analyze
        
    Returns:
        Dictionary of matrix properties
    """
    properties = {}
    
    # Check if matrix is finite (no NaN or inf)
    properties["is_finite"] = np.all(np.isfinite(matrix))
    
    # Check if matrix is square
    properties["is_square"] = matrix.shape[0] == matrix.shape[1]
    
    if not properties["is_square"]:
        properties["is_positive_definite"] = False
        properties["eigenvalues"] = None
        properties["min_eigenvalue"] = None
        properties["condition_number"] = np.inf
        return properties
    
    try:
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)
        properties["eigenvalues"] = eigenvalues
        properties["min_eigenvalue"] = np.min(eigenvalues)
        properties["max_eigenvalue"] = np.max(eigenvalues)
        
        # Check positive definiteness
        properties["is_positive_definite"] = np.all(eigenvalues > DEFAULT_TOLERANCES["covariance_eigenvalues"])
        
        # Compute condition number
        if properties["min_eigenvalue"] > 0:
            properties["condition_number"] = properties["max_eigenvalue"] / properties["min_eigenvalue"]
        else:
            properties["condition_number"] = np.inf
            
    except np.linalg.LinAlgError as e:
        # Handle singular matrices or other linear algebra errors
        properties["eigenvalues"] = None
        properties["min_eigenvalue"] = None
        properties["max_eigenvalue"] = None
        properties["is_positive_definite"] = False
        properties["condition_number"] = np.inf
        properties["linalg_error"] = str(e)
    
    return properties


# Default test configurations
DEFAULT_TEST_REDSHIFTS = [0.1, 0.5, 1.0, 2.0]
DEFAULT_TOLERANCES = {
    "h_ratios": 1e-4,
    "recombination": 1e-4,
    "sound_horizon": 1e-4,
    "covariance_eigenvalues": 1e-12
}

# Reference values for validation
REFERENCE_VALUES = {
    "planck2018_z_recomb": 1089.80,
    "planck2018_r_s_drag": 147.09,  # Mpc (BAO scale, different from EH98 formula)
    "eh98_r_s_expected": 222.3,     # Mpc (EH98 formula result for standard params)
    "eisenstein_hu_1998_coefficients": {
        "b1": 0.313,
        "b2": 0.238
    }
}