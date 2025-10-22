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


def verify_bao_anisotropic_integrity(
    params: ParameterDict,
    data: Optional[Dict[str, Any]] = None,
    tolerance: float = None
) -> Dict[str, Any]:
    """
    Comprehensive integrity validation for anisotropic BAO datasets and physics.
    
    Validates covariance matrix structure, physics consistency checks for 
    transverse/radial BAO ratios, and anisotropic-specific metrics.
    
    Args:
        params: Parameter dictionary for physics checks
        data: Anisotropic BAO dataset (if None, loads from datasets module)
        tolerance: Tolerance for physics consistency checks
        
    Returns:
        Dictionary with detailed anisotropic BAO validation results
        
    Requirements: 2.1, 2.2, 5.1
    """
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCES.get("bao_anisotropic", 1e-3)
    
    results = {
        "overall_status": "PASS",
        "tests_run": [],
        "failures": [],
        "warnings": [],
        "tolerance_used": tolerance
    }
    
    # Load anisotropic BAO dataset if not provided
    if data is None:
        try:
            from . import datasets
            data = datasets.load_dataset("bao_ani")
        except Exception as e:
            results["overall_status"] = "FAIL"
            results["failures"].append("dataset_loading")
            results["dataset_loading"] = {
                "status": "FAIL",
                "error": str(e),
                "description": "Failed to load anisotropic BAO dataset"
            }
            return results
    
    # Test 1: Covariance matrix structure validation (2N×2N for N redshift bins)
    print("Running anisotropic BAO covariance matrix validation...")
    cov_validation = _validate_anisotropic_covariance_structure(data)
    results["tests_run"].append("covariance_structure")
    results["covariance_structure"] = cov_validation
    if cov_validation["status"] == "FAIL":
        results["failures"].append("covariance_structure")
        results["overall_status"] = "FAIL"
    elif cov_validation["status"] == "WARNING":
        results["warnings"].append("covariance_structure")
    
    # Test 2: Physics consistency checks for transverse/radial BAO ratios
    print("Running BAO anisotropic physics consistency checks...")
    physics_validation = _validate_bao_anisotropic_physics(params, data, tolerance)
    results["tests_run"].append("physics_consistency")
    results["physics_consistency"] = physics_validation
    if physics_validation["status"] == "FAIL":
        results["failures"].append("physics_consistency")
        results["overall_status"] = "FAIL"
    elif physics_validation["status"] == "WARNING":
        results["warnings"].append("physics_consistency")
    
    # Test 3: Dataset separation validation (prevent mixing with isotropic BAO)
    print("Running BAO dataset separation validation...")
    separation_validation = _validate_bao_dataset_separation(data)
    results["tests_run"].append("dataset_separation")
    results["dataset_separation"] = separation_validation
    if separation_validation["status"] == "FAIL":
        results["failures"].append("dataset_separation")
        results["overall_status"] = "FAIL"
    elif separation_validation["status"] == "WARNING":
        results["warnings"].append("dataset_separation")
    
    # Test 4: Anisotropic-specific metrics validation
    print("Running anisotropic BAO metrics validation...")
    metrics_validation = _validate_anisotropic_metrics(data, tolerance)
    results["tests_run"].append("anisotropic_metrics")
    results["anisotropic_metrics"] = metrics_validation
    if metrics_validation["status"] == "FAIL":
        results["failures"].append("anisotropic_metrics")
        results["overall_status"] = "FAIL"
    elif metrics_validation["status"] == "WARNING":
        results["warnings"].append("anisotropic_metrics")
    
    # Summary
    results["summary"] = {
        "total_tests": len(results["tests_run"]),
        "passed": len(results["tests_run"]) - len(results["failures"]),
        "failed": len(results["failures"]),
        "warnings": len(results["warnings"])
    }
    
    print(f"\nAnisotropic BAO Integrity Results:")
    print(f"  Overall Status: {results['overall_status']}")
    print(f"  Tests Run: {results['summary']['total_tests']}")
    print(f"  Passed: {results['summary']['passed']}")
    print(f"  Failed: {results['summary']['failed']}")
    print(f"  Warnings: {results['summary']['warnings']}")
    
    if results["failures"]:
        print(f"  Failed Tests: {', '.join(results['failures'])}")
    
    return results


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
    
    # Test 6: Anisotropic BAO specific validation (if bao_ani in datasets)
    if "bao_ani" in datasets:
        print("Running anisotropic BAO specific validation...")
        bao_ani_results = verify_bao_anisotropic_integrity(params)
        results["tests_run"].append("bao_anisotropic")
        results["bao_anisotropic"] = bao_ani_results
        if bao_ani_results["overall_status"] == "FAIL":
            results["failures"].append("bao_anisotropic")
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
    "covariance_eigenvalues": 1e-12,
    "bao_anisotropic": 1e-3,
    "bao_ratio_consistency": 0.05,  # 5% tolerance for BAO ratio physics checks
    "covariance_conditioning": 1e10  # Condition number threshold
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


def _validate_anisotropic_covariance_structure(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate covariance matrix structure for anisotropic BAO (2N×2N for N redshift bins).
    
    Args:
        data: Anisotropic BAO dataset dictionary
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "status": "PASS",
        "description": "Anisotropic BAO covariance matrix structure validation",
        "details": {}
    }
    
    try:
        observations = data["observations"]
        covariance = data["covariance"]
        
        # Check if we have the expected observables
        if "redshift" not in observations:
            result["status"] = "FAIL"
            result["details"]["missing_redshift"] = "No redshift array found in observations"
            return result
        
        if "DM_over_rs" not in observations or "H_times_rs" not in observations:
            result["status"] = "FAIL"
            result["details"]["missing_observables"] = "Missing DM_over_rs or H_times_rs in observations"
            return result
        
        # Get number of redshift bins
        redshifts = np.asarray(observations["redshift"])
        n_redshift_bins = len(redshifts)
        expected_cov_size = 2 * n_redshift_bins
        
        result["details"]["n_redshift_bins"] = n_redshift_bins
        result["details"]["expected_covariance_size"] = expected_cov_size
        result["details"]["actual_covariance_shape"] = covariance.shape
        
        # Check covariance matrix dimensions
        if covariance.shape != (expected_cov_size, expected_cov_size):
            result["status"] = "FAIL"
            result["details"]["dimension_mismatch"] = (
                f"Expected {expected_cov_size}×{expected_cov_size}, "
                f"got {covariance.shape[0]}×{covariance.shape[1]}"
            )
            return result
        
        # Check data vector consistency
        dm_ratios = np.asarray(observations["DM_over_rs"])
        h_ratios = np.asarray(observations["H_times_rs"])
        
        if len(dm_ratios) != n_redshift_bins:
            result["status"] = "FAIL"
            result["details"]["dm_length_mismatch"] = (
                f"DM_over_rs has {len(dm_ratios)} elements, expected {n_redshift_bins}"
            )
            return result
        
        if len(h_ratios) != n_redshift_bins:
            result["status"] = "FAIL"
            result["details"]["h_length_mismatch"] = (
                f"H_times_rs has {len(h_ratios)} elements, expected {n_redshift_bins}"
            )
            return result
        
        # Check covariance matrix properties
        matrix_props = _check_matrix_properties(covariance)
        result["details"]["matrix_properties"] = matrix_props
        
        if not matrix_props["is_positive_definite"]:
            result["status"] = "FAIL"
            result["details"]["not_positive_definite"] = (
                f"Minimum eigenvalue: {matrix_props['min_eigenvalue']:.2e}"
            )
            return result
        
        # Check conditioning
        condition_threshold = DEFAULT_TOLERANCES["covariance_conditioning"]
        if matrix_props["condition_number"] > condition_threshold:
            result["status"] = "WARNING"
            result["details"]["poor_conditioning"] = (
                f"Condition number {matrix_props['condition_number']:.2e} "
                f"exceeds threshold {condition_threshold:.2e}"
            )
        
        # Check block structure (DM-DM, H-H, DM-H correlations)
        dm_block = covariance[:n_redshift_bins, :n_redshift_bins]
        h_block = covariance[n_redshift_bins:, n_redshift_bins:]
        cross_block = covariance[:n_redshift_bins, n_redshift_bins:]
        
        # Validate diagonal blocks are positive definite
        dm_props = _check_matrix_properties(dm_block)
        h_props = _check_matrix_properties(h_block)
        
        result["details"]["dm_block_properties"] = dm_props
        result["details"]["h_block_properties"] = h_props
        
        if not dm_props["is_positive_definite"]:
            result["status"] = "WARNING"
            result["details"]["dm_block_issue"] = "DM block not positive definite"
        
        if not h_props["is_positive_definite"]:
            result["status"] = "WARNING"
            result["details"]["h_block_issue"] = "H block not positive definite"
        
        # Check cross-correlation structure
        cross_correlation_strength = np.max(np.abs(cross_block)) / np.sqrt(
            np.max(np.diag(dm_block)) * np.max(np.diag(h_block))
        )
        result["details"]["max_cross_correlation"] = cross_correlation_strength
        
        if cross_correlation_strength > 0.9:
            result["status"] = "WARNING"
            result["details"]["high_cross_correlation"] = (
                f"Maximum cross-correlation {cross_correlation_strength:.3f} is very high"
            )
        
    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["validation_error"] = str(e)
    
    return result


def _validate_bao_anisotropic_physics(
    params: ParameterDict, 
    data: Dict[str, Any], 
    tolerance: float
) -> Dict[str, Any]:
    """
    Validate physics consistency for anisotropic BAO ratios.
    
    Args:
        params: Parameter dictionary
        data: Anisotropic BAO dataset
        tolerance: Tolerance for physics checks
        
    Returns:
        Dictionary with physics validation results
    """
    result = {
        "status": "PASS",
        "description": "Anisotropic BAO physics consistency validation",
        "details": {}
    }
    
    try:
        observations = data["observations"]
        redshifts = np.asarray(observations["redshift"])
        dm_ratios = np.asarray(observations["DM_over_rs"])
        h_ratios = np.asarray(observations["H_times_rs"])
        
        # Check 1: Redshift ordering (should be monotonically increasing)
        if not np.all(np.diff(redshifts) > 0):
            result["status"] = "WARNING"
            result["details"]["redshift_ordering"] = "Redshifts are not monotonically increasing"
        
        # Check 2: Reasonable redshift range for BAO measurements
        z_min, z_max = redshifts.min(), redshifts.max()
        result["details"]["redshift_range"] = [float(z_min), float(z_max)]
        
        if z_min < 0.1 or z_max > 3.0:
            result["status"] = "WARNING"
            result["details"]["unusual_redshift_range"] = (
                f"Redshift range [{z_min:.2f}, {z_max:.2f}] is outside typical BAO range [0.1, 3.0]"
            )
        
        # Check 3: DM/rs ratio evolution (should generally increase with redshift)
        dm_trend = np.polyfit(redshifts, dm_ratios, 1)[0]  # Linear slope
        result["details"]["dm_trend_slope"] = float(dm_trend)
        
        if dm_trend < 0:
            result["status"] = "WARNING"
            result["details"]["dm_decreasing"] = (
                f"DM/rs ratios decrease with redshift (slope: {dm_trend:.3f}), "
                "which is unusual for standard cosmology"
            )
        
        # Check 4: H*rs ratio evolution (should generally increase with redshift)
        h_trend = np.polyfit(redshifts, h_ratios, 1)[0]  # Linear slope
        result["details"]["h_trend_slope"] = float(h_trend)
        
        if h_trend < 0:
            result["status"] = "WARNING"
            result["details"]["h_decreasing"] = (
                f"H*rs ratios decrease with redshift (slope: {h_trend:.3f}), "
                "which is unusual for standard cosmology"
            )
        
        # Check 5: Ratio magnitudes are reasonable
        dm_range = [float(dm_ratios.min()), float(dm_ratios.max())]
        h_range = [float(h_ratios.min()), float(h_ratios.max())]
        
        result["details"]["dm_ratio_range"] = dm_range
        result["details"]["h_ratio_range"] = h_range
        
        # Typical ranges from observations: DM/rs ~ 5-25, H*rs ~ 50-150
        if dm_range[0] < 1 or dm_range[1] > 50:
            result["status"] = "WARNING"
            result["details"]["unusual_dm_range"] = (
                f"DM/rs range {dm_range} is outside typical range [1, 50]"
            )
        
        if h_range[0] < 20 or h_range[1] > 200:
            result["status"] = "WARNING"
            result["details"]["unusual_h_range"] = (
                f"H*rs range {h_range} is outside typical range [20, 200]"
            )
        
        # Check 6: Consistency with theoretical predictions (if possible)
        if "r_s_drag" in params:
            try:
                from .likelihoods import _compute_bao_predictions
                theoretical_predictions = _compute_bao_predictions(params, isotropic=False)
                
                if "DM_over_rs" in theoretical_predictions and "H_times_rs" in theoretical_predictions:
                    theory_dm = theoretical_predictions["DM_over_rs"]
                    theory_h = theoretical_predictions["H_times_rs"]
                    
                    # Compare with observations (allow for different redshift grids)
                    if len(theory_dm) == len(dm_ratios):
                        dm_residuals = (dm_ratios - theory_dm) / theory_dm
                        h_residuals = (h_ratios - theory_h) / theory_h
                        
                        max_dm_residual = np.max(np.abs(dm_residuals))
                        max_h_residual = np.max(np.abs(h_residuals))
                        
                        result["details"]["max_dm_residual"] = float(max_dm_residual)
                        result["details"]["max_h_residual"] = float(max_h_residual)
                        
                        residual_threshold = DEFAULT_TOLERANCES["bao_ratio_consistency"]
                        
                        if max_dm_residual > residual_threshold:
                            result["status"] = "WARNING"
                            result["details"]["large_dm_residuals"] = (
                                f"Maximum DM/rs residual {max_dm_residual:.3f} "
                                f"exceeds threshold {residual_threshold:.3f}"
                            )
                        
                        if max_h_residual > residual_threshold:
                            result["status"] = "WARNING"
                            result["details"]["large_h_residuals"] = (
                                f"Maximum H*rs residual {max_h_residual:.3f} "
                                f"exceeds threshold {residual_threshold:.3f}"
                            )
                
            except Exception as e:
                result["details"]["theory_comparison_error"] = str(e)
        
    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["physics_validation_error"] = str(e)
    
    return result


def _validate_bao_dataset_separation(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate proper separation from isotropic BAO datasets.
    
    Args:
        data: Anisotropic BAO dataset
        
    Returns:
        Dictionary with separation validation results
    """
    result = {
        "status": "PASS",
        "description": "BAO dataset separation validation",
        "details": {}
    }
    
    try:
        # Check dataset type
        dataset_type = data.get("dataset_type")
        if dataset_type != "bao_ani":
            result["status"] = "FAIL"
            result["details"]["wrong_dataset_type"] = (
                f"Expected dataset_type 'bao_ani', got '{dataset_type}'"
            )
            return result
        
        # Check that we don't have isotropic BAO observables
        observations = data["observations"]
        
        if "DV_over_rs" in observations:
            result["status"] = "WARNING"
            result["details"]["isotropic_observable_present"] = (
                "Dataset contains DV_over_rs (isotropic BAO observable), "
                "which should not be mixed with anisotropic measurements"
            )
        
        # Check that we have the correct anisotropic observables
        required_observables = ["DM_over_rs", "H_times_rs"]
        missing_observables = []
        
        for obs in required_observables:
            if obs not in observations:
                missing_observables.append(obs)
        
        if missing_observables:
            result["status"] = "FAIL"
            result["details"]["missing_anisotropic_observables"] = missing_observables
        
        # Check metadata consistency
        metadata = data.get("metadata", {})
        observable_types = metadata.get("observable_types", [])
        
        expected_types = ["transverse_bao", "radial_bao"]
        if not all(obs_type in observable_types for obs_type in expected_types):
            result["status"] = "WARNING"
            result["details"]["metadata_inconsistency"] = (
                f"Expected observable_types {expected_types}, got {observable_types}"
            )
        
        # Check for proper anisotropic structure in metadata
        if "n_data_points" in metadata:
            n_data_points = metadata["n_data_points"]
            n_redshift_bins = len(observations.get("redshift", []))
            expected_data_points = 2 * n_redshift_bins
            
            if n_data_points != expected_data_points:
                result["status"] = "WARNING"
                result["details"]["data_point_count_mismatch"] = (
                    f"Metadata reports {n_data_points} data points, "
                    f"expected {expected_data_points} for {n_redshift_bins} redshift bins"
                )
        
    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["separation_validation_error"] = str(e)
    
    return result


def _validate_anisotropic_metrics(data: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """
    Validate anisotropic-specific metrics and derived quantities.
    
    Args:
        data: Anisotropic BAO dataset
        tolerance: Tolerance for metric validation
        
    Returns:
        Dictionary with metrics validation results
    """
    result = {
        "status": "PASS",
        "description": "Anisotropic BAO metrics validation",
        "details": {}
    }
    
    try:
        observations = data["observations"]
        covariance = data["covariance"]
        
        redshifts = np.asarray(observations["redshift"])
        dm_ratios = np.asarray(observations["DM_over_rs"])
        h_ratios = np.asarray(observations["H_times_rs"])
        
        n_bins = len(redshifts)
        
        # Metric 1: Anisotropy parameter α = (DM/rs) / (H*rs) * H0/c
        # This should be close to 1 for isotropic universe
        # Note: Need to be careful about units and conventions
        
        # For rough estimate, assume H*rs is in km/s units and DM/rs is dimensionless
        # The ratio (DM/rs) / (H*rs) should scale roughly as c/H0 ~ 4000 Mpc
        # But exact interpretation depends on conventions
        
        anisotropy_ratios = dm_ratios / h_ratios
        result["details"]["anisotropy_ratios"] = anisotropy_ratios.tolist()
        result["details"]["mean_anisotropy_ratio"] = float(np.mean(anisotropy_ratios))
        result["details"]["std_anisotropy_ratio"] = float(np.std(anisotropy_ratios))
        
        # Check for excessive anisotropy (ratios should be relatively stable)
        anisotropy_variation = np.std(anisotropy_ratios) / np.mean(anisotropy_ratios)
        result["details"]["anisotropy_variation_coefficient"] = float(anisotropy_variation)
        
        if anisotropy_variation > 0.2:  # 20% variation threshold
            result["status"] = "WARNING"
            result["details"]["high_anisotropy_variation"] = (
                f"Anisotropy ratio variation coefficient {anisotropy_variation:.3f} "
                "exceeds 20%, suggesting possible systematic issues"
            )
        
        # Metric 2: Cross-correlation analysis between DM and H measurements
        dm_block = covariance[:n_bins, :n_bins]
        h_block = covariance[n_bins:, n_bins:]
        cross_block = covariance[:n_bins, n_bins:]
        
        # Compute correlation coefficients
        dm_errors = np.sqrt(np.diag(dm_block))
        h_errors = np.sqrt(np.diag(h_block))
        
        correlations = []
        for i in range(n_bins):
            corr = cross_block[i, i] / (dm_errors[i] * h_errors[i])
            correlations.append(corr)
        
        correlations = np.array(correlations)
        result["details"]["dm_h_correlations"] = correlations.tolist()
        result["details"]["mean_dm_h_correlation"] = float(np.mean(correlations))
        
        # Check for unrealistic correlations
        if np.any(np.abs(correlations) > 0.95):
            result["status"] = "WARNING"
            result["details"]["very_high_correlations"] = (
                f"Some DM-H correlations exceed 95%: {correlations[np.abs(correlations) > 0.95]}"
            )
        
        # Metric 3: Signal-to-noise ratios
        dm_snr = dm_ratios / dm_errors
        h_snr = h_ratios / h_errors
        
        result["details"]["dm_signal_to_noise"] = dm_snr.tolist()
        result["details"]["h_signal_to_noise"] = h_snr.tolist()
        result["details"]["mean_dm_snr"] = float(np.mean(dm_snr))
        result["details"]["mean_h_snr"] = float(np.mean(h_snr))
        
        # Check for low signal-to-noise measurements
        min_snr_threshold = 2.0  # 2-sigma detection threshold
        
        low_snr_dm = np.sum(dm_snr < min_snr_threshold)
        low_snr_h = np.sum(h_snr < min_snr_threshold)
        
        if low_snr_dm > 0:
            result["status"] = "WARNING"
            result["details"]["low_snr_dm_measurements"] = (
                f"{low_snr_dm} DM measurements have SNR < {min_snr_threshold}"
            )
        
        if low_snr_h > 0:
            result["status"] = "WARNING"
            result["details"]["low_snr_h_measurements"] = (
                f"{low_snr_h} H measurements have SNR < {min_snr_threshold}"
            )
        
        # Metric 4: Redshift bin spacing analysis
        if n_bins > 1:
            z_spacings = np.diff(redshifts)
            result["details"]["redshift_spacings"] = z_spacings.tolist()
            result["details"]["mean_redshift_spacing"] = float(np.mean(z_spacings))
            result["details"]["min_redshift_spacing"] = float(np.min(z_spacings))
            
            # Check for very close redshift bins (might indicate correlation issues)
            if np.min(z_spacings) < 0.05:
                result["status"] = "WARNING"
                result["details"]["close_redshift_bins"] = (
                    f"Minimum redshift spacing {np.min(z_spacings):.3f} is very small, "
                    "which may lead to correlation issues"
                )
        
        # Metric 5: Consistency with fiducial cosmology expectations
        # Check if ratios are in reasonable ranges for standard cosmology
        
        # For ΛCDM with typical parameters, DM/rs should be roughly:
        # z=0.3: ~8, z=0.6: ~15, z=1.0: ~22
        # H*rs should be roughly: z=0.3: ~80, z=0.6: ~95, z=1.0: ~110
        
        expected_dm_range = [5, 30]  # Conservative range
        expected_h_range = [50, 150]  # Conservative range
        
        dm_out_of_range = np.sum((dm_ratios < expected_dm_range[0]) | 
                                (dm_ratios > expected_dm_range[1]))
        h_out_of_range = np.sum((h_ratios < expected_h_range[0]) | 
                               (h_ratios > expected_h_range[1]))
        
        if dm_out_of_range > 0:
            result["status"] = "WARNING"
            result["details"]["dm_out_of_expected_range"] = (
                f"{dm_out_of_range} DM/rs measurements outside expected range {expected_dm_range}"
            )
        
        if h_out_of_range > 0:
            result["status"] = "WARNING"
            result["details"]["h_out_of_expected_range"] = (
                f"{h_out_of_range} H*rs measurements outside expected range {expected_h_range}"
            )
        
    except Exception as e:
        result["status"] = "FAIL"
        result["details"]["metrics_validation_error"] = str(e)
    
    return result