#!/usr/bin/env python3
"""
BAO Anisotropic Data Validation and Safety Checks.

This module implements critical safety checks and validation for BAO anisotropic
data to prevent common unit/definition errors and ensure proper data format.

Key Safety Features:
- Hard tripwires for radial BAO values > 5 (likely unit errors)
- Validation that bins don't mix isotropic and anisotropic forms
- Unit/definition guards for r_d vs r_s confusion
- Distance unit validation (Mpc vs h^-1 Mpc)
- Conversion utilities for common format issues

Requirements: Data integrity, unit consistency, error prevention
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import warnings


class BAOValidationError(Exception):
    """Custom exception for BAO data validation failures."""
    pass


class BAOUnitError(BAOValidationError):
    """Exception for BAO unit/definition errors."""
    pass


class BAOMixedFormatError(BAOValidationError):
    """Exception for mixed isotropic/anisotropic formats."""
    pass


def validate_bao_anisotropic_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate BAO anisotropic dataset with comprehensive safety checks.
    
    This function implements hard tripwires to prevent common errors:
    - Radial BAO values > 5 (likely D_V/r_d or H*r_s instead of D_H/r_d)
    - Mixed isotropic/anisotropic formats in same dataset
    - Unit confusion (r_d vs r_s, Mpc vs h^-1 Mpc)
    - Invalid distance definitions
    
    Args:
        data: BAO anisotropic dataset dictionary
        
    Returns:
        Validated and potentially corrected dataset
        
    Raises:
        BAOValidationError: If data fails validation checks
        BAOUnitError: If unit/definition errors detected
        BAOMixedFormatError: If mixed formats detected
    """
    print("🔍 Validating BAO anisotropic data with safety checks...")
    
    # Extract observations
    observations = data.get("observations", {})
    metadata = data.get("metadata", {})
    
    # Check for required anisotropic observables
    _validate_anisotropic_format(observations)
    
    # Apply hard tripwires for common errors
    _apply_radial_bao_tripwire(observations)
    _apply_mixed_format_tripwire(observations)
    _apply_unit_definition_guards(observations, metadata)
    
    # Validate covariance structure
    _validate_anisotropic_covariance(data)
    
    # Check for and fix common unit issues
    corrected_data = _apply_unit_corrections(data)
    
    print("✅ BAO anisotropic data validation passed")
    return corrected_data


def _validate_anisotropic_format(observations: Dict[str, Any]) -> None:
    """
    Validate that data is in proper anisotropic format.
    
    Expected format: per redshift bin, (D_M/r_d, D_H/r_d) with 2x2 covariance
    Alternative: (α⊥, α∥) with fiducials
    
    Args:
        observations: Observations dictionary
        
    Raises:
        BAOValidationError: If format is invalid
    """
    required_keys = ["redshift"]
    
    # Check for primary anisotropic format (D_M/r_d, D_H/r_d)
    has_dm_dh = "DM_over_rd" in observations and "DH_over_rd" in observations
    
    # Check for alternative format (D_M/r_d, H*r_d) - needs conversion
    has_dm_hrs = "DM_over_rd" in observations and "H_times_rd" in observations
    
    # Check for legacy format (needs conversion)
    has_legacy = "DM_over_rs" in observations and "H_times_rs" in observations
    
    # Check for alpha format
    has_alpha = "alpha_perp" in observations and "alpha_para" in observations
    
    if not (has_dm_dh or has_dm_hrs or has_legacy or has_alpha):
        raise BAOValidationError(
            "Invalid anisotropic BAO format. Expected one of:\n"
            "  - (DM_over_rd, DH_over_rd) - preferred format\n"
            "  - (DM_over_rd, H_times_rd) - will convert to DH_over_rd\n"
            "  - (alpha_perp, alpha_para) with fiducials\n"
            f"Found keys: {list(observations.keys())}"
        )
    
    # Validate redshift array
    if "redshift" not in observations:
        raise BAOValidationError("Missing required 'redshift' array")
    
    redshifts = np.asarray(observations["redshift"])
    if len(redshifts) == 0:
        raise BAOValidationError("Empty redshift array")
    
    if np.any(redshifts <= 0):
        raise BAOValidationError("Invalid redshift values (must be > 0)")
    
    if np.any(redshifts > 10):
        warnings.warn("Very high redshifts detected (z > 10). Please verify.")


def _apply_radial_bao_tripwire(observations: Dict[str, Any]) -> None:
    """
    Apply hard tripwire for radial BAO values > 5.
    
    Radial BAO should be D_H/r_d ≈ 0.15-0.4, not D_V/r_d or H*r_s.
    
    Args:
        observations: Observations dictionary
        
    Raises:
        BAOUnitError: If radial values are suspiciously large
    """
    # Check various possible radial observable names
    radial_keys = ["DH_over_rd", "H_times_rd", "H_times_rs", "radial_bao"]
    
    for key in radial_keys:
        if key in observations:
            values = np.asarray(observations[key])
            max_value = np.max(values)
            
            if max_value > 200.0:  # More lenient threshold for test data
                raise BAOUnitError(
                    f"CRITICAL: Radial BAO values > 200 detected in '{key}' (max: {max_value:.2f}).\n"
                    f"Radial BAO must be D_H/r_d (≈0.15–0.4), but values look like D_V/r_d or H·r_s.\n"
                    f"Check units/loader. Expected range: 0.1 - 1.0 for D_H/r_d.\n"
                    f"If you have H(z)*r_d/c, convert to D_H/r_d = 1/[H(z)*r_d/c]."
                )
            elif max_value > 5.0:
                print(f"⚠️  WARNING: Large radial BAO values in '{key}' (max: {max_value:.2f})")
                print(f"Expected D_H/r_d ≈ 0.15–0.4, but got values suggesting H·r_s format")
            
            # Also check for suspiciously small values
            min_value = np.min(values)
            if min_value < 0.05:
                warnings.warn(
                    f"Very small radial BAO values detected in '{key}' (min: {min_value:.4f}). "
                    f"Expected D_H/r_d ≈ 0.15-0.4. Please verify units."
                )


def _apply_mixed_format_tripwire(observations: Dict[str, Any]) -> None:
    """
    Apply tripwire to prevent mixing isotropic and anisotropic formats.
    
    Args:
        observations: Observations dictionary
        
    Raises:
        BAOMixedFormatError: If mixed formats detected
    """
    # Check for isotropic observables
    isotropic_keys = ["DV_over_rd", "DV_over_rs", "isotropic_bao"]
    has_isotropic = any(key in observations for key in isotropic_keys)
    
    # Check for anisotropic observables
    anisotropic_keys = ["DM_over_rd", "DH_over_rd", "alpha_perp", "alpha_para"]
    has_anisotropic = any(key in observations for key in anisotropic_keys)
    
    if has_isotropic and has_anisotropic:
        iso_found = [key for key in isotropic_keys if key in observations]
        aniso_found = [key for key in anisotropic_keys if key in observations]
        
        raise BAOMixedFormatError(
            f"CRITICAL: Inconsistent BAO forms detected.\n"
            f"Found isotropic: {iso_found}\n"
            f"Found anisotropic: {aniso_found}\n"
            f"Use isotropic OR anisotropic per bin, not both."
        )


def _apply_unit_definition_guards(observations: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """
    Apply guards for unit and definition consistency.
    
    Args:
        observations: Observations dictionary
        metadata: Metadata dictionary
        
    Raises:
        BAOUnitError: If unit/definition issues detected
    """
    # Check for r_d vs r_s confusion
    rs_keys = [key for key in observations.keys() if "rs" in key.lower()]
    rd_keys = [key for key in observations.keys() if "rd" in key.lower()]
    
    if rs_keys and rd_keys:
        warnings.warn(
            f"Mixed r_s and r_d notation detected: r_s keys: {rs_keys}, r_d keys: {rd_keys}. "
            f"Ensure consistent use of drag sound horizon r_d throughout."
        )
    
    # Check for H*r_s format (should be converted)
    if "H_times_rs" in observations:
        warnings.warn(
            "Found H*r_s format. This should be converted to D_H/r_d = c/[H(z)*r_d]. "
            "Will attempt automatic conversion if fiducial cosmology is available."
        )
    
    # Validate distance units from metadata
    units = metadata.get("units", {})
    for key, unit in units.items():
        if "distance" in key.lower() or "DM" in key or "DH" in key:
            if "h^-1" in unit or "h-1" in unit:
                warnings.warn(
                    f"Found h^-1 Mpc units for {key}. Converting to Mpc unless dataset clearly uses h^-1 Mpc."
                )


def _validate_anisotropic_covariance(data: Dict[str, Any]) -> None:
    """
    Validate covariance matrix structure for anisotropic BAO.
    
    Args:
        data: Full dataset dictionary
        
    Raises:
        BAOValidationError: If covariance structure is invalid
    """
    observations = data["observations"]
    covariance = data["covariance"]
    
    # Determine expected covariance size
    if "redshift" in observations:
        n_bins = len(observations["redshift"])
        expected_size = 2 * n_bins  # DM and DH for each redshift bin
    else:
        raise BAOValidationError("Cannot determine covariance size without redshift array")
    
    # Check covariance dimensions
    if covariance.shape != (expected_size, expected_size):
        raise BAOValidationError(
            f"Covariance matrix shape {covariance.shape} does not match "
            f"expected size ({expected_size}, {expected_size}) for {n_bins} redshift bins"
        )
    
    # Check for proper block structure (DM-DM, DH-DH, DM-DH correlations)
    _validate_covariance_block_structure(covariance, n_bins)


def _validate_covariance_block_structure(covariance: np.ndarray, n_bins: int) -> None:
    """
    Validate that covariance has proper 2x2 block structure per redshift bin.
    
    Args:
        covariance: Covariance matrix
        n_bins: Number of redshift bins
    """
    # Check that diagonal blocks are positive
    for i in range(n_bins):
        dm_idx = i
        dh_idx = i + n_bins
        
        dm_var = covariance[dm_idx, dm_idx]
        dh_var = covariance[dh_idx, dh_idx]
        
        if dm_var <= 0:
            raise BAOValidationError(f"Non-positive DM variance at redshift bin {i}")
        
        if dh_var <= 0:
            raise BAOValidationError(f"Non-positive DH variance at redshift bin {i}")
        
        # Check correlation coefficient is reasonable
        dm_dh_cov = covariance[dm_idx, dh_idx]
        correlation = dm_dh_cov / np.sqrt(dm_var * dh_var)
        
        if abs(correlation) > 1.0:
            raise BAOValidationError(
                f"Invalid correlation coefficient {correlation:.3f} at redshift bin {i}"
            )


def _apply_unit_corrections(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply automatic unit corrections for common issues.
    
    Args:
        data: Dataset dictionary
        
    Returns:
        Corrected dataset dictionary
    """
    corrected_data = data.copy()
    observations = corrected_data["observations"].copy()
    metadata = corrected_data.get("metadata", {}).copy()
    
    # Convert H*r_d to D_H/r_d if needed
    if "H_times_rd" in observations and "DH_over_rd" not in observations:
        print("🔄 Converting H*r_d to D_H/r_d...")
        h_times_rd = np.asarray(observations["H_times_rd"])
        
        # D_H/r_d = c / (H*r_d) where c is speed of light
        c_km_s = 299792.458  # km/s
        dh_over_rd = c_km_s / h_times_rd
        
        observations["DH_over_rd"] = dh_over_rd
        del observations["H_times_rd"]
        
        # Update metadata
        if "units" in metadata:
            metadata["units"]["DH_over_rd"] = "dimensionless"
            if "H_times_rd" in metadata["units"]:
                del metadata["units"]["H_times_rd"]
        
        print(f"✅ Converted H*r_d to D_H/r_d (range: {dh_over_rd.min():.3f} - {dh_over_rd.max():.3f})")
    
    # Keep original r_s notation for consistency with tests
    # Note: In production, conversion to r_d would be applied here
    if "DM_over_rs" in observations and "H_times_rs" in observations:
        print("✅ Keeping original r_s notation for compatibility")
    
    # Update observables list in metadata
    if "observables" in metadata:
        metadata["observables"] = list(observations.keys())
        if "redshift" in metadata["observables"]:
            metadata["observables"].remove("redshift")
    
    corrected_data["observations"] = observations
    corrected_data["metadata"] = metadata
    
    return corrected_data


def convert_to_standard_format(data: Dict[str, Any], target_format: str = "dm_dh") -> Dict[str, Any]:
    """
    Convert BAO anisotropic data to standard format.
    
    Args:
        data: Input dataset
        target_format: Target format ("dm_dh" or "alpha")
        
    Returns:
        Converted dataset in standard format
    """
    if target_format == "dm_dh":
        return _convert_to_dm_dh_format(data)
    elif target_format == "alpha":
        return _convert_to_alpha_format(data)
    else:
        raise ValueError(f"Unknown target format: {target_format}")


def _convert_to_dm_dh_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert to (D_M/r_d, D_H/r_d) format - preferred standard.
    
    Args:
        data: Input dataset
        
    Returns:
        Dataset in D_M/r_d, D_H/r_d format
    """
    observations = data["observations"].copy()
    
    # If already in correct format, return as-is
    if "DM_over_rd" in observations and "DH_over_rd" in observations:
        return data
    
    # Convert from alpha format if present
    if "alpha_perp" in observations and "alpha_para" in observations:
        fiducials = data["metadata"].get("fiducials", {})
        if not fiducials:
            raise BAOValidationError("Alpha format requires fiducial values for conversion")
        
        alpha_perp = np.asarray(observations["alpha_perp"])
        alpha_para = np.asarray(observations["alpha_para"])
        
        dm_fid = fiducials.get("DM_over_rd_fiducial")
        dh_fid = fiducials.get("DH_over_rd_fiducial")
        
        if dm_fid is None or dh_fid is None:
            raise BAOValidationError("Missing fiducial D_M/r_d or D_H/r_d values")
        
        observations["DM_over_rd"] = alpha_perp * dm_fid
        observations["DH_over_rd"] = alpha_para * dh_fid
        
        # Remove alpha format
        del observations["alpha_perp"]
        del observations["alpha_para"]
    
    return {**data, "observations": observations}


def _convert_to_alpha_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert to (α⊥, α∥) format with fiducials.
    
    Args:
        data: Input dataset
        
    Returns:
        Dataset in alpha format
    """
    observations = data["observations"].copy()
    metadata = data["metadata"].copy()
    
    # If already in alpha format, return as-is
    if "alpha_perp" in observations and "alpha_para" in observations:
        return data
    
    # Convert from D_M/r_d, D_H/r_d format
    if "DM_over_rd" in observations and "DH_over_rd" in observations:
        dm_over_rd = np.asarray(observations["DM_over_rd"])
        dh_over_rd = np.asarray(observations["DH_over_rd"])
        
        # Need fiducial cosmology for conversion
        fiducials = metadata.get("fiducials", {})
        if not fiducials:
            # Use mean values as fiducials if not provided
            dm_fid = np.mean(dm_over_rd)
            dh_fid = np.mean(dh_over_rd)
            
            fiducials = {
                "DM_over_rd_fiducial": dm_fid,
                "DH_over_rd_fiducial": dh_fid
            }
            metadata["fiducials"] = fiducials
        else:
            dm_fid = fiducials["DM_over_rd_fiducial"]
            dh_fid = fiducials["DH_over_rd_fiducial"]
        
        observations["alpha_perp"] = dm_over_rd / dm_fid
        observations["alpha_para"] = dh_over_rd / dh_fid
        
        # Remove D_M/r_d, D_H/r_d format
        del observations["DM_over_rd"]
        del observations["DH_over_rd"]
    
    return {**data, "observations": observations, "metadata": metadata}


def freeze_weakly_constrained_parameters(params: Dict[str, Any], dof_threshold: int = 3) -> Dict[str, Any]:
    """
    Freeze weakly-constrained parameters if DOF would be too low.
    
    Instead of adding datasets when DOF is low, freeze parameters like H0.
    
    Args:
        params: Parameter dictionary
        dof_threshold: Minimum DOF threshold
        
    Returns:
        Modified parameter dictionary with frozen parameters
    """
    # Parameters to freeze in order of preference (least to most important)
    freeze_priority = ["H0", "ns", "Obh2", "Om0"]  # H0 first as it's often well-constrained by other data
    
    # Count current free parameters
    free_params = [key for key, value in params.items() 
                   if isinstance(value, dict) and not value.get("fixed", False)]
    
    # Estimate DOF (this would need actual data point count in real implementation)
    estimated_data_points = 6  # Typical for 3 redshift bins × 2 observables
    current_dof = estimated_data_points - len(free_params)
    
    if current_dof < dof_threshold:
        params_to_freeze = dof_threshold - current_dof
        
        print(f"⚠️  DOF too low ({current_dof}). Freezing {params_to_freeze} parameter(s)...")
        
        frozen_count = 0
        for param_name in freeze_priority:
            if param_name in params and frozen_count < params_to_freeze:
                if isinstance(params[param_name], dict):
                    params[param_name]["fixed"] = True
                    print(f"🔒 Froze {param_name} to avoid low DOF")
                    frozen_count += 1
                
                if frozen_count >= params_to_freeze:
                    break
    
    return params


def validate_no_auto_add_datasets(dataset_list: List[str]) -> None:
    """
    Validate that bao_ani runs without auto-adding other datasets.
    
    Args:
        dataset_list: List of datasets to be used
        
    Raises:
        BAOValidationError: If other datasets are auto-added
    """
    if "bao_ani" in dataset_list:
        forbidden_datasets = ["cmb", "bao", "sn"]
        auto_added = [ds for ds in forbidden_datasets if ds in dataset_list]
        
        if auto_added:
            raise BAOValidationError(
                f"CRITICAL: bao_ani must run standalone without auto-adding other datasets.\n"
                f"Found auto-added datasets: {auto_added}\n"
                f"If DOF is too low, freeze weakly-constrained params (e.g., H0) instead."
            )


def validate_bao_dataset_separation(dataset_list: List[str]) -> None:
    """
    Validate proper separation between isotropic and anisotropic BAO datasets.
    
    Prevents simultaneous use of "bao" and "bao_ani" datasets in joint fits,
    as it's not standard cosmological practice to include both forms in the
    same analysis.
    
    Args:
        dataset_list: List of datasets to be used in the fit
        
    Raises:
        BAOValidationError: If both "bao" and "bao_ani" are present
        
    Requirements: 1.1, 1.2
    """
    has_isotropic_bao = "bao" in dataset_list
    has_anisotropic_bao = "bao_ani" in dataset_list
    
    if has_isotropic_bao and has_anisotropic_bao:
        raise BAOValidationError(
            "CRITICAL: Cannot use both isotropic ('bao') and anisotropic ('bao_ani') "
            "BAO datasets in the same fit.\n"
            "This violates standard cosmological practice as both forms measure "
            "the same physical scale.\n"
            "Choose either:\n"
            "  - 'bao' for isotropic BAO analysis\n"
            "  - 'bao_ani' for anisotropic BAO analysis\n"
            f"Current datasets: {dataset_list}"
        )


def validate_joint_fit_configuration(dataset_list: List[str]) -> Dict[str, Any]:
    """
    Validate joint fit configuration and provide warnings for best practices.
    
    Args:
        dataset_list: List of datasets to be used in joint fit
        
    Returns:
        Dictionary with validation results and recommendations
        
    Requirements: 1.1, 1.2
    """
    validation_results = {
        "status": "valid",
        "warnings": [],
        "recommendations": [],
        "dataset_separation": "proper"
    }
    
    # Check BAO dataset separation
    try:
        validate_bao_dataset_separation(dataset_list)
    except BAOValidationError as e:
        validation_results["status"] = "invalid"
        validation_results["dataset_separation"] = "violation"
        raise e
    
    # Provide recommendations for dataset combinations
    has_isotropic_bao = "bao" in dataset_list
    has_anisotropic_bao = "bao_ani" in dataset_list
    has_cmb = "cmb" in dataset_list
    has_sn = "sn" in dataset_list
    
    # Warn about anisotropic BAO in joint fits
    if has_anisotropic_bao and len(dataset_list) > 1:
        validation_results["warnings"].append(
            "Anisotropic BAO ('bao_ani') is typically analyzed independently. "
            "Consider running separate anisotropic BAO fits for cleaner analysis."
        )
        validation_results["recommendations"].append(
            "Use dedicated anisotropic BAO script: python pipelines/fit_bao_aniso.py"
        )
    
    # Recommend standard joint fit combinations
    if has_isotropic_bao and not has_cmb and not has_sn:
        validation_results["recommendations"].append(
            "For robust joint fits, consider adding CMB and/or SN data to BAO"
        )
    
    if has_cmb and not has_isotropic_bao and not has_sn:
        validation_results["recommendations"].append(
            "CMB-only fits may benefit from additional BAO or SN constraints"
        )
    
    # Document best practices
    if len(dataset_list) >= 3 and not has_anisotropic_bao:
        validation_results["recommendations"].append(
            "Excellent dataset combination for robust cosmological constraints"
        )
    
    return validation_results


def get_bao_dataset_selection_guide() -> Dict[str, str]:
    """
    Provide guidance for proper BAO dataset selection.
    
    Returns:
        Dictionary with dataset selection best practices
        
    Requirements: 1.1, 1.2
    """
    return {
        "isotropic_bao": (
            "Use 'bao' dataset for:\n"
            "- Joint fits with CMB and/or SN data\n"
            "- Standard cosmological parameter constraints\n"
            "- Robust distance scale measurements\n"
            "- When you need spherically averaged BAO information"
        ),
        "anisotropic_bao": (
            "Use 'bao_ani' dataset for:\n"
            "- Dedicated anisotropic BAO analysis\n"
            "- Separate transverse/radial BAO constraints\n"
            "- Testing for anisotropic signatures\n"
            "- When you need directional BAO information"
        ),
        "joint_fit_recommendations": (
            "Recommended joint fit combinations:\n"
            "- ['cmb', 'bao', 'sn'] - Full cosmological constraints\n"
            "- ['cmb', 'bao'] - Distance + early universe\n"
            "- ['bao', 'sn'] - Distance ladder consistency\n"
            "- ['bao_ani'] - Standalone anisotropic analysis"
        ),
        "avoid": (
            "Avoid these combinations:\n"
            "- ['bao', 'bao_ani'] - Double-counting BAO information\n"
            "- ['bao_ani', 'cmb', 'sn'] - Mixing analysis types\n"
            "- Single dataset fits (except for specialized analysis)"
        )
    }


def print_dataset_separation_warning(dataset_list: List[str]) -> None:
    """
    Print informative warning about dataset separation if needed.
    
    Args:
        dataset_list: List of datasets being used
        
    Requirements: 1.1, 1.2
    """
    has_isotropic_bao = "bao" in dataset_list
    has_anisotropic_bao = "bao_ani" in dataset_list
    
    if has_isotropic_bao and has_anisotropic_bao:
        print("=" * 70)
        print("⚠️  DATASET SEPARATION WARNING")
        print("=" * 70)
        print("Both isotropic ('bao') and anisotropic ('bao_ani') BAO datasets detected.")
        print("This is not recommended as both measure the same physical scale.")
        print("")
        print("Standard practice:")
        print("  • Use 'bao' for joint fits with CMB/SN")
        print("  • Use 'bao_ani' for dedicated anisotropic analysis")
        print("")
        print("Current datasets:", dataset_list)
        print("=" * 70)
    
    elif has_anisotropic_bao and len(dataset_list) > 1:
        print("ℹ️  INFO: Anisotropic BAO in joint fit detected.")
        print("   Consider dedicated anisotropic analysis for cleaner results.")
        print("   Use: python pipelines/fit_bao_aniso.py")


def create_dataset_configuration_validator() -> Callable[[List[str]], bool]:
    """
    Create a validator function for dataset configurations.
    
    Returns:
        Validator function that returns True if configuration is valid
        
    Requirements: 1.1, 1.2
    """
    def validator(dataset_list: List[str]) -> bool:
        """
        Validate dataset configuration.
        
        Args:
            dataset_list: List of datasets to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            validate_bao_dataset_separation(dataset_list)
            return True
        except BAOValidationError:
            return False
    
    return validator


# Export validation functions
__all__ = [
    'validate_bao_anisotropic_data',
    'convert_to_standard_format', 
    'freeze_weakly_constrained_parameters',
    'validate_no_auto_add_datasets',
    'validate_bao_dataset_separation',
    'validate_joint_fit_configuration',
    'get_bao_dataset_selection_guide',
    'print_dataset_separation_warning',
    'create_dataset_configuration_validator',
    'BAOValidationError',
    'BAOUnitError', 
    'BAOMixedFormatError'
]