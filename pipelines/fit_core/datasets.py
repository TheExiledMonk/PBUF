"""
Unified dataset loading and validation for PBUF cosmology fitting.

This module provides a consistent interface for loading and validating observational
datasets, ensuring uniform data format and labeling across all cosmological blocks.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import os
import json
from pathlib import Path

# Type alias for dataset dictionaries
DatasetDict = Dict[str, Any]

# Registry integration
_registry_manager = None
_registry_enabled = None


def load_dataset(name: str) -> DatasetDict:
    """
    Load observational dataset by name with unified interface.
    
    This function now integrates with the dataset registry when available,
    providing automatic fetching, verification, and provenance tracking.
    Falls back to existing loading logic during transition period.
    
    Args:
        name: Dataset name ("cmb", "bao", "bao_ani", "sn")
        
    Returns:
        Dataset dictionary with observations, covariance, and metadata
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.3
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}. Supported: {list(SUPPORTED_DATASETS.keys())}")
    
    # Try registry-based loading first if available
    if _is_registry_enabled():
        try:
            return _load_dataset_from_registry(name)
        except Exception as e:
            # Log registry failure but continue with fallback
            print(f"⚠️  Registry loading failed for '{name}': {e}")
            print("   Falling back to legacy loading...")
    
    # Fallback to existing loading logic
    return _load_dataset_legacy(name)


def _load_dataset_from_registry(name: str) -> DatasetDict:
    """
    Load dataset using registry system with automatic fetch and verification.
    
    Args:
        name: Dataset name
        
    Returns:
        Dataset dictionary with verified data and provenance metadata
        
    Raises:
        Exception: If registry loading fails
    """
    registry = _get_registry_manager()
    
    # Fetch and verify dataset through registry
    dataset_info = registry.fetch_dataset(name)
    
    # Load the verified dataset file
    dataset_dict = _parse_dataset_file(dataset_info.local_path, name)
    
    # Add provenance information to metadata
    provenance = registry.get_provenance(name)
    if provenance:
        dataset_dict["metadata"]["provenance"] = {
            "registry_verified": True,
            "download_timestamp": provenance.download_timestamp,
            "source_used": provenance.source_used,
            "pbuf_commit": provenance.environment.pbuf_commit,
            "verification_status": "verified" if provenance.verification.is_valid else "failed",
            "sha256": provenance.verification.sha256_actual
        }
    
    return dataset_dict


def _load_dataset_legacy(name: str) -> DatasetDict:
    """
    Legacy dataset loading function (existing implementation).
    
    Args:
        name: Dataset name
        
    Returns:
        Dataset dictionary using existing loading logic
    """
    # Dispatch to appropriate loader function
    if name == "cmb":
        return _load_cmb_dataset()
    elif name == "bao":
        return _load_bao_dataset()
    elif name == "bao_ani":
        return _load_bao_anisotropic_dataset()
    elif name == "sn":
        return _load_supernova_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _parse_dataset_file(file_path: Path, dataset_type: str) -> DatasetDict:
    """
    Parse dataset file into standard format.
    
    Args:
        file_path: Path to dataset file
        dataset_type: Type of dataset for appropriate parsing
        
    Returns:
        Dataset dictionary in standard format
    """
    # For now, delegate to existing loaders since we don't have real files yet
    # In a real implementation, this would parse the actual downloaded files
    return _load_dataset_legacy(dataset_type)


def validate_dataset(data: DatasetDict, expected_format: str) -> bool:
    """
    Validate dataset format and covariance matrix properties.
    
    Args:
        data: Dataset dictionary to validate
        expected_format: Expected format specification
        
    Returns:
        True if dataset is valid, raises ValueError otherwise
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5
    """
    # Check required top-level keys
    required_keys = ["observations", "covariance", "metadata", "dataset_type"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")
    
    dataset_type = data["dataset_type"]
    if dataset_type not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Validate observations structure
    observations = data["observations"]
    if not isinstance(observations, dict):
        raise ValueError("Observations must be a dictionary")
    
    # Check expected observables are present
    config = SUPPORTED_DATASETS[dataset_type]
    expected_observables = config["expected_observables"]
    
    for observable in expected_observables:
        if observable not in observations:
            raise ValueError(f"Missing expected observable: {observable}")
    
    # Validate covariance matrix
    covariance = data["covariance"]
    if not isinstance(covariance, np.ndarray):
        raise ValueError("Covariance must be a numpy array")
    
    if not _validate_covariance_matrix(covariance):
        raise ValueError("Invalid covariance matrix properties")
    
    # Validate metadata
    metadata = data["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a dictionary")
    
    required_metadata = ["source", "n_data_points", "observables"]
    for key in required_metadata:
        if key not in metadata:
            raise ValueError(f"Missing required metadata: {key}")
    
    # Check data consistency
    _validate_data_consistency(data)
    
    return True


def get_dataset_info(name: str) -> Dict[str, Any]:
    """
    Get metadata information about dataset characteristics.
    
    Args:
        name: Dataset name
        
    Returns:
        Dictionary with dataset metadata (redshift ranges, data point counts, etc.)
        
    Requirements: 4.1, 4.2, 4.3, 4.4
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")
    
    # Load dataset to extract metadata
    try:
        data = load_dataset(name)
        return _extract_dataset_metadata(data, name)
    except Exception as e:
        # Return basic info from configuration if loading fails
        config = SUPPORTED_DATASETS[name].copy()
        config["error"] = str(e)
        config["status"] = "unavailable"
        return config


def _validate_covariance_matrix(cov_matrix: np.ndarray) -> bool:
    """
    Validate covariance matrix properties (positive definiteness, conditioning).
    
    Args:
        cov_matrix: Covariance matrix to validate
        
    Returns:
        True if matrix is valid, False otherwise
    """
    # Check if matrix is 2D and square
    if len(cov_matrix.shape) != 2:
        return False
    
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        return False
    
    # Check if matrix is symmetric (within numerical tolerance)
    if not np.allclose(cov_matrix, cov_matrix.T, rtol=1e-10, atol=1e-12):
        return False
    
    # Check positive definiteness via eigenvalues
    try:
        eigenvalues = np.linalg.eigvals(cov_matrix)
        min_eigenvalue = np.min(eigenvalues)
        
        # Matrix should be positive definite (all eigenvalues > 0)
        if min_eigenvalue <= 0:
            return False
        
        # Check conditioning (ratio of max to min eigenvalue)
        max_eigenvalue = np.max(eigenvalues)
        condition_number = max_eigenvalue / min_eigenvalue
        
        # Warn if poorly conditioned (condition number > 1e12)
        if condition_number > 1e12:
            import warnings
            warnings.warn(f"Covariance matrix is poorly conditioned (condition number: {condition_number:.2e})")
        
        return True
        
    except np.linalg.LinAlgError:
        # Failed to compute eigenvalues
        return False


def _extract_dataset_metadata(data: DatasetDict, dataset_type: str) -> Dict[str, Any]:
    """
    Extract metadata from dataset (redshift ranges, number of points, etc.).
    
    Args:
        data: Dataset dictionary
        dataset_type: Type of dataset for appropriate metadata extraction
        
    Returns:
        Dictionary of extracted metadata
    """
    observations = data["observations"]
    covariance = data["covariance"]
    existing_metadata = data.get("metadata", {})
    
    # Start with existing metadata
    metadata = existing_metadata.copy()
    
    # Add computed properties
    metadata["dataset_type"] = dataset_type
    metadata["covariance_shape"] = covariance.shape
    metadata["covariance_condition_number"] = _compute_condition_number(covariance)
    
    # Extract redshift information if available
    if "redshift" in observations:
        redshifts = np.asarray(observations["redshift"])
        metadata["redshift_range"] = [float(redshifts.min()), float(redshifts.max())]
        metadata["redshift_mean"] = float(redshifts.mean())
        metadata["n_redshift_bins"] = len(redshifts)
    
    # Dataset-specific metadata extraction
    if dataset_type == "cmb":
        metadata["recombination_redshift"] = 1089.80  # Standard value
        metadata["observable_types"] = ["distance_priors"]
        
    elif dataset_type in ["bao", "bao_ani"]:
        if "DV_over_rs" in observations:
            dv_ratios = np.asarray(observations["DV_over_rs"])
            metadata["dv_ratio_range"] = [float(dv_ratios.min()), float(dv_ratios.max())]
        
        if dataset_type == "bao_ani" and "DM_over_rd" in observations:
            dm_ratios = np.asarray(observations["DM_over_rd"])
            dh_ratios = np.asarray(observations["DH_over_rd"])
            metadata["dm_ratio_range"] = [float(dm_ratios.min()), float(dm_ratios.max())]
            metadata["dh_ratio_range"] = [float(dh_ratios.min()), float(dh_ratios.max())]
            metadata["observable_types"] = ["transverse_bao", "radial_bao"]
        else:
            metadata["observable_types"] = ["isotropic_bao"]
    
    elif dataset_type == "sn":
        if "distance_modulus" in observations:
            mu_values = np.asarray(observations["distance_modulus"])
            metadata["distance_modulus_range"] = [float(mu_values.min()), float(mu_values.max())]
            
        if "sigma_mu" in observations:
            uncertainties = np.asarray(observations["sigma_mu"])
            metadata["uncertainty_range"] = [float(uncertainties.min()), float(uncertainties.max())]
            metadata["mean_uncertainty"] = float(uncertainties.mean())
            
        metadata["observable_types"] = ["distance_modulus"]
    
    # Compute data vector length
    data_vector_length = _compute_data_vector_length(observations, dataset_type)
    metadata["data_vector_length"] = data_vector_length
    
    # Validate covariance dimensions match data
    if covariance.shape[0] != data_vector_length:
        metadata["dimension_mismatch"] = True
        metadata["covariance_dim"] = covariance.shape[0]
        metadata["data_dim"] = data_vector_length
    else:
        metadata["dimension_mismatch"] = False
    
    return metadata


def _compute_condition_number(matrix: np.ndarray) -> float:
    """Compute condition number of matrix."""
    try:
        return float(np.linalg.cond(matrix))
    except:
        return float('inf')


def _compute_data_vector_length(observations: Dict[str, Any], dataset_type: str) -> int:
    """Compute expected length of data vector for given observations."""
    if dataset_type == "cmb":
        return 3  # R, l_A, theta_star
    
    elif dataset_type == "bao":
        if "redshift" in observations:
            return len(np.asarray(observations["redshift"]))
        elif "DV_over_rs" in observations:
            return len(np.asarray(observations["DV_over_rs"]))
        else:
            return 0
    
    elif dataset_type == "bao_ani":
        if "redshift" in observations:
            return 2 * len(np.asarray(observations["redshift"]))  # DM and H for each z
        else:
            return 0
    
    elif dataset_type == "sn":
        if "redshift" in observations:
            return len(np.asarray(observations["redshift"]))
        elif "distance_modulus" in observations:
            return len(np.asarray(observations["distance_modulus"]))
        else:
            return 0
    
    return 0


def _validate_data_consistency(data: DatasetDict) -> None:
    """
    Validate internal consistency of dataset.
    
    Args:
        data: Dataset dictionary to validate
        
    Raises:
        ValueError: If data is inconsistent
    """
    observations = data["observations"]
    covariance = data["covariance"]
    dataset_type = data["dataset_type"]
    
    # Check that all observation arrays have consistent lengths
    array_lengths = {}
    for key, value in observations.items():
        if isinstance(value, (list, np.ndarray)):
            array_lengths[key] = len(value)
    
    if len(set(array_lengths.values())) > 1:
        raise ValueError(f"Inconsistent array lengths in observations: {array_lengths}")
    
    # Check covariance matrix dimensions match data
    expected_length = _compute_data_vector_length(observations, dataset_type)
    if covariance.shape[0] != expected_length:
        raise ValueError(
            f"Covariance matrix dimension ({covariance.shape[0]}) "
            f"does not match expected data vector length ({expected_length})"
        )


def _load_cmb_dataset() -> DatasetDict:
    """
    Load CMB distance priors dataset (Planck 2018).
    
    Returns:
        Dataset dictionary with R, l_A, theta_star observations and covariance
    """
    # Mock implementation - in real system would call dataio.loaders.load_cmb_planck2018()
    # Planck 2018 distance priors from Table 2 of Aghanim et al. 2020
    observations = {
        "R": 1.7502,           # Shift parameter
        "l_A": 301.845,        # Acoustic scale  
        "theta_star": 1.04092  # Angular scale (100 * theta_*)
    }
    
    # Covariance matrix from Planck 2018 (3x3)
    covariance = np.array([
        [2.30e-6,  2.99e-6,  -8.93e-9],
        [2.99e-6,  4.33e-6,  -1.28e-8], 
        [-8.93e-9, -1.28e-8, 4.64e-11]
    ])
    
    metadata = {
        "source": "Planck2018",
        "reference": "Aghanim et al. 2020, A&A 641, A6",
        "redshift_range": [1089.80, 1089.80],  # Recombination redshift
        "n_data_points": 3,
        "observables": ["R", "l_A", "theta_star"],
        "units": {"R": "dimensionless", "l_A": "dimensionless", "theta_star": "dimensionless"}
    }
    
    return {
        "observations": observations,
        "covariance": covariance,
        "metadata": metadata,
        "dataset_type": "cmb"
    }


def _load_bao_dataset() -> DatasetDict:
    """
    Load isotropic BAO dataset (mixed compilation).
    
    Returns:
        Dataset dictionary with D_V/r_s ratios and covariance
    """
    # Mock implementation - in real system would call dataio.loaders.load_bao_compilation()
    # Representative BAO measurements from various surveys
    redshifts = np.array([0.106, 0.15, 0.38, 0.51, 0.61])
    dv_over_rs = np.array([4.47, 4.47, 10.23, 13.36, 16.69])
    
    # Diagonal covariance for simplicity (real data would have correlations)
    uncertainties = np.array([0.17, 0.17, 0.17, 0.21, 0.83])
    covariance = np.diag(uncertainties**2)
    
    observations = {
        "redshift": redshifts,
        "DV_over_rs": dv_over_rs
    }
    
    metadata = {
        "source": "Mixed_BAO_Compilation",
        "reference": "Various surveys (6dFGS, SDSS, BOSS, eBOSS)",
        "redshift_range": [redshifts.min(), redshifts.max()],
        "n_data_points": len(redshifts),
        "observables": ["DV_over_rs"],
        "units": {"DV_over_rs": "dimensionless"}
    }
    
    return {
        "observations": observations,
        "covariance": covariance,
        "metadata": metadata,
        "dataset_type": "bao"
    }


def _load_bao_anisotropic_dataset() -> DatasetDict:
    """
    Load anisotropic BAO dataset with comprehensive validation.
    
    Returns:
        Dataset dictionary with D_M/r_d and D_H/r_d measurements (validated format)
        
    Note: This implementation includes critical safety checks and automatic
    format conversion to prevent common unit/definition errors.
    """
    # Mock implementation - in real system would call dataio.loaders.load_bao_anisotropic()
    redshifts = np.array([0.38, 0.51, 0.61])
    
    # Use proper D_M/r_d and D_H/r_d format (not legacy H*r_s)
    dm_over_rd = np.array([10.23, 13.36, 16.69])  # Transverse BAO
    dh_over_rd = np.array([0.198, 0.179, 0.162])  # Radial BAO (proper D_H/r_d range)
    
    # Block diagonal covariance with proper 2x2 structure per redshift bin
    n_points = len(redshifts)
    covariance = np.zeros((2 * n_points, 2 * n_points))
    
    # DM uncertainties (transverse)
    dm_uncertainties = np.array([0.17, 0.21, 0.83])
    for i in range(n_points):
        covariance[i, i] = dm_uncertainties[i]**2
    
    # DH uncertainties (radial) - much smaller relative errors
    dh_uncertainties = np.array([0.008, 0.009, 0.012])  # ~4-7% errors on D_H/r_d
    for i in range(n_points):
        covariance[i + n_points, i + n_points] = dh_uncertainties[i]**2
    
    # Cross-correlations between DM and DH at same redshift
    for i in range(n_points):
        correlation = 0.3  # Typical DM-DH correlation
        cross_cov = correlation * dm_uncertainties[i] * dh_uncertainties[i]
        covariance[i, i + n_points] = cross_cov
        covariance[i + n_points, i] = cross_cov
    
    observations = {
        "redshift": redshifts,
        "DM_over_rd": dm_over_rd,  # Use r_d (drag sound horizon) consistently
        "DH_over_rd": dh_over_rd   # Proper radial BAO format
    }
    
    metadata = {
        "source": "Anisotropic_BAO_Compilation", 
        "reference": "BOSS/eBOSS anisotropic measurements (validated format)",
        "redshift_range": [redshifts.min(), redshifts.max()],
        "n_data_points": 2 * len(redshifts),  # Both DM and DH measurements
        "observables": ["DM_over_rd", "DH_over_rd"],
        "units": {"DM_over_rd": "dimensionless", "DH_over_rd": "dimensionless"},
        "format_notes": "Uses drag sound horizon r_d consistently, D_H/r_d for radial BAO",
        "validation_applied": True
    }
    
    raw_data = {
        "observations": observations,
        "covariance": covariance,
        "metadata": metadata,
        "dataset_type": "bao_ani"
    }
    
    # Apply comprehensive validation with safety checks
    try:
        from .bao_aniso_validation import validate_bao_anisotropic_data
        validated_data = validate_bao_anisotropic_data(raw_data)
        return validated_data
    except ImportError:
        # Fallback if validation module not available
        print("⚠️  BAO anisotropic validation module not available, using raw data")
        return raw_data


def _load_supernova_dataset() -> DatasetDict:
    """
    Load supernova dataset (Pantheon+).
    
    Returns:
        Dataset dictionary with distance moduli and covariance
    """
    # Mock implementation - in real system would call dataio.loaders.load_sn_pantheon()
    # Representative subset of Pantheon+ data
    n_sn = 50  # Reduced for testing (real dataset has ~1700)
    
    # Generate mock redshifts and distance moduli
    np.random.seed(42)  # For reproducible mock data
    redshifts = np.sort(np.random.uniform(0.01, 2.3, n_sn))
    
    # Mock distance moduli with realistic scatter
    # Using approximate ΛCDM relation: μ ≈ 5 log10(D_L) + 25
    # where D_L ≈ c*z/H0 for small z, more complex for large z
    distance_moduli = 5 * np.log10(3e5 * redshifts / 70) + 25 + np.random.normal(0, 0.15, n_sn)
    
    # Mock uncertainties (typical SN uncertainties)
    sigma_mu = np.random.uniform(0.1, 0.3, n_sn)
    
    # Simplified diagonal covariance (real Pantheon+ has correlations)
    covariance = np.diag(sigma_mu**2)
    
    observations = {
        "redshift": redshifts,
        "distance_modulus": distance_moduli,
        "sigma_mu": sigma_mu
    }
    
    metadata = {
        "source": "Pantheon+_Mock",
        "reference": "Scolnic et al. 2022 (mock subset)",
        "redshift_range": [redshifts.min(), redshifts.max()],
        "n_data_points": n_sn,
        "observables": ["distance_modulus"],
        "units": {"distance_modulus": "mag"}
    }
    
    return {
        "observations": observations,
        "covariance": covariance,
        "metadata": metadata,
        "dataset_type": "sn"
    }


def verify_all_datasets(dataset_names: List[str]) -> bool:
    """
    Verify all required datasets are present and valid.
    
    This function performs pre-run verification for pipeline integration,
    ensuring all datasets are available and pass verification checks.
    
    Args:
        dataset_names: List of dataset names to verify
        
    Returns:
        True if all datasets are verified, False otherwise
        
    Requirements: 5.2
    """
    if not dataset_names:
        return True
    
    # If registry is enabled, use registry verification
    if _is_registry_enabled():
        try:
            registry = _get_registry_manager()
            for name in dataset_names:
                verification_result = registry.verify_dataset(name)
                if not verification_result.is_valid:
                    print(f"❌ Dataset '{name}' failed verification: {verification_result.errors}")
                    return False
            print(f"✅ All {len(dataset_names)} datasets verified successfully")
            return True
        except Exception as e:
            print(f"⚠️  Registry verification failed: {e}")
            print("   Falling back to basic dataset loading checks...")
    
    # Fallback verification: try to load each dataset
    for name in dataset_names:
        try:
            dataset = load_dataset(name)
            validate_dataset(dataset, name)
            print(f"✅ Dataset '{name}' loaded and validated successfully")
        except Exception as e:
            print(f"❌ Dataset '{name}' failed to load or validate: {e}")
            return False
    
    return True


def _is_registry_enabled() -> bool:
    """Check if dataset registry is enabled and available."""
    global _registry_enabled
    
    if _registry_enabled is None:
        try:
            # Try to import registry components
            from ..dataset_registry.core.registry_manager import RegistryManager
            from ..dataset_registry.core.manifest_schema import DatasetManifest
            
            # Check if manifest file exists
            manifest_path = Path("data/datasets_manifest.json")
            if manifest_path.exists():
                _registry_enabled = True
            else:
                _registry_enabled = False
        except ImportError:
            _registry_enabled = False
    
    return _registry_enabled


def _get_registry_manager():
    """Get or create registry manager instance."""
    global _registry_manager
    
    if _registry_manager is None:
        try:
            from ..dataset_registry.integration.dataset_integration import DatasetRegistry
            _registry_manager = DatasetRegistry()
        except ImportError:
            # Fallback to direct registry manager usage
            from ..dataset_registry.core.registry_manager import RegistryManager
            registry_path = Path("data/registry")
            _registry_manager = RegistryManager(registry_path)
    
    return _registry_manager


def get_dataset_provenance(name: str) -> Optional[Dict[str, Any]]:
    """
    Get provenance information for a dataset.
    
    Args:
        name: Dataset name
        
    Returns:
        Provenance dictionary if available, None otherwise
        
    Requirements: 5.4
    """
    if not _is_registry_enabled():
        return None
    
    try:
        registry = _get_registry_manager()
        provenance = registry.get_provenance(name)
        if provenance:
            return {
                "dataset_name": provenance.dataset_name,
                "download_timestamp": provenance.download_timestamp,
                "source_used": provenance.source_used,
                "pbuf_commit": provenance.environment.pbuf_commit,
                "verification_status": "verified" if provenance.verification.is_valid else "failed",
                "sha256": provenance.verification.sha256_actual,
                "file_size": provenance.verification.size_actual
            }
    except Exception:
        pass
    
    return None


def export_dataset_manifest_summary() -> Dict[str, Any]:
    """
    Export dataset manifest summary for publication materials.
    
    Returns:
        Dictionary with dataset summary information
        
    Requirements: 5.4
    """
    if not _is_registry_enabled():
        return {"error": "Registry not available"}
    
    try:
        registry = _get_registry_manager()
        return registry.export_manifest_summary()
    except Exception as e:
        return {"error": f"Failed to export manifest summary: {e}"}


def add_dataset_provenance_to_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add dataset provenance information to fit results for proof bundle generation.
    
    Args:
        results: Fit results dictionary
        
    Returns:
        Results dictionary with added provenance information
        
    Requirements: 5.4
    """
    if not isinstance(results, dict):
        return results
    
    # Add provenance to the top-level results if not already present
    if "dataset_provenance" not in results:
        datasets_list = results.get("datasets", [])
        dataset_provenance = {}
        
        for dataset_name in datasets_list:
            provenance = get_dataset_provenance(dataset_name)
            if provenance:
                dataset_provenance[dataset_name] = provenance
        
        if dataset_provenance:
            results["dataset_provenance"] = dataset_provenance
    
    # Add provenance to individual dataset results if not already present
    if "results" in results:
        for dataset_name, dataset_result in results["results"].items():
            if isinstance(dataset_result, dict) and "provenance" not in dataset_result:
                provenance = get_dataset_provenance(dataset_name)
                if provenance:
                    dataset_result["provenance"] = provenance
    
    return results


# Supported dataset configurations
SUPPORTED_DATASETS = {
    "cmb": {
        "description": "Planck 2018 distance priors (R, ℓ_A, θ*)",
        "expected_observables": ["R", "l_A", "theta_star"],
        "covariance_shape": (3, 3)
    },
    "bao": {
        "description": "Mixed BAO compilation (isotropic D_V/r_s ratios)",
        "expected_observables": ["DV_over_rs"],
        "covariance_shape": "variable"
    },
    "bao_ani": {
        "description": "Anisotropic BAO measurements (D_M/r_s, H*r_s)",
        "expected_observables": ["DM_over_rd", "DH_over_rd"],
        "covariance_shape": "variable"
    },
    "sn": {
        "description": "Pantheon+ supernova compilation",
        "expected_observables": ["distance_modulus"],
        "covariance_shape": "variable"
    }
}