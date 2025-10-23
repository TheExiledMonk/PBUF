"""
Standardized dataset schema for the PBUF data preparation framework.

This module defines the unified internal format that all cosmological datasets
are transformed into, ensuring consistency across different data types.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class StandardDataset:
    """
    Standardized dataset format (analysis-ready dataset) for all cosmological data types.
    
    This schema ensures all datasets entering the PBUF fitting pipelines share
    consistent structure, units, metadata schema, and provenance traceability.
    
    Attributes:
        z: Redshift array
        observable: Measured quantities (μ, D_M/r_d, H(z), etc.)
        uncertainty: One-sigma uncertainties
        covariance: Full covariance matrix (N×N or None)
        metadata: Source, citation, processing info
    """
    z: np.ndarray
    observable: np.ndarray
    uncertainty: np.ndarray
    covariance: Optional[np.ndarray]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate basic array properties after initialization."""
        # Validate types first
        self._validate_array_types()
        # Then validate shapes
        self._validate_array_shapes()
    
    def _validate_array_types(self):
        """Check if required fields are numpy arrays."""
        required_arrays = ['z', 'observable', 'uncertainty']
        for field in required_arrays:
            value = getattr(self, field)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Field '{field}' must be numpy array, got {type(value).__name__}")
        
        if self.covariance is not None:
            if not isinstance(self.covariance, np.ndarray):
                raise ValueError(f"Covariance must be numpy array, got {type(self.covariance).__name__}")
    
    def _validate_array_shapes(self):
        """Ensure all arrays have consistent shapes."""
        n_redshift_points = len(self.z)
        
        # Check dataset type for special validation rules
        dataset_type = self.metadata.get('dataset_type', '')
        
        # Check if this is anisotropic BAO data (2 measurements per redshift)
        is_anisotropic_bao = (
            dataset_type == 'bao' and 
            self.metadata.get('measurement_type') == 'anisotropic'
        ) or dataset_type == 'bao_ani'
        
        # Check if this is CMB data (3 measurements for 1 redshift)
        is_cmb = dataset_type == 'cmb'
        
        if is_anisotropic_bao:
            # For anisotropic BAO: observable and uncertainty should be 2x unique redshift points
            # The z array is expanded with duplicates, so we need to count unique values
            n_unique_redshifts = len(np.unique(self.z))
            expected_obs_length = 2 * n_unique_redshifts
            if len(self.observable) != expected_obs_length:
                raise ValueError(f"Anisotropic BAO observable array length {len(self.observable)} != 2 * unique redshift points {expected_obs_length}")
            
            if len(self.uncertainty) != expected_obs_length:
                raise ValueError(f"Anisotropic BAO uncertainty array length {len(self.uncertainty)} != 2 * unique redshift points {expected_obs_length}")
            

        elif is_cmb:
            # For CMB: 3 observables (R, l_A, theta_star) with repeated redshift
            expected_obs_length = 3
            if len(self.observable) != expected_obs_length:
                raise ValueError(f"CMB observable array length {len(self.observable)} != expected length {expected_obs_length}")
            
            if len(self.uncertainty) != expected_obs_length:
                raise ValueError(f"CMB uncertainty array length {len(self.uncertainty)} != expected length {expected_obs_length}")
            
            if len(self.z) != expected_obs_length:
                raise ValueError(f"CMB redshift array length {len(self.z)} != expected length {expected_obs_length}")
            

        else:
            # For isotropic data: all arrays should have same length as redshift
            if len(self.observable) != n_redshift_points:
                raise ValueError(f"Observable array length {len(self.observable)} != redshift array length {n_redshift_points}")
            
            if len(self.uncertainty) != n_redshift_points:
                raise ValueError(f"Uncertainty array length {len(self.uncertainty)} != redshift array length {n_redshift_points}")
        
        # Validate covariance matrix shape if provided
        if self.covariance is not None:
            obs_length = len(self.observable)
            if self.covariance.shape != (obs_length, obs_length):
                raise ValueError(f"Covariance matrix shape {self.covariance.shape} != ({obs_length}, {obs_length})")

    
    def validate_schema(self) -> bool:
        """
        Validate schema compliance (fields and datatypes).
        
        Returns:
            bool: True if schema is valid
            
        Raises:
            ValueError: If schema validation fails with detailed error messages
        """
        # First validate array shapes
        self._validate_array_shapes()
        # Check required fields exist and are numpy arrays
        required_arrays = ['z', 'observable', 'uncertainty']
        for field in required_arrays:
            if not hasattr(self, field):
                raise ValueError(f"Required field '{field}' is missing from dataset")
            
            value = getattr(self, field)
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Field '{field}' must be numpy array, got {type(value).__name__}")
            
            # Check array is not empty
            if value.size == 0:
                raise ValueError(f"Field '{field}' cannot be empty array")
            
            # Check array is 1-dimensional for main fields
            if value.ndim != 1:
                raise ValueError(f"Field '{field}' must be 1-dimensional array, got {value.ndim}D")
            
            # Check for appropriate numeric dtype
            if not np.issubdtype(value.dtype, np.number):
                raise ValueError(f"Field '{field}' must contain numeric data, got dtype {value.dtype}")
        
        # Check metadata is dictionary
        if not hasattr(self, 'metadata'):
            raise ValueError("Required field 'metadata' is missing from dataset")
        
        if not isinstance(self.metadata, dict):
            raise ValueError(f"Metadata must be dictionary, got {type(self.metadata).__name__}")
        
        # Check covariance is array or None
        if hasattr(self, 'covariance'):
            if self.covariance is not None:
                if not isinstance(self.covariance, np.ndarray):
                    raise ValueError(f"Covariance must be numpy array or None, got {type(self.covariance).__name__}")
                

                

                
                # Check for appropriate numeric dtype
                if not np.issubdtype(self.covariance.dtype, np.number):
                    raise ValueError(f"Covariance matrix must contain numeric data, got dtype {self.covariance.dtype}")
        
        # Validate required metadata fields
        required_metadata_keys = ['source']
        for key in required_metadata_keys:
            if key not in self.metadata:
                raise ValueError(f"Required metadata field '{key}' is missing")
            
            if not isinstance(self.metadata[key], str):
                raise ValueError(f"Metadata field '{key}' must be string, got {type(self.metadata[key]).__name__}")
        
        # Validate optional but recommended metadata fields
        if 'dataset_type' in self.metadata:
            if not isinstance(self.metadata['dataset_type'], str):
                raise ValueError(f"Metadata field 'dataset_type' must be string, got {type(self.metadata['dataset_type']).__name__}")
        
        return True
    
    def validate_numerical(self) -> bool:
        """
        Validate numerical sanity (no NaNs, infinities, negative variances).
        
        Returns:
            bool: True if numerical validation passes
            
        Raises:
            ValueError: If numerical validation fails
        """
        # Check for complex numbers in main arrays
        for field_name in ['z', 'observable', 'uncertainty']:
            field_data = getattr(self, field_name)
            
            # Check for complex numbers
            if np.iscomplexobj(field_data):
                complex_mask = np.iscomplex(field_data)
                if np.any(complex_mask):
                    complex_count = np.sum(complex_mask)
                    complex_indices = np.where(complex_mask)[0][:5]  # Show first 5 indices
                    raise ValueError(f"Complex values found in {field_name} ({complex_count} total, indices: {complex_indices.tolist()})")
        
        # Check for NaNs and infinities in main arrays
        for field_name in ['z', 'observable', 'uncertainty']:
            field_data = getattr(self, field_name)
            
            # Check for NaN values
            nan_mask = np.isnan(field_data)
            if np.any(nan_mask):
                nan_count = np.sum(nan_mask)
                nan_indices = np.where(nan_mask)[0][:5]  # Show first 5 indices
                raise ValueError(f"NaN values found in {field_name} ({nan_count} total, indices: {nan_indices.tolist()})")
            
            # Check for infinite values
            inf_mask = np.isinf(field_data)
            if np.any(inf_mask):
                inf_count = np.sum(inf_mask)
                inf_indices = np.where(inf_mask)[0][:5]  # Show first 5 indices
                raise ValueError(f"Infinite values found in {field_name} ({inf_count} total, indices: {inf_indices.tolist()})")
        
        # Check for negative uncertainties (variances must be non-negative)
        negative_mask = self.uncertainty < 0
        if np.any(negative_mask):
            negative_count = np.sum(negative_mask)
            negative_indices = np.where(negative_mask)[0][:5]
            min_uncertainty = np.min(self.uncertainty)
            raise ValueError(f"Negative uncertainty values found ({negative_count} total, min: {min_uncertainty}, indices: {negative_indices.tolist()})")
        
        # Check for zero uncertainties (may indicate missing error bars)
        zero_mask = self.uncertainty == 0
        if np.any(zero_mask):
            zero_count = np.sum(zero_mask)
            if zero_count > len(self.uncertainty) * 0.1:  # More than 10% zero uncertainties is suspicious
                zero_indices = np.where(zero_mask)[0][:5]
                raise ValueError(f"Suspicious number of zero uncertainties found ({zero_count} total, {zero_count/len(self.uncertainty)*100:.1f}%, indices: {zero_indices.tolist()})")
        
        # Check for negative redshifts (unphysical)
        if np.any(self.z < 0):
            negative_z_mask = self.z < 0
            negative_z_count = np.sum(negative_z_mask)
            negative_z_indices = np.where(negative_z_mask)[0][:5]
            min_z = np.min(self.z)
            raise ValueError(f"Negative redshift values found ({negative_z_count} total, min: {min_z}, indices: {negative_z_indices.tolist()})")
        
        # Check covariance matrix if present
        if self.covariance is not None:
            # Check for NaN values
            if np.any(np.isnan(self.covariance)):
                nan_count = np.sum(np.isnan(self.covariance))
                raise ValueError(f"NaN values found in covariance matrix ({nan_count} total)")
            
            # Check for infinite values
            if np.any(np.isinf(self.covariance)):
                inf_count = np.sum(np.isinf(self.covariance))
                raise ValueError(f"Infinite values found in covariance matrix ({inf_count} total)")
            
            # Check diagonal elements are non-negative (variances)
            diagonal = np.diag(self.covariance)
            if np.any(diagonal < 0):
                negative_var_count = np.sum(diagonal < 0)
                min_var = np.min(diagonal)
                raise ValueError(f"Negative variance values found on covariance diagonal ({negative_var_count} total, min: {min_var})")
        
        return True
    
    def validate_covariance(self) -> bool:
        """
        Validate covariance matrix properties (symmetry, positive-definiteness, correlation coefficients).
        
        Returns:
            bool: True if covariance validation passes
            
        Raises:
            ValueError: If covariance validation fails
        """
        if self.covariance is None:
            return True
        
        # Check covariance matrix size matches data
        expected_size = len(self.observable)
        actual_shape = self.covariance.shape
        
        if len(actual_shape) != 2:
            raise ValueError(f"Covariance matrix must be 2-dimensional, got {len(actual_shape)}D array with shape {actual_shape}")
        
        if actual_shape[0] != actual_shape[1]:
            raise ValueError(f"Covariance matrix must be square, got shape {actual_shape}")
        
        if actual_shape[0] != expected_size:
            raise ValueError(f"Covariance matrix size {actual_shape} doesn't match data size ({expected_size}, {expected_size})")
        
        n = self.covariance.shape[0]
        
        # Check for NaN values first
        if np.any(np.isnan(self.covariance)):
            nan_count = np.sum(np.isnan(self.covariance))
            raise ValueError(f"NaN values found in covariance matrix ({nan_count} total)")
        
        # Check for infinite values
        if np.any(np.isinf(self.covariance)):
            inf_count = np.sum(np.isinf(self.covariance))
            raise ValueError(f"Infinite values found in covariance matrix ({inf_count} total)")
        
        # Check symmetry with detailed error reporting
        symmetry_diff = self.covariance - self.covariance.T
        max_asymmetry = np.max(np.abs(symmetry_diff))
        if not np.allclose(self.covariance, self.covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Covariance matrix is not symmetric (max asymmetry: {max_asymmetry})")
        
        # Check diagonal elements are positive (variances)
        diagonal = np.diag(self.covariance)
        if np.any(diagonal <= 0):
            negative_indices = np.where(diagonal <= 0)[0]
            min_diagonal = np.min(diagonal)
            raise ValueError(f"Covariance matrix has non-positive diagonal elements ({len(negative_indices)} total, min: {min_diagonal}, indices: {negative_indices[:5].tolist()})")
        
        # Check positive-definiteness using eigenvalue decomposition
        try:
            eigenvalues = np.linalg.eigvals(self.covariance)
            min_eigenvalue = np.min(eigenvalues)
            
            if min_eigenvalue <= 0:
                negative_eigenvalue_count = np.sum(eigenvalues <= 0)
                condition_number = np.max(eigenvalues) / max(min_eigenvalue, 1e-16)
                raise ValueError(f"Covariance matrix is not positive-definite (min eigenvalue: {min_eigenvalue}, {negative_eigenvalue_count} non-positive eigenvalues, condition number: {condition_number})")
            
            # Check condition number for numerical stability
            condition_number = np.max(eigenvalues) / min_eigenvalue
            if condition_number > 1e12:
                raise ValueError(f"Covariance matrix is poorly conditioned (condition number: {condition_number})")
                
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Failed to compute eigenvalues for covariance matrix: {str(e)}")
        
        # Validate correlation coefficients are in range [-1, 1]
        try:
            # Compute correlation matrix from covariance
            std_devs = np.sqrt(diagonal)
            correlation_matrix = self.covariance / np.outer(std_devs, std_devs)
            
            # Check diagonal elements are 1 (within tolerance)
            diag_corr = np.diag(correlation_matrix)
            if not np.allclose(diag_corr, 1.0, rtol=1e-10):
                max_diag_error = np.max(np.abs(diag_corr - 1.0))
                raise ValueError(f"Correlation matrix diagonal elements not equal to 1 (max error: {max_diag_error})")
            
            # Check off-diagonal correlation coefficients are in [-1, 1]
            # Set diagonal to 0 to check only off-diagonal elements
            off_diag_corr = correlation_matrix.copy()
            np.fill_diagonal(off_diag_corr, 0)
            
            if np.any(off_diag_corr < -1) or np.any(off_diag_corr > 1):
                invalid_mask = (off_diag_corr < -1) | (off_diag_corr > 1)
                invalid_indices = np.where(invalid_mask)
                invalid_values = off_diag_corr[invalid_mask]
                min_corr = np.min(off_diag_corr)
                max_corr = np.max(off_diag_corr)
                
                raise ValueError(f"Correlation coefficients outside [-1, 1] range (min: {min_corr}, max: {max_corr}, {len(invalid_values)} invalid values)")
            
            # Check for perfect correlations (may indicate issues)
            perfect_corr_mask = (np.abs(off_diag_corr) >= 0.9999) & (off_diag_corr != 0)
            if np.any(perfect_corr_mask):
                perfect_indices = np.where(perfect_corr_mask)
                perfect_count = len(perfect_indices[0])
                if perfect_count > n * 0.1:  # More than 10% perfect correlations is suspicious
                    raise ValueError(f"Suspicious number of near-perfect correlations found ({perfect_count} pairs with |r| >= 0.9999)")
                    
        except (ValueError, ZeroDivisionError) as e:
            if "Correlation coefficients" in str(e) or "Correlation matrix" in str(e):
                raise e
            else:
                raise ValueError(f"Failed to validate correlation coefficients: {str(e)}")
        
        return True
    
    def validate_redshift_range(self, min_z: float = 0.0, max_z: float = 10.0) -> bool:
        """
        Validate redshift range consistency against known catalog limits.
        
        Args:
            min_z: Minimum allowed redshift
            max_z: Maximum allowed redshift
            
        Returns:
            bool: True if redshift range is valid
            
        Raises:
            ValueError: If redshift validation fails
        """
        if np.any(self.z < min_z):
            min_found = np.min(self.z)
            raise ValueError(f"Redshift values below minimum {min_z} found (min: {min_found})")
        
        if np.any(self.z > max_z):
            max_found = np.max(self.z)
            raise ValueError(f"Redshift values above maximum {max_z} found (max: {max_found})")
        
        return True
    
    def validate_all(self, min_z: float = 0.0, max_z: float = 10.0) -> bool:
        """
        Run all validation checks.
        
        Args:
            min_z: Minimum allowed redshift
            max_z: Maximum allowed redshift
            
        Returns:
            bool: True if all validations pass
        """
        self.validate_schema()
        self.validate_numerical()
        self.validate_covariance()
        self.validate_redshift_range(min_z, max_z)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for serialization.
        
        Returns:
            Dict containing all dataset fields
        """
        return {
            'z': self.z,
            'observable': self.observable,
            'uncertainty': self.uncertainty,
            'covariance': self.covariance,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StandardDataset':
        """
        Create StandardDataset from dictionary.
        
        Args:
            data: Dictionary containing dataset fields
            
        Returns:
            StandardDataset instance
        """
        return cls(
            z=data['z'],
            observable=data['observable'],
            uncertainty=data['uncertainty'],
            covariance=data.get('covariance'),
            metadata=data.get('metadata', {})
        )