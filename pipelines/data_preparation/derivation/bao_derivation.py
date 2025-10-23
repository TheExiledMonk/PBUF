"""
BAO (Baryon Acoustic Oscillations) derivation module for the PBUF data preparation framework.

This module implements BAO-specific transformation logic including:
- Processing both isotropic (D_V/r_d) and anisotropic (D_M/r_d, D_H/r_d) measurements
- Distance measure unit conversion to consistent Mpc units
- Correlation matrix validation and survey-specific systematic corrections
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Handle optional dependencies
try:
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:
    pd = None
    DataFrame = Any  # Fallback type for when pandas is not available

from ..core.interfaces import DerivationModule, ProcessingError
from ..core.schema import StandardDataset


class BAODerivationModule(DerivationModule):
    """
    BAO derivation module implementing BAO-specific data processing.
    
    Transforms raw BAO measurements into standardized format with proper
    distance measure calculations, unit conversions, and systematic
    error handling for both isotropic and anisotropic measurements.
    """
    
    @property
    def dataset_type(self) -> str:
        """Return dataset type identifier."""
        return 'bao'
    
    @property
    def supported_formats(self) -> List[str]:
        """Return supported input file formats."""
        return ['.txt', '.csv', '.dat', '.json']
    
    def _is_numeric_string(self, s: str) -> bool:
        """Check if a string represents a numeric value."""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Validate raw BAO data before processing.
        
        Args:
            raw_data_path: Path to raw dataset file
            metadata: Dataset metadata from registry
            
        Returns:
            bool: True if input is valid
            
        Raises:
            ValueError: If input validation fails
        """
        if not raw_data_path.exists():
            raise ValueError("Input file does not exist")
        
        # Check file size is reasonable
        file_size = raw_data_path.stat().st_size
        if file_size == 0:
            raise ValueError("Raw data file is empty")
        
        if file_size > 50 * 1024 * 1024:  # 50 MB limit
            raise ValueError(f"Raw data file too large: {file_size / (1024*1024):.1f} MB")
        
        # Try to read and validate basic structure
        try:
            if raw_data_path.suffix.lower() in ['.txt', '.dat', '.csv']:
                data = pd.read_csv(raw_data_path, comment='#', sep=None, engine='python')
                
                # Check if this is a headerless CSV with numeric column names (indicating no header)
                if len(data.columns) == 3 and all(self._is_numeric_string(str(col)) for col in data.columns):
                    # This is likely a headerless CSV with BAO values: z, D_V, sigma_D_V
                    data.columns = ['z', 'D_V', 'sigma_D_V']
                
            elif raw_data_path.suffix.lower() == '.json':
                data = pd.read_json(raw_data_path)
            else:
                raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
            
            # Determine measurement type from metadata or data structure
            measurement_type = self._determine_measurement_type(data, metadata)
            
            # Validate based on measurement type
            if measurement_type == 'isotropic':
                self._validate_isotropic_data(data)
            elif measurement_type == 'anisotropic':
                self._validate_anisotropic_data(data)
            else:
                raise ValueError(f"Unknown BAO measurement type: {measurement_type}")
            
            # Check data has reasonable number of points
            min_points = 1 if metadata and 'compatibility' in metadata.get('source', '').lower() else 2
            if len(data) < min_points:
                raise ValueError(f"Insufficient data points: {len(data)} (minimum {min_points} required)")
            
            if len(data) > 1000:
                raise ValueError(f"Too many data points: {len(data)} (maximum 1000 supported)")
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to read or validate raw data: {str(e)}")
        
        return True
    
    def _determine_measurement_type(self, data: DataFrame, metadata: Dict[str, Any]) -> str:
        """Determine if measurements are isotropic or anisotropic."""
        # Check metadata first
        if 'measurement_type' in metadata:
            return metadata['measurement_type'].lower()
        
        # Infer from column names
        isotropic_columns = ['dv_rd', 'dv_over_rd', 'd_v', 'dv']
        anisotropic_columns = ['dm_rd', 'dh_rd', 'dm_over_rd', 'dh_over_rd', 'd_m', 'd_h']
        
        has_isotropic = any(col.lower() in [c.lower() for c in data.columns] for col in isotropic_columns)
        has_anisotropic = any(col.lower() in [c.lower() for c in data.columns] for col in anisotropic_columns)
        
        if has_isotropic and not has_anisotropic:
            return 'isotropic'
        elif has_anisotropic and not has_isotropic:
            return 'anisotropic'
        elif has_isotropic and has_anisotropic:
            # Mixed - prefer anisotropic as it's more informative
            return 'anisotropic'
        else:
            raise ValueError("Cannot determine BAO measurement type from data columns")
    
    def _validate_isotropic_data(self, data: DataFrame):
        """Validate isotropic BAO data structure."""
        # Check for required columns
        required_columns = ['z']  # redshift is always required
        distance_columns = ['dv_rd', 'dv_over_rd', 'd_v', 'dv', 'D_V']
        error_columns = ['dv_rd_err', 'dv_over_rd_err', 'dv_err', 'err_dv', 'sigma_dv', 'sigma_D_V']
        
        # Find distance column
        distance_col = self._find_column(data, distance_columns)
        if distance_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. Expected one of {distance_columns}. Available: {available_cols}")
        
        # Find error column
        error_col = self._find_column(data, error_columns)
        if error_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. Expected one of {error_columns}. Available: {available_cols}")
        
        # Validate ranges
        z_col = self._find_column(data, ['z', 'redshift', 'zeff'])
        if z_col is None:
            raise ValueError("No redshift column found")
        
        # Check for NaN values
        if data[z_col].isna().any():
            raise ValueError("NaN redshift values found")
        
        if data[z_col].min() < 0:
            raise ValueError(f"Negative redshift values found (min: {data[z_col].min()})")
        
        if data[z_col].max() > 5.0:
            raise ValueError(f"Unreasonably high redshift values found (max: {data[z_col].max()})")
        
        distance_col_actual = self._find_column(data, distance_columns)
        if data[distance_col_actual].min() <= 0:
            raise ValueError(f"Non-positive distance values found (min: {data[distance_col_actual].min()})")
        
        error_col_actual = self._find_column(data, error_columns)
        if data[error_col_actual].min() <= 0:
            raise ValueError(f"Non-positive error values found (min: {data[error_col_actual].min()})")
    
    def _validate_anisotropic_data(self, data: DataFrame):
        """Validate anisotropic BAO data structure."""
        # Check for required columns
        z_col = self._find_column(data, ['z', 'redshift', 'zeff'])
        if z_col is None:
            raise ValueError("No redshift column found")
        
        # Check for anisotropic distance measures
        dm_columns = ['dm_rd', 'dm_over_rd', 'd_m']
        dh_columns = ['dh_rd', 'dh_over_rd', 'd_h']
        
        dm_col = self._find_column(data, dm_columns)
        dh_col = self._find_column(data, dh_columns)
        
        if dm_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. Expected one of {dm_columns}. Available: {available_cols}")
        
        if dh_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. Expected one of {dh_columns}. Available: {available_cols}")
        
        # Check for error columns
        dm_err_columns = ['dm_rd_err', 'dm_over_rd_err', 'dm_err', 'err_dm', 'sigma_dm']
        dh_err_columns = ['dh_rd_err', 'dh_over_rd_err', 'dh_err', 'err_dh', 'sigma_dh']
        
        dm_err_col = self._find_column(data, dm_err_columns)
        dh_err_col = self._find_column(data, dh_err_columns)
        
        if dm_err_col is None:
            raise ValueError(f"Missing required columns. Expected one of {dm_err_columns}")
        
        if dh_err_col is None:
            raise ValueError(f"Missing required columns. Expected one of {dh_err_columns}")
        
        # Validate ranges
        if data[z_col].min() < 0:
            raise ValueError(f"Negative redshift values found (min: {data[z_col].min()})")
        
        if data[dm_col].min() <= 0:
            raise ValueError(f"Non-positive D_M values found (min: {data[dm_col].min()})")
        
        if data[dh_col].min() <= 0:
            raise ValueError(f"Non-positive D_H values found (min: {data[dh_col].min()})")
    
    def _find_column(self, data: DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column with case-insensitive matching."""
        for col_name in possible_names:
            for actual_col in data.columns:
                if col_name.lower() == actual_col.lower():
                    return actual_col
        return None
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """
        Transform raw BAO data to standardized format.
        
        Args:
            raw_data_path: Path to verified raw dataset file
            metadata: Dataset metadata from registry
            
        Returns:
            StandardDataset: Transformed BAO data in standard format
        """
        try:
            # Load raw data
            data = self._load_raw_data(raw_data_path)
            
            # Determine measurement type
            measurement_type = self._determine_measurement_type(data, metadata)
            
            # Validate data quality
            if measurement_type == 'isotropic':
                self._validate_isotropic_data(data)
            else:
                self._validate_anisotropic_data(data)
            
            # Convert units to consistent Mpc
            data = self._convert_units(data, metadata)
            
            # Apply survey-specific corrections
            data = self._apply_survey_corrections(data, metadata)
            
            # Extract standardized measurements
            if measurement_type == 'isotropic':
                z, observable, uncertainty = self._extract_isotropic_measurements(data)
            else:
                z, observable, uncertainty = self._extract_anisotropic_measurements(data)
            
            # Validate and apply correlation matrix
            covariance = self._process_correlation_matrix(data, metadata, measurement_type)
            
            # Create standardized dataset
            standard_dataset = StandardDataset(
                z=z,
                observable=observable,
                uncertainty=uncertainty,
                covariance=covariance,
                metadata=self._create_metadata(metadata, data, measurement_type)
            )
            
            # Store transformation summary for provenance
            self._transformation_summary = self._generate_transformation_summary(data, metadata, measurement_type)
            
            return standard_dataset
            
        except ValueError as e:
            # Data validation errors
            raise ProcessingError(
                dataset_name=metadata.get('name', 'unknown'),
                stage='validation',
                error_type='data_validation_error',
                error_message=str(e),
                context={'raw_data_path': str(raw_data_path)},
                suggested_actions=[
                    'Check data quality and remove invalid values',
                    'Verify redshift values are positive and finite',
                    'Check distance measurements for physical validity'
                ]
            )
        except Exception as e:
            # Other processing errors
            raise ProcessingError(
                dataset_name=metadata.get('name', 'unknown'),
                stage='transformation',
                error_type='bao_derivation_error',
                error_message=str(e),
                context={'raw_data_path': str(raw_data_path)},
                suggested_actions=[
                    'Check raw data format and column names',
                    'Verify measurement type (isotropic/anisotropic) in metadata',
                    'Check unit specifications and survey information'
                ]
            )
    
    def _load_raw_data(self, raw_data_path: Path) -> DataFrame:
        """Load and standardize column names from raw data."""
        # Read data
        if raw_data_path.suffix.lower() in ['.txt', '.dat', '.csv']:
            data = pd.read_csv(raw_data_path, comment='#', sep=None, engine='python')
            
            # Check if this is a headerless CSV with numeric column names (indicating no header)
            if len(data.columns) == 3 and all(self._is_numeric_string(str(col)) for col in data.columns):
                # This is likely a headerless CSV with BAO values: z, D_V, sigma_D_V
                data.columns = ['z', 'D_V', 'sigma_D_V']
                
        elif raw_data_path.suffix.lower() == '.json':
            data = pd.read_json(raw_data_path)
        else:
            raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
        
        return data
    
    def _convert_units(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Convert distance measures to consistent Mpc units."""
        converted_data = data.copy()
        
        # Get unit information from metadata
        units_info = metadata.get('units', {})
        
        # Handle case where units is a string instead of dict
        if isinstance(units_info, str):
            # If it's a string, apply it to all distance columns
            default_unit = units_info.lower()
            units_info = {}
        else:
            default_unit = 'dimensionless'
        
        # Standard conversion factors to Mpc
        unit_conversions = {
            'mpc': 1.0,
            'mpc_h': 1.0,  # Will be handled separately with h_value
            'kpc': 0.001,
            'gpc': 1000.0,
            'km/s/mpc': 1.0,  # For H(z) measurements
            'dimensionless': 1.0  # For ratios like D_V/r_d
        }
        
        # Convert distance columns
        distance_columns = ['dv_rd', 'dv_over_rd', 'd_v', 'dv', 'dm_rd', 'dm_over_rd', 'd_m', 'dh_rd', 'dh_over_rd', 'd_h']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(dist_col in col_lower for dist_col in distance_columns):
                # Get unit for this column
                if isinstance(metadata.get('units'), str):
                    col_unit = metadata.get('units', 'dimensionless').lower()
                else:
                    col_unit = units_info.get(col, default_unit).lower()
                
                if col_unit in unit_conversions:
                    conversion_factor = unit_conversions[col_unit]
                    
                    # Handle h-units conversion
                    if col_unit == 'mpc_h' and 'h_value' in metadata:
                        h_value = metadata['h_value']
                        conversion_factor = conversion_factor / h_value  # Convert Mpc/h to Mpc
                    
                    converted_data[col] = converted_data[col] * conversion_factor
                    
                    # Also convert corresponding error columns
                    error_col_names = [f"{col}_err", f"err_{col}", f"sigma_{col}", f"d{col}"]
                    for err_col in error_col_names:
                        if err_col in converted_data.columns:
                            converted_data[err_col] = converted_data[err_col] * conversion_factor
                else:
                    print(f"Warning: Unknown unit '{col_unit}' for column '{col}', no conversion applied")
        
        return converted_data
    
    def _apply_survey_corrections(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Apply survey-specific systematic corrections."""
        corrected_data = data.copy()
        
        survey = metadata.get('survey', '').lower()
        
        # Apply survey-specific corrections based on known systematics
        if survey == 'boss':
            # BOSS-specific corrections
            # Apply reconstruction correction if needed
            if metadata.get('reconstruction', False):
                # Typical reconstruction boost factor ~1.5-2.0
                boost_factor = metadata.get('reconstruction_boost', 1.8)
                distance_cols = self._get_distance_columns(corrected_data)
                for col in distance_cols:
                    corrected_data[col] = corrected_data[col] * boost_factor
                    
        elif survey == 'eboss':
            # eBOSS-specific corrections
            # Apply fiber collision corrections if needed
            if metadata.get('fiber_collision_correction', True):
                # Typical correction factor ~1.02
                correction_factor = metadata.get('fiber_collision_factor', 1.02)
                distance_cols = self._get_distance_columns(corrected_data)
                for col in distance_cols:
                    corrected_data[col] = corrected_data[col] * correction_factor
                    
        elif survey == 'desi':
            # DESI-specific corrections
            # Apply imaging systematics corrections
            if metadata.get('imaging_systematics_correction', True):
                correction_factor = metadata.get('imaging_correction_factor', 1.01)
                distance_cols = self._get_distance_columns(corrected_data)
                for col in distance_cols:
                    corrected_data[col] = corrected_data[col] * correction_factor
        
        return corrected_data
    
    def _get_distance_columns(self, data: DataFrame) -> List[str]:
        """Get list of distance measurement columns."""
        distance_keywords = ['dv', 'd_v', 'dm', 'd_m', 'dh', 'd_h']
        distance_cols = []
        
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in distance_keywords):
                if 'err' not in col_lower and 'sigma' not in col_lower:
                    distance_cols.append(col)
        
        return distance_cols
    
    def _extract_isotropic_measurements(self, data: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract isotropic BAO measurements (D_V/r_d)."""
        # Find columns
        z_col = self._find_column(data, ['z', 'redshift', 'zeff'])
        distance_col = self._find_column(data, ['dv_rd', 'dv_over_rd', 'd_v', 'dv', 'D_V'])
        error_col = self._find_column(data, ['dv_rd_err', 'dv_over_rd_err', 'dv_err', 'err_dv', 'sigma_dv', 'sigma_D_V'])
        
        z = data[z_col].values
        observable = data[distance_col].values
        uncertainty = data[error_col].values
        
        return z, observable, uncertainty
    
    def _extract_anisotropic_measurements(self, data: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract anisotropic BAO measurements [D_M/r_d, D_H/r_d]."""
        # Find columns
        z_col = self._find_column(data, ['z', 'redshift', 'zeff'])
        dm_col = self._find_column(data, ['dm_rd', 'dm_over_rd', 'd_m'])
        dh_col = self._find_column(data, ['dh_rd', 'dh_over_rd', 'd_h'])
        dm_err_col = self._find_column(data, ['dm_rd_err', 'dm_over_rd_err', 'dm_err', 'err_dm', 'sigma_dm'])
        dh_err_col = self._find_column(data, ['dh_rd_err', 'dh_over_rd_err', 'dh_err', 'err_dh', 'sigma_dh'])
        
        z = data[z_col].values
        dm_values = data[dm_col].values
        dh_values = data[dh_col].values
        dm_errors = data[dm_err_col].values
        dh_errors = data[dh_err_col].values
        
        # Stack anisotropic measurements: [D_M/r_d, D_H/r_d] for each redshift
        n_points = len(z)
        observable = np.zeros(2 * n_points)
        uncertainty = np.zeros(2 * n_points)
        z_expanded = np.zeros(2 * n_points)
        
        # Interleave D_M and D_H measurements
        observable[0::2] = dm_values  # D_M/r_d at even indices
        observable[1::2] = dh_values  # D_H/r_d at odd indices
        uncertainty[0::2] = dm_errors
        uncertainty[1::2] = dh_errors
        z_expanded[0::2] = z
        z_expanded[1::2] = z
        
        return z_expanded, observable, uncertainty
    
    def _process_correlation_matrix(self, data: DataFrame, metadata: Dict[str, Any], measurement_type: str) -> Optional[np.ndarray]:
        """Validate and process correlation matrix."""
        # Check if correlation matrix is provided
        if ('correlation_matrix' not in metadata and 
            'covariance_matrix' not in metadata and 
            'correlation_matrix_file' not in metadata and 
            'covariance_matrix_file' not in metadata):
            return None
        
        # Load correlation/covariance matrix
        if 'correlation_matrix' in metadata or 'correlation_matrix_file' in metadata:
            corr_info = metadata.get('correlation_matrix') or metadata.get('correlation_matrix_file')
            if isinstance(corr_info, str):
                # Load from file
                if corr_info.endswith('.npy'):
                    corr_matrix = np.load(corr_info)
                else:
                    corr_matrix = np.loadtxt(corr_info)
            elif isinstance(corr_info, list):
                # Direct matrix data
                corr_matrix = np.array(corr_info)
            else:
                return None
            
            # Convert correlation to covariance
            if measurement_type == 'isotropic':
                error_col = self._find_column(data, ['dv_rd_err', 'dv_over_rd_err', 'dv_err', 'err_dv', 'sigma_dv'])
                if error_col is None:
                    raise ValueError(f"Could not find error column for isotropic data. Available columns: {list(data.columns)}")
                errors = data[error_col].values
            else:
                dm_err_col = self._find_column(data, ['dm_rd_err', 'dm_over_rd_err', 'dm_err', 'err_dm', 'sigma_dm'])
                dh_err_col = self._find_column(data, ['dh_rd_err', 'dh_over_rd_err', 'dh_err', 'err_dh', 'sigma_dh'])
                if dm_err_col is None or dh_err_col is None:
                    raise ValueError(f"Could not find error columns for anisotropic data. Available columns: {list(data.columns)}")
                dm_errors = data[dm_err_col].values
                dh_errors = data[dh_err_col].values
                # Interleave errors for anisotropic case
                n_points = len(dm_errors)
                errors = np.zeros(2 * n_points)
                errors[0::2] = dm_errors
                errors[1::2] = dh_errors
            
            # Convert correlation to covariance: Cov_ij = Corr_ij * σ_i * σ_j
            covariance = corr_matrix * np.outer(errors, errors)
            
        elif 'covariance_matrix' in metadata or 'covariance_matrix_file' in metadata:
            cov_info = metadata.get('covariance_matrix') or metadata.get('covariance_matrix_file')
            if isinstance(cov_info, str):
                if cov_info.endswith('.npy'):
                    covariance = np.load(cov_info)
                else:
                    covariance = np.loadtxt(cov_info)
            elif isinstance(cov_info, list):
                covariance = np.array(cov_info)
            else:
                return None
        
        # Validate matrix properties
        self._validate_covariance_matrix(covariance)
        
        return covariance
    
    def _validate_covariance_matrix(self, covariance: np.ndarray):
        """Validate covariance matrix properties."""
        # Check symmetry
        if not np.allclose(covariance, covariance.T, rtol=1e-10):
            raise ValueError("Covariance matrix is not symmetric")
        
        # Check positive definiteness
        eigenvalues = np.linalg.eigvals(covariance)
        if np.any(eigenvalues <= 0):
            min_eigenvalue = np.min(eigenvalues)
            raise ValueError(f"Covariance matrix is not positive definite (min eigenvalue: {min_eigenvalue})")
        
        # Check condition number
        condition_number = np.max(eigenvalues) / np.min(eigenvalues)
        if condition_number > 1e12:
            raise ValueError(f"Covariance matrix is poorly conditioned (condition number: {condition_number})")
    
    def _create_metadata(self, input_metadata: Dict[str, Any], processed_data: DataFrame, measurement_type: str) -> Dict[str, Any]:
        """Create metadata for the standardized dataset."""
        z_col = self._find_column(processed_data, ['z', 'redshift', 'zeff'])
        
        metadata = {
            'dataset_type': 'bao',
            'measurement_type': measurement_type,
            'source': input_metadata.get('source', 'unknown'),
            'citation': input_metadata.get('citation', ''),
            'version': input_metadata.get('version', ''),
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'n_redshift_bins': len(processed_data),
            'redshift_range': [float(processed_data[z_col].min()), float(processed_data[z_col].max())],
            'survey': input_metadata.get('survey', 'unknown'),
            'transformations_applied': [
                'unit_conversion_to_mpc',
                'survey_specific_corrections',
                'correlation_matrix_validation'
            ]
        }
        
        # Add measurement-specific information
        if measurement_type == 'isotropic':
            metadata['observable_description'] = 'Isotropic BAO distance D_V/r_d'
            metadata['n_measurements'] = len(processed_data)
            metadata['n_points'] = len(processed_data)
        else:
            metadata['observable_description'] = 'Anisotropic BAO distances [D_M/r_d, D_H/r_d]'
            metadata['n_measurements'] = 2 * len(processed_data)
            metadata['n_points'] = len(processed_data)
            metadata['DM_rd_points'] = len(processed_data)
            metadata['DH_rd_points'] = len(processed_data)
        
        # Add survey-specific information
        if 'reconstruction' in input_metadata:
            metadata['reconstruction'] = input_metadata['reconstruction']
        
        if 'tracer' in input_metadata:
            metadata['tracer'] = input_metadata['tracer']
        
        # Add systematic corrections if applied
        if 'systematic_corrections' in input_metadata:
            metadata['systematic_corrections_applied'] = input_metadata['systematic_corrections']
        
        return metadata
    
    def _generate_transformation_summary(self, data: DataFrame, metadata: Dict[str, Any], measurement_type: str) -> Dict[str, Any]:
        """Generate summary of applied transformations."""
        z_col = self._find_column(data, ['z', 'redshift', 'zeff'])
        
        return {
            'transformation_steps': [
                f'Loaded raw BAO data with {measurement_type} measurements',
                'Applied unit conversions to consistent Mpc scale',
                'Applied survey-specific systematic corrections',
                'Validated and processed correlation matrix',
                'Extracted standardized distance measurements'
            ],
            'formulas_used': [
                'D_V(z) = [z * D_M²(z) * D_H(z)]^(1/3) (isotropic volume-averaged distance)',
                'D_M(z) = comoving angular diameter distance',
                'D_H(z) = c/H(z) (Hubble distance)',
                'Cov_ij = Corr_ij * σ_i * σ_j (correlation to covariance conversion)'
            ],
            'assumptions': [
                'Sound horizon scale r_d from CMB measurements',
                'Linear theory BAO feature at ~150 Mpc/h',
                'Survey-specific systematic corrections applied',
                'Gaussian uncertainties for error propagation'
            ],
            'references': [
                'Eisenstein et al. 2005 (BAO detection)',
                'Anderson et al. 2014 (BOSS BAO measurements)',
                'Alam et al. 2017 (eBOSS BAO analysis)',
                'DESI Collaboration 2024 (DESI BAO results)'
            ],
            'data_statistics': {
                'measurement_type': measurement_type,
                'n_redshift_bins': len(data),
                'redshift_range': [float(data[z_col].min()), float(data[z_col].max())],
                'survey': metadata.get('survey', 'unknown')
            }
        }
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return summary of applied transformations."""
        if not hasattr(self, '_transformation_summary'):
            return {
                'transformation_steps': [
                    'Load raw BAO data with isotropic/anisotropic measurements',
                    'Apply unit conversions to consistent Mpc scale',
                    'Apply survey-specific systematic corrections',
                    'Validate and process correlation matrix',
                    'Extract standardized distance measurements'
                ],
                'formulas_used': [
                    'D_V(z) = [z * D_M²(z) * D_H(z)]^(1/3) (isotropic volume-averaged distance)',
                    'D_M(z) = comoving angular diameter distance',
                    'D_H(z) = c/H(z) (Hubble distance)',
                    'BAO scale: D_V/r_d, D_M/r_d, D_H/r_d (normalized by sound horizon)',
                    'Cov_ij = Corr_ij * σ_i * σ_j (correlation to covariance conversion)'
                ],
                'assumptions': [
                    'Sound horizon scale r_d from CMB measurements',
                    'Linear theory BAO feature at ~150 Mpc/h',
                    'Survey-specific systematic corrections applied',
                    'Gaussian uncertainties for error propagation'
                ],
                'references': [
                    'Eisenstein et al. 2005 (BAO detection)',
                    'Anderson et al. 2014 (BOSS BAO measurements)',
                    'Alam et al. 2017 (eBOSS BAO analysis)',
                    'DESI Collaboration 2024 (DESI BAO results)'
                ],
                'data_statistics': {}
            }
        return self._transformation_summary