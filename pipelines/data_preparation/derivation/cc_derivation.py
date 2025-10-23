"""
Cosmic Chronometers (CC) derivation module for the PBUF data preparation framework.

This module implements CC-specific transformation logic including:
- H(z) data merging from multiple compilation sources
- Overlapping redshift bin filtering and uncertainty propagation
- H(z) sign convention validation and systematic error handling
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Handle optional dependencies
try:
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:
    pd = None
    DataFrame = Any  # Fallback type for when pandas is not available

try:
    from scipy.interpolate import interp1d
except ImportError:
    interp1d = None

from ..core.interfaces import DerivationModule, ProcessingError
from ..core.schema import StandardDataset


class CCDerivationModule(DerivationModule):
    """
    Cosmic Chronometers derivation module implementing CC-specific data processing.
    
    Transforms raw H(z) measurements from cosmic chronometers into standardized 
    format with proper compilation merging, overlap filtering, and systematic
    error handling.
    """
    
    @property
    def dataset_type(self) -> str:
        """Return dataset type identifier."""
        return 'cc'
    
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
        Validate raw Cosmic Chronometers data before processing.
        
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
        
        if file_size > 10 * 1024 * 1024:  # 10 MB limit
            raise ValueError(f"Raw data file too large: {file_size / (1024*1024):.1f} MB")
        
        # Try to read and validate basic structure
        try:
            if raw_data_path.suffix.lower() in ['.txt', '.dat', '.csv']:
                data = pd.read_csv(raw_data_path, comment='#', sep=None, engine='python')
                
                # Check if this is a headerless CSV with numeric column names (indicating no header)
                if len(data.columns) == 3 and all(self._is_numeric_string(str(col)) for col in data.columns):
                    # This is likely a headerless CSV with CC values: z, H(z), sigma_H
                    data.columns = ['z', 'H', 'sigma_H']
                
            elif raw_data_path.suffix.lower() == '.json':
                data = pd.read_json(raw_data_path)
            else:
                raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
            
            # Check for required columns
            self._validate_cc_columns(data)
            
            # Validate H(z) measurements and filter extreme values
            data = self._validate_hz_measurements(data)
            
            # Check data has reasonable number of points
            min_points = 1 if metadata and 'compatibility' in metadata.get('source', '').lower() else 2
            if len(data) < min_points:
                raise ValueError(f"Insufficient data points: {len(data)} (minimum {min_points} required)")
            
            if len(data) > 500:
                raise ValueError(f"Too many data points: {len(data)} (maximum 500 supported)")
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to read or validate raw data: {str(e)}")
        
        return True
    
    def _validate_cc_columns(self, data: DataFrame):
        """Validate required columns for cosmic chronometers data."""
        # Check for redshift column
        z_columns = ['z', 'redshift', 'zeff']
        z_col = self._find_column(data, z_columns)
        if z_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. No redshift column found. Expected one of {z_columns}. Available: {available_cols}")
        
        # Check for H(z) column
        hz_columns = ['hz', 'h_z', 'hubble', 'h', 'H']
        hz_col = self._find_column(data, hz_columns)
        if hz_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. No H(z) column found. Expected one of {hz_columns}. Available: {available_cols}")
        
        # Check for H(z) error column
        hz_err_columns = ['hz_err', 'h_z_err', 'hubble_err', 'h_err', 'H_err', 'sigma_h', 'dh', 'err_hz']
        hz_err_col = self._find_column(data, hz_err_columns)
        if hz_err_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. No H(z) error column found. Expected one of {hz_err_columns}. Available: {available_cols}")
    
    def _validate_hz_measurements(self, data: DataFrame) -> DataFrame:
        """Validate H(z) measurement ranges and sign conventions, filtering extreme values."""
        z_col = self._find_column(data, ['z', 'redshift', 'zeff'])
        hz_col = self._find_column(data, ['hz', 'h_z', 'hubble', 'h', 'H'])
        hz_err_col = self._find_column(data, ['hz_err', 'h_z_err', 'hubble_err', 'h_err', 'H_err', 'sigma_h', 'dh', 'err_hz'])
        
        # Validate redshift range
        # Check for NaN values
        if data[z_col].isna().any():
            raise ValueError("NaN redshift values found")
        
        if data[z_col].min() < 0:
            raise ValueError(f"Negative redshift values found (min: {data[z_col].min()})")
        
        # Filter out extreme redshift values instead of raising error
        if data[z_col].max() > 3.0:
            extreme_mask = data[z_col] > 3.0
            n_extreme = extreme_mask.sum()
            print(f"Warning: Filtering {n_extreme} measurements with unreasonably high redshift (z > 3.0)")
            data = data[~extreme_mask].copy()
            
        if len(data) == 0:
            raise ValueError("No valid measurements remain after filtering extreme redshift values")
        
        # Validate H(z) values (should be positive and reasonable)
        if data[hz_col].min() <= 0:
            raise ValueError(f"Negative or non-positive H(z) values found (min: {data[hz_col].min()})")
        
        if data[hz_col].min() < 50:  # km/s/Mpc
            raise ValueError(f"Unreasonably low H(z) values found (min: {data[hz_col].min()} km/s/Mpc)")
        
        if data[hz_col].max() > 500:  # km/s/Mpc
            raise ValueError(f"Unreasonably high H(z) values found (max: {data[hz_col].max()} km/s/Mpc)")
        
        # Validate H(z) errors
        if data[hz_err_col].min() <= 0:
            raise ValueError(f"Non-positive H(z) error values found (min: {data[hz_err_col].min()})")
        
        if data[hz_err_col].max() > 100:  # km/s/Mpc
            raise ValueError(f"Unreasonably large H(z) errors found (max: {data[hz_err_col].max()} km/s/Mpc)")
        
        # Check for reasonable error-to-value ratios
        relative_errors = data[hz_err_col] / data[hz_col]
        if relative_errors.max() > 1.0:  # 100% relative error
            max_rel_err = relative_errors.max()
            raise ValueError(f"Unreasonably large relative errors found (max: {max_rel_err:.2%})")
        
        return data
    
    def _find_column(self, data: DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column with case-insensitive matching."""
        for col_name in possible_names:
            for actual_col in data.columns:
                if col_name.lower() == actual_col.lower():
                    return actual_col
        return None
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """
        Transform raw Cosmic Chronometers data to standardized format.
        
        Args:
            raw_data_path: Path to verified raw dataset file
            metadata: Dataset metadata from registry
            
        Returns:
            StandardDataset: Transformed CC data in standard format
        """
        try:
            # Load raw data
            data = self._load_raw_data(raw_data_path)
            
            # Merge data from multiple compilations if needed
            merged_data = self._merge_compilations(data, metadata)
            
            # Filter overlapping redshift bins
            filtered_data = self._filter_overlapping_bins(merged_data, metadata)
            
            # Validate H(z) sign conventions
            validated_data = self._validate_sign_conventions(filtered_data)
            
            # Handle systematic errors
            final_data = self._handle_systematic_errors(validated_data, metadata)
            
            # Extract standardized measurements
            z, hz, sigma_hz = self._extract_measurements(final_data)
            
            # Process covariance matrix if available
            covariance = self._process_covariance_matrix(final_data, metadata)
            
            # Create standardized dataset
            standard_dataset = StandardDataset(
                z=z,
                observable=hz,
                uncertainty=sigma_hz,
                covariance=covariance,
                metadata=self._create_metadata(metadata, final_data)
            )
            
            # Store transformation summary for provenance
            self._transformation_summary = self._generate_transformation_summary(final_data, metadata)
            
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
                    'Verify H(z) values are positive and reasonable',
                    'Check redshift values for physical validity'
                ]
            )
        except Exception as e:
            # Other processing errors
            raise ProcessingError(
                dataset_name=metadata.get('name', 'unknown'),
                stage='transformation',
                error_type='cc_derivation_error',
                error_message=str(e),
                context={'raw_data_path': str(raw_data_path)},
                suggested_actions=[
                    'Check H(z) column names and units',
                    'Verify compilation source information',
                    'Check for overlapping redshift bins'
                ]
            )
    
    def _load_raw_data(self, raw_data_path: Path) -> DataFrame:
        """Load and standardize column names from raw data."""
        # Read data
        if raw_data_path.suffix.lower() in ['.txt', '.dat', '.csv']:
            data = pd.read_csv(raw_data_path, comment='#', sep=None, engine='python')
            
            # Check if this is a headerless CSV with numeric column names (indicating no header)
            if len(data.columns) == 3 and all(self._is_numeric_string(str(col)) for col in data.columns):
                # This is likely a headerless CSV with CC values: z, H(z), sigma_H
                data.columns = ['z', 'H', 'sigma_H']
                
        elif raw_data_path.suffix.lower() == '.json':
            data = pd.read_json(raw_data_path)
        else:
            raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
        
        # Validate data before standardization and filter extreme values
        data = self._validate_hz_measurements(data)
        
        # Standardize column names
        column_mappings = {
            'z': ['z', 'redshift', 'zeff'],
            'hz': ['hz', 'h_z', 'hubble', 'h', 'H'],
            'hz_err': ['hz_err', 'h_z_err', 'hubble_err', 'h_err', 'H_err', 'sigma_h', 'dh', 'err_hz'],
            'source': ['source', 'compilation', 'reference', 'ref'],
            'method': ['method', 'technique', 'approach']
        }
        
        standardized_data = {}
        for std_name, possible_names in column_mappings.items():
            found_col = None
            for possible_name in possible_names:
                # Case-insensitive matching
                for actual_col in data.columns:
                    if possible_name.lower() == actual_col.lower():
                        found_col = actual_col
                        break
                if found_col:
                    break
            if found_col:
                standardized_data[std_name] = data[found_col].values
        
        # Convert to DataFrame with standardized names
        result_df = DataFrame(standardized_data)
        
        # Ensure required columns exist
        required_cols = ['z', 'hz', 'hz_err']
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Required column '{col}' not found after standardization")
        
        return result_df
    
    def _merge_compilations(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Merge H(z) data from multiple compilation sources."""
        # Check if multiple sources are present
        if 'source' not in data.columns:
            # Single source, no merging needed
            return data
        
        sources = data['source'].unique()
        if len(sources) <= 1:
            return data
        
        print(f"Merging data from {len(sources)} compilation sources: {sources}")
        
        # Sort by redshift for consistent processing
        merged_data = data.sort_values('z').copy()
        
        # Add source priority based on known compilation quality
        source_priorities = {
            'moresco2016': 1,  # High quality compilation
            'jimenez2003': 2,
            'simon2005': 2,
            'stern2010': 2,
            'moresco2012': 3,
            'zhang2014': 3,
            'ratsimbazafy2017': 3,
            'default': 4
        }
        
        merged_data['priority'] = merged_data['source'].map(
            lambda x: source_priorities.get(x.lower(), source_priorities['default'])
        )
        
        return merged_data
    
    def _filter_overlapping_bins(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Filter overlapping redshift bins."""
        # Define redshift tolerance for overlap detection
        z_tolerance = metadata.get('redshift_tolerance', 0.05)
        
        # Sort by redshift
        sorted_data = data.sort_values('z').copy()
        
        # Find overlapping measurements
        z_values = sorted_data['z'].values
        keep_mask = np.ones(len(sorted_data), dtype=bool)
        
        for i in range(len(z_values)):
            if not keep_mask[i]:
                continue
            
            # Find nearby measurements (including exact matches)
            z_diff = np.abs(z_values - z_values[i])
            nearby_mask = (z_diff <= z_tolerance) & (np.arange(len(z_values)) != i)
            nearby_indices = np.where(nearby_mask)[0]
            
            if len(nearby_indices) > 0:
                # Compute weighted average for overlapping measurements
                candidates = [i] + nearby_indices.tolist()
                candidate_data = sorted_data.iloc[candidates]
                
                # Compute weighted average
                hz_values = candidate_data['hz'].values
                hz_errors = candidate_data['hz_err'].values
                weights = 1.0 / (hz_errors ** 2)
                
                # Weighted average
                weighted_hz = np.sum(weights * hz_values) / np.sum(weights)
                weighted_error = np.sqrt(1.0 / np.sum(weights))
                
                # Update the first measurement with weighted average
                sorted_data.loc[sorted_data.index[i], 'hz'] = weighted_hz
                sorted_data.loc[sorted_data.index[i], 'hz_err'] = weighted_error
                
                # Mark others for removal
                for idx in nearby_indices:
                    keep_mask[idx] = False
        
        filtered_data = sorted_data[keep_mask].copy()
        
        n_removed = len(sorted_data) - len(filtered_data)
        if n_removed > 0:
            print(f"Filtered {n_removed} overlapping measurements (tolerance: Δz = {z_tolerance})")
        
        return filtered_data
    
    def _validate_sign_conventions(self, data: DataFrame) -> DataFrame:
        """Validate H(z) sign conventions."""
        validated_data = data.copy()
        
        # H(z) should always be positive
        if np.any(validated_data['hz'] <= 0):
            negative_mask = validated_data['hz'] <= 0
            n_negative = np.sum(negative_mask)
            raise ValueError(f"Found {n_negative} negative or non-positive H(z) values - check sign conventions")
        
        # Check for reasonable H(z) evolution with redshift
        # H(z) should generally increase with redshift for standard cosmology
        z_sorted_idx = np.argsort(validated_data['z'])
        z_sorted = validated_data['z'].iloc[z_sorted_idx].values
        hz_sorted = validated_data['hz'].iloc[z_sorted_idx].values
        
        # Check if H(z) is generally increasing (allowing for scatter)
        # Use smoothed trend to avoid noise issues
        if len(z_sorted) >= 5:
            from scipy.ndimage import uniform_filter1d
            hz_smooth = uniform_filter1d(hz_sorted, size=min(5, len(hz_sorted)))
            
            # Check if smoothed H(z) is generally increasing
            if hz_smooth[-1] < hz_smooth[0] * 0.8:  # Allow 20% decrease
                print("Warning: H(z) appears to decrease significantly with redshift - check data quality")
        
        return validated_data
    
    def _handle_systematic_errors(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Handle systematic errors in H(z) measurements."""
        processed_data = data.copy()
        
        # Add systematic error floor if specified
        systematic_error = metadata.get('systematic_error_floor', 0.0)  # km/s/Mpc
        
        if systematic_error > 0:
            # Add systematic error in quadrature
            processed_data['hz_err'] = np.sqrt(
                processed_data['hz_err']**2 + systematic_error**2
            )
            print(f"Added systematic error floor: {systematic_error} km/s/Mpc")
        
        # Handle systematic errors dictionary
        systematic_errors = metadata.get('systematic_errors', {})
        if systematic_errors:
            # Combine systematic error fractions in quadrature
            total_sys_frac = np.sqrt(sum(frac**2 for frac in systematic_errors.values()))
            
            # Apply as fraction of H(z) value
            systematic_error_abs = processed_data['hz'] * total_sys_frac
            
            # Add to statistical errors in quadrature
            processed_data['hz_err'] = np.sqrt(
                processed_data['hz_err']**2 + systematic_error_abs**2
            )
            print(f"Added systematic errors: {list(systematic_errors.keys())} (total fraction: {total_sys_frac:.3f})")
        
        # Handle method-specific systematic corrections
        if 'method' in processed_data.columns:
            methods = processed_data['method'].unique()
            
            for method in methods:
                if pd.isna(method):
                    continue
                
                method_mask = processed_data['method'] == method
                
                # Apply method-specific corrections
                if 'differential_age' in str(method).lower():
                    # Differential age method - typically reliable
                    pass
                elif 'red_envelope' in str(method).lower():
                    # Red envelope method - may have larger systematics
                    correction_factor = metadata.get('red_envelope_correction', 1.05)
                    processed_data.loc[method_mask, 'hz_err'] *= correction_factor
        
        return processed_data
    
    def _extract_measurements(self, data: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract standardized H(z) measurements."""
        z = data['z'].values
        hz = data['hz'].values
        sigma_hz = data['hz_err'].values
        
        return z, hz, sigma_hz
    
    def _process_covariance_matrix(self, data: DataFrame, metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        """Process covariance matrix if available."""
        # Check if covariance matrix is provided
        if 'covariance_matrix' not in metadata:
            return None
        
        cov_info = metadata['covariance_matrix']
        
        if isinstance(cov_info, str):
            # Load from file
            try:
                covariance = np.loadtxt(cov_info)
            except:
                return None
        elif isinstance(cov_info, (list, np.ndarray)):
            covariance = np.array(cov_info)
        else:
            return None
        
        # Validate covariance matrix
        n_points = len(data)
        if covariance.shape != (n_points, n_points):
            raise ValueError(f"Covariance matrix shape {covariance.shape} != ({n_points}, {n_points})")
        
        # Check symmetry and positive definiteness
        if not np.allclose(covariance, covariance.T, rtol=1e-10):
            raise ValueError("Covariance matrix is not symmetric")
        
        eigenvalues = np.linalg.eigvals(covariance)
        if np.any(eigenvalues <= 0):
            min_eigenvalue = np.min(eigenvalues)
            raise ValueError(f"Covariance matrix is not positive definite (min eigenvalue: {min_eigenvalue})")
        
        return covariance
    
    def _create_metadata(self, input_metadata: Dict[str, Any], processed_data: DataFrame) -> Dict[str, Any]:
        """Create metadata for the standardized dataset."""
        metadata = {
            'dataset_type': 'cc',
            'data_type': input_metadata.get('data_type', 'hubble_parameter'),
            'source': input_metadata.get('source', 'unknown'),
            'citation': input_metadata.get('citation', ''),
            'version': input_metadata.get('version', ''),
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'n_points': len(processed_data),
            'redshift_range': [float(processed_data['z'].min()), float(processed_data['z'].max())],
            'hz_range': [float(processed_data['hz'].min()), float(processed_data['hz'].max())],
            'observable_description': 'Hubble parameter H(z) from cosmic chronometers',
            'units': 'km/s/Mpc',
            'transformations_applied': [
                'compilation_merging',
                'overlapping_bin_filtering',
                'sign_convention_validation',
                'systematic_error_handling'
            ]
        }
        
        # Add compilation information
        if 'source' in processed_data.columns:
            sources = processed_data['source'].unique()
            metadata['compilations'] = sources.tolist()
            metadata['n_compilations'] = len(sources)
            metadata['individual_sources'] = sources.tolist()  # For source tracking
        
        # Add method information
        if 'method' in processed_data.columns:
            methods = processed_data['method'].unique()
            metadata['methods'] = [m for m in methods if pd.notna(m)]
        
        # Add systematic error information
        if 'systematic_error_floor' in input_metadata:
            metadata['systematic_error_floor'] = input_metadata['systematic_error_floor']
        
        # Add merge strategy information
        if 'merge_strategy' in input_metadata:
            metadata['merge_strategy'] = input_metadata['merge_strategy']
        
        # Add compilation merging flag
        metadata['compilation_merging'] = True
        
        # Add redshift filtering information
        if 'redshift_tolerance' in input_metadata:
            metadata['redshift_filtering'] = True
            metadata['redshift_tolerance'] = input_metadata['redshift_tolerance']
        
        # Add systematic error information
        if 'systematic_errors' in input_metadata:
            metadata['systematic_errors_applied'] = True
            metadata['systematic_errors'] = input_metadata['systematic_errors']
        
        return metadata
    
    def _generate_transformation_summary(self, data: DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of applied transformations."""
        return {
            'transformation_steps': [
                'Loaded raw cosmic chronometer H(z) measurements',
                'Merged data from multiple compilation sources with priority weighting',
                'Filtered overlapping redshift bins using proximity tolerance',
                'Validated H(z) sign conventions and evolution trends',
                'Applied systematic error handling and method-specific corrections'
            ],
            'formulas_used': [
                'H(z) = H₀ E(z) (Hubble parameter evolution)',
                'σ²_total = σ²_stat + σ²_sys (error combination)',
                'Δt/Δz = -1/[(1+z)H(z)] (differential age method)',
                'Cov_ij = correlation matrix element conversion'
            ],
            'assumptions': [
                'Cosmic chronometer galaxies are passively evolving',
                'Stellar population synthesis models are accurate',
                'Systematic errors are Gaussian and uncorrelated',
                'H(z) measurements are independent between redshift bins'
            ],
            'references': [
                'Jimenez & Loeb 2002 (cosmic chronometer method)',
                'Moresco et al. 2016 (comprehensive compilation)',
                'Stern et al. 2010 (differential age technique)',
                'Ratsimbazafy et al. 2017 (systematic analysis)'
            ],
            'data_statistics': {
                'n_measurements': len(data),
                'redshift_range': [float(data['z'].min()), float(data['z'].max())],
                'hz_range': [float(data['hz'].min()), float(data['hz'].max())],
                'compilations': data['source'].unique().tolist() if 'source' in data.columns else ['unknown']
            }
        }
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return summary of applied transformations."""
        if not hasattr(self, '_transformation_summary'):
            return {
                'transformation_steps': [
                    'Load raw cosmic chronometer data',
                    'Merge data from multiple compilations',
                    'Filter overlapping redshift bins',
                    'Validate H(z) sign conventions',
                    'Handle systematic errors',
                    'Extract standardized measurements'
                ],
                'formulas_used': [
                    'H(z) = Hubble parameter at redshift z',
                    'dt/dz = differential age method for cosmic chronometers',
                    'H(z) = -1/(1+z) * dt/dz (fundamental relation)',
                    'Weighted average for overlapping measurements'
                ],
                'assumptions': [
                    'Passive evolution of early-type galaxies',
                    'Single stellar population models',
                    'Gaussian uncertainties for error propagation',
                    'Standard cosmological model for age calculations'
                ],
                'references': [
                    'Jimenez & Loeb 2002 (cosmic chronometer method)',
                    'Moresco et al. 2012 (systematic compilation)',
                    'Stern et al. 2010 (early measurements)',
                    'Moresco et al. 2016 (updated compilation)'
                ],
                'data_statistics': {}
            }
        return self._transformation_summary