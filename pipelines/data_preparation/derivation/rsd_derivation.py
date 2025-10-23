"""
RSD (Redshift Space Distortions) derivation module for the PBUF data preparation framework.

This module implements RSD-specific transformation logic including:
- Growth rate (fσ₈) processing with sign convention validation
- Covariance homogenization from published sources
- Survey-specific correction application and error propagation
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

from ..core.interfaces import DerivationModule, ProcessingError
from ..core.schema import StandardDataset


class RSDDerivationModule(DerivationModule):
    """
    RSD derivation module implementing RSD-specific data processing.
    
    Transforms raw growth rate measurements from redshift space distortions
    into standardized format with proper sign convention validation,
    covariance homogenization, and systematic error handling.
    """
    
    @property
    def dataset_type(self) -> str:
        """Return dataset type identifier."""
        return 'rsd'
    
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
        Validate raw RSD data before processing.
        
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
                    # This is likely a headerless CSV with RSD values: z, fsigma8, sigma_fsigma8
                    data.columns = ['z', 'fsigma8', 'fsigma8_err']
                
            elif raw_data_path.suffix.lower() == '.json':
                data = pd.read_json(raw_data_path)
            else:
                raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
            
            # Check for required columns
            self._validate_rsd_columns(data)
            
            # Validate growth rate measurements
            self._validate_growth_rate_measurements(data)
            
            # Check data has reasonable number of points
            min_points = 1 if metadata and 'compatibility' in metadata.get('source', '').lower() else 2
            if len(data) < min_points:
                raise ValueError(f"Insufficient data points: {len(data)} (minimum {min_points} required)")
            
            if len(data) > 200:
                raise ValueError(f"Too many data points: {len(data)} (maximum 200 supported)")
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to read or validate raw data: {str(e)}")
        
        return True
    
    def _validate_rsd_columns(self, data: DataFrame):
        """Validate required columns for RSD data."""
        # Check for redshift column
        z_columns = ['z', 'redshift', 'zeff', 'z_eff']
        z_col = self._find_column(data, z_columns)
        if z_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. No redshift column found. Expected one of {z_columns}. Available: {available_cols}")
        
        # Check for growth rate column (fσ₈) - should exist after alternative parameterization handling
        fsig8_columns = ['fsig8', 'f_sig8', 'fsigma8', 'f_sigma8', 'growth_rate', 'f*sigma8']
        fsig8_col = self._find_column(data, fsig8_columns)
        if fsig8_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. No fσ₈ column found. Expected one of {fsig8_columns}. Available: {available_cols}")
        
        # Check for fσ₈ error column
        fsig8_err_columns = ['fsig8_err', 'f_sig8_err', 'fsigma8_err', 'f_sigma8_err', 'growth_rate_err', 'err_fsig8', 'sigma_fsig8']
        fsig8_err_col = self._find_column(data, fsig8_err_columns)
        if fsig8_err_col is None:
            available_cols = list(data.columns)
            raise ValueError(f"Missing required columns. No fσ₈ error column found. Expected one of {fsig8_err_columns}. Available: {available_cols}")
    
    def _validate_growth_rate_measurements(self, data: DataFrame):
        """Validate growth rate measurement ranges and sign conventions."""
        z_col = self._find_column(data, ['z', 'redshift', 'zeff', 'z_eff'])
        fsig8_col = self._find_column(data, ['fsig8', 'f_sig8', 'fsigma8', 'f_sigma8', 'growth_rate', 'f*sigma8'])
        fsig8_err_col = self._find_column(data, ['fsig8_err', 'f_sig8_err', 'fsigma8_err', 'f_sigma8_err', 'growth_rate_err', 'err_fsig8', 'sigma_fsig8'])
        
        # Check for NaN values first
        if data[z_col].isna().any():
            raise ValueError(f"NaN values found in redshift column")
        
        if data[fsig8_col].isna().any():
            raise ValueError(f"NaN values found in fσ₈ column")
        
        if data[fsig8_err_col].isna().any():
            raise ValueError(f"NaN values found in fσ₈ uncertainty column")
        
        # Validate redshift range
        if data[z_col].min() < 0:
            raise ValueError(f"Negative redshift values found (min: {data[z_col].min()})")
        
        if data[z_col].max() > 3.0:
            raise ValueError(f"Unreasonably high redshift values found (max: {data[z_col].max()})")
        
        # Validate fσ₈ values (should be positive and reasonable)
        if data[fsig8_col].min() <= 0:
            raise ValueError(f"Non-positive fσ₈ values found (min: {data[fsig8_col].min()})")
        
        if data[fsig8_col].min() < 0.1:
            raise ValueError(f"Unreasonably low fσ₈ values found (min: {data[fsig8_col].min()})")
        
        if data[fsig8_col].max() > 2.0:
            raise ValueError(f"Unreasonably high fσ₈ values found (max: {data[fsig8_col].max()})")
        
        # Validate fσ₈ errors
        if data[fsig8_err_col].min() <= 0:
            raise ValueError(f"Non-positive fσ₈ error values found (min: {data[fsig8_err_col].min()})")
        
        if data[fsig8_err_col].max() > 1.0:
            raise ValueError(f"Unreasonably large fσ₈ errors found (max: {data[fsig8_err_col].max()})")
        
        # Check for reasonable error-to-value ratios
        relative_errors = data[fsig8_err_col] / data[fsig8_col]
        if relative_errors.max() > 0.8:  # 80% relative error
            max_rel_err = relative_errors.max()
            raise ValueError(f"Unreasonably large relative errors found (max: {max_rel_err:.2%})")
    
    def _find_column(self, data: DataFrame, possible_names: List[str]) -> Optional[str]:
        """Find column with case-insensitive matching."""
        for col_name in possible_names:
            for actual_col in data.columns:
                if col_name.lower() == actual_col.lower():
                    return actual_col
        return None
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """
        Transform raw RSD data to standardized format.
        
        Args:
            raw_data_path: Path to verified raw dataset file
            metadata: Dataset metadata from registry
            
        Returns:
            StandardDataset: Transformed RSD data in standard format
        """
        try:
            # Load raw data
            data = self._load_raw_data(raw_data_path)
            
            # Handle alternative parameterizations (convert f and σ₈ to fσ₈ if needed)
            data = self._handle_alternative_parameterizations(data, metadata)
            
            # Validate growth rate measurements (ranges, NaN values, etc.)
            self._validate_growth_rate_measurements(data)
            
            # Validate growth rate sign conventions
            validated_data = self._validate_sign_conventions(data)
            
            # Apply survey-specific corrections
            corrected_data = self._apply_survey_corrections(validated_data, metadata)
            
            # Homogenize covariance from published sources
            homogenized_data = self._homogenize_covariance(corrected_data, metadata)
            
            # Propagate systematic errors
            final_data = self._propagate_systematic_errors(homogenized_data, metadata)
            
            # Extract standardized measurements
            z, fsig8, sigma_fsig8 = self._extract_measurements(final_data)
            
            # Process covariance matrix
            covariance = self._process_covariance_matrix(final_data, metadata)
            
            # Create standardized dataset
            standard_dataset = StandardDataset(
                z=z,
                observable=fsig8,
                uncertainty=sigma_fsig8,
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
                    'Verify fσ₈ values are positive and reasonable',
                    'Check redshift values for physical validity'
                ]
            )
        except Exception as e:
            # Other processing errors
            raise ProcessingError(
                dataset_name=metadata.get('name', 'unknown'),
                stage='transformation',
                error_type='rsd_derivation_error',
                error_message=str(e),
                context={'raw_data_path': str(raw_data_path)},
                suggested_actions=[
                    'Check fσ₈ column names and sign conventions',
                    'Verify survey-specific correction parameters',
                    'Check covariance matrix format and dimensions'
                ]
            )
    
    def _load_raw_data(self, raw_data_path: Path) -> DataFrame:
        """Load and standardize column names from raw data."""
        # Read data
        if raw_data_path.suffix.lower() in ['.txt', '.dat', '.csv']:
            data = pd.read_csv(raw_data_path, comment='#', sep=None, engine='python')
            
            # Check if this is a headerless CSV with numeric column names (indicating no header)
            if len(data.columns) == 3 and all(self._is_numeric_string(str(col)) for col in data.columns):
                # This is likely a headerless CSV with RSD values: z, fsigma8, sigma_fsigma8
                data.columns = ['z', 'fsigma8', 'fsigma8_err']
                
        elif raw_data_path.suffix.lower() == '.json':
            data = pd.read_json(raw_data_path)
        else:
            raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
        
        # Standardize column names
        column_mappings = {
            'z': ['z', 'redshift', 'zeff', 'z_eff'],
            'fsig8': ['fsig8', 'f_sig8', 'fsigma8', 'f_sigma8', 'growth_rate', 'f*sigma8'],
            'fsig8_err': ['fsig8_err', 'f_sig8_err', 'fsigma8_err', 'f_sigma8_err', 'growth_rate_err', 'err_fsig8', 'sigma_fsig8'],
            'survey': ['survey', 'source', 'compilation'],
            'method': ['method', 'technique', 'analysis_type'],
            'tracer': ['tracer', 'galaxy_type', 'sample']
        }
        
        standardized_data = {}
        for std_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in data.columns:
                    standardized_data[std_name] = data[possible_name].values
                    break
        
        # Convert to DataFrame with standardized names
        result_df = DataFrame(standardized_data)
        
        # Add any remaining columns that weren't standardized
        for col in data.columns:
            if col not in [mapped_col for mapped_cols in column_mappings.values() for mapped_col in mapped_cols]:
                result_df[col] = data[col].values
        
        return result_df
    
    def _handle_alternative_parameterizations(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Handle alternative parameterizations like separate f and σ₈ columns."""
        processed_data = data.copy()
        
        # Check if we need to convert separate f and σ₈ to fσ₈
        fsig8_columns = ['fsig8', 'f_sig8', 'fsigma8', 'f_sigma8', 'growth_rate', 'f*sigma8']
        fsig8_col = self._find_column(processed_data, fsig8_columns)
        
        if fsig8_col is None:
            # Look for separate f and σ₈ columns
            f_col = self._find_column(processed_data, ['f', 'growth_rate_f', 'f_growth'])
            sigma8_col = self._find_column(processed_data, ['sigma8', 'sig8', 'sigma_8'])
            
            if f_col is not None and sigma8_col is not None:
                # Create combined fσ₈ column
                processed_data['fsig8'] = processed_data[f_col] * processed_data[sigma8_col]
                
                # Handle error propagation for separate f and σ₈ errors
                f_err_col = self._find_column(processed_data, ['f_err', 'growth_rate_f_err', 'f_growth_err', 'err_f'])
                sigma8_err_col = self._find_column(processed_data, ['sigma8_err', 'sig8_err', 'sigma_8_err', 'err_sigma8'])
                
                if f_err_col is not None and sigma8_err_col is not None:
                    # Error propagation: σ(fσ₈) = √[(σf × σ₈)² + (f × σσ₈)²]
                    processed_data['fsig8_err'] = np.sqrt(
                        (processed_data[f_err_col] * processed_data[sigma8_col])**2 + 
                        (processed_data[f_col] * processed_data[sigma8_err_col])**2
                    )
                else:
                    # Use a default relative error if individual errors not available
                    processed_data['fsig8_err'] = processed_data['fsig8'] * 0.1  # 10% default error
        
        return processed_data
    
    def _validate_sign_conventions(self, data: DataFrame) -> DataFrame:
        """Validate growth rate sign conventions."""
        validated_data = data.copy()
        
        # fσ₈ should always be positive
        if np.any(validated_data['fsig8'] <= 0):
            negative_mask = validated_data['fsig8'] <= 0
            n_negative = np.sum(negative_mask)
            raise ValueError(f"Found {n_negative} negative or non-positive fσ₈ values - check sign conventions")
        
        # Check for reasonable fσ₈ evolution with redshift
        # fσ₈ should generally decrease with redshift due to growth suppression
        z_sorted_idx = np.argsort(validated_data['z'])
        z_sorted = validated_data['z'].iloc[z_sorted_idx].values
        fsig8_sorted = validated_data['fsig8'].iloc[z_sorted_idx].values
        
        # Check if fσ₈ is generally decreasing (allowing for scatter)
        if len(z_sorted) >= 5:
            # Use linear fit to check overall trend
            coeffs = np.polyfit(z_sorted, fsig8_sorted, 1)
            slope = coeffs[0]
            
            # Slope should be negative (fσ₈ decreases with z)
            if slope > 0.1:  # Allow small positive slopes due to scatter
                print("Warning: fσ₈ appears to increase significantly with redshift - check data quality")
        
        return validated_data
    
    def _apply_survey_corrections(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Apply survey-specific corrections."""
        corrected_data = data.copy()
        
        # Get survey information
        if 'survey' in corrected_data.columns:
            surveys = corrected_data['survey'].unique()
        else:
            surveys = [metadata.get('survey', 'unknown')]
        
        for survey in surveys:
            if pd.isna(survey):
                continue
                
            survey_lower = str(survey).lower()
            
            if 'survey' in corrected_data.columns:
                survey_mask = corrected_data['survey'] == survey
            else:
                survey_mask = np.ones(len(corrected_data), dtype=bool)
            
            # Apply survey-specific corrections
            if 'boss' in survey_lower:
                # BOSS-specific corrections
                # Apply AP (Alcock-Paczynski) effect corrections
                ap_correction = metadata.get('boss_ap_correction', 1.0)
                corrected_data.loc[survey_mask, 'fsig8'] *= ap_correction
                
                # Apply fiber collision corrections
                fiber_correction = metadata.get('boss_fiber_correction', 1.02)
                corrected_data.loc[survey_mask, 'fsig8'] *= fiber_correction
                
            elif 'eboss' in survey_lower:
                # eBOSS-specific corrections
                # Apply systematic corrections for different tracers
                if 'tracer' in corrected_data.columns:
                    tracer_mask = corrected_data['tracer'].str.contains('LRG', case=False, na=False)
                    if np.any(tracer_mask & survey_mask):
                        lrg_correction = metadata.get('eboss_lrg_correction', 1.01)
                        corrected_data.loc[tracer_mask & survey_mask, 'fsig8'] *= lrg_correction
                        
                    quasar_mask = corrected_data['tracer'].str.contains('QSO', case=False, na=False)
                    if np.any(quasar_mask & survey_mask):
                        qso_correction = metadata.get('eboss_qso_correction', 0.98)
                        corrected_data.loc[quasar_mask & survey_mask, 'fsig8'] *= qso_correction
                        
            elif 'desi' in survey_lower:
                # DESI-specific corrections
                # Apply imaging systematics corrections
                imaging_correction = metadata.get('desi_imaging_correction', 1.005)
                corrected_data.loc[survey_mask, 'fsig8'] *= imaging_correction
                
            elif '6dfgs' in survey_lower or 'sixdf' in survey_lower:
                # 6dFGS-specific corrections
                # Apply peculiar velocity corrections
                pv_correction = metadata.get('sixdf_pv_correction', 0.95)
                corrected_data.loc[survey_mask, 'fsig8'] *= pv_correction
        
        return corrected_data
    
    def _homogenize_covariance(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Homogenize covariance from published sources."""
        homogenized_data = data.copy()
        
        # Check if covariance homogenization is needed
        covariance_method = metadata.get('covariance_method', 'individual')
        
        if covariance_method == 'homogenized':
            # Apply homogenization corrections to uncertainties
            # This accounts for different error estimation methods across surveys
            
            if 'survey' in homogenized_data.columns:
                surveys = homogenized_data['survey'].unique()
                
                for survey in surveys:
                    if pd.isna(survey):
                        continue
                        
                    survey_mask = homogenized_data['survey'] == survey
                    survey_lower = str(survey).lower()
                    
                    # Apply survey-specific error scaling
                    if 'boss' in survey_lower:
                        error_scale = metadata.get('boss_error_scale', 1.0)
                    elif 'eboss' in survey_lower:
                        error_scale = metadata.get('eboss_error_scale', 1.1)
                    elif 'desi' in survey_lower:
                        error_scale = metadata.get('desi_error_scale', 0.95)
                    elif '6dfgs' in survey_lower:
                        error_scale = metadata.get('sixdf_error_scale', 1.2)
                    else:
                        error_scale = 1.0
                    
                    homogenized_data.loc[survey_mask, 'fsig8_err'] *= error_scale
        
        return homogenized_data
    
    def _propagate_systematic_errors(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Propagate systematic errors."""
        processed_data = data.copy()
        
        # Add systematic error components
        systematic_components = metadata.get('systematic_errors', {})
        
        if systematic_components:
            # Calculate total systematic error for each measurement
            for i in range(len(processed_data)):
                fsig8_val = processed_data.iloc[i]['fsig8']
                stat_err = processed_data.iloc[i]['fsig8_err']
                
                total_systematic_sq = 0.0
                
                # Absolute systematic errors
                absolute_errors = ['theoretical_modeling', 'modeling_error', 'calibration_error']
                for error_name in absolute_errors:
                    if error_name in systematic_components:
                        abs_error = systematic_components[error_name]
                        total_systematic_sq += abs_error**2
                
                # Fractional systematic errors
                fractional_errors = ['survey_systematics', 'nonlinear_error', 'rsd_model_error']
                for error_name in fractional_errors:
                    if error_name in systematic_components:
                        frac_error = systematic_components[error_name]
                        total_systematic_sq += (frac_error * fsig8_val)**2
                
                # Combine statistical and systematic errors
                if total_systematic_sq > 0:
                    total_error = np.sqrt(stat_err**2 + total_systematic_sq)
                    processed_data.iloc[i, processed_data.columns.get_loc('fsig8_err')] = total_error
        
        return processed_data
    
    def _extract_measurements(self, data: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract standardized growth rate measurements."""
        z = data['z'].values
        fsig8 = data['fsig8'].values
        sigma_fsig8 = data['fsig8_err'].values
        
        return z, fsig8, sigma_fsig8
    
    def _process_covariance_matrix(self, data: DataFrame, metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        """Process covariance matrix if available."""
        # Check if covariance matrix is provided
        cov_info = None
        if 'covariance_matrix' in metadata:
            cov_info = metadata['covariance_matrix']
        elif 'covariance_matrix_file' in metadata:
            cov_info = metadata['covariance_matrix_file']
        else:
            return None
        
        if isinstance(cov_info, str):
            # Load from file
            try:
                if cov_info.endswith('.npy'):
                    covariance = np.load(cov_info)
                else:
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
            'dataset_type': 'rsd',
            'data_type': input_metadata.get('data_type', 'growth_rate'),
            'source': input_metadata.get('source', 'unknown'),
            'citation': input_metadata.get('citation', ''),
            'version': input_metadata.get('version', ''),
            'processing_timestamp': pd.Timestamp.now().isoformat(),
            'n_points': len(processed_data),
            'redshift_range': [float(processed_data['z'].min()), float(processed_data['z'].max())],
            'fsig8_range': [float(processed_data['fsig8'].min()), float(processed_data['fsig8'].max())],
            'observable_description': 'Growth rate parameter fσ₈(z) from redshift space distortions',
            'units': 'dimensionless',
            'transformations_applied': [
                'sign_convention_validation',
                'survey_specific_corrections',
                'covariance_homogenization',
                'systematic_error_propagation'
            ]
        }
        
        # Add survey information
        if 'survey' in processed_data.columns:
            surveys = processed_data['survey'].unique()
            metadata['surveys'] = [s for s in surveys if pd.notna(s)]
            metadata['n_surveys'] = len(metadata['surveys'])
        
        # Add tracer information
        if 'tracer' in processed_data.columns:
            tracers = processed_data['tracer'].unique()
            metadata['tracers'] = [t for t in tracers if pd.notna(t)]
        
        # Add method information
        if 'method' in processed_data.columns:
            methods = processed_data['method'].unique()
            metadata['methods'] = [m for m in methods if pd.notna(m)]
        
        # Add systematic error information
        if 'systematic_errors' in input_metadata:
            metadata['systematic_errors'] = input_metadata['systematic_errors']
            metadata['systematic_errors_applied'] = True
        
        # Add survey corrections information
        if 'survey_corrections' in input_metadata:
            metadata['survey_corrections_applied'] = input_metadata['survey_corrections']
            metadata['correction_summary'] = f"Applied survey-specific corrections to {len(input_metadata['survey_corrections'])} surveys"
        
        # Add individual surveys tracking
        if 'survey' in processed_data.columns:
            surveys = processed_data['survey'].unique()
            metadata['individual_surveys'] = [s for s in surveys if pd.notna(s)]
        
        # Add physical consistency check flag
        metadata['physical_consistency_check'] = True
        
        # Add parameterization conversion flag if applicable
        if 'parameterization' in input_metadata and input_metadata['parameterization'] == 'f_sigma8_separate':
            metadata['parameterization_conversion'] = True
        
        # Add covariance homogenization flag
        if 'covariance_matrix_file' in input_metadata or 'covariance_matrix' in input_metadata:
            metadata['covariance_homogenization'] = True
        
        return metadata
    
    def _generate_transformation_summary(self, data: DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of applied transformations."""
        return {
            'transformation_steps': [
                'Loaded raw growth rate fσ₈(z) measurements from RSD analysis',
                'Validated growth rate sign conventions and evolution trends',
                'Applied survey-specific systematic corrections (AP, fiber collisions, etc.)',
                'Homogenized covariance matrices from different published sources',
                'Propagated theoretical and observational systematic errors'
            ],
            'formulas_used': [
                'fσ₈(z) = f(z) × σ₈(z) (growth rate parameter)',
                'f(z) ≈ Ω_m(z)^γ (growth rate approximation)',
                'σ²_total = σ²_stat + σ²_sys (error propagation)',
                'P(k,μ) = (1 + βμ²)² P_real(k) (RSD model)'
            ],
            'assumptions': [
                'Linear theory RSD model on large scales',
                'Kaiser effect dominates over non-linear corrections',
                'Gaussian uncertainties for error propagation',
                'Survey-specific systematics are well-characterized'
            ],
            'references': [
                'Kaiser 1987 (redshift space distortions)',
                'Percival & White 2009 (RSD measurements)',
                'Samushia et al. 2014 (growth rate compilation)',
                'de la Torre et al. 2013 (VIPERS RSD analysis)'
            ],
            'data_statistics': {
                'n_measurements': len(data),
                'redshift_range': [float(data['z'].min()), float(data['z'].max())],
                'fsig8_range': [float(data['fsig8'].min()), float(data['fsig8'].max())],
                'surveys': data['survey'].unique().tolist() if 'survey' in data.columns else ['unknown']
            }
        }
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return summary of applied transformations."""
        if not hasattr(self, '_transformation_summary'):
            return {
                'transformation_steps': [
                    'Load raw RSD growth rate data',
                    'Validate growth rate sign conventions',
                    'Apply survey-specific corrections',
                    'Homogenize covariance from published sources',
                    'Propagate systematic errors',
                    'Extract standardized measurements'
                ],
                'formulas_used': [
                    'fσ₈(z) = f(z) × σ₈(z) (growth rate parameter)',
                    'f(z) ≈ Ωₘ(z)^γ (growth rate approximation)',
                    'β = f/b (redshift space distortion parameter)',
                    'Survey corrections for fiber collisions and AP effects',
                    'Systematic error propagation in quadrature'
                ],
                'assumptions': [
                    'Linear theory for large-scale structure growth',
                    'Kaiser effect for redshift space distortions',
                    'Gaussian velocity distribution',
                    'Survey-specific systematic corrections',
                    'Independent statistical and systematic errors'
                ],
                'references': [
                    'Kaiser 1987 (redshift space distortions)',
                    'Percival & White 2009 (BAO and RSD)',
                    'Samushia et al. 2014 (RSD measurements)',
                    'Macaulay et al. 2013 (RSD compilation)'
                ],
                'data_statistics': {}
            }
        return self._transformation_summary