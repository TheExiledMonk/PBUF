"""
Supernova (SN) derivation module for the PBUF data preparation framework.

This module implements SN-specific transformation logic including:
- Magnitude to distance modulus conversion
- Duplicate removal by coordinate matching
- Calibration homogenization
- Systematic covariance matrix application
- z-μ-σμ extraction
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

from astropy.coordinates import SkyCoord
from astropy import units as u

from ..core.interfaces import DerivationModule, ProcessingError
from ..core.schema import StandardDataset


class SNDerivationModule(DerivationModule):
    """
    Supernova derivation module implementing SN-specific data processing.
    
    Transforms raw supernova data into standardized format with proper
    distance modulus calculations, duplicate removal, and systematic
    error handling.
    """
    
    @property
    def dataset_type(self) -> str:
        """Return dataset type identifier."""
        return 'sn'
    
    @property
    def supported_formats(self) -> List[str]:
        """Return supported input file formats."""
        return ['.txt', '.csv', '.dat', '.fits']
    
    def _is_numeric_string(self, s: str) -> bool:
        """Check if a string represents a numeric value."""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def validate_input(self, raw_data_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Validate raw supernova data before processing.
        
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
        
        if file_size > 100 * 1024 * 1024:  # 100 MB limit
            raise ValueError(f"Raw data file too large: {file_size / (1024*1024):.1f} MB")
        
        # Try to read and validate basic structure
        try:
            if raw_data_path.suffix.lower() in ['.txt', '.dat', '.csv']:
                # Try reading as text file
                data = pd.read_csv(raw_data_path, comment='#', sep=None, engine='python')
                
                # Check if this is a headerless CSV with numeric column names (indicating no header)
                if len(data.columns) == 3 and all(self._is_numeric_string(str(col)) for col in data.columns):
                    # This is likely a headerless CSV with SN values: z, mb, dmb
                    data.columns = ['z', 'mb', 'dmb']
                
            else:
                raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
            
            # Check for minimum required columns
            required_columns = ['z', 'mb', 'dmb']  # redshift, apparent magnitude, magnitude error
            missing_columns = []
            
            # Check for various naming conventions
            column_mappings = {
                'z': ['z', 'redshift', 'zcmb', 'zhel'],
                'mb': ['mb', 'mag', 'magnitude', 'apparent_mag', 'm_b'],
                'dmb': ['dmb', 'mag_err', 'magnitude_err', 'dm_b', 'emag']
            }
            
            found_columns = {}
            for req_col, possible_names in column_mappings.items():
                found = False
                for possible_name in possible_names:
                    if possible_name in data.columns:
                        found_columns[req_col] = possible_name
                        found = True
                        break
                
                if not found:
                    missing_columns.append(req_col)
            
            if missing_columns:
                available_cols = list(data.columns)
                raise ValueError(f"Missing required columns {missing_columns}. Available columns: {available_cols}")
            
            # Check data has reasonable number of points
            min_points = 1 if metadata and 'compatibility' in metadata.get('source', '').lower() else 2
            if len(data) < min_points:
                raise ValueError(f"Insufficient data points: {len(data)} (minimum {min_points} required)")
            
            if len(data) > 10000:
                raise ValueError(f"Too many data points: {len(data)} (maximum 10000 supported)")
            
            # Basic range checks
            z_col = found_columns['z']
            mb_col = found_columns['mb']
            dmb_col = found_columns['dmb']
            
            if data[z_col].min() < 0:
                raise ValueError(f"Negative redshift values found (min: {data[z_col].min()})")
            
            if data[z_col].max() > 5.0:
                raise ValueError(f"Unreasonably high redshift values found (max: {data[z_col].max()})")
            
            if data[dmb_col].min() <= 0:
                raise ValueError(f"Non-positive magnitude errors found (min: {data[dmb_col].min()})")
            
            if data[dmb_col].max() > 5.0:
                raise ValueError(f"Unreasonably large magnitude errors found (max: {data[dmb_col].max()})")
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"Failed to read or validate raw data: {str(e)}")
        
        return True
    
    def derive(self, raw_data_path: Path, metadata: Dict[str, Any]) -> StandardDataset:
        """
        Transform raw supernova data to standardized format.
        
        Args:
            raw_data_path: Path to verified raw dataset file
            metadata: Dataset metadata from registry
            
        Returns:
            StandardDataset: Transformed SN data in standard format
        """
        try:
            # Load raw data
            data = self._load_raw_data(raw_data_path)
            
            # Validate data quality
            self._validate_data_quality(data)
            
            # Remove duplicates by coordinate matching
            data = self._remove_duplicates(data)
            
            # Homogenize calibration systems
            data = self._homogenize_calibration(data, metadata)
            
            # Convert magnitudes to distance modulus
            z, mu, sigma_mu = self._convert_to_distance_modulus(data)
            
            # Apply systematic covariance if available
            covariance = self._apply_systematic_covariance(data, metadata)
            
            # Create standardized dataset
            standard_dataset = StandardDataset(
                z=z,
                observable=mu,
                uncertainty=sigma_mu,
                covariance=covariance,
                metadata=self._create_metadata(metadata, data)
            )
            
            # Store transformation summary for provenance
            self._transformation_summary = self._generate_transformation_summary(data, metadata)
            
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
                    'Check magnitude values for NaN or infinite values'
                ]
            )
        except Exception as e:
            # Other processing errors
            raise ProcessingError(
                dataset_name=metadata.get('name', 'unknown'),
                stage='transformation',
                error_type='sn_derivation_error',
                error_message=str(e),
                context={'raw_data_path': str(raw_data_path)},
                suggested_actions=[
                    'Check raw data format and column names',
                    'Verify coordinate columns are present for duplicate removal',
                    'Check calibration system metadata'
                ]
            )
    
    def _validate_data_quality(self, data: DataFrame):
        """Validate data quality and check for invalid values."""
        # Check for required columns
        required_cols = ['z', 'mb', 'dmb']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Validate redshift values
        if data['z'].isna().any():
            raise ValueError("NaN redshift values found")
        
        if (data['z'] < 0).any():
            raise ValueError("Negative redshift values found")
        
        if (data['z'] > 5.0).any():
            raise ValueError("Unreasonably high redshift values found (max > 5.0)")
        
        # Validate magnitude values
        if data['mb'].isna().any():
            raise ValueError("NaN magnitude values found")
        
        if not np.isfinite(data['mb']).all():
            raise ValueError("Non-finite magnitude values found")
        
        # Validate magnitude errors
        if data['dmb'].isna().any():
            raise ValueError("NaN magnitude error values found")
        
        if (data['dmb'] <= 0).any():
            raise ValueError("Non-positive magnitude error values found")
    
    def _load_raw_data(self, raw_data_path: Path) -> DataFrame:
        """Load and standardize column names from raw data."""
        # Read data
        if raw_data_path.suffix.lower() in ['.txt', '.dat', '.csv']:
            data = pd.read_csv(raw_data_path, comment='#', sep=None, engine='python')
            
            # Check if this is a headerless CSV with numeric column names (indicating no header)
            if len(data.columns) == 3 and all(self._is_numeric_string(str(col)) for col in data.columns):
                # This is likely a headerless CSV with SN values: z, mb, dmb
                data.columns = ['z', 'mb', 'dmb']
                
        else:
            raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")
        
        # Standardize column names
        column_mappings = {
            'z': ['z', 'redshift', 'zcmb', 'zhel'],
            'mb': ['mb', 'mag', 'magnitude', 'apparent_mag', 'm_b'],
            'dmb': ['dmb', 'mag_err', 'magnitude_err', 'dm_b', 'emag'],
            'ra': ['ra', 'RA', 'right_ascension', 'alpha'],
            'dec': ['dec', 'DEC', 'declination', 'delta'],
            'x1': ['x1', 'stretch', 's'],
            'dx1': ['dx1', 'stretch_err', 'ds'],
            'c': ['c', 'color'],
            'dc': ['dc', 'color_err']
        }
        
        standardized_data = {}
        for std_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in data.columns:
                    standardized_data[std_name] = data[possible_name].values
                    break
        
        # Convert to DataFrame with standardized names
        result_df = pd.DataFrame(standardized_data)
        
        # Ensure required columns exist
        required_cols = ['z', 'mb', 'dmb']
        for col in required_cols:
            if col not in result_df.columns:
                raise ValueError(f"Required column '{col}' not found after standardization")
        
        return result_df
    
    def _remove_duplicates(self, data: DataFrame) -> DataFrame:
        """Remove duplicate entries by coordinate matching."""
        if 'ra' not in data.columns or 'dec' not in data.columns:
            # No coordinates available, use redshift-based duplicate removal
            return self._remove_duplicates_by_redshift(data)
        
        # Use coordinate-based duplicate removal
        coords = SkyCoord(ra=data['ra'].values * u.degree, 
                         dec=data['dec'].values * u.degree)
        
        # Find duplicates within 1 arcsecond
        duplicate_mask = np.zeros(len(data), dtype=bool)
        for i in range(len(coords)):
            if duplicate_mask[i]:
                continue
            
            # Find matches within 1 arcsecond
            separations = coords[i].separation(coords[i+1:])
            matches = separations < 1.0 * u.arcsec
            
            if np.any(matches):
                # Mark duplicates (keep first occurrence)
                match_indices = np.where(matches)[0] + i + 1
                duplicate_mask[match_indices] = True
        
        # Remove duplicates
        clean_data = data[~duplicate_mask].copy()
        
        n_removed = np.sum(duplicate_mask)
        if n_removed > 0:
            print(f"Removed {n_removed} duplicate entries by coordinate matching")
        
        return clean_data
    
    def _remove_duplicates_by_redshift(self, data: DataFrame) -> DataFrame:
        """Remove duplicates based on redshift proximity when coordinates unavailable."""
        # Sort by redshift
        sorted_data = data.sort_values('z').copy()
        
        # Find duplicates within Δz = 0.001
        z_values = sorted_data['z'].values
        duplicate_mask = np.zeros(len(sorted_data), dtype=bool)
        
        for i in range(len(z_values) - 1):
            if duplicate_mask[i]:
                continue
            
            # Find nearby redshifts
            z_diff = np.abs(z_values[i+1:] - z_values[i])
            matches = z_diff < 0.001
            
            if np.any(matches):
                # Mark duplicates (keep first occurrence)
                match_indices = np.where(matches)[0] + i + 1
                duplicate_mask[match_indices] = True
        
        # Remove duplicates
        clean_data = sorted_data[~duplicate_mask].copy()
        
        n_removed = np.sum(duplicate_mask)
        if n_removed > 0:
            print(f"Removed {n_removed} duplicate entries by redshift matching")
        
        return clean_data
    
    def _homogenize_calibration(self, data: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """Homogenize calibration systems (SALT2, MLCS2k2, etc.)."""
        calibration_system = metadata.get('calibration_system', 'unknown')
        
        # Apply calibration-specific corrections
        corrected_data = data.copy()
        
        if calibration_system.lower() == 'salt2':
            # SALT2 calibration - no correction needed (reference)
            pass
        elif calibration_system.lower() == 'mlcs2k2':
            # Apply MLCS2k2 to SALT2 conversion
            # Typical offset: Δμ ≈ 0.05 mag
            corrected_data['mb'] = corrected_data['mb'] - 0.05
        elif calibration_system.lower() == 'sifto':
            # Apply SIFTO to SALT2 conversion
            # Typical offset: Δμ ≈ 0.02 mag
            corrected_data['mb'] = corrected_data['mb'] - 0.02
        else:
            # Unknown calibration system - issue warning but proceed
            print(f"Warning: Unknown calibration system '{calibration_system}', no correction applied")
        
        return corrected_data
    
    def _convert_to_distance_modulus(self, data: DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert apparent magnitudes to distance modulus."""
        z = data['z'].values
        mb = data['mb'].values
        dmb = data['dmb'].values
        
        # Apply K-corrections and other corrections if available
        if 'x1' in data.columns and 'c' in data.columns:
            # SALT2 light curve parameters available
            x1 = data['x1'].values
            c = data['c'].values
            
            # Standard SALT2 distance modulus formula
            # μ = mb - M + αx1 - βc
            # Using typical values: M = -19.3, α = 0.14, β = 3.1
            M = -19.3  # Absolute magnitude
            alpha = 0.14  # Stretch correction
            beta = 3.1   # Color correction
            
            mu = mb - M + alpha * x1 - beta * c
            
            # Propagate uncertainties
            dx1 = data.get('dx1', np.zeros_like(x1))
            dc = data.get('dc', np.zeros_like(c))
            
            # Error propagation: σ²μ = σ²mb + α²σ²x1 + β²σ²c
            sigma_mu = np.sqrt(dmb**2 + (alpha * dx1)**2 + (beta * dc)**2)
            
        else:
            # Simple magnitude to distance modulus conversion
            # Assume fiducial absolute magnitude
            M_fiducial = -19.3
            mu = mb - M_fiducial
            sigma_mu = dmb
        
        # Ensure all return values are numpy arrays
        return np.asarray(z), np.asarray(mu), np.asarray(sigma_mu)
    
    def _apply_systematic_covariance(self, data: DataFrame, metadata: Dict[str, Any]) -> Optional[np.ndarray]:
        """Apply systematic covariance matrix if available."""
        # Check if systematic covariance is provided in metadata
        if 'systematic_covariance' not in metadata and 'systematic_covariance_file' not in metadata:
            return None
        
        # Handle file-based covariance matrix
        if 'systematic_covariance_file' in metadata:
            cov_file = metadata['systematic_covariance_file']
            if isinstance(cov_file, str):
                if cov_file.endswith('.npy'):
                    sys_cov = np.load(cov_file)
                else:
                    sys_cov = np.loadtxt(cov_file)
                return sys_cov
        
        sys_cov_info = metadata['systematic_covariance']
        
        # For now, implement a simple systematic covariance model
        # In practice, this would load from external files or compute based on survey properties
        n_points = len(data)
        
        if sys_cov_info.get('type') == 'diagonal':
            # Diagonal systematic uncertainties
            sys_error = sys_cov_info.get('magnitude', 0.1)  # 0.1 mag systematic
            sys_cov = np.diag(np.full(n_points, sys_error**2))
            
        elif sys_cov_info.get('type') == 'redshift_correlated':
            # Redshift-correlated systematic uncertainties
            z_values = data['z'].values
            sys_error = sys_cov_info.get('magnitude', 0.1)
            correlation_length = sys_cov_info.get('correlation_length', 0.1)
            
            # Create correlation matrix
            z_diff = np.abs(z_values[:, np.newaxis] - z_values[np.newaxis, :])
            correlation = np.exp(-z_diff / correlation_length)
            sys_cov = (sys_error**2) * correlation
            
        else:
            # No systematic covariance
            return None
        
        # Combine with statistical uncertainties
        stat_cov = np.diag(data['dmb'].values**2)
        total_cov = stat_cov + sys_cov
        
        return total_cov
    
    def _create_metadata(self, input_metadata: Dict[str, Any], processed_data: DataFrame) -> Dict[str, Any]:
        """Create metadata for the standardized dataset."""
        metadata = {
            'dataset_type': 'sn',
            'source': input_metadata.get('source', 'unknown'),
            'citation': input_metadata.get('citation', ''),
            'version': input_metadata.get('version', ''),
            'processing_date': pd.Timestamp.now().isoformat(),
            'n_points': len(processed_data),
            'redshift_range': [float(processed_data['z'].min()), float(processed_data['z'].max())],
            'calibration_system': input_metadata.get('calibration_system', 'unknown'),
            'transformations_applied': [
                'duplicate_removal',
                'calibration_homogenization', 
                'magnitude_to_distance_modulus',
                'systematic_covariance_application'
            ]
        }
        
        # Add survey-specific information if available
        if 'survey' in input_metadata:
            metadata['survey'] = input_metadata['survey']
        
        if 'telescope' in input_metadata:
            metadata['telescope'] = input_metadata['telescope']
        
        # Add calibration homogenization flag
        metadata['calibration_homogenization'] = True
        
        return metadata
    
    def _generate_transformation_summary(self, data: DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of applied transformations."""
        return {
            'transformation_steps': [
                'Loaded raw supernova data with standardized column mapping',
                'Removed duplicate entries using coordinate or redshift matching',
                'Applied calibration system homogenization to SALT2 reference',
                'Converted apparent magnitudes to distance modulus using SALT2 formula',
                'Applied systematic covariance matrix if available'
            ],
            'formulas_used': [
                'μ = mb - M + αx1 - βc (SALT2 distance modulus)',
                'σ²μ = σ²mb + α²σ²x1 + β²σ²c (uncertainty propagation)',
                'Cov_total = Cov_stat + Cov_sys (covariance combination)'
            ],
            'assumptions': [
                'Fiducial absolute magnitude M = -19.3 mag',
                'SALT2 parameters: α = 0.14, β = 3.1',
                'Coordinate matching threshold: 1 arcsecond',
                'Redshift matching threshold: Δz = 0.001'
            ],
            'references': [
                'Betoule et al. 2014 (JLA compilation)',
                'Guy et al. 2007 (SALT2 light curve fitter)',
                'Conley et al. 2011 (supernova systematics)'
            ],
            'data_statistics': {
                'n_input_points': len(data),
                'redshift_range': [float(data['z'].min()), float(data['z'].max())],
                'calibration_system': metadata.get('calibration_system', 'unknown')
            }
        }
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """Return summary of applied transformations."""
        if not hasattr(self, '_transformation_summary'):
            return {
                'transformation_steps': [
                    'Load raw supernova photometry data',
                    'Validate data quality (redshift, magnitudes)',
                    'Remove duplicates by coordinate matching',
                    'Homogenize calibration systems',
                    'Convert magnitudes to distance modulus',
                    'Apply systematic covariance matrix'
                ],
                'formulas_used': [
                    'μ = m - M (distance modulus formula)',
                    'μ = mb - Mx1 + αx1 - βc (SALT2 standardization)',
                    'DL = 10^((μ + 25)/5) Mpc (luminosity distance)',
                    'Systematic covariance: Csys + Cstat (error combination)',
                    'Calibration corrections for different surveys'
                ],
                'assumptions': [
                    'Standard candle assumption after light curve corrections',
                    'SALT2 or similar light curve fitter parameterization',
                    'Gaussian uncertainties for error propagation',
                    'Survey-specific calibration corrections',
                    'Systematic errors uncorrelated with statistical errors'
                ],
                'references': [
                    'Riess et al. 1998 (SN Ia cosmology)',
                    'Perlmutter et al. 1999 (accelerating universe)',
                    'Guy et al. 2007 (SALT2 light curve fitter)',
                    'Betoule et al. 2014 (JLA compilation)',
                    'Scolnic et al. 2018 (Pantheon compilation)'
                ],
                'data_statistics': {}
            }
        return self._transformation_summary