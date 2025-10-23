"""
Format converter for compatibility with existing fit pipelines.

This module provides conversion between StandardDataset format and the existing
DatasetDict format used by pipelines/fit_core/datasets.py, ensuring seamless
integration with current fitting workflows.
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np

from ..core.schema import StandardDataset


class FormatConverter:
    """
    Converts between StandardDataset and existing DatasetDict formats.
    
    Provides seamless integration with pipelines/fit_core/datasets.py interface
    while maintaining backward compatibility with existing fitting workflows.
    
    Requirements: 6.3
    """
    
    @staticmethod
    def standard_to_dataset_dict(
        dataset: StandardDataset,
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Convert StandardDataset to existing DatasetDict format.
        
        Args:
            dataset: StandardDataset instance to convert
            dataset_type: Type of dataset ('cmb', 'bao', 'bao_ani', 'sn')
            
        Returns:
            DatasetDict compatible with existing fit_core/datasets.py interface
            
        Requirements: 6.3
        """
        # Validate dataset type
        supported_types = ['cmb', 'bao', 'bao_ani', 'sn', 'cc', 'rsd']
        if dataset_type not in supported_types:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported types: {supported_types}")
        
        # Validate dataset before conversion with appropriate redshift limits
        if dataset_type == 'cmb':
            # CMB data has redshift ~1090 (recombination)
            dataset.validate_all(min_z=0.0, max_z=2000.0)
        else:
            # Standard cosmological surveys
            dataset.validate_all(min_z=0.0, max_z=10.0)
        
        # Create base DatasetDict structure with expected interface
        dataset_dict = {
            'dataset_type': dataset_type,
            'observations': dataset.observable,  # Main observable array
            'uncertainties': dataset.uncertainty,  # Uncertainty array
            'redshifts': dataset.z,  # Redshift array
            'covariance': dataset.covariance if dataset.covariance is not None else None,
            'metadata': dataset.metadata.copy()
        }
        
        # Add dataset-specific observations for specialized handling
        if dataset_type == 'cmb':
            # CMB has special structure with R, l_A, theta_star
            if len(dataset.observable) == 3:
                dataset_dict['observations'] = {
                    'R': float(dataset.observable[0]),
                    'l_A': float(dataset.observable[1]),
                    'theta_star': float(dataset.observable[2])
                }
        elif dataset_type == 'bao_ani':
            # Anisotropic BAO has special structure with DM and DH measurements
            n_z = len(dataset.z)
            dm_values = dataset.observable[:n_z]  # First half: DM measurements
            dh_values = dataset.observable[n_z:]  # Second half: DH measurements
            dataset_dict['observations'] = {
                'redshift': dataset.z,
                'DM_over_rd': dm_values,
                'DH_over_rd': dh_values
            }
        elif dataset_type == 'bao':
            # Isotropic BAO has structured observations
            dataset_dict['observations'] = {
                'redshift': dataset.z,
                'DV_over_rs': dataset.observable
            }
        elif dataset_type == 'sn':
            # SN has structured observations
            dataset_dict['observations'] = {
                'redshift': dataset.z,
                'distance_modulus': dataset.observable,
                'sigma_mu': dataset.uncertainty
            }
        elif dataset_type == 'cc':
            # Cosmic Chronometers have H(z) measurements
            dataset_dict['observations'] = {
                'redshift': dataset.z,
                'H_z': dataset.observable,
                'sigma_H': dataset.uncertainty
            }
        elif dataset_type == 'rsd':
            # RSD have f*sigma8 measurements
            dataset_dict['observations'] = {
                'redshift': dataset.z,
                'f_sigma8': dataset.observable,
                'sigma_f_sigma8': dataset.uncertainty
            }
        
        # Add metadata required by existing interface
        dataset_dict['metadata'].update({
            'n_data_points': len(dataset.z),
            'source': dataset.metadata.get('source', 'unknown')
        })
        
        return dataset_dict
    
    @staticmethod
    def _convert_cmb_observations(dataset: StandardDataset) -> Dict[str, Any]:
        """Convert CMB StandardDataset to observations dict."""
        if len(dataset.observable) != 3:
            raise ValueError(f"CMB dataset must have exactly 3 observables, got {len(dataset.observable)}")
        
        # CMB observables are R, l_A, theta_star
        return {
            'R': float(dataset.observable[0]),
            'l_A': float(dataset.observable[1]),
            'theta_star': float(dataset.observable[2])
        }
    
    @staticmethod
    def _convert_bao_observations(dataset: StandardDataset) -> Dict[str, Any]:
        """Convert isotropic BAO StandardDataset to observations dict."""
        return {
            'redshift': dataset.z,
            'DV_over_rs': dataset.observable
        }
    
    @staticmethod
    def _convert_bao_anisotropic_observations(dataset: StandardDataset) -> Dict[str, Any]:
        """Convert anisotropic BAO StandardDataset to observations dict."""
        n_points = len(dataset.z)
        
        # Anisotropic BAO has 2*n_points observables: DM/rd and DH/rd for each redshift
        if len(dataset.observable) != 2 * n_points:
            raise ValueError(
                f"Anisotropic BAO dataset must have 2*n_redshifts observables, "
                f"got {len(dataset.observable)} for {n_points} redshifts"
            )
        
        # Split observables into DM/rd and DH/rd
        dm_over_rd = dataset.observable[:n_points]
        dh_over_rd = dataset.observable[n_points:]
        
        return {
            'redshift': dataset.z,
            'DM_over_rd': dm_over_rd,
            'DH_over_rd': dh_over_rd
        }
    
    @staticmethod
    def _convert_sn_observations(dataset: StandardDataset) -> Dict[str, Any]:
        """Convert supernova StandardDataset to observations dict."""
        return {
            'redshift': dataset.z,
            'distance_modulus': dataset.observable,
            'sigma_mu': dataset.uncertainty
        }
    
    @staticmethod
    def dataset_dict_to_standard(
        dataset_dict: Dict[str, Any],
        dataset_type: Optional[str] = None
    ) -> StandardDataset:
        """
        Convert existing DatasetDict to StandardDataset format.
        
        Args:
            dataset_dict: DatasetDict from existing fit_core/datasets.py
            dataset_type: Type of dataset (auto-detected if None)
            
        Returns:
            StandardDataset instance
            
        Requirements: 6.3
        """
        # Auto-detect dataset type if not provided
        if dataset_type is None:
            dataset_type = dataset_dict.get('dataset_type')
            if dataset_type is None:
                raise ValueError("Dataset type must be provided or present in dataset_dict")
        
        observations = dataset_dict['observations']
        covariance = dataset_dict.get('covariance')
        metadata = dataset_dict.get('metadata', {})
        
        # Convert observations based on dataset type
        if dataset_type == 'cmb':
            z, observable, uncertainty = FormatConverter._extract_cmb_data(observations)
        elif dataset_type == 'bao':
            z, observable, uncertainty = FormatConverter._extract_bao_data(observations, covariance)
        elif dataset_type == 'bao_ani':
            z, observable, uncertainty = FormatConverter._extract_bao_anisotropic_data(observations, covariance)
        elif dataset_type == 'sn':
            z, observable, uncertainty = FormatConverter._extract_sn_data(observations)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Ensure metadata includes dataset type
        metadata = metadata.copy()
        metadata['dataset_type'] = dataset_type
        
        return StandardDataset(
            z=z,
            observable=observable,
            uncertainty=uncertainty,
            covariance=covariance,
            metadata=metadata
        )
    
    @staticmethod
    def _extract_cmb_data(observations: Dict[str, Any]) -> tuple:
        """Extract CMB data from observations dict."""
        # CMB has fixed redshift (recombination) repeated for each observable
        z = np.array([1089.80, 1089.80, 1089.80])
        
        # Extract R, l_A, theta_star
        observable = np.array([
            observations['R'],
            observations['l_A'],
            observations['theta_star']
        ])
        
        # CMB uncertainties come from covariance diagonal
        uncertainty = np.array([0.0, 0.0, 0.0])  # Will be overridden by covariance
        
        return z, observable, uncertainty
    
    @staticmethod
    def _extract_bao_data(observations: Dict[str, Any], covariance: np.ndarray) -> tuple:
        """Extract isotropic BAO data from observations dict."""
        z = np.asarray(observations['redshift'])
        observable = np.asarray(observations['DV_over_rs'])
        
        # Extract uncertainties from covariance diagonal
        if covariance is not None:
            uncertainty = np.sqrt(np.diag(covariance))
        else:
            uncertainty = np.zeros_like(observable)
        
        return z, observable, uncertainty
    
    @staticmethod
    def _extract_bao_anisotropic_data(observations: Dict[str, Any], covariance: np.ndarray) -> tuple:
        """Extract anisotropic BAO data from observations dict."""
        z = np.asarray(observations['redshift'])
        dm_over_rd = np.asarray(observations['DM_over_rd'])
        dh_over_rd = np.asarray(observations['DH_over_rd'])
        
        # Combine DM and DH observables
        observable = np.concatenate([dm_over_rd, dh_over_rd])
        
        # Extract uncertainties from covariance diagonal
        if covariance is not None:
            uncertainty = np.sqrt(np.diag(covariance))
        else:
            uncertainty = np.zeros_like(observable)
        
        return z, observable, uncertainty
    
    @staticmethod
    def _extract_sn_data(observations: Dict[str, Any]) -> tuple:
        """Extract supernova data from observations dict."""
        z = np.asarray(observations['redshift'])
        observable = np.asarray(observations['distance_modulus'])
        
        # Use provided uncertainties or extract from sigma_mu
        if 'sigma_mu' in observations:
            uncertainty = np.asarray(observations['sigma_mu'])
        else:
            uncertainty = np.zeros_like(observable)
        
        return z, observable, uncertainty
    
    @staticmethod
    def validate_conversion_compatibility(
        original_dict: Dict[str, Any],
        converted_dataset: StandardDataset,
        dataset_type: str,
        tolerance: float = 1e-10
    ) -> bool:
        """
        Validate that conversion preserves data integrity.
        
        Args:
            original_dict: Original DatasetDict
            converted_dataset: Converted StandardDataset
            dataset_type: Type of dataset
            tolerance: Numerical tolerance for comparisons
            
        Returns:
            True if conversion is valid, raises ValueError otherwise
            
        Requirements: 6.3
        """
        # Convert back to DatasetDict and compare
        reconverted_dict = FormatConverter.standard_to_dataset_dict(converted_dataset, dataset_type)
        
        # Compare observations
        original_obs = original_dict['observations']
        reconverted_obs = reconverted_dict['observations']
        
        for key in original_obs:
            if key not in reconverted_obs:
                raise ValueError(f"Missing observable after conversion: {key}")
            
            original_val = np.asarray(original_obs[key])
            reconverted_val = np.asarray(reconverted_obs[key])
            
            if not np.allclose(original_val, reconverted_val, rtol=tolerance, atol=tolerance):
                max_diff = np.max(np.abs(original_val - reconverted_val))
                raise ValueError(
                    f"Observable '{key}' differs after conversion (max diff: {max_diff}, tolerance: {tolerance})"
                )
        
        # Compare covariance matrices
        original_cov = original_dict.get('covariance')
        reconverted_cov = reconverted_dict.get('covariance')
        
        if original_cov is not None and reconverted_cov is not None:
            if not np.allclose(original_cov, reconverted_cov, rtol=tolerance, atol=tolerance):
                max_diff = np.max(np.abs(original_cov - reconverted_cov))
                raise ValueError(
                    f"Covariance matrix differs after conversion (max diff: {max_diff}, tolerance: {tolerance})"
                )
        elif original_cov is not None or reconverted_cov is not None:
            raise ValueError("Covariance matrix presence differs after conversion")
        
        return True
    
    @staticmethod
    def create_compatibility_wrapper(
        preparation_framework,
        fallback_loader_func
    ):
        """
        Create a wrapper function for seamless integration with existing load_dataset().
        
        Args:
            preparation_framework: DataPreparationFramework instance
            fallback_loader_func: Fallback function for legacy loading
            
        Returns:
            Wrapper function compatible with existing interface
            
        Requirements: 6.3
        """
        def load_dataset_wrapper(name: str) -> Dict[str, Any]:
            """
            Enhanced load_dataset function with preparation framework integration.
            
            Args:
                name: Dataset name
                
            Returns:
                DatasetDict compatible with existing interface
            """
            try:
                # Try preparation framework first
                standard_dataset = preparation_framework.prepare_dataset(name)
                
                # Convert to existing format
                dataset_dict = FormatConverter.standard_to_dataset_dict(standard_dataset, name)
                
                # Add framework metadata
                dataset_dict['metadata']['preparation_framework'] = {
                    'used': True,
                    'version': '1.0.0',
                    'processing_timestamp': standard_dataset.metadata.get('processing_timestamp')
                }
                
                return dataset_dict
                
            except Exception as e:
                # Fall back to legacy loading
                print(f"⚠️  Preparation framework failed for '{name}': {e}")
                print("   Falling back to legacy loading...")
                
                dataset_dict = fallback_loader_func(name)
                
                # Mark as legacy loaded
                if 'metadata' not in dataset_dict:
                    dataset_dict['metadata'] = {}
                dataset_dict['metadata']['preparation_framework'] = {
                    'used': False,
                    'fallback_reason': str(e)
                }
                
                return dataset_dict
        
        return load_dataset_wrapper
    
    @staticmethod
    def batch_convert_datasets(
        dataset_dicts: Dict[str, Dict[str, Any]],
        output_manager,
        output_formats: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert multiple DatasetDict instances to StandardDataset and save.
        
        Args:
            dataset_dicts: Dictionary mapping dataset names to DatasetDict instances
            output_manager: OutputManager instance for saving
            output_formats: List of output formats to save
            
        Returns:
            Dictionary mapping dataset names to conversion results
            
        Requirements: 6.3
        """
        if output_formats is None:
            output_formats = ['csv', 'numpy']
        
        results = {}
        
        for dataset_name, dataset_dict in dataset_dicts.items():
            try:
                # Convert to StandardDataset
                dataset_type = dataset_dict.get('dataset_type', dataset_name)
                standard_dataset = FormatConverter.dataset_dict_to_standard(dataset_dict, dataset_type)
                
                # Save in requested formats
                output_paths = output_manager.save_dataset(
                    standard_dataset,
                    dataset_name,
                    formats=output_formats
                )
                
                results[dataset_name] = {
                    'status': 'success',
                    'output_paths': output_paths,
                    'dataset_type': dataset_type,
                    'n_points': len(standard_dataset.z)
                }
                
            except Exception as e:
                results[dataset_name] = {
                    'status': 'error',
                    'error': str(e),
                    'dataset_type': dataset_dict.get('dataset_type', 'unknown')
                }
        
        return results