"""
Data models for CMB raw parameter integration.

This module defines the core data structures for handling raw cosmological parameters
and derived distance priors in the CMB preparation pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class ParameterSet:
    """
    Raw cosmological parameters from Planck-style parameter files.
    
    Contains the fundamental cosmological parameters that are used to derive
    CMB distance priors through background integrators.
    
    Attributes:
        H0: Hubble constant [km/s/Mpc]
        Omega_m: Matter density parameter (dimensionless)
        Omega_b_h2: Baryon density × h² (dimensionless)
        n_s: Scalar spectral index (dimensionless)
        tau: Optical depth to reionization (dimensionless)
        A_s: Scalar amplitude (optional, dimensionless)
    """
    H0: float                        # Hubble constant [km/s/Mpc]
    Omega_m: float                   # Matter density parameter
    Omega_b_h2: float               # Baryon density × h²
    n_s: float                      # Scalar spectral index
    tau: float                      # Optical depth to reionization
    A_s: Optional[float] = None     # Scalar amplitude (if available)
    
    def __post_init__(self):
        """Validate parameter values after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate parameter values against physical bounds.
        
        Returns:
            bool: True if all parameters are valid
            
        Raises:
            ValueError: If any parameter is outside physical bounds
        """
        # Define physical parameter bounds based on Planck 2018 + reasonable extensions
        bounds = {
            'H0': (50.0, 80.0),           # km/s/Mpc
            'Omega_m': (0.2, 0.5),        # Matter density (more realistic range)
            'Omega_b_h2': (0.015, 0.035), # Baryon density (Planck range)
            'n_s': (0.9, 1.1),            # Spectral index
            'tau': (0.01, 0.15),          # Optical depth
            'A_s': (1e-10, 5e-9)          # Scalar amplitude
        }
        
        # Check each parameter
        for param_name, (min_val, max_val) in bounds.items():
            value = getattr(self, param_name)
            
            # Skip None values for optional parameters
            if value is None and param_name == 'A_s':
                continue
                
            if value is None:
                raise ValueError(f"Required parameter {param_name} cannot be None")
            
            # Check for NaN or infinite values
            if not np.isfinite(value):
                raise ValueError(f"Parameter {param_name} = {value} is not finite")
            
            # Check physical bounds
            if value < min_val or value > max_val:
                raise ValueError(
                    f"Parameter {param_name} = {value} outside physical bounds "
                    f"[{min_val}, {max_val}]"
                )
        
        # Additional consistency checks
        self._validate_consistency()
        
        return True
    
    def _validate_consistency(self):
        """Validate parameter consistency relationships."""
        # Check that Omega_m is reasonable given H0
        # For typical cosmologies: 0.2 < Omega_m < 0.4
        if self.Omega_m < 0.15 or self.Omega_m > 0.45:
            raise ValueError(f"Omega_m = {self.Omega_m} outside typical range [0.15, 0.45]")
        
        # Check baryon fraction is reasonable
        # Omega_b ≈ Omega_b_h2 / h² where h = H0/100
        h = self.H0 / 100.0
        Omega_b = self.Omega_b_h2 / (h * h)
        baryon_fraction = Omega_b / self.Omega_m
        
        if baryon_fraction < 0.1 or baryon_fraction > 0.25:
            raise ValueError(
                f"Baryon fraction Ω_b/Ω_m = {baryon_fraction:.3f} outside "
                f"reasonable range [0.1, 0.25]"
            )
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary for processing.
        
        Returns:
            Dict mapping parameter names to values
        """
        result = {
            'H0': self.H0,
            'Omega_m': self.Omega_m,
            'Omega_b_h2': self.Omega_b_h2,
            'n_s': self.n_s,
            'tau': self.tau
        }
        
        if self.A_s is not None:
            result['A_s'] = self.A_s
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterSet':
        """
        Create ParameterSet from dictionary.
        
        Args:
            data: Dictionary containing parameter values
            
        Returns:
            ParameterSet instance
            
        Raises:
            ValueError: If required parameters are missing
        """
        required_params = ['H0', 'Omega_m', 'Omega_b_h2', 'n_s', 'tau']
        
        # Check for required parameters
        missing_params = [p for p in required_params if p not in data]
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        return cls(
            H0=float(data['H0']),
            Omega_m=float(data['Omega_m']),
            Omega_b_h2=float(data['Omega_b_h2']),
            n_s=float(data['n_s']),
            tau=float(data['tau']),
            A_s=float(data['A_s']) if 'A_s' in data and data['A_s'] is not None else None
        )
    
    def copy(self) -> 'ParameterSet':
        """Create a copy of this parameter set."""
        return ParameterSet(
            H0=self.H0,
            Omega_m=self.Omega_m,
            Omega_b_h2=self.Omega_b_h2,
            n_s=self.n_s,
            tau=self.tau,
            A_s=self.A_s
        )
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        names = ['H0', 'Omega_m', 'Omega_b_h2', 'n_s', 'tau']
        if self.A_s is not None:
            names.append('A_s')
        return names


@dataclass
class DistancePriors:
    """
    CMB distance priors derived from raw cosmological parameters.
    
    Contains the standard CMB observables used in cosmological fitting:
    - R: Shift parameter (dimensionless)
    - l_A: Acoustic scale (multipole number)
    - Omega_b_h2: Baryon density (pass-through from parameters)
    - theta_star: Angular scale at last scattering
    
    Attributes:
        R: Shift parameter (dimensionless)
        l_A: Acoustic scale (multipole number)
        Omega_b_h2: Baryon density × h² (dimensionless)
        theta_star: Angular scale at last scattering (dimensionless)
    """
    R: float                        # Shift parameter
    l_A: float                      # Acoustic scale
    Omega_b_h2: float              # Baryon density (pass-through)
    theta_star: float              # Angular scale
    
    def __post_init__(self):
        """Validate distance prior values after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate distance prior values against expected ranges.
        
        Returns:
            bool: True if all values are valid
            
        Raises:
            ValueError: If any value is outside expected bounds
        """
        # Define expected ranges based on Planck 2018 results with margins
        bounds = {
            'R': (1.0, 2.5),              # Shift parameter
            'l_A': (250.0, 400.0),        # Acoustic scale (expanded range)
            'Omega_b_h2': (0.01, 0.05),   # Baryon density
            'theta_star': (0.001, 0.02)   # Angular scale (more realistic range)
        }
        
        # Check each parameter
        for param_name, (min_val, max_val) in bounds.items():
            value = getattr(self, param_name)
            
            # Check for NaN or infinite values
            if not np.isfinite(value):
                raise ValueError(f"Distance prior {param_name} = {value} is not finite")
            
            # Check physical bounds
            if value < min_val or value > max_val:
                raise ValueError(
                    f"Distance prior {param_name} = {value} outside expected bounds "
                    f"[{min_val}, {max_val}]"
                )
        
        # Additional consistency checks
        self._validate_consistency()
        
        return True
    
    def _validate_consistency(self):
        """Validate consistency relationships between distance priors."""
        # For now, skip detailed consistency checks as they depend on the specific
        # cosmological model and units used. Focus on basic sanity checks.
        
        # Basic sanity checks: all values should be positive and finite
        if not (self.R > 0 and np.isfinite(self.R)):
            raise ValueError(f"Invalid R value: {self.R}")
        
        if not (self.l_A > 0 and np.isfinite(self.l_A)):
            raise ValueError(f"Invalid l_A value: {self.l_A}")
        
        if not (self.theta_star > 0 and np.isfinite(self.theta_star)):
            raise ValueError(f"Invalid theta_star value: {self.theta_star}")
        
        if not (self.Omega_b_h2 > 0 and np.isfinite(self.Omega_b_h2)):
            raise ValueError(f"Invalid Omega_b_h2 value: {self.Omega_b_h2}")
        
        # Additional check: θ* should be much smaller than 1 (it's in radians)
        if self.theta_star > 0.1:  # Very generous upper bound
            raise ValueError(f"theta_star = {self.theta_star} seems too large (should be << 1 radian)")
    
    @property
    def values(self) -> np.ndarray:
        """
        Return distance priors as array for numerical operations.
        
        Returns:
            Array containing [R, l_A, Omega_b_h2, theta_star]
        """
        return np.array([self.R, self.l_A, self.Omega_b_h2, self.theta_star])
    
    @property
    def parameter_names(self) -> List[str]:
        """Get list of distance prior parameter names."""
        return ['R', 'l_A', 'Omega_b_h2', 'theta_star']
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary for processing.
        
        Returns:
            Dict mapping parameter names to values
        """
        return {
            'R': self.R,
            'l_A': self.l_A,
            'Omega_b_h2': self.Omega_b_h2,
            'theta_star': self.theta_star
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistancePriors':
        """
        Create DistancePriors from dictionary.
        
        Args:
            data: Dictionary containing distance prior values
            
        Returns:
            DistancePriors instance
            
        Raises:
            ValueError: If required parameters are missing
        """
        required_params = ['R', 'l_A', 'Omega_b_h2', 'theta_star']
        
        # Check for required parameters
        missing_params = [p for p in required_params if p not in data]
        if missing_params:
            raise ValueError(f"Missing required distance priors: {missing_params}")
        
        return cls(
            R=float(data['R']),
            l_A=float(data['l_A']),
            Omega_b_h2=float(data['Omega_b_h2']),
            theta_star=float(data['theta_star'])
        )
    
    def copy(self) -> 'DistancePriors':
        """Create a copy of this distance priors set."""
        return DistancePriors(
            R=self.R,
            l_A=self.l_A,
            Omega_b_h2=self.Omega_b_h2,
            theta_star=self.theta_star
        )
    
    def get_uncertainties(self, covariance: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get uncertainties from covariance matrix diagonal.
        
        Args:
            covariance: Covariance matrix (4x4)
            
        Returns:
            Array of uncertainties (standard deviations)
        """
        if covariance is not None:
            if covariance.shape != (4, 4):
                raise ValueError(f"Covariance matrix must be 4x4, got {covariance.shape}")
            return np.sqrt(np.diag(covariance))
        else:
            # Return default uncertainties if no covariance provided
            return np.array([0.01, 1.0, 0.0001, 0.001])  # Rough estimates


@dataclass
class CMBConfig:
    """
    Configuration settings for CMB raw parameter processing.
    
    Controls the behavior of the CMB derivation pipeline, including whether to use
    raw parameters, numerical computation settings, and fallback behavior.
    
    Attributes:
        use_raw_parameters: Enable raw parameter processing
        z_recombination: Recombination redshift for distance calculations
        jacobian_step_size: Step size for numerical differentiation
        validation_tolerance: Tolerance for covariance validation
        fallback_to_legacy: Auto-fallback if raw params unavailable
        cache_computations: Cache expensive computations
        performance_monitoring: Enable performance metrics collection
    """
    use_raw_parameters: bool = True          # Enable raw parameter processing
    z_recombination: float = 1089.8         # Recombination redshift
    jacobian_step_size: float = 1e-6       # Numerical differentiation step
    validation_tolerance: float = 1e-8      # Covariance validation tolerance
    fallback_to_legacy: bool = True         # Auto-fallback if raw params unavailable
    cache_computations: bool = True         # Cache expensive computations
    performance_monitoring: bool = False    # Enable performance metrics collection
    
    def __post_init__(self):
        """Validate configuration values after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate z_recombination
        if not isinstance(self.z_recombination, (int, float)):
            raise ValueError("z_recombination must be numeric")
        
        if self.z_recombination < 1000 or self.z_recombination > 1200:
            raise ValueError(
                f"z_recombination = {self.z_recombination} outside reasonable range [1000, 1200]"
            )
        
        # Validate jacobian_step_size
        if not isinstance(self.jacobian_step_size, (int, float)):
            raise ValueError("jacobian_step_size must be numeric")
        
        if self.jacobian_step_size <= 0 or self.jacobian_step_size >= 1e-3:
            raise ValueError(
                f"jacobian_step_size = {self.jacobian_step_size} outside reasonable range (0, 1e-3]"
            )
        
        # Validate validation_tolerance
        if not isinstance(self.validation_tolerance, (int, float)):
            raise ValueError("validation_tolerance must be numeric")
        
        if self.validation_tolerance <= 0 or self.validation_tolerance >= 1e-5:
            raise ValueError(
                f"validation_tolerance = {self.validation_tolerance} outside reasonable range (0, 1e-5]"
            )
        
        # Validate boolean flags
        bool_params = [
            'use_raw_parameters', 'fallback_to_legacy', 
            'cache_computations', 'performance_monitoring'
        ]
        
        for param in bool_params:
            value = getattr(self, param)
            if not isinstance(value, bool):
                raise ValueError(f"{param} must be boolean, got {type(value).__name__}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dict containing all configuration parameters
        """
        return {
            'use_raw_parameters': self.use_raw_parameters,
            'z_recombination': self.z_recombination,
            'jacobian_step_size': self.jacobian_step_size,
            'validation_tolerance': self.validation_tolerance,
            'fallback_to_legacy': self.fallback_to_legacy,
            'cache_computations': self.cache_computations,
            'performance_monitoring': self.performance_monitoring
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CMBConfig':
        """
        Create CMBConfig from dictionary.
        
        Args:
            data: Dictionary containing configuration parameters
            
        Returns:
            CMBConfig instance with validated parameters
        """
        # Use defaults for missing parameters
        defaults = cls()
        
        return cls(
            use_raw_parameters=data.get('use_raw_parameters', defaults.use_raw_parameters),
            z_recombination=data.get('z_recombination', defaults.z_recombination),
            jacobian_step_size=data.get('jacobian_step_size', defaults.jacobian_step_size),
            validation_tolerance=data.get('validation_tolerance', defaults.validation_tolerance),
            fallback_to_legacy=data.get('fallback_to_legacy', defaults.fallback_to_legacy),
            cache_computations=data.get('cache_computations', defaults.cache_computations),
            performance_monitoring=data.get('performance_monitoring', defaults.performance_monitoring)
        )
    
    @classmethod
    def get_default_config(cls) -> 'CMBConfig':
        """
        Get default configuration for production use.
        
        Returns:
            CMBConfig with production-ready defaults
        """
        return cls(
            use_raw_parameters=True,
            z_recombination=1089.8,
            jacobian_step_size=1e-6,
            validation_tolerance=1e-8,
            fallback_to_legacy=True,
            cache_computations=True,
            performance_monitoring=False
        )
    
    @classmethod
    def get_development_config(cls) -> 'CMBConfig':
        """
        Get configuration optimized for development and testing.
        
        Returns:
            CMBConfig with development-friendly settings
        """
        return cls(
            use_raw_parameters=True,
            z_recombination=1089.8,
            jacobian_step_size=1e-5,  # Slightly larger for faster computation
            validation_tolerance=1e-7,  # Slightly relaxed for testing
            fallback_to_legacy=True,
            cache_computations=False,  # Disable caching for testing
            performance_monitoring=True  # Enable monitoring for development
        )
    
    def copy(self) -> 'CMBConfig':
        """Create a copy of this configuration."""
        return CMBConfig(
            use_raw_parameters=self.use_raw_parameters,
            z_recombination=self.z_recombination,
            jacobian_step_size=self.jacobian_step_size,
            validation_tolerance=self.validation_tolerance,
            fallback_to_legacy=self.fallback_to_legacy,
            cache_computations=self.cache_computations,
            performance_monitoring=self.performance_monitoring
        )
    
    def update_from_environment(self) -> 'CMBConfig':
        """
        Update configuration from environment variables.
        
        Environment variables:
        - CMB_USE_RAW_PARAMETERS: "true"/"false"
        - CMB_Z_RECOMBINATION: float value
        - CMB_JACOBIAN_STEP_SIZE: float value
        - CMB_VALIDATION_TOLERANCE: float value
        - CMB_FALLBACK_TO_LEGACY: "true"/"false"
        - CMB_CACHE_COMPUTATIONS: "true"/"false"
        - CMB_PERFORMANCE_MONITORING: "true"/"false"
        
        Returns:
            Updated CMBConfig instance
        """
        import os
        
        # Helper function to parse boolean environment variables
        def parse_bool(value: str) -> bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        
        # Update from environment variables if present
        env_mappings = {
            'CMB_USE_RAW_PARAMETERS': ('use_raw_parameters', parse_bool),
            'CMB_Z_RECOMBINATION': ('z_recombination', float),
            'CMB_JACOBIAN_STEP_SIZE': ('jacobian_step_size', float),
            'CMB_VALIDATION_TOLERANCE': ('validation_tolerance', float),
            'CMB_FALLBACK_TO_LEGACY': ('fallback_to_legacy', parse_bool),
            'CMB_CACHE_COMPUTATIONS': ('cache_computations', parse_bool),
            'CMB_PERFORMANCE_MONITORING': ('performance_monitoring', parse_bool)
        }
        
        updated_config = self.copy()
        
        for env_var, (attr_name, converter) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    setattr(updated_config, attr_name, converted_value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid environment variable {env_var}={env_value}: {e}"
                    )
        
        # Validate the updated configuration
        updated_config.validate()
        
        return updated_config


# Configuration integration utilities
def load_cmb_config_from_file(config_path: str) -> CMBConfig:
    """
    Load CMB configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        CMBConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Extract CMB-specific configuration if nested
        if 'cmb' in config_data:
            config_data = config_data['cmb']
        elif 'derivation' in config_data and 'cmb' in config_data['derivation']:
            config_data = config_data['derivation']['cmb']
        
        return CMBConfig.from_dict(config_data)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")


def save_cmb_config_to_file(config: CMBConfig, config_path: str):
    """
    Save CMB configuration to JSON file.
    
    Args:
        config: CMBConfig instance to save
        config_path: Path where to save the configuration
    """
    import json
    from pathlib import Path
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def get_integrated_cmb_config() -> CMBConfig:
    """
    Get CMB configuration integrated with the data preparation framework.
    
    This function attempts to load configuration from multiple sources in order:
    1. Environment variables
    2. Framework configuration file
    3. Default configuration
    
    Returns:
        CMBConfig instance with integrated settings
    """
    # Start with default configuration
    config = CMBConfig.get_default_config()
    
    # Try to load from framework configuration
    try:
        # Look for framework config file in standard locations
        config_paths = [
            'config/data_preparation.json',
            '.kiro/config/data_preparation.json',
            'pipelines/data_preparation/config.json'
        ]
        
        for config_path in config_paths:
            try:
                framework_config = load_cmb_config_from_file(config_path)
                config = framework_config
                break
            except (FileNotFoundError, ValueError):
                continue
                
    except Exception:
        # Fall back to default if framework config loading fails
        pass
    
    # Override with environment variables
    try:
        config = config.update_from_environment()
    except Exception:
        # Fall back to current config if environment parsing fails
        pass
    
    return config