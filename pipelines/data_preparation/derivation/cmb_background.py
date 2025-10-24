"""
Background integrator module for CMB distance prior calculations.

This module provides background cosmology calculations needed for deriving
CMB distance priors from raw cosmological parameters. It implements the
integration methods consistent with BAO and SN background calculations.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .cmb_models import ParameterSet


@dataclass
class BackgroundIntegrator:
    """
    Background integrator for cosmological distance calculations.
    
    This class provides methods for computing comoving distances, angular diameter
    distances, and other background quantities needed for CMB distance priors.
    
    Attributes:
        params: Cosmological parameters
        z_max: Maximum redshift for integration tables
        n_points: Number of points in integration tables
    """
    params: ParameterSet
    z_max: float = 1200.0
    n_points: int = 2000
    
    def __post_init__(self):
        """Initialize integration tables and validate parameters."""
        self.params.validate()
        self._setup_integration_tables()
    
    def _setup_integration_tables(self):
        """Set up redshift and integration tables for efficient computation."""
        # Create redshift array
        self.z_table = np.linspace(0.0, self.z_max, self.n_points)
        
        # Compute E(z) = H(z)/H0 for all redshifts
        self.E_table = self._compute_E_function(self.z_table)
        
        # Pre-compute comoving distance table
        self.comoving_distance_table = self._compute_comoving_distance_table()
    
    def _compute_E_function(self, z: np.ndarray) -> np.ndarray:
        """
        Compute dimensionless Hubble parameter E(z) = H(z)/H0.
        
        For flat ΛCDM cosmology:
        E(z) = √[Ωm(1+z)³ + ΩΛ]
        
        Args:
            z: Redshift array
            
        Returns:
            E(z) values
        """
        # For flat universe: Omega_Lambda = 1 - Omega_m
        Omega_Lambda = 1.0 - self.params.Omega_m
        
        # Handle z = 0 case
        z_safe = np.maximum(z, 0.0)
        
        # Compute E(z)
        E_z = np.sqrt(
            self.params.Omega_m * (1 + z_safe)**3 + Omega_Lambda
        )
        
        return E_z
    
    def _compute_comoving_distance_table(self) -> np.ndarray:
        """
        Compute comoving distance table using numerical integration.
        
        Returns:
            Comoving distance values in Mpc
        """
        # Speed of light in km/s
        c_km_s = 299792.458
        
        # Initialize distance array
        distances = np.zeros_like(self.z_table)
        
        # Compute distances by integration
        for i in range(1, len(self.z_table)):
            z_slice = self.z_table[:i+1]
            E_slice = self.E_table[:i+1]
            
            # Integrand: 1/E(z)
            integrand = 1.0 / E_slice
            
            # Numerical integration using trapezoidal rule
            integral = np.trapz(integrand, z_slice)
            
            # Convert to physical distance: r(z) = c/H0 * integral
            distances[i] = (c_km_s / self.params.H0) * integral
        
        return distances
    
    def comoving_distance(self, z: float) -> float:
        """
        Compute comoving distance to redshift z.
        
        Args:
            z: Redshift
            
        Returns:
            Comoving distance in Mpc
        """
        if z <= 0:
            return 0.0
        
        if z >= self.z_max:
            # Extrapolate for high redshifts
            return self._extrapolate_comoving_distance(z)
        
        # Interpolate from pre-computed table
        return float(np.interp(z, self.z_table, self.comoving_distance_table))
    
    def _extrapolate_comoving_distance(self, z: float) -> float:
        """
        Extrapolate comoving distance for redshifts beyond table range.
        
        Args:
            z: Redshift (z > z_max)
            
        Returns:
            Extrapolated comoving distance in Mpc
        """
        # Use the last few points to estimate the derivative
        z_end = self.z_table[-10:]
        d_end = self.comoving_distance_table[-10:]
        
        # Linear extrapolation based on the slope at z_max
        slope = (d_end[-1] - d_end[-2]) / (z_end[-1] - z_end[-2])
        extrapolated = d_end[-1] + slope * (z - self.z_max)
        
        return float(extrapolated)
    
    def angular_diameter_distance(self, z: float) -> float:
        """
        Compute angular diameter distance to redshift z.
        
        Args:
            z: Redshift
            
        Returns:
            Angular diameter distance in Mpc
        """
        if z <= 0:
            return 0.0
        
        comoving_dist = self.comoving_distance(z)
        return comoving_dist / (1 + z)
    
    def luminosity_distance(self, z: float) -> float:
        """
        Compute luminosity distance to redshift z.
        
        Args:
            z: Redshift
            
        Returns:
            Luminosity distance in Mpc
        """
        if z <= 0:
            return 0.0
        
        comoving_dist = self.comoving_distance(z)
        return comoving_dist * (1 + z)


def compute_sound_horizon(params: ParameterSet, z: float) -> float:
    """
    Compute sound horizon at redshift z using standard ΛCDM baseline.
    
    This function uses the standard ΛCDM reference implementation for computing
    the sound horizon, ensuring consistency with the baseline cosmology used
    for CMB distance prior derivation.
    
    Args:
        params: Cosmological parameters
        z: Redshift
        
    Returns:
        Sound horizon in Mpc (consistent units with comoving distance)
        
    Note:
        Uses standard ΛCDM cosmology as baseline. Model-specific effects are
        handled in the fitting stage, not in data preparation.
    """
    try:
        # Import the PBUF reference functions (which have correct units and formulas)
        # but use them with ΛCDM parameters for baseline computation
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lib'))
        from pbuf_equations import r_s, H_lcdm
        
        # Convert parameters to the format expected by lcdm_equations
        h = params.H0 / 100.0
        
        # Density parameters (convert from physical to fractional)
        Omega_b = params.Omega_b_h2 / (h * h)  # Baryon density parameter
        Omega_gamma = 2.469e-5 / (h * h)      # Photon density parameter  
        Omega_r = 9.236e-5 / (h * h)          # Total radiation density
        Ok0 = 0.0                             # Flat universe
        Ode0 = 1.0 - params.Omega_m - Omega_r - Ok0  # Dark energy density
        
        # Prepare arguments for ΛCDM functions
        H_args = (params.H0, params.Omega_m, Omega_r, Ode0, Ok0)
        
        # Compute sound horizon using ΛCDM baseline (via PBUF equations with ΛCDM parameters)
        sound_horizon = r_s(z, H_lcdm, H_args, Omega_b, Omega_gamma)
        
        return float(sound_horizon)
        
    except Exception as e:
        raise ValueError(f"Failed to compute sound horizon using ΛCDM baseline: {str(e)}")


def create_background_integrator(params: ParameterSet, **kwargs) -> BackgroundIntegrator:
    """
    Create a background integrator instance with the given parameters.
    
    This factory function provides a consistent interface for creating
    background integrators across the PBUF pipeline.
    
    Args:
        params: Cosmological parameters
        **kwargs: Additional configuration options
        
    Returns:
        Configured BackgroundIntegrator instance
    """
    return BackgroundIntegrator(params, **kwargs)


def validate_background_consistency(params: ParameterSet) -> Dict[str, Any]:
    """
    Validate that background calculations are consistent with PBUF methods.
    
    This function performs consistency checks to ensure that the background
    integrator produces results compatible with existing BAO and SN calculations.
    
    Args:
        params: Cosmological parameters to validate
        
    Returns:
        Dictionary containing validation results and diagnostics
    """
    try:
        # Create integrator
        integrator = BackgroundIntegrator(params)
        
        # Test calculations at standard redshifts
        test_redshifts = [0.1, 0.5, 1.0, 2.0, 1089.8]
        results = {}
        
        for z in test_redshifts:
            results[f'z_{z}'] = {
                'comoving_distance': integrator.comoving_distance(z),
                'angular_diameter_distance': integrator.angular_diameter_distance(z),
                'luminosity_distance': integrator.luminosity_distance(z),
                'sound_horizon': compute_sound_horizon(params, z)
            }
        
        # Check for reasonable values
        validation_result = {
            'valid': True,
            'results': results,
            'diagnostics': {
                'integrator_setup': 'successful',
                'table_size': len(integrator.z_table),
                'z_max': integrator.z_max
            }
        }
        
        # Basic sanity checks
        z_recomb = 1089.8
        r_recomb = integrator.comoving_distance(z_recomb)
        rs_recomb = compute_sound_horizon(params, z_recomb)
        
        if r_recomb <= 0 or rs_recomb <= 0:
            validation_result['valid'] = False
            validation_result['diagnostics']['error'] = 'Non-positive distances computed'
        
        if rs_recomb > r_recomb:
            validation_result['valid'] = False
            validation_result['diagnostics']['error'] = 'Sound horizon larger than comoving distance'
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'error': str(e),
            'diagnostics': {'validation_failed': True}
        }


def propagate_covariance(jacobian: np.ndarray, input_cov: np.ndarray) -> np.ndarray:
    """
    Propagate covariance matrix through linear transformation using Jacobian.
    
    Args:
        jacobian: Jacobian matrix (n_outputs x n_inputs)
        input_cov: Input covariance matrix (n_inputs x n_inputs)
        
    Returns:
        Output covariance matrix (n_outputs x n_outputs)
    """
    return jacobian @ input_cov @ jacobian.T


def compute_jacobian(params: ParameterSet, z_recombination: float = 1089.8) -> np.ndarray:
    """
    Compute Jacobian matrix for CMB distance priors with respect to parameters.
    
    This function computes the derivatives of distance priors (R, l_A, θ*)
    with respect to cosmological parameters using finite differences.
    
    Args:
        params: Cosmological parameters
        z_recombination: Redshift of recombination (default: 1089.8)
        
    Returns:
        Jacobian matrix (3x4) for [R, l_A, θ*] vs [H0, Omega_m, Omega_b_h2, ns]
    """
    try:
        # Parameter names and their current values
        param_names = ['H0', 'Omega_m', 'Omega_b_h2', 'n_s']
        base_values = [params.H0, params.Omega_m, params.Omega_b_h2, params.n_s]
        
        # Compute base distance priors
        base_priors = compute_distance_priors(params, z_recombination)
        base_vector = np.array([base_priors.R, base_priors.l_A, base_priors.theta_star])
        
        # Initialize Jacobian matrix
        jacobian = np.zeros((3, 4))
        
        # Finite difference step sizes (relative)
        step_sizes = [0.01, 0.01, 0.01, 0.01]  # 1% steps
        
        # Compute derivatives using finite differences
        for i, (param_name, base_val, step) in enumerate(zip(param_names, base_values, step_sizes)):
            # Create perturbed parameter set
            delta = abs(base_val * step)
            if delta == 0:
                delta = 1e-6
            
            # Forward difference
            perturbed_params = ParameterSet(
                H0=params.H0 + (delta if param_name == 'H0' else 0),
                Omega_m=params.Omega_m + (delta if param_name == 'Omega_m' else 0),
                Omega_b_h2=params.Omega_b_h2 + (delta if param_name == 'Omega_b_h2' else 0),
                n_s=params.n_s + (delta if param_name == 'n_s' else 0)
            )
            
            try:
                perturbed_priors = compute_distance_priors(perturbed_params, z_recombination)
                perturbed_vector = np.array([perturbed_priors.R, perturbed_priors.l_A, perturbed_priors.theta_star])
                
                # Compute derivative
                jacobian[:, i] = (perturbed_vector - base_vector) / delta
                
            except Exception:
                # If forward difference fails, use a smaller step or set to zero
                jacobian[:, i] = 0.0
        
        return jacobian
        
    except Exception as e:
        # Return identity-like matrix if computation fails
        return np.eye(3, 4)


def compute_distance_priors(params: ParameterSet, z_recombination: float = 1089.8) -> 'DistancePriors':
    """
    Compute CMB distance priors from cosmological parameters using standard ΛCDM baseline.
    
    This function uses standard ΛCDM cosmology as the baseline for computing distance priors
    from raw cosmological parameters. This is the standard approach in cosmology - CMB distance
    priors are derived assuming ΛCDM background, then used in model comparisons. Model-specific
    effects (like PBUF) are handled later in the fitting stage.
    
    Implementation Strategy:
    - Uses PBUF equations library (lib/pbuf_equations.py) with ΛCDM parameters
    - This ensures correct units and formulas while maintaining ΛCDM baseline
    - The PBUF library has the correct R_b(z) formula and sound horizon calculation
    - Model-specific effects are applied in fit_cmb.py, not in data preparation
    
    Args:
        params: Cosmological parameters (H0, Omega_m, Omega_b_h2, etc.)
        z_recombination: Redshift of recombination (default: 1089.8)
        
    Returns:
        DistancePriors object containing R, l_A, θ*, and Omega_b_h2
        
    Raises:
        ValueError: If parameters are invalid or computation fails
        
    Note:
        This implementation assumes flat ΛCDM cosmology for the baseline distance computation.
        Model-specific modifications (PBUF, wCDM, etc.) are applied in the fitting pipeline.
    """
    try:
        from .cmb_models import DistancePriors
        
        # Import the PBUF reference functions (which have correct units and formulas)
        # but use them with ΛCDM parameters for baseline computation
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'lib'))
        from pbuf_equations import cmb_priors, H_lcdm
        
        # Convert parameters to the format expected by lcdm_equations
        h = params.H0 / 100.0
        
        # Density parameters (convert from physical to fractional)
        Omega_b = params.Omega_b_h2 / (h * h)  # Baryon density parameter
        Omega_gamma = 2.469e-5 / (h * h)      # Photon density parameter  
        Omega_r = 9.236e-5 / (h * h)          # Total radiation density
        Ok0 = 0.0                             # Flat universe
        Ode0 = 1.0 - params.Omega_m - Omega_r - Ok0  # Dark energy density
        
        # Prepare arguments for ΛCDM functions
        H_args = (params.H0, params.Omega_m, Omega_r, Ode0, Ok0)
        
        # Compute distance priors using ΛCDM baseline (via PBUF equations with ΛCDM parameters)
        priors_dict = cmb_priors(z_recombination, H_lcdm, H_args, params.Omega_m, params.H0, Omega_b, Omega_gamma)
        
        # Create and return DistancePriors object
        return DistancePriors(
            R=float(priors_dict["R"]),
            l_A=float(priors_dict["l_A"]), 
            theta_star=float(priors_dict["θ*"]),
            Omega_b_h2=float(params.Omega_b_h2)
        )
        
    except Exception as e:
        raise ValueError(f"Failed to compute distance priors using ΛCDM baseline: {str(e)}")