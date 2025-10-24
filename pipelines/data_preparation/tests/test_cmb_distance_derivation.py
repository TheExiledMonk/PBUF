"""
Unit tests for CMB distance prior derivation functionality.

Tests the computation of R, ℓ_A, θ* from raw cosmological parameters,
numerical integration consistency, and parameter sensitivity analysis.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from pipelines.data_preparation.derivation.cmb_models import (
    ParameterSet, DistancePriors, CMBConfig
)
from pipelines.data_preparation.derivation.cmb_exceptions import (
    DerivationError, NumericalInstabilityError
)

# Mock the background integrator functions if not available
try:
    from pipelines.data_preparation.derivation.cmb_background import (
        BackgroundIntegrator, compute_sound_horizon, create_background_integrator
    )
    BACKGROUND_INTEGRATOR_AVAILABLE = True
except ImportError:
    BACKGROUND_INTEGRATOR_AVAILABLE = False
    
    # Create mock classes for testing
    class MockBackgroundIntegrator:
        def __init__(self, params):
            self.params = params
        
        def comoving_distance(self, z):
            # Mock implementation based on typical values
            if z == 1089.8:
                return 14000.0  # Mpc (approximate)
            return 14000.0 * (z / 1089.8)
        
        def angular_diameter_distance(self, z):
            return self.comoving_distance(z) / (1 + z)
    
    def mock_compute_sound_horizon(params, z):
        # Mock sound horizon at recombination (~147 Mpc for Planck cosmology)
        return 147.0
    
    def mock_create_background_integrator(params):
        return MockBackgroundIntegrator(params)
    
    BackgroundIntegrator = MockBackgroundIntegrator
    compute_sound_horizon = mock_compute_sound_horizon
    create_background_integrator = mock_create_background_integrator


class TestDistancePriorComputation:
    """Test cases for distance prior computation from raw parameters."""
    
    def setup_method(self):
        """Set up test fixtures with Planck 2018 values."""
        # Planck 2018 cosmological parameters
        self.planck_params = ParameterSet(
            H0=67.36,
            Omega_m=0.3153,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544,
            A_s=2.1e-9
        )
        
        # Expected Planck 2018 distance priors (approximate)
        self.expected_priors = {
            'R': 1.7502,
            'l_A': 301.76,
            'theta_star': 1.04119
        }
        
        # Default configuration
        self.config = CMBConfig()
    
    def test_compute_shift_parameter_planck_values(self):
        """Test R computation with Planck 2018 parameters."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_shift_parameter
        
        # Mock the background integrator class
        with patch('pipelines.data_preparation.derivation.cmb_derivation.BackgroundIntegrator') as mock_integrator_class:
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0  # Mpc
            mock_integrator_class.return_value = mock_bg
            
            R = compute_shift_parameter(self.planck_params, self.config.z_recombination)
            
            # R = √(Ωm H₀²) × r(z*)/c
            # Expected: R ≈ 1.75 for Planck cosmology
            assert isinstance(R, float)
            assert 1.5 < R < 2.0  # Reasonable range
            
            # Check that background integrator was called correctly
            mock_integrator_class.assert_called_once()
            mock_bg.comoving_distance.assert_called_once_with(self.config.z_recombination)
    
    def test_compute_acoustic_scale_planck_values(self):
        """Test ℓ_A computation with Planck 2018 parameters."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_acoustic_scale
        
        # Mock both background integrator and sound horizon
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.angular_diameter_distance.return_value = 12.8  # Mpc (at z*)
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0  # Mpc
            
            l_A = compute_acoustic_scale(self.planck_params, self.config.z_recombination)
            
            # ℓ_A = π × r(z*)/r_s(z*)
            # Expected: ℓ_A ≈ 302 for Planck cosmology
            assert isinstance(l_A, float)
            assert 250 < l_A < 350  # Reasonable range
            
            # Check function calls
            mock_integrator.assert_called_once()
            mock_bg.angular_diameter_distance.assert_called_once_with(self.config.z_recombination)
            mock_sound_horizon.assert_called_once_with(self.planck_params, self.config.z_recombination)
    
    def test_compute_angular_scale_planck_values(self):
        """Test θ* computation with Planck 2018 parameters."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_angular_scale
        
        # Mock both background integrator and sound horizon
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.angular_diameter_distance.return_value = 12.8  # Mpc (at z*)
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0  # Mpc
            
            theta_star = compute_angular_scale(self.planck_params, self.config.z_recombination)
            
            # θ* = r_s(z*)/r(z*)
            # Expected: θ* ≈ 1.041 for Planck cosmology
            assert isinstance(theta_star, float)
            assert 0.8 < theta_star < 1.3  # Reasonable range
            
            # Check function calls
            mock_integrator.assert_called_once()
            mock_bg.angular_diameter_distance.assert_called_once_with(self.config.z_recombination)
            mock_sound_horizon.assert_called_once_with(self.planck_params, self.config.z_recombination)
    
    def test_compute_distance_priors_complete(self):
        """Test complete distance prior computation."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        # Mock the background integrator imports at the module level
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0  # Mpc (tuned for Planck consistency)
            mock_bg.angular_diameter_distance.return_value = 12.86  # Mpc
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05  # Mpc (tuned for Planck consistency)
            
            priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            # Check that we get a DistancePriors object
            assert isinstance(priors, DistancePriors)
            
            # Check that all values are reasonable
            assert 1.5 < priors.R < 2.0
            assert 250 < priors.l_A < 350
            assert priors.Omega_b_h2 == self.planck_params.Omega_b_h2  # Pass-through
            assert 0.005 < priors.theta_star < 0.02  # θ* is in radians, should be small
            
            # Verify validation passes
            assert priors.validate() is True
    
    def test_distance_prior_accuracy_against_planck(self):
        """Test distance prior accuracy against published Planck values."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        # Use realistic mock values that should produce Planck-like results
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            # Set up mocks to return values that give correct Planck results
            mock_bg = Mock()
            # Tuned to give approximately correct Planck 2018 values
            mock_bg.comoving_distance.return_value = 14026.0  # Mpc
            mock_bg.angular_diameter_distance.return_value = 12.86  # Mpc
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05  # Mpc
            
            priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            # Check accuracy against Planck 2018 values (within 1σ)
            # Allow for some tolerance due to mocking
            assert abs(priors.R - self.expected_priors['R']) < 0.1
            assert abs(priors.l_A - self.expected_priors['l_A']) < 5.0
            assert abs(priors.theta_star - self.expected_priors['theta_star']) < 0.05
    
    def test_parameter_sensitivity_h0(self):
        """Test sensitivity of distance priors to H0 variations."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Baseline computation
            baseline_priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            # Increase H0 by 1%
            high_h0_params = self.planck_params.copy()
            high_h0_params.H0 *= 1.01
            high_h0_priors = compute_distance_priors(high_h0_params, self.config.z_recombination)
            
            # R should increase with H0 (R ∝ √(Ωm H₀²))
            assert high_h0_priors.R > baseline_priors.R
            
            # ℓ_A and θ* should be less sensitive to H0
            relative_change_R = abs(high_h0_priors.R - baseline_priors.R) / baseline_priors.R
            relative_change_l_A = abs(high_h0_priors.l_A - baseline_priors.l_A) / baseline_priors.l_A
            
            # R should be more sensitive to H0 than ℓ_A
            assert relative_change_R > relative_change_l_A
    
    def test_parameter_sensitivity_omega_m(self):
        """Test sensitivity of distance priors to Omega_m variations."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Baseline computation
            baseline_priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            # Increase Omega_m by 1%
            high_omega_m_params = self.planck_params.copy()
            high_omega_m_params.Omega_m *= 1.01
            high_omega_m_priors = compute_distance_priors(high_omega_m_params, self.config.z_recombination)
            
            # R should increase with Omega_m (R ∝ √(Ωm H₀²))
            assert high_omega_m_priors.R > baseline_priors.R
            
            # Check that changes are reasonable
            relative_change = abs(high_omega_m_priors.R - baseline_priors.R) / baseline_priors.R
            assert 0.001 < relative_change < 0.1  # Should be sensitive but not extreme
    
    def test_recombination_redshift_variation(self):
        """Test distance prior computation with different recombination redshifts."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            # Mock functions that depend on redshift
            def mock_comoving_distance(z):
                return 14000.0 * (z / 1089.8)  # Scale with redshift
            
            def mock_angular_diameter_distance(z):
                return mock_comoving_distance(z) / (1 + z)
            
            def mock_sound_horizon_func(params, z):
                return 147.0 * (1089.8 / z)**0.1  # Weak redshift dependence
            
            mock_bg = Mock()
            mock_bg.comoving_distance.side_effect = mock_comoving_distance
            mock_bg.angular_diameter_distance.side_effect = mock_angular_diameter_distance
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.side_effect = mock_sound_horizon_func
            
            # Test with different recombination redshifts
            z_values = [1089.0, 1089.8, 1090.5]
            priors_list = []
            
            for z_recomb in z_values:
                priors = compute_distance_priors(self.planck_params, z_recomb)
                priors_list.append(priors)
                
                # All should be valid
                assert priors.validate() is True
            
            # Distance priors should vary with recombination redshift
            assert priors_list[0].R != priors_list[2].R
            assert priors_list[0].l_A != priors_list[2].l_A
            assert priors_list[0].theta_star != priors_list[2].theta_star


class TestNumericalIntegrationConsistency:
    """Test consistency with existing PBUF background integration methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planck_params = ParameterSet(
            H0=67.36,
            Omega_m=0.3153,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        self.config = CMBConfig()
    
    def test_background_integrator_consistency(self):
        """Test that background integrator is used consistently."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator:
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            
            with patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
                mock_sound_horizon.return_value = 147.0
                
                # Compute distance priors
                priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
                
                # Verify that the same integrator instance is used consistently
                mock_integrator.assert_called()
                
                # Check that parameters are passed correctly to integrator
                call_args = mock_integrator.call_args[0][0]  # First argument (params)
                assert call_args.H0 == self.planck_params.H0
                assert call_args.Omega_m == self.planck_params.Omega_m
    
    def test_sound_horizon_computation_consistency(self):
        """Test that sound horizon computation is consistent."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Compute distance priors
            priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            # Verify sound horizon is called with correct parameters
            mock_sound_horizon.assert_called()
            call_args = mock_sound_horizon.call_args
            
            # Check parameters
            params_arg = call_args[0][0]
            z_arg = call_args[0][1]
            
            assert params_arg.H0 == self.planck_params.H0
            assert z_arg == self.config.z_recombination
    
    def test_integration_method_consistency(self):
        """Test that integration methods are consistent with BAO/SN calculations."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        # This test ensures that the same background physics is used across modules
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator:
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            
            with patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
                mock_sound_horizon.return_value = 147.0
                
                # Multiple calls should use the same integration methods
                priors1 = compute_distance_priors(self.planck_params, 1089.8)
                priors2 = compute_distance_priors(self.planck_params, 1089.8)
                
                # Results should be identical for identical inputs
                assert priors1.R == priors2.R
                assert priors1.l_A == priors2.l_A
                assert priors1.theta_star == priors2.theta_star


class TestParameterSensitivityAndDerivatives:
    """Test parameter sensitivity and derivative computation accuracy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planck_params = ParameterSet(
            H0=67.36,
            Omega_m=0.3153,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        self.config = CMBConfig()
    
    def test_finite_difference_derivative_accuracy(self):
        """Test accuracy of finite difference derivatives."""
        from pipelines.data_preparation.derivation.cmb_derivation import (
            compute_distance_priors, finite_difference_derivative
        )
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Define function to differentiate
            def R_function(params):
                priors = compute_distance_priors(params, self.config.z_recombination)
                return priors.R
            
            # Compute derivative with respect to H0
            derivative = finite_difference_derivative(
                R_function, self.planck_params, 'H0', self.config.jacobian_step_size
            )
            
            # Derivative should be positive (R increases with H0)
            assert derivative > 0
            
            # Check that derivative is reasonable magnitude
            # dR/dH0 should be roughly R/H0 for the √(H0²) dependence
            expected_magnitude = self.planck_params.H0 / 100.0  # Rough estimate
            assert 0.1 * expected_magnitude < abs(derivative) < 10 * expected_magnitude
    
    def test_jacobian_computation_accuracy(self):
        """Test Jacobian matrix computation accuracy."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_jacobian
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Compute Jacobian matrix
            jacobian = compute_jacobian(
                self.planck_params, 
                self.config.z_recombination, 
                self.config.jacobian_step_size
            )
            
            # Check Jacobian shape: (4 observables, 5 parameters)
            assert jacobian.shape == (4, 5)
            
            # Check that Jacobian elements are finite
            assert np.all(np.isfinite(jacobian))
            
            # Check that some derivatives have expected signs
            param_names = self.planck_params.get_parameter_names()
            h0_index = param_names.index('H0')
            omega_m_index = param_names.index('Omega_m')
            
            # dR/dH0 and dR/dOmega_m should be positive
            assert jacobian[0, h0_index] > 0  # dR/dH0 > 0
            assert jacobian[0, omega_m_index] > 0  # dR/dOmega_m > 0
    
    def test_step_size_optimization(self):
        """Test automatic step size optimization for derivatives."""
        from pipelines.data_preparation.derivation.cmb_derivation import optimize_step_size
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Define test function
            def test_function(params):
                from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
                priors = compute_distance_priors(params, self.config.z_recombination)
                return priors.R
            
            # Optimize step size for H0 derivative
            optimal_step = optimize_step_size(test_function, self.planck_params, 'H0')
            
            # Step size should be reasonable
            assert 1e-8 < optimal_step < 1e-3
            
            # Should be different from default step size (indicating optimization occurred)
            assert optimal_step != self.config.jacobian_step_size
    
    def test_derivative_consistency_check(self):
        """Test consistency of derivatives computed different ways."""
        from pipelines.data_preparation.derivation.cmb_derivation import (
            compute_distance_priors, finite_difference_derivative
        )
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Define function
            def R_function(params):
                priors = compute_distance_priors(params, self.config.z_recombination)
                return priors.R
            
            # Compute derivative with different step sizes
            step1 = 1e-6
            step2 = 1e-5
            
            deriv1 = finite_difference_derivative(R_function, self.planck_params, 'H0', step1)
            deriv2 = finite_difference_derivative(R_function, self.planck_params, 'H0', step2)
            
            # Derivatives should be similar (within 10% for reasonable step sizes)
            relative_diff = abs(deriv1 - deriv2) / abs(deriv1)
            assert relative_diff < 0.1


class TestErrorHandlingAndValidation:
    """Test error handling and validation in distance prior derivation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planck_params = ParameterSet(
            H0=67.36,
            Omega_m=0.3153,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        self.config = CMBConfig()
    
    def test_integration_failure_handling(self):
        """Test handling of background integration failures."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        # Mock integrator that raises an exception
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator:
            mock_integrator.side_effect = RuntimeError("Integration failed")
            
            with pytest.raises(DerivationError) as exc_info:
                compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            assert "Integration failed" in str(exc_info.value) or "computation failed" in str(exc_info.value).lower()
            assert exc_info.value.error_code == "CMB_DERIVATION_ERROR"
    
    def test_invalid_parameter_handling(self):
        """Test handling of invalid cosmological parameters."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        # Create parameters with invalid values
        invalid_params = ParameterSet(
            H0=-67.36,  # Negative H0 (invalid)
            Omega_m=0.3153,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        
        # This should fail during parameter validation, not derivation
        with pytest.raises((ValueError, DerivationError)):
            compute_distance_priors(invalid_params, self.config.z_recombination)
    
    def test_extreme_parameter_values(self):
        """Test behavior with extreme but valid parameter values."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        # Create parameters at the edge of valid ranges
        extreme_params = ParameterSet(
            H0=79.9,  # High but valid H0
            Omega_m=0.49,  # High but valid Omega_m
            Omega_b_h2=0.049,  # High but valid Omega_b_h2
            n_s=1.09,  # High but valid n_s
            tau=0.149  # High but valid tau
        )
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Should not raise exception with extreme but valid parameters
            priors = compute_distance_priors(extreme_params, self.config.z_recombination)
            
            # Results should still be reasonable
            assert isinstance(priors, DistancePriors)
            assert priors.validate() is True
    
    def test_numerical_instability_detection(self):
        """Test detection of numerical instabilities."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_jacobian
        
        # Mock integrator that returns NaN values
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = np.nan  # Unstable result
            mock_bg.angular_diameter_distance.return_value = np.nan
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = np.nan
            
            with pytest.raises((NumericalInstabilityError, DerivationError, ValueError)):
                compute_jacobian(self.planck_params, self.config.z_recombination, 1e-6)
    
    def test_invalid_recombination_redshift(self):
        """Test handling of invalid recombination redshift values."""
        from pipelines.data_preparation.derivation.cmb_derivation import compute_distance_priors
        
        with patch('pipelines.data_preparation.derivation.cmb_derivation.create_background_integrator') as mock_integrator, \
             patch('pipelines.data_preparation.derivation.cmb_derivation.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14000.0
            mock_bg.angular_diameter_distance.return_value = 12.8
            mock_integrator.return_value = mock_bg
            mock_sound_horizon.return_value = 147.0
            
            # Test with invalid redshift values
            invalid_z_values = [-100.0, 0.0, 10.0, 10000.0]
            
            for z_invalid in invalid_z_values:
                with pytest.raises((ValueError, DerivationError)):
                    compute_distance_priors(self.planck_params, z_invalid)


class TestDistancePriorsModel:
    """Test cases for DistancePriors data model validation."""
    
    def test_distance_priors_creation_valid(self):
        """Test creating DistancePriors with valid values."""
        priors = DistancePriors(
            R=1.7502,
            l_A=301.76,
            Omega_b_h2=0.02237,
            theta_star=1.04119
        )
        
        assert priors.R == 1.7502
        assert priors.l_A == 301.76
        assert priors.Omega_b_h2 == 0.02237
        assert priors.theta_star == 1.04119
    
    def test_distance_priors_validation_valid(self):
        """Test validation with valid distance prior values."""
        priors = DistancePriors(
            R=1.7502,
            l_A=301.76,
            Omega_b_h2=0.02237,
            theta_star=1.04119
        )
        
        assert priors.validate() is True
    
    def test_distance_priors_validation_invalid_r(self):
        """Test validation with invalid R value."""
        with pytest.raises(ValueError) as exc_info:
            DistancePriors(
                R=-1.7502,  # Negative R (invalid)
                l_A=301.76,
                Omega_b_h2=0.02237,
                theta_star=1.04119
            )
        
        assert "R" in str(exc_info.value)
        assert "outside expected bounds" in str(exc_info.value)
    
    def test_distance_priors_validation_invalid_l_a(self):
        """Test validation with invalid ℓ_A value."""
        with pytest.raises(ValueError) as exc_info:
            DistancePriors(
                R=1.7502,
                l_A=1000.0,  # Too high
                Omega_b_h2=0.02237,
                theta_star=1.04119
            )
        
        assert "l_A" in str(exc_info.value)
    
    def test_distance_priors_validation_nan_values(self):
        """Test validation with NaN values."""
        with pytest.raises(ValueError) as exc_info:
            DistancePriors(
                R=np.nan,  # Invalid
                l_A=301.76,
                Omega_b_h2=0.02237,
                theta_star=1.04119
            )
        
        assert "not finite" in str(exc_info.value)
    
    def test_distance_priors_consistency_validation(self):
        """Test consistency validation between distance priors."""
        # Test with inconsistent values that violate R ∝ l_A * θ_*
        with pytest.raises(ValueError) as exc_info:
            DistancePriors(
                R=1.0,  # Too low for given l_A and θ_*
                l_A=301.76,
                Omega_b_h2=0.02237,
                theta_star=1.04119
            )
        
        assert "inconsistent ratio" in str(exc_info.value)
    
    def test_distance_priors_values_property(self):
        """Test values property returns correct array."""
        priors = DistancePriors(
            R=1.7502,
            l_A=301.76,
            Omega_b_h2=0.02237,
            theta_star=1.04119
        )
        
        values = priors.values
        expected = np.array([1.7502, 301.76, 0.02237, 1.04119])
        
        np.testing.assert_array_equal(values, expected)
    
    def test_distance_priors_to_dict(self):
        """Test converting DistancePriors to dictionary."""
        priors = DistancePriors(
            R=1.7502,
            l_A=301.76,
            Omega_b_h2=0.02237,
            theta_star=1.04119
        )
        
        result = priors.to_dict()
        expected = {
            'R': 1.7502,
            'l_A': 301.76,
            'Omega_b_h2': 0.02237,
            'theta_star': 1.04119
        }
        
        assert result == expected
    
    def test_distance_priors_from_dict(self):
        """Test creating DistancePriors from dictionary."""
        data = {
            'R': 1.7502,
            'l_A': 301.76,
            'Omega_b_h2': 0.02237,
            'theta_star': 1.04119
        }
        
        priors = DistancePriors.from_dict(data)
        
        assert priors.R == 1.7502
        assert priors.l_A == 301.76
        assert priors.Omega_b_h2 == 0.02237
        assert priors.theta_star == 1.04119
    
    def test_distance_priors_get_uncertainties(self):
        """Test getting uncertainties from covariance matrix."""
        priors = DistancePriors(
            R=1.7502,
            l_A=301.76,
            Omega_b_h2=0.02237,
            theta_star=1.04119
        )
        
        # Test with covariance matrix
        cov_matrix = np.diag([0.01**2, 1.0**2, 0.0001**2, 0.001**2])
        uncertainties = priors.get_uncertainties(cov_matrix)
        
        expected = np.array([0.01, 1.0, 0.0001, 0.001])
        np.testing.assert_array_equal(uncertainties, expected)
        
        # Test without covariance matrix (should return defaults)
        default_uncertainties = priors.get_uncertainties()
        assert len(default_uncertainties) == 4
        assert np.all(default_uncertainties > 0)


if __name__ == '__main__':
    pytest.main([__file__])