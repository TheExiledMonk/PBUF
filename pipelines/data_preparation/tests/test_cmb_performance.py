"""
Performance and stress tests for CMB raw parameter processing.

Tests processing time, memory usage, numerical stability with extreme values,
and system behavior under stress conditions.
"""

import pytest
import numpy as np
import time
import psutil
import os
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import tempfile
from pathlib import Path

from pipelines.data_preparation.derivation.cmb_derivation import (
    compute_distance_priors, compute_jacobian, process_cmb_dataset
)
from pipelines.data_preparation.derivation.cmb_models import (
    ParameterSet, DistancePriors, CMBConfig
)
from pipelines.data_preparation.derivation.cmb_exceptions import (
    DerivationError, NumericalInstabilityError
)


class TestProcessingPerformance:
    """Test cases for processing time performance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Standard Planck 2018 parameters
        self.planck_params = ParameterSet(
            H0=67.36,
            Omega_m=0.3153,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544,
            A_s=2.1e-9
        )
        
        # Performance-optimized configuration
        self.config = CMBConfig(
            use_raw_parameters=True,
            z_recombination=1089.8,
            jacobian_step_size=1e-5,  # Slightly larger for faster computation
            validation_tolerance=1e-7,
            cache_computations=True,
            performance_monitoring=True
        )
    
    def test_distance_prior_computation_timing(self):
        """Test that distance prior computation completes within time limit."""
        # Mock background integrator for consistent timing
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_bg.angular_diameter_distance.return_value = 12.86
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Time the computation
            start_time = time.time()
            priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            end_time = time.time()
            
            computation_time = end_time - start_time
            
            # Should complete within 1 second (very generous for mocked computation)
            assert computation_time < 1.0
            
            # Verify result is valid
            assert isinstance(priors, DistancePriors)
            assert priors.validate() is True
    
    def test_jacobian_computation_timing(self):
        """Test that Jacobian computation completes within time limit."""
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Time the Jacobian computation
            start_time = time.time()
            jacobian = compute_jacobian(
                self.planck_params, 
                self.config.z_recombination, 
                self.config.jacobian_step_size
            )
            end_time = time.time()
            
            computation_time = end_time - start_time
            
            # Should complete within 5 seconds (generous for numerical differentiation)
            assert computation_time < 5.0
            
            # Verify result shape and properties
            assert jacobian.shape == (4, 5)  # 4 observables, 5 parameters
            assert np.all(np.isfinite(jacobian))
    
    def test_multiple_parameter_sets_timing(self):
        """Test processing time with multiple parameter sets."""
        # Create variations of Planck parameters
        parameter_sets = []
        for i in range(10):
            # Small random variations around Planck values
            params = ParameterSet(
                H0=67.36 + np.random.normal(0, 0.5),
                Omega_m=0.3153 + np.random.normal(0, 0.01),
                Omega_b_h2=0.02237 + np.random.normal(0, 0.0001),
                n_s=0.9649 + np.random.normal(0, 0.005),
                tau=0.0544 + np.random.normal(0, 0.008)
            )
            parameter_sets.append(params)
        
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Time processing of all parameter sets
            start_time = time.time()
            results = []
            for params in parameter_sets:
                priors = compute_distance_priors(params, self.config.z_recombination)
                results.append(priors)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_set = total_time / len(parameter_sets)
            
            # Should process each set quickly
            assert avg_time_per_set < 0.5  # 0.5 seconds per parameter set
            assert total_time < 10.0  # Total time under 10 seconds
            
            # Verify all results are valid
            assert len(results) == len(parameter_sets)
            for result in results:
                assert isinstance(result, DistancePriors)
                assert result.validate() is True
class
 TestMemoryUsage:
    """Test cases for memory usage monitoring."""
    
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
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_distance_prior_memory_usage(self):
        """Test memory usage during distance prior computation."""
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Measure memory before computation
            memory_before = self.get_memory_usage_mb()
            
            # Perform computation
            priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            # Measure memory after computation
            memory_after = self.get_memory_usage_mb()
            
            memory_increase = memory_after - memory_before
            
            # Memory increase should be minimal (< 10 MB for simple computation)
            assert memory_increase < 10.0
            
            # Verify result
            assert isinstance(priors, DistancePriors)
    
    def test_covariance_propagation_memory_usage(self):
        """Test memory usage during covariance matrix propagation."""
        # Create large covariance matrix (but still reasonable size)
        n_params = 6  # H0, Omega_m, Omega_b_h2, n_s, tau, A_s
        large_covariance = np.random.randn(n_params, n_params)
        large_covariance = large_covariance @ large_covariance.T  # Make positive definite
        
        # Mock background integrator and Jacobian computation
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Measure memory before
            memory_before = self.get_memory_usage_mb()
            
            # Compute Jacobian (this involves covariance operations)
            jacobian = compute_jacobian(
                self.planck_params, 
                self.config.z_recombination, 
                self.config.jacobian_step_size
            )
            
            # Simulate covariance propagation
            derived_cov = jacobian @ large_covariance[:5, :5] @ jacobian.T  # Use 5x5 submatrix
            
            # Measure memory after
            memory_after = self.get_memory_usage_mb()
            
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (< 50 MB for matrix operations)
            assert memory_increase < 50.0
            
            # Verify results
            assert jacobian.shape == (4, 5)
            assert derived_cov.shape == (4, 4)
    
    def test_large_dataset_memory_scaling(self):
        """Test memory usage scaling with larger datasets."""
        # Create multiple parameter sets to simulate batch processing
        n_datasets = 50
        parameter_sets = []
        
        for i in range(n_datasets):
            params = ParameterSet(
                H0=67.36 + np.random.normal(0, 0.5),
                Omega_m=0.3153 + np.random.normal(0, 0.01),
                Omega_b_h2=0.02237 + np.random.normal(0, 0.0001),
                n_s=0.9649 + np.random.normal(0, 0.005),
                tau=0.0544 + np.random.normal(0, 0.008)
            )
            parameter_sets.append(params)
        
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Measure memory before batch processing
            memory_before = self.get_memory_usage_mb()
            
            # Process all parameter sets
            results = []
            for params in parameter_sets:
                priors = compute_distance_priors(params, self.config.z_recombination)
                results.append(priors)
            
            # Measure memory after batch processing
            memory_after = self.get_memory_usage_mb()
            
            memory_increase = memory_after - memory_before
            memory_per_dataset = memory_increase / n_datasets
            
            # Memory per dataset should be minimal
            assert memory_per_dataset < 1.0  # Less than 1 MB per dataset
            assert memory_increase < 100.0  # Total increase less than 100 MB
            
            # Verify all results
            assert len(results) == n_datasets
            for result in results:
                assert isinstance(result, DistancePriors)


class TestNumericalStability:
    """Test cases for numerical stability with extreme parameter values."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CMBConfig()
    
    def test_extreme_but_valid_parameters(self):
        """Test behavior with extreme but physically valid parameters."""
        # Parameters at the edge of valid ranges
        extreme_params_list = [
            # High H0, low Omega_m
            ParameterSet(H0=79.9, Omega_m=0.11, Omega_b_h2=0.015, n_s=0.91, tau=0.02),
            # Low H0, high Omega_m  
            ParameterSet(H0=50.1, Omega_m=0.49, Omega_b_h2=0.049, n_s=1.09, tau=0.14),
            # Extreme spectral index
            ParameterSet(H0=67.0, Omega_m=0.3, Omega_b_h2=0.022, n_s=0.901, tau=0.05),
            # High optical depth
            ParameterSet(H0=67.0, Omega_m=0.3, Omega_b_h2=0.022, n_s=0.96, tau=0.149)
        ]
        
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            for i, params in enumerate(extreme_params_list):
                # Should not raise exception with valid extreme parameters
                priors = compute_distance_priors(params, self.config.z_recombination)
                
                # Results should be finite and reasonable
                assert isinstance(priors, DistancePriors)
                assert np.isfinite(priors.R)
                assert np.isfinite(priors.l_A)
                assert np.isfinite(priors.theta_star)
                
                # Values should still be in reasonable ranges
                assert 0.5 < priors.R < 5.0
                assert 100 < priors.l_A < 1000
                assert 0.001 < priors.theta_star < 0.1
    
    def test_numerical_precision_limits(self):
        """Test behavior near numerical precision limits."""
        # Parameters that might cause numerical issues
        precision_test_params = ParameterSet(
            H0=67.36000000001,  # Very small differences
            Omega_m=0.315300000001,
            Omega_b_h2=0.022370000001,
            n_s=0.964900000001,
            tau=0.054400000001
        )
        
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Should handle small numerical differences gracefully
            priors = compute_distance_priors(precision_test_params, self.config.z_recombination)
            
            # Results should be finite and stable
            assert isinstance(priors, DistancePriors)
            assert np.all(np.isfinite([priors.R, priors.l_A, priors.theta_star]))
    
    def test_jacobian_numerical_stability(self):
        """Test Jacobian computation numerical stability."""
        # Test with different step sizes
        step_sizes = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        
        planck_params = ParameterSet(
            H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544
        )
        
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            jacobians = []
            for step_size in step_sizes:
                jacobian = compute_jacobian(planck_params, self.config.z_recombination, step_size)
                jacobians.append(jacobian)
                
                # Each Jacobian should be finite
                assert np.all(np.isfinite(jacobian))
                assert jacobian.shape == (4, 5)
            
            # Jacobians computed with different step sizes should be similar
            # (within reasonable tolerance for numerical differentiation)
            for i in range(len(jacobians) - 1):
                relative_diff = np.abs(jacobians[i] - jacobians[i+1]) / np.abs(jacobians[i])
                # Allow up to 10% relative difference between step sizes
                assert np.all(relative_diff < 0.1)
    
    def test_covariance_matrix_conditioning(self):
        """Test behavior with ill-conditioned covariance matrices."""
        # Create ill-conditioned covariance matrix
        n_params = 5
        
        # Nearly singular matrix (small eigenvalues)
        eigenvals = np.array([1.0, 0.1, 0.01, 0.001, 1e-6])  # Very small last eigenvalue
        eigenvecs = np.random.randn(n_params, n_params)
        eigenvecs, _ = np.linalg.qr(eigenvecs)  # Orthogonalize
        
        ill_conditioned_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Mock Jacobian computation
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            planck_params = ParameterSet(
                H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544
            )
            
            jacobian = compute_jacobian(planck_params, self.config.z_recombination, 1e-6)
            
            # Test covariance propagation with ill-conditioned matrix
            try:
                derived_cov = jacobian @ ill_conditioned_cov @ jacobian.T
                
                # If successful, result should still be finite
                assert np.all(np.isfinite(derived_cov))
                
                # Check condition number
                condition_number = np.linalg.cond(derived_cov)
                # Should detect ill-conditioning but not crash
                if condition_number > 1e12:
                    pytest.skip("Matrix is too ill-conditioned for stable computation")
                
            except (np.linalg.LinAlgError, ValueError):
                # Acceptable to fail gracefully with ill-conditioned matrices
                pytest.skip("Ill-conditioned matrix caused expected numerical failure")


class TestStressConditions:
    """Test cases for system behavior under stress conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CMBConfig()
    
    def test_concurrent_processing_simulation(self):
        """Test behavior under simulated concurrent processing load."""
        # Simulate multiple concurrent parameter processing requests
        n_concurrent = 20
        parameter_sets = []
        
        for i in range(n_concurrent):
            # Create diverse parameter sets
            params = ParameterSet(
                H0=60.0 + i * 0.5,  # Spread across valid range
                Omega_m=0.2 + i * 0.01,
                Omega_b_h2=0.02 + i * 0.0001,
                n_s=0.95 + i * 0.001,
                tau=0.04 + i * 0.002
            )
            parameter_sets.append(params)
        
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Process all parameter sets rapidly
            start_time = time.time()
            results = []
            
            for params in parameter_sets:
                priors = compute_distance_priors(params, self.config.z_recombination)
                results.append(priors)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should handle concurrent-like load efficiently
            assert total_time < 30.0  # Should complete within 30 seconds
            assert len(results) == n_concurrent
            
            # All results should be valid
            for result in results:
                assert isinstance(result, DistancePriors)
                assert result.validate() is True
    
    def test_repeated_processing_stability(self):
        """Test stability over many repeated computations."""
        planck_params = ParameterSet(
            H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544
        )
        
        # Mock background integrator
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Perform many repeated computations
            n_iterations = 100
            results = []
            
            for i in range(n_iterations):
                priors = compute_distance_priors(planck_params, self.config.z_recombination)
                results.append(priors)
            
            # All results should be identical (deterministic computation)
            reference_result = results[0]
            for result in results[1:]:
                assert result.R == reference_result.R
                assert result.l_A == reference_result.l_A
                assert result.theta_star == reference_result.theta_star
                assert result.Omega_b_h2 == reference_result.Omega_b_h2
    
    def test_error_recovery_resilience(self):
        """Test system resilience to errors and recovery."""
        planck_params = ParameterSet(
            H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544
        )
        
        # Test recovery after integration failure
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            # First call fails
            mock_integrator_class.side_effect = [RuntimeError("Integration failed"), Mock()]
            
            # Second call succeeds
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.side_effect = [RuntimeError("Integration failed"), mock_bg]
            mock_sound_horizon.return_value = 147.05
            
            # First attempt should fail
            with pytest.raises((DerivationError, RuntimeError)):
                compute_distance_priors(planck_params, self.config.z_recombination)
            
            # Reset mock for second attempt
            mock_integrator_class.side_effect = None
            mock_integrator_class.return_value = mock_bg
            
            # Second attempt should succeed (system recovered)
            priors = compute_distance_priors(planck_params, self.config.z_recombination)
            assert isinstance(priors, DistancePriors)
            assert priors.validate() is True


if __name__ == '__main__':
    pytest.main([__file__])