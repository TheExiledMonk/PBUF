"""
Performance and stress tests for CMB raw parameter processing.

This module implements comprehensive performance testing and stress testing
for the CMB raw parameter integration feature, covering:
- Processing time benchmarks with typical Planck parameter datasets
- Memory usage monitoring during covariance propagation
- Numerical stability testing with extreme parameter values
- System behavior under stress conditions and concurrent processing
- Resource usage optimization and scalability testing

Requirements covered: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
import numpy as np
import time
import psutil
import os
import gc
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple
import tempfile
from pathlib import Path
import json
import sys
from contextlib import contextmanager
import tracemalloc

# Import CMB processing modules
from pipelines.data_preparation.derivation.cmb_derivation import (
    compute_distance_priors, compute_jacobian, process_cmb_dataset,
    parse_parameter_file, extract_covariance_matrix, ParameterFormat
)
from pipelines.data_preparation.derivation.cmb_models import (
    ParameterSet, DistancePriors, CMBConfig
)
from pipelines.data_preparation.derivation.cmb_exceptions import (
    DerivationError, NumericalInstabilityError, CovarianceError
)


class PerformanceMonitor:
    """Utility class for monitoring system performance during tests."""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process(os.getpid())
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.get_memory_usage_mb()
        self.peak_memory = self.start_memory
        tracemalloc.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return performance metrics."""
        end_time = time.time()
        end_memory = self.get_memory_usage_mb()
        
        # Get tracemalloc statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'elapsed_time': end_time - self.start_time,
            'memory_start_mb': self.start_memory,
            'memory_end_mb': end_memory,
            'memory_peak_mb': max(self.peak_memory, peak / 1024 / 1024),
            'memory_increase_mb': end_memory - self.start_memory,
            'tracemalloc_current_mb': current / 1024 / 1024,
            'tracemalloc_peak_mb': peak / 1024 / 1024
        }
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.get_memory_usage_mb()
        self.peak_memory = max(self.peak_memory, current_memory)


@contextmanager
def performance_monitor():
    """Context manager for performance monitoring."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        metrics = monitor.stop_monitoring()
        # Store metrics for test reporting
        if hasattr(pytest, 'current_test_metrics'):
            pytest.current_test_metrics.update(metrics)


class TestProcessingTimePerformance:
    """Test cases for processing time performance with typical Planck datasets."""
    
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
            jacobian_step_size=1e-5,  # Balanced accuracy/speed
            validation_tolerance=1e-7,
            cache_computations=True,
            performance_monitoring=True
        )
        
        # Create mock background integrator for consistent timing
        self.mock_bg = Mock()
        self.mock_bg.comoving_distance.return_value = 14026.0
        self.mock_bg.angular_diameter_distance.return_value = 12.86
    
    def test_single_parameter_set_processing_time(self):
        """Test processing time for single Planck parameter set (Requirement 8.1)."""
        with performance_monitor() as monitor:
            with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                 patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                
                mock_integrator_class.return_value = self.mock_bg
                mock_sound_horizon.return_value = 147.05
                
                # Process single parameter set
                priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
        
        metrics = monitor.stop_monitoring()
        
        # Should complete within 10 seconds (Requirement 8.1)
        assert metrics['elapsed_time'] < 10.0, f"Processing took {metrics['elapsed_time']:.2f}s, expected < 10s"
        
        # Verify result validity
        assert isinstance(priors, DistancePriors)
        assert priors.validate() is True
        
        # Log performance for monitoring
        print(f"Single parameter set processing: {metrics['elapsed_time']:.3f}s")
    
    def test_jacobian_computation_performance(self):
        """Test Jacobian computation performance (Requirement 8.1)."""
        with performance_monitor() as monitor:
            with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                 patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                
                mock_integrator_class.return_value = self.mock_bg
                mock_sound_horizon.return_value = 147.05
                
                # Compute Jacobian matrix
                jacobian = compute_jacobian(
                    self.planck_params, 
                    self.config.z_recombination, 
                    self.config.jacobian_step_size
                )
        
        metrics = monitor.stop_monitoring()
        
        # Jacobian computation should complete within reasonable time
        assert metrics['elapsed_time'] < 30.0, f"Jacobian computation took {metrics['elapsed_time']:.2f}s, expected < 30s"
        
        # Verify result properties
        assert jacobian.shape == (4, 6)  # 4 observables, 6 parameters (including A_s)
        assert np.all(np.isfinite(jacobian))
        
        print(f"Jacobian computation: {metrics['elapsed_time']:.3f}s")
    
    def test_batch_processing_performance(self):
        """Test performance with batch processing of multiple parameter sets."""
        # Create batch of parameter sets (typical MCMC chain size)
        n_sets = 100
        parameter_sets = []
        
        for i in range(n_sets):
            # Small variations around Planck values
            params = ParameterSet(
                H0=67.36 + np.random.normal(0, 0.5),
                Omega_m=0.3153 + np.random.normal(0, 0.01),
                Omega_b_h2=0.02237 + np.random.normal(0, 0.0001),
                n_s=0.9649 + np.random.normal(0, 0.005),
                tau=0.0544 + np.random.normal(0, 0.008)
            )
            parameter_sets.append(params)
        
        with performance_monitor() as monitor:
            with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                 patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                
                mock_integrator_class.return_value = self.mock_bg
                mock_sound_horizon.return_value = 147.05
                
                # Process all parameter sets
                results = []
                for params in parameter_sets:
                    priors = compute_distance_priors(params, self.config.z_recombination)
                    results.append(priors)
        
        metrics = monitor.stop_monitoring()
        
        # Calculate performance metrics
        total_time = metrics['elapsed_time']
        avg_time_per_set = total_time / n_sets
        
        # Performance requirements
        assert avg_time_per_set < 1.0, f"Average time per set: {avg_time_per_set:.3f}s, expected < 1.0s"
        assert total_time < 300.0, f"Total batch processing: {total_time:.2f}s, expected < 300s"
        
        # Verify all results
        assert len(results) == n_sets
        for result in results:
            assert isinstance(result, DistancePriors)
            assert result.validate() is True
        
        print(f"Batch processing ({n_sets} sets): {total_time:.2f}s, avg: {avg_time_per_set:.3f}s per set")
    
    def test_large_covariance_matrix_performance(self):
        """Test performance with large covariance matrices."""
        # Create large covariance matrix (extended parameter space)
        n_params = 20  # Extended cosmological parameter set
        large_covariance = np.random.randn(n_params, n_params)
        large_covariance = large_covariance @ large_covariance.T  # Make positive definite
        
        # Create extended Jacobian (mock)
        jacobian = np.random.randn(4, n_params)  # 4 observables, 20 parameters
        
        with performance_monitor() as monitor:
            # Simulate covariance propagation
            for _ in range(10):  # Multiple iterations to test consistency
                derived_cov = jacobian @ large_covariance @ jacobian.T
                
                # Validate matrix properties
                assert derived_cov.shape == (4, 4)
                assert np.allclose(derived_cov, derived_cov.T)  # Symmetry
        
        metrics = monitor.stop_monitoring()
        
        # Should handle large matrices efficiently
        assert metrics['elapsed_time'] < 5.0, f"Large covariance processing: {metrics['elapsed_time']:.2f}s, expected < 5s"
        
        print(f"Large covariance matrix processing: {metrics['elapsed_time']:.3f}s")
    
    def test_parameter_file_parsing_performance(self):
        """Test performance of parameter file parsing across different formats."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create test files in different formats
            test_params = {
                'H0': 67.36,
                'Omega_m': 0.3153,
                'Omega_b_h2': 0.02237,
                'n_s': 0.9649,
                'tau': 0.0544
            }
            
            # JSON file
            json_file = temp_dir / 'params.json'
            with open(json_file, 'w') as f:
                json.dump(test_params, f)
            
            # NumPy file
            npy_file = temp_dir / 'params.npy'
            param_array = np.array([67.36, 0.3153, 0.02237, 0.9649, 0.0544])
            np.save(npy_file, param_array)
            
            # Test parsing performance for each format
            formats_to_test = [
                (json_file, ParameterFormat.JSON),
                (npy_file, ParameterFormat.NUMPY)
            ]
            
            for file_path, format_type in formats_to_test:
                with performance_monitor() as monitor:
                    for _ in range(50):  # Multiple parses to test consistency
                        params = parse_parameter_file(str(file_path), format_type)
                        assert isinstance(params, ParameterSet)
                
                metrics = monitor.stop_monitoring()
                
                # Parsing should be very fast
                avg_parse_time = metrics['elapsed_time'] / 50
                assert avg_parse_time < 0.1, f"Average parse time for {format_type}: {avg_parse_time:.4f}s, expected < 0.1s"
                
                print(f"Parameter parsing ({format_type.value}): {avg_parse_time:.4f}s per file")
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestMemoryUsageMonitoring:
    """Test cases for memory usage monitoring during covariance propagation (Requirement 8.2)."""
    
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
        
        # Force garbage collection before tests
        gc.collect()
    
    def test_distance_prior_memory_usage(self):
        """Test memory usage during distance prior computation (Requirement 8.2)."""
        with performance_monitor() as monitor:
            with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                 patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                
                mock_bg = Mock()
                mock_bg.comoving_distance.return_value = 14026.0
                mock_integrator_class.return_value = mock_bg
                mock_sound_horizon.return_value = 147.05
                
                # Perform computation
                priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
        
        metrics = monitor.stop_monitoring()
        
        # Memory increase should be minimal for single computation
        assert metrics['memory_increase_mb'] < 50.0, f"Memory increase: {metrics['memory_increase_mb']:.2f}MB, expected < 50MB"
        
        # Verify result
        assert isinstance(priors, DistancePriors)
        
        print(f"Distance prior computation memory: {metrics['memory_increase_mb']:.2f}MB")
    
    def test_covariance_propagation_memory_scaling(self):
        """Test memory usage scaling during covariance matrix propagation."""
        # Test with different matrix sizes
        matrix_sizes = [5, 10, 20, 50]
        memory_usage = []
        
        for n_params in matrix_sizes:
            # Create covariance matrix
            covariance = np.random.randn(n_params, n_params)
            covariance = covariance @ covariance.T
            
            # Create Jacobian
            jacobian = np.random.randn(4, n_params)
            
            with performance_monitor() as monitor:
                # Perform covariance propagation
                for _ in range(10):  # Multiple iterations
                    derived_cov = jacobian @ covariance @ jacobian.T
                    
                    # Force computation to complete
                    _ = np.sum(derived_cov)
            
            metrics = monitor.stop_monitoring()
            memory_usage.append((n_params, metrics['memory_increase_mb']))
            
            # Memory usage should remain under 1GB for standard datasets (Requirement 8.2)
            assert metrics['memory_increase_mb'] < 1024.0, f"Memory usage for {n_params}x{n_params} matrix: {metrics['memory_increase_mb']:.2f}MB, expected < 1024MB"
        
        # Check memory scaling is reasonable (should be roughly O(n²))
        for i in range(1, len(memory_usage)):
            prev_size, prev_memory = memory_usage[i-1]
            curr_size, curr_memory = memory_usage[i]
            
            # Memory should scale reasonably with matrix size
            size_ratio = (curr_size / prev_size) ** 2
            memory_ratio = curr_memory / max(prev_memory, 1.0)  # Avoid division by zero
            
            # Allow some overhead but shouldn't be excessive
            assert memory_ratio < size_ratio * 2, f"Memory scaling too aggressive: {memory_ratio:.2f} vs expected ~{size_ratio:.2f}"
        
        print(f"Memory scaling test: {memory_usage}")
    
    def test_batch_processing_memory_efficiency(self):
        """Test memory efficiency during batch processing."""
        # Create large batch of parameter sets
        n_sets = 200
        parameter_sets = []
        
        for i in range(n_sets):
            params = ParameterSet(
                H0=67.36 + np.random.normal(0, 0.5),
                Omega_m=0.3153 + np.random.normal(0, 0.01),
                Omega_b_h2=0.02237 + np.random.normal(0, 0.0001),
                n_s=0.9649 + np.random.normal(0, 0.005),
                tau=0.0544 + np.random.normal(0, 0.008)
            )
            parameter_sets.append(params)
        
        with performance_monitor() as monitor:
            with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                 patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                
                mock_bg = Mock()
                mock_bg.comoving_distance.return_value = 14026.0
                mock_integrator_class.return_value = mock_bg
                mock_sound_horizon.return_value = 147.05
                
                # Process with memory monitoring
                results = []
                for i, params in enumerate(parameter_sets):
                    priors = compute_distance_priors(params, self.config.z_recombination)
                    results.append(priors)
                    
                    # Monitor memory every 50 iterations
                    if i % 50 == 0:
                        monitor.update_peak_memory()
        
        metrics = monitor.stop_monitoring()
        
        # Memory per dataset should be reasonable
        memory_per_dataset = metrics['memory_increase_mb'] / n_sets
        assert memory_per_dataset < 5.0, f"Memory per dataset: {memory_per_dataset:.3f}MB, expected < 5MB"
        
        # Total memory increase should be reasonable
        assert metrics['memory_increase_mb'] < 500.0, f"Total memory increase: {metrics['memory_increase_mb']:.2f}MB, expected < 500MB"
        
        print(f"Batch processing memory efficiency: {memory_per_dataset:.3f}MB per dataset")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated processing."""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Perform many iterations to detect leaks
            for i in range(100):
                params = ParameterSet(
                    H0=67.36 + np.random.normal(0, 0.1),
                    Omega_m=0.3153 + np.random.normal(0, 0.001),
                    Omega_b_h2=0.02237 + np.random.normal(0, 0.00001),
                    n_s=0.9649 + np.random.normal(0, 0.001),
                    tau=0.0544 + np.random.normal(0, 0.001)
                )
                
                priors = compute_distance_priors(params, self.config.z_recombination)
                
                # Explicitly delete to help garbage collection
                del priors
                
                # Force garbage collection every 20 iterations
                if i % 20 == 0:
                    gc.collect()
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (no significant leaks)
        assert memory_increase < 100.0, f"Potential memory leak detected: {memory_increase:.2f}MB increase after 100 iterations"
        
        print(f"Memory leak test: {memory_increase:.2f}MB increase over 100 iterations")


class TestNumericalStabilityExtreme:
    """Test cases for numerical stability with extreme parameter values (Requirement 8.3)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CMBConfig()
    
    def test_extreme_valid_parameter_ranges(self):
        """Test behavior with extreme but physically valid parameters."""
        # Parameters at the edge of valid ranges (within validation bounds)
        extreme_cases = [
            # Case 1: High H0, low Omega_m
            {
                'name': 'high_H0_low_Om',
                'params': ParameterSet(H0=79.5, Omega_m=0.16, Omega_b_h2=0.018, n_s=0.92, tau=0.03),
                'expected_ranges': {'R': (0.5, 3.0), 'l_A': (200, 500), 'theta_star': (0.005, 0.05)}
            },
            # Case 2: Low H0, high Omega_m  
            {
                'name': 'low_H0_high_Om',
                'params': ParameterSet(H0=51.0, Omega_m=0.44, Omega_b_h2=0.025, n_s=1.08, tau=0.13),
                'expected_ranges': {'R': (1.0, 5.0), 'l_A': (100, 400), 'theta_star': (0.005, 0.05)}
            },
            # Case 3: Extreme spectral index
            {
                'name': 'extreme_ns',
                'params': ParameterSet(H0=67.0, Omega_m=0.3, Omega_b_h2=0.022, n_s=0.91, tau=0.05),
                'expected_ranges': {'R': (1.0, 3.0), 'l_A': (200, 400), 'theta_star': (0.005, 0.05)}
            },
            # Case 4: High optical depth
            {
                'name': 'high_tau',
                'params': ParameterSet(H0=67.0, Omega_m=0.3, Omega_b_h2=0.022, n_s=0.96, tau=0.14),
                'expected_ranges': {'R': (1.0, 3.0), 'l_A': (200, 400), 'theta_star': (0.005, 0.05)}
            }
        ]
        
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            for case in extreme_cases:
                params = case['params']
                expected_ranges = case['expected_ranges']
                
                # Should not raise exception with valid extreme parameters
                priors = compute_distance_priors(params, self.config.z_recombination)
                
                # Results should be finite and within expected ranges
                assert isinstance(priors, DistancePriors)
                assert np.isfinite(priors.R), f"R not finite for case {case['name']}"
                assert np.isfinite(priors.l_A), f"l_A not finite for case {case['name']}"
                assert np.isfinite(priors.theta_star), f"theta_star not finite for case {case['name']}"
                
                # Check values are in reasonable ranges
                r_min, r_max = expected_ranges['R']
                assert r_min < priors.R < r_max, f"R={priors.R:.3f} outside range [{r_min}, {r_max}] for case {case['name']}"
                
                la_min, la_max = expected_ranges['l_A']
                assert la_min < priors.l_A < la_max, f"l_A={priors.l_A:.3f} outside range [{la_min}, {la_max}] for case {case['name']}"
                
                theta_min, theta_max = expected_ranges['theta_star']
                assert theta_min < priors.theta_star < theta_max, f"theta_star={priors.theta_star:.6f} outside range [{theta_min}, {theta_max}] for case {case['name']}"
                
                print(f"Extreme case {case['name']}: R={priors.R:.3f}, l_A={priors.l_A:.1f}, θ*={priors.theta_star:.6f}")
    
    def test_numerical_precision_limits(self):
        """Test behavior near numerical precision limits."""
        # Parameters with very small differences (testing numerical precision)
        base_params = ParameterSet(
            H0=67.36,
            Omega_m=0.3153,
            Omega_b_h2=0.02237,
            n_s=0.9649,
            tau=0.0544
        )
        
        # Create variations at different precision levels
        precision_levels = [1e-6, 1e-8, 1e-10, 1e-12]
        
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            base_priors = compute_distance_priors(base_params, self.config.z_recombination)
            
            for precision in precision_levels:
                # Create slightly perturbed parameters
                perturbed_params = ParameterSet(
                    H0=base_params.H0 + precision,
                    Omega_m=base_params.Omega_m + precision,
                    Omega_b_h2=base_params.Omega_b_h2 + precision,
                    n_s=base_params.n_s + precision,
                    tau=base_params.tau + precision
                )
                
                # Should handle small numerical differences gracefully
                priors = compute_distance_priors(perturbed_params, self.config.z_recombination)
                
                # Results should be finite and stable
                assert isinstance(priors, DistancePriors)
                assert np.all(np.isfinite([priors.R, priors.l_A, priors.theta_star]))
                
                # Differences should be small and proportional to input precision
                r_diff = abs(priors.R - base_priors.R)
                la_diff = abs(priors.l_A - base_priors.l_A)
                theta_diff = abs(priors.theta_star - base_priors.theta_star)
                
                # Differences should be reasonable (not amplified excessively)
                assert r_diff < 1.0, f"R difference too large at precision {precision}: {r_diff}"
                assert la_diff < 10.0, f"l_A difference too large at precision {precision}: {la_diff}"
                assert theta_diff < 0.01, f"theta_star difference too large at precision {precision}: {theta_diff}"
                
                print(f"Precision {precision}: ΔR={r_diff:.2e}, Δl_A={la_diff:.2e}, Δθ*={theta_diff:.2e}")
    
    def test_jacobian_numerical_stability_comprehensive(self):
        """Test Jacobian computation numerical stability across parameter space."""
        # Test with different step sizes and parameter values
        step_sizes = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        
        # Test parameters spanning the valid range
        test_parameter_sets = [
            # Standard Planck
            ParameterSet(H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544),
            # High H0 case
            ParameterSet(H0=75.0, Omega_m=0.25, Omega_b_h2=0.020, n_s=0.97, tau=0.06),
            # Low H0 case
            ParameterSet(H0=60.0, Omega_m=0.40, Omega_b_h2=0.025, n_s=0.95, tau=0.04)
        ]
        
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            for params in test_parameter_sets:
                jacobians = []
                
                for step_size in step_sizes:
                    jacobian = compute_jacobian(params, self.config.z_recombination, step_size)
                    jacobians.append(jacobian)
                    
                    # Each Jacobian should be finite
                    assert np.all(np.isfinite(jacobian)), f"Non-finite Jacobian at step size {step_size}"
                    assert jacobian.shape == (4, 5), f"Wrong Jacobian shape: {jacobian.shape}"
                
                # Check consistency across step sizes
                for i in range(len(jacobians) - 1):
                    j1, j2 = jacobians[i], jacobians[i+1]
                    
                    # Compute relative differences
                    rel_diff = np.abs(j1 - j2) / (np.abs(j1) + 1e-10)  # Avoid division by zero
                    
                    # Most elements should be consistent within reasonable tolerance
                    consistent_elements = np.sum(rel_diff < 0.1)
                    total_elements = rel_diff.size
                    consistency_ratio = consistent_elements / total_elements
                    
                    assert consistency_ratio > 0.8, f"Jacobian inconsistency: only {consistency_ratio:.2f} of elements consistent"
                
                print(f"Jacobian stability test passed for params: H0={params.H0}, Ωm={params.Omega_m}")
    
    def test_ill_conditioned_covariance_handling(self):
        """Test behavior with ill-conditioned covariance matrices."""
        # Create matrices with different condition numbers
        condition_numbers = [1e3, 1e6, 1e9, 1e12]
        
        for cond_num in condition_numbers:
            # Create ill-conditioned matrix with specified condition number
            n_params = 5
            U, _, Vt = np.linalg.svd(np.random.randn(n_params, n_params))
            
            # Set singular values to create desired condition number
            singular_values = np.logspace(0, -np.log10(cond_num), n_params)
            S = np.diag(singular_values)
            
            ill_conditioned_cov = U @ S @ Vt
            
            # Create mock Jacobian
            jacobian = np.random.randn(4, n_params)
            
            try:
                # Test covariance propagation
                derived_cov = jacobian @ ill_conditioned_cov @ jacobian.T
                
                # If successful, result should still be finite
                assert np.all(np.isfinite(derived_cov)), f"Non-finite result with condition number {cond_num:.1e}"
                
                # Check that result is still symmetric
                assert np.allclose(derived_cov, derived_cov.T, atol=1e-10), f"Lost symmetry with condition number {cond_num:.1e}"
                
                # Verify condition number of result
                result_cond = np.linalg.cond(derived_cov)
                print(f"Condition number {cond_num:.1e} -> {result_cond:.1e}")
                
            except (np.linalg.LinAlgError, ValueError) as e:
                # Acceptable to fail gracefully with very ill-conditioned matrices
                if cond_num > 1e10:
                    print(f"Expected failure with condition number {cond_num:.1e}: {str(e)}")
                else:
                    raise AssertionError(f"Unexpected failure with condition number {cond_num:.1e}: {str(e)}")


class TestStressConditions:
    """Test cases for system behavior under stress conditions (Requirements 8.4, 8.5)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CMBConfig()
    
    def test_concurrent_processing_simulation(self):
        """Test behavior under simulated concurrent processing load."""
        # Simulate multiple concurrent parameter processing requests
        n_concurrent = 50
        parameter_sets = []
        
        for i in range(n_concurrent):
            # Create diverse parameter sets with valid combinations
            # Use smaller variations to stay within validation bounds
            params = ParameterSet(
                H0=67.36 + (i - n_concurrent/2) * 0.2,  # ±5 km/s/Mpc around Planck
                Omega_m=0.315 + (i - n_concurrent/2) * 0.002,  # ±0.05 around Planck
                Omega_b_h2=0.02237 + (i - n_concurrent/2) * 0.00005,  # Small variations
                n_s=0.9649 + (i - n_concurrent/2) * 0.001,  # Small variations
                tau=0.0544 + (i - n_concurrent/2) * 0.001  # Small variations
            )
            parameter_sets.append(params)
        
        with performance_monitor() as monitor:
            with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                 patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                
                mock_bg = Mock()
                mock_bg.comoving_distance.return_value = 14026.0
                mock_integrator_class.return_value = mock_bg
                mock_sound_horizon.return_value = 147.05
                
                # Process all parameter sets rapidly (simulating concurrent load)
                results = []
                for params in parameter_sets:
                    priors = compute_distance_priors(params, self.config.z_recombination)
                    results.append(priors)
        
        metrics = monitor.stop_monitoring()
        
        # Should handle concurrent-like load efficiently
        total_time = metrics['elapsed_time']
        assert total_time < 120.0, f"Concurrent processing took {total_time:.2f}s, expected < 120s"
        assert len(results) == n_concurrent
        
        # All results should be valid
        for i, result in enumerate(results):
            assert isinstance(result, DistancePriors), f"Invalid result type at index {i}"
            assert result.validate() is True, f"Invalid result at index {i}"
        
        # Check for result diversity (parameters were different)
        r_values = [r.R for r in results]
        assert len(set(np.round(r_values, 3))) > n_concurrent * 0.8, "Results not sufficiently diverse"
        
        print(f"Concurrent processing simulation: {total_time:.2f}s for {n_concurrent} parameter sets")
    
    def test_repeated_processing_stability(self):
        """Test stability over many repeated computations."""
        planck_params = ParameterSet(
            H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544
        )
        
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Perform many repeated computations
            n_iterations = 500
            results = []
            
            with performance_monitor() as monitor:
                for i in range(n_iterations):
                    priors = compute_distance_priors(planck_params, self.config.z_recombination)
                    results.append(priors)
                    
                    # Monitor memory periodically
                    if i % 100 == 0:
                        monitor.update_peak_memory()
            
            metrics = monitor.stop_monitoring()
        
        # All results should be identical (deterministic computation)
        reference_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result.R == reference_result.R, f"R mismatch at iteration {i}"
            assert result.l_A == reference_result.l_A, f"l_A mismatch at iteration {i}"
            assert result.theta_star == reference_result.theta_star, f"theta_star mismatch at iteration {i}"
            assert result.Omega_b_h2 == reference_result.Omega_b_h2, f"Omega_b_h2 mismatch at iteration {i}"
        
        # Performance should remain stable
        avg_time_per_iteration = metrics['elapsed_time'] / n_iterations
        assert avg_time_per_iteration < 0.1, f"Average time per iteration: {avg_time_per_iteration:.4f}s, expected < 0.1s"
        
        # Memory usage should be stable (no significant growth)
        assert metrics['memory_increase_mb'] < 200.0, f"Memory increase: {metrics['memory_increase_mb']:.2f}MB, expected < 200MB"
        
        print(f"Stability test: {n_iterations} iterations, {avg_time_per_iteration:.4f}s avg, {metrics['memory_increase_mb']:.2f}MB memory")
    
    def test_error_recovery_resilience(self):
        """Test system resilience to errors and recovery."""
        planck_params = ParameterSet(
            H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544
        )
        
        # Test recovery after various types of failures
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            # Test 1: Integration failure followed by success
            mock_integrator_class.side_effect = [
                RuntimeError("Integration failed"),  # First call fails
                Mock()  # Second call succeeds
            ]
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
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
            
            print("Error recovery test: Successfully recovered from integration failure")
    
    def test_resource_exhaustion_handling(self):
        """Test behavior under simulated resource exhaustion."""
        # Test with progressively larger workloads to approach resource limits
        workload_sizes = [10, 50, 100, 200]
        
        for n_sets in workload_sizes:
            # Create parameter sets
            parameter_sets = []
            for i in range(n_sets):
                params = ParameterSet(
                    H0=67.36 + np.random.normal(0, 0.5),
                    Omega_m=0.3153 + np.random.normal(0, 0.01),
                    Omega_b_h2=0.02237 + np.random.normal(0, 0.0001),
                    n_s=0.9649 + np.random.normal(0, 0.005),
                    tau=0.0544 + np.random.normal(0, 0.008)
                )
                parameter_sets.append(params)
            
            try:
                with performance_monitor() as monitor:
                    with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                         patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                        
                        mock_bg = Mock()
                        mock_bg.comoving_distance.return_value = 14026.0
                        mock_integrator_class.return_value = mock_bg
                        mock_sound_horizon.return_value = 147.05
                        
                        # Process with resource monitoring
                        results = []
                        for params in parameter_sets:
                            priors = compute_distance_priors(params, self.config.z_recombination)
                            results.append(priors)
                
                metrics = monitor.stop_monitoring()
                
                # Verify all results are valid
                assert len(results) == n_sets
                for result in results:
                    assert isinstance(result, DistancePriors)
                    assert result.validate() is True
                
                # Check resource usage is reasonable
                memory_per_set = metrics['memory_increase_mb'] / n_sets
                time_per_set = metrics['elapsed_time'] / n_sets
                
                print(f"Workload {n_sets}: {time_per_set:.4f}s/set, {memory_per_set:.3f}MB/set")
                
                # Resource usage should scale reasonably
                assert memory_per_set < 10.0, f"Memory per set too high: {memory_per_set:.3f}MB"
                assert time_per_set < 1.0, f"Time per set too high: {time_per_set:.4f}s"
                
            except MemoryError:
                # If we hit memory limits, that's expected for very large workloads
                print(f"Memory limit reached at workload size {n_sets}")
                break
            except Exception as e:
                # Other exceptions should not occur under normal stress
                raise AssertionError(f"Unexpected error at workload size {n_sets}: {str(e)}")
    
    def test_multithreaded_processing_safety(self):
        """Test thread safety of CMB processing functions."""
        n_threads = 4
        n_iterations_per_thread = 25
        
        # Shared results storage (thread-safe)
        results_lock = threading.Lock()
        all_results = []
        errors = []
        
        def worker_function(thread_id):
            """Worker function for multithreaded processing."""
            thread_results = []
            
            try:
                with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
                     patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
                    
                    mock_bg = Mock()
                    mock_bg.comoving_distance.return_value = 14026.0
                    mock_integrator_class.return_value = mock_bg
                    mock_sound_horizon.return_value = 147.05
                    
                    for i in range(n_iterations_per_thread):
                        # Create unique parameters for each thread/iteration
                        params = ParameterSet(
                            H0=67.36 + thread_id * 0.1 + i * 0.01,
                            Omega_m=0.3153 + thread_id * 0.001 + i * 0.0001,
                            Omega_b_h2=0.02237 + thread_id * 0.00001,
                            n_s=0.9649 + thread_id * 0.0001,
                            tau=0.0544 + thread_id * 0.001
                        )
                        
                        priors = compute_distance_priors(params, 1089.8)
                        thread_results.append((thread_id, i, priors))
                
                # Store results thread-safely
                with results_lock:
                    all_results.extend(thread_results)
                    
            except Exception as e:
                with results_lock:
                    errors.append((thread_id, str(e)))
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for thread_id in range(n_threads):
            thread = threading.Thread(target=worker_function, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Check results
        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        assert len(all_results) == n_threads * n_iterations_per_thread, f"Expected {n_threads * n_iterations_per_thread} results, got {len(all_results)}"
        
        # Verify all results are valid
        for thread_id, iteration, result in all_results:
            assert isinstance(result, DistancePriors), f"Invalid result from thread {thread_id}, iteration {iteration}"
            assert result.validate() is True, f"Invalid result from thread {thread_id}, iteration {iteration}"
        
        # Check performance
        total_time = end_time - start_time
        expected_sequential_time = n_threads * n_iterations_per_thread * 0.01  # Rough estimate
        
        print(f"Multithreaded processing: {total_time:.2f}s for {n_threads} threads × {n_iterations_per_thread} iterations")
        print(f"Thread safety test passed with {len(all_results)} successful computations")


class TestPerformanceRegression:
    """Test cases for performance regression detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.planck_params = ParameterSet(
            H0=67.36, Omega_m=0.3153, Omega_b_h2=0.02237, n_s=0.9649, tau=0.0544
        )
        self.config = CMBConfig()
    
    def test_performance_benchmarks(self):
        """Establish performance benchmarks for regression testing."""
        benchmarks = {}
        
        with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class, \
             patch('pipelines.data_preparation.derivation.cmb_background.compute_sound_horizon') as mock_sound_horizon:
            
            mock_bg = Mock()
            mock_bg.comoving_distance.return_value = 14026.0
            mock_integrator_class.return_value = mock_bg
            mock_sound_horizon.return_value = 147.05
            
            # Benchmark 1: Single distance prior computation
            with performance_monitor() as monitor:
                for _ in range(100):
                    priors = compute_distance_priors(self.planck_params, self.config.z_recombination)
            
            metrics = monitor.stop_monitoring()
            benchmarks['distance_prior_avg_time'] = metrics['elapsed_time'] / 100
            benchmarks['distance_prior_memory'] = metrics['memory_increase_mb']
            
            # Benchmark 2: Jacobian computation
            with performance_monitor() as monitor:
                jacobian = compute_jacobian(self.planck_params, self.config.z_recombination, 1e-6)
            
            metrics = monitor.stop_monitoring()
            benchmarks['jacobian_time'] = metrics['elapsed_time']
            benchmarks['jacobian_memory'] = metrics['memory_increase_mb']
            
            # Benchmark 3: Covariance propagation
            covariance = np.eye(5) * 0.01
            
            with performance_monitor() as monitor:
                for _ in range(50):
                    derived_cov = jacobian @ covariance @ jacobian.T
            
            metrics = monitor.stop_monitoring()
            benchmarks['covariance_avg_time'] = metrics['elapsed_time'] / 50
            benchmarks['covariance_memory'] = metrics['memory_increase_mb']
        
        # Performance assertions (these define our performance requirements)
        assert benchmarks['distance_prior_avg_time'] < 0.1, f"Distance prior computation too slow: {benchmarks['distance_prior_avg_time']:.4f}s"
        assert benchmarks['jacobian_time'] < 5.0, f"Jacobian computation too slow: {benchmarks['jacobian_time']:.2f}s"
        assert benchmarks['covariance_avg_time'] < 0.01, f"Covariance propagation too slow: {benchmarks['covariance_avg_time']:.4f}s"
        
        assert benchmarks['distance_prior_memory'] < 10.0, f"Distance prior memory usage too high: {benchmarks['distance_prior_memory']:.2f}MB"
        assert benchmarks['jacobian_memory'] < 50.0, f"Jacobian memory usage too high: {benchmarks['jacobian_memory']:.2f}MB"
        assert benchmarks['covariance_memory'] < 20.0, f"Covariance memory usage too high: {benchmarks['covariance_memory']:.2f}MB"
        
        # Print benchmarks for reference
        print("Performance Benchmarks:")
        for key, value in benchmarks.items():
            if 'time' in key:
                print(f"  {key}: {value:.4f}s")
            elif 'memory' in key:
                print(f"  {key}: {value:.2f}MB")
        
        return benchmarks


if __name__ == '__main__':
    # Run with performance reporting
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x'  # Stop on first failure for performance tests
    ])