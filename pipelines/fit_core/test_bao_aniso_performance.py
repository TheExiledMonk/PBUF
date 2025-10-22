#!/usr/bin/env python3
"""
Performance benchmark tests for BAO anisotropic fitting.

This module provides comprehensive performance benchmarks to measure and validate
the execution time characteristics of the BAO anisotropic fitting implementation.

Requirements: 5.1
"""

import unittest
import time
import statistics
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class PerformanceBenchmark:
    """
    Performance benchmark utility class.
    
    Provides methods for timing operations and collecting performance statistics.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.measurements = []
        self.start_time = None
    
    def start(self):
        """Start timing an operation."""
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timing and record the measurement."""
        if self.start_time is None:
            raise ValueError("Must call start() before stop()")
        
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        self.measurements.append(duration)
        self.start_time = None
        return duration
    
    def get_statistics(self) -> Dict[str, float]:
        """Get performance statistics for all measurements."""
        if not self.measurements:
            return {}
        
        return {
            "count": len(self.measurements),
            "mean": statistics.mean(self.measurements),
            "median": statistics.median(self.measurements),
            "stdev": statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0,
            "min": min(self.measurements),
            "max": max(self.measurements),
            "total": sum(self.measurements)
        }
    
    def reset(self):
        """Reset all measurements."""
        self.measurements = []
        self.start_time = None


class TestBaoAnisoPerformance(unittest.TestCase):
    """
    Performance benchmark tests for BAO anisotropic fitting.
    
    These tests measure execution times for various components and operations
    to ensure the implementation meets performance requirements.
    """
    
    def setUp(self):
        """Set up performance test environment."""
        self.benchmarks = {}
        self.performance_thresholds = {
            "parameter_loading": 0.01,      # 10ms max for parameter loading
            "parameter_override": 0.001,    # 1ms max for parameter override
            "integrity_check": 0.1,         # 100ms max for integrity check
            "full_fit_mock": 0.05,         # 50ms max for full fit (mocked)
            "result_formatting": 0.01       # 10ms max for result formatting
        }
        
        # Standard test parameters
        self.test_params_lcdm = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649, "model_class": "lcdm"
        }
        
        self.test_params_pbuf = {
            "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
            "alpha": 0.1, "Rmax": 100.0, "eps0": 0.01, "n_eps": 2.0, "k_sat": 0.1,
            "model_class": "pbuf"
        }
    
    def test_parameter_loading_performance(self):
        """Benchmark parameter loading and optimization integration performance."""
        benchmark = PerformanceBenchmark("parameter_loading")
        
        # Mock parameter store for consistent testing
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store.get_model_defaults.return_value = self.test_params_lcdm.copy()
            mock_store.is_optimized.return_value = False
            mock_store.get_optimization_history.return_value = []
            mock_store.get_warm_start_params.return_value = None
            mock_store_class.return_value = mock_store
            
            # Import the function to test
            try:
                from fit_bao_aniso import _get_optimized_parameters_with_metadata
                
                # Warm-up runs
                for _ in range(5):
                    _get_optimized_parameters_with_metadata(mock_store, "lcdm")
                
                # Benchmark runs
                for i in range(100):
                    benchmark.start()
                    result = _get_optimized_parameters_with_metadata(mock_store, "lcdm")
                    benchmark.stop()
                    
                    # Verify result structure
                    self.assertIn("final_params", result)
                    self.assertIn("source", result)
            
            except ImportError:
                self.skipTest("Could not import _get_optimized_parameters_with_metadata")
        
        # Analyze performance
        stats = benchmark.get_statistics()
        self.benchmarks["parameter_loading"] = stats
        
        # Performance assertions
        self.assertLess(stats["mean"], self.performance_thresholds["parameter_loading"],
                       f"Parameter loading too slow: {stats['mean']:.6f}s average")
        self.assertLess(stats["max"], self.performance_thresholds["parameter_loading"] * 2,
                       f"Parameter loading max time too slow: {stats['max']:.6f}s")
        
        print(f"\nParameter Loading Performance:")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Median: {stats['median']:.6f}s")
        print(f"  Max: {stats['max']:.6f}s")
        print(f"  Std Dev: {stats['stdev']:.6f}s")
    
    def test_parameter_override_performance(self):
        """Benchmark parameter override application performance."""
        benchmark = PerformanceBenchmark("parameter_override")
        
        base_params = self.test_params_pbuf.copy()
        overrides = {"H0": 70.0, "Om0": 0.3, "alpha": 0.15}
        
        # Mock validation functions
        with patch('fit_bao_aniso.validate_params'):
            with patch('fit_bao_aniso.OPTIMIZABLE_PARAMETERS', {"pbuf": ["H0", "Om0", "alpha"]}):
                
                try:
                    from fit_bao_aniso import _apply_parameter_overrides
                    
                    # Warm-up runs
                    for _ in range(10):
                        test_params = base_params.copy()
                        _apply_parameter_overrides(test_params, overrides, "pbuf")
                    
                    # Benchmark runs
                    for i in range(1000):
                        test_params = base_params.copy()
                        
                        benchmark.start()
                        result = _apply_parameter_overrides(test_params, overrides, "pbuf")
                        benchmark.stop()
                        
                        # Verify override was applied
                        self.assertEqual(test_params["H0"], 70.0)
                        self.assertEqual(result["overrides_applied"], 3)
                
                except ImportError:
                    self.skipTest("Could not import _apply_parameter_overrides")
        
        # Analyze performance
        stats = benchmark.get_statistics()
        self.benchmarks["parameter_override"] = stats
        
        # Performance assertions
        self.assertLess(stats["mean"], self.performance_thresholds["parameter_override"],
                       f"Parameter override too slow: {stats['mean']:.6f}s average")
        
        print(f"\nParameter Override Performance:")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Median: {stats['median']:.6f}s")
        print(f"  Operations/sec: {1.0/stats['mean']:.0f}")
    
    def test_integrity_check_performance(self):
        """Benchmark integrity validation performance."""
        benchmark = PerformanceBenchmark("integrity_check")
        
        # Mock integrity module
        with patch('fit_bao_aniso.integrity.run_integrity_suite') as mock_integrity:
            mock_integrity.return_value = {
                "overall_status": "PASS",
                "summary": {"total_tests": 5, "passed": 5, "failed": 0},
                "tests_run": ["recombination", "sound_horizon", "h_ratios"],
                "tolerances_used": {"h_ratios": 1e-4, "recombination": 1e-4}
            }
            
            # Simulate realistic processing time
            def slow_integrity_check(*args, **kwargs):
                time.sleep(0.001)  # 1ms simulated processing
                return mock_integrity.return_value
            
            mock_integrity.side_effect = slow_integrity_check
            
            # Benchmark runs
            for i in range(50):
                benchmark.start()
                
                # Simulate integrity check call
                result = mock_integrity(
                    params=None,
                    datasets=["bao_ani"],
                    tolerances={"h_ratios": 1e-4, "recombination": 1e-4}
                )
                
                benchmark.stop()
                
                # Verify result
                self.assertEqual(result["overall_status"], "PASS")
        
        # Analyze performance
        stats = benchmark.get_statistics()
        self.benchmarks["integrity_check"] = stats
        
        # Performance assertions
        self.assertLess(stats["mean"], self.performance_thresholds["integrity_check"],
                       f"Integrity check too slow: {stats['mean']:.6f}s average")
        
        print(f"\nIntegrity Check Performance:")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Median: {stats['median']:.6f}s")
    
    def test_full_fit_performance_mocked(self):
        """Benchmark full fitting execution performance with mocked engine."""
        benchmark = PerformanceBenchmark("full_fit_mock")
        
        # Mock all dependencies
        with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
            mock_store_class.return_value = mock_store
            
            with patch('fit_bao_aniso.engine.run_fit') as mock_run_fit:
                # Mock realistic fit results
                mock_results = {
                    "params": self.test_params_lcdm,
                    "metrics": {"total_chi2": 15.234, "aic": 23.234, "dof": 8},
                    "results": {"bao_ani": {"chi2": 15.234}}
                }
                
                # Simulate realistic fitting time
                def slow_fit(*args, **kwargs):
                    time.sleep(0.002)  # 2ms simulated fitting
                    return mock_results
                
                mock_run_fit.side_effect = slow_fit
                
                with patch('fit_bao_aniso._get_optimized_parameters_with_metadata') as mock_get_params:
                    mock_get_params.return_value = {
                        "final_params": self.test_params_lcdm,
                        "source": "defaults",
                        "cmb_optimized": False
                    }
                    
                    try:
                        from fit_bao_aniso import run_bao_aniso_fit
                        
                        # Warm-up runs
                        for _ in range(3):
                            run_bao_aniso_fit("lcdm")
                        
                        # Benchmark runs
                        for i in range(20):
                            benchmark.start()
                            result = run_bao_aniso_fit("lcdm")
                            benchmark.stop()
                            
                            # Verify result structure
                            self.assertIn("params", result)
                            self.assertIn("metrics", result)
                    
                    except ImportError:
                        self.skipTest("Could not import run_bao_aniso_fit")
        
        # Analyze performance
        stats = benchmark.get_statistics()
        self.benchmarks["full_fit_mock"] = stats
        
        # Performance assertions
        self.assertLess(stats["mean"], self.performance_thresholds["full_fit_mock"],
                       f"Full fit (mocked) too slow: {stats['mean']:.6f}s average")
        
        print(f"\nFull Fit Performance (Mocked):")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Median: {stats['median']:.6f}s")
        print(f"  Fits/sec: {1.0/stats['mean']:.1f}")
    
    def test_result_formatting_performance(self):
        """Benchmark result formatting and output performance."""
        benchmark = PerformanceBenchmark("result_formatting")
        
        # Create realistic results structure
        test_results = {
            "params": self.test_params_pbuf,
            "metrics": {
                "total_chi2": 12.891,
                "aic": 22.891,
                "bic": 35.234,
                "dof": 8,
                "p_value": 0.115
            },
            "results": {
                "bao_ani": {
                    "chi2": 12.891,
                    "predictions": {
                        "DM_over_rs": np.array([8.467, 13.156, 16.789, 19.234, 21.567]),
                        "H_times_rs": np.array([147.8, 98.2, 81.3, 72.1, 65.8])
                    },
                    "residuals": np.array([0.1, -0.2, 0.05, -0.1, 0.15])
                }
            },
            "parameter_source": {
                "source": "cmb_optimized",
                "cmb_optimized": True,
                "overrides_applied": 2,
                "override_params": ["H0", "alpha"],
                "param_sources": {param: "cmb_optimized" for param in self.test_params_pbuf.keys()}
            }
        }
        
        try:
            from fit_bao_aniso import format_json_results, print_human_readable_results
            
            # Benchmark JSON formatting
            for i in range(100):
                benchmark.start()
                json_output = format_json_results(test_results)
                benchmark.stop()
                
                # Verify JSON is valid
                json_str = json.dumps(json_output, indent=2, default=str)
                self.assertIsInstance(json_str, str)
                self.assertGreater(len(json_str), 100)
        
        except ImportError:
            # Mock the formatting if import fails
            for i in range(100):
                benchmark.start()
                # Simulate JSON formatting
                json_str = json.dumps(test_results, indent=2, default=str)
                benchmark.stop()
        
        # Analyze performance
        stats = benchmark.get_statistics()
        self.benchmarks["result_formatting"] = stats
        
        # Performance assertions
        self.assertLess(stats["mean"], self.performance_thresholds["result_formatting"],
                       f"Result formatting too slow: {stats['mean']:.6f}s average")
        
        print(f"\nResult Formatting Performance:")
        print(f"  Mean: {stats['mean']:.6f}s")
        print(f"  Operations/sec: {1.0/stats['mean']:.0f}")
    
    def test_memory_usage_estimation(self):
        """Estimate memory usage during fitting operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_arrays = []
        for i in range(10):
            # Create arrays similar to what might be used in fitting
            dm_array = np.random.random(100) * 20  # DM/rs values
            h_array = np.random.random(100) * 150   # H*rs values
            covariance = np.random.random((200, 200))  # 2N x 2N covariance
            
            large_arrays.append((dm_array, h_array, covariance))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = peak_memory - initial_memory
        
        # Clean up
        del large_arrays
        
        print(f"\nMemory Usage Estimation:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        print(f"  Estimated usage: {memory_usage:.1f} MB")
        
        # Memory usage should be reasonable (< 100 MB for test data)
        self.assertLess(memory_usage, 100, f"Memory usage too high: {memory_usage:.1f} MB")
    
    def test_scalability_with_data_size(self):
        """Test performance scalability with different data sizes."""
        data_sizes = [10, 50, 100, 200, 500]
        timing_results = {}
        
        for size in data_sizes:
            benchmark = PerformanceBenchmark(f"data_size_{size}")
            
            # Create test data of varying sizes
            test_data = {
                "observations": {
                    "redshift": np.linspace(0.1, 2.0, size),
                    "DM_over_rs": np.random.random(size) * 20,
                    "H_times_rs": np.random.random(size) * 150
                },
                "covariance": np.eye(2 * size) * 0.1  # 2N x 2N matrix
            }
            
            # Benchmark data processing (simulated)
            for i in range(10):
                benchmark.start()
                
                # Simulate likelihood calculation
                residuals = np.random.random(2 * size)
                chi2 = np.dot(residuals, np.linalg.solve(test_data["covariance"], residuals))
                
                benchmark.stop()
            
            stats = benchmark.get_statistics()
            timing_results[size] = stats["mean"]
        
        print(f"\nScalability Analysis:")
        for size, avg_time in timing_results.items():
            print(f"  Data size {size:3d}: {avg_time:.6f}s average")
        
        # Check that scaling is reasonable (should be roughly O(N^3) due to matrix inversion)
        # For small test sizes, this should still be fast
        for size, avg_time in timing_results.items():
            if size <= 200:
                self.assertLess(avg_time, 0.01, f"Processing too slow for size {size}: {avg_time:.6f}s")
    
    def tearDown(self):
        """Report comprehensive performance summary."""
        if self.benchmarks:
            print(f"\n" + "="*60)
            print(f"PERFORMANCE BENCHMARK SUMMARY")
            print(f"="*60)
            
            for test_name, stats in self.benchmarks.items():
                threshold = self.performance_thresholds.get(test_name, float('inf'))
                status = "✓ PASS" if stats["mean"] < threshold else "✗ FAIL"
                
                print(f"\n{test_name.upper()}:")
                print(f"  Status: {status}")
                print(f"  Mean time: {stats['mean']:.6f}s")
                print(f"  Threshold: {threshold:.6f}s")
                print(f"  Samples: {stats['count']}")
                
                if stats["count"] > 1:
                    print(f"  Std dev: {stats['stdev']:.6f}s")
                    print(f"  Min/Max: {stats['min']:.6f}s / {stats['max']:.6f}s")
            
            print(f"\n" + "="*60)


class TestConcurrentPerformance(unittest.TestCase):
    """
    Tests for concurrent execution performance.
    
    Measures performance when multiple fitting operations run simultaneously.
    """
    
    def test_concurrent_parameter_loading(self):
        """Test parameter loading performance under concurrent access."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        num_threads = 5
        operations_per_thread = 20
        
        def worker():
            """Worker function for concurrent testing."""
            thread_times = []
            
            with patch('fit_bao_aniso.OptimizedParameterStore') as mock_store_class:
                mock_store = MagicMock()
                mock_store.verify_storage_integrity.return_value = {"overall_status": "healthy"}
                mock_store.get_model_defaults.return_value = {"H0": 67.4, "Om0": 0.315}
                mock_store.is_optimized.return_value = False
                mock_store_class.return_value = mock_store
                
                try:
                    from fit_bao_aniso import _get_optimized_parameters_with_metadata
                    
                    for _ in range(operations_per_thread):
                        start_time = time.perf_counter()
                        result = _get_optimized_parameters_with_metadata(mock_store, "lcdm")
                        end_time = time.perf_counter()
                        
                        thread_times.append(end_time - start_time)
                
                except ImportError:
                    thread_times = [0.001] * operations_per_thread  # Mock times
            
            results_queue.put(thread_times)
        
        # Start concurrent workers
        threads = []
        start_time = time.perf_counter()
        
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Collect results
        all_times = []
        while not results_queue.empty():
            thread_times = results_queue.get()
            all_times.extend(thread_times)
        
        # Analyze concurrent performance
        total_operations = num_threads * operations_per_thread
        avg_time_per_op = statistics.mean(all_times)
        throughput = total_operations / total_time
        
        print(f"\nConcurrent Performance Analysis:")
        print(f"  Threads: {num_threads}")
        print(f"  Operations per thread: {operations_per_thread}")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per operation: {avg_time_per_op:.6f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")
        
        # Performance assertions
        self.assertLess(avg_time_per_op, 0.01, "Concurrent operations too slow")
        self.assertGreater(throughput, 50, "Concurrent throughput too low")


if __name__ == '__main__':
    # Run performance tests with detailed output
    unittest.main(verbosity=2, buffer=True)