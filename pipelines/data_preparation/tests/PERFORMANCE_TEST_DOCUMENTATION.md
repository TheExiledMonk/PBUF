# CMB Performance and Stress Testing Documentation

## Overview

This document describes the comprehensive performance and stress testing suite for the CMB raw parameter integration feature. The tests ensure that the system meets all performance requirements (8.1-8.5) and behaves correctly under various stress conditions.

## Test Categories

### 1. Processing Time Performance Tests (Requirement 8.1)

**Purpose**: Verify that CMB parameter processing completes within acceptable time limits.

**Test Cases**:
- `test_single_parameter_set_processing_time`: Single Planck parameter set processing < 10 seconds
- `test_jacobian_computation_performance`: Jacobian computation < 30 seconds  
- `test_batch_processing_performance`: Batch processing < 1 second per parameter set
- `test_large_covariance_matrix_performance`: Large matrix operations < 5 seconds
- `test_parameter_file_parsing_performance`: File parsing < 0.1 seconds per file

**Key Metrics**:
- Processing time per parameter set
- Jacobian computation time
- File parsing speed
- Covariance propagation time

### 2. Memory Usage Monitoring Tests (Requirement 8.2)

**Purpose**: Monitor memory usage during covariance propagation and ensure it stays under 1GB for standard datasets.

**Test Cases**:
- `test_distance_prior_memory_usage`: Basic computation memory < 50MB
- `test_covariance_propagation_memory_scaling`: Matrix operations < 1GB
- `test_batch_processing_memory_efficiency`: Memory per dataset < 5MB
- `test_memory_leak_detection`: No significant memory leaks over 100 iterations

**Key Metrics**:
- Memory usage per computation
- Memory scaling with matrix size
- Peak memory usage
- Memory leak detection

### 3. Numerical Stability Tests (Requirement 8.3)

**Purpose**: Test numerical stability with extreme parameter values and edge cases.

**Test Cases**:
- `test_extreme_valid_parameter_ranges`: Parameters at physical bounds
- `test_numerical_precision_limits`: Behavior near machine precision
- `test_jacobian_numerical_stability_comprehensive`: Derivative accuracy across parameter space
- `test_ill_conditioned_covariance_handling`: Behavior with ill-conditioned matrices

**Key Metrics**:
- Result accuracy with extreme parameters
- Numerical precision handling
- Jacobian consistency across step sizes
- Matrix condition number tolerance

### 4. Stress Condition Tests (Requirements 8.4, 8.5)

**Purpose**: Test system behavior under stress conditions and concurrent processing.

**Test Cases**:
- `test_concurrent_processing_simulation`: Simulated concurrent load
- `test_repeated_processing_stability`: Stability over many iterations
- `test_error_recovery_resilience`: Recovery from various error conditions
- `test_resource_exhaustion_handling`: Behavior approaching resource limits
- `test_multithreaded_processing_safety`: Thread safety verification

**Key Metrics**:
- Concurrent processing throughput
- System stability over time
- Error recovery capability
- Thread safety validation

### 5. Performance Regression Tests

**Purpose**: Establish performance benchmarks and detect regressions.

**Test Cases**:
- `test_performance_benchmarks`: Baseline performance measurements

**Key Metrics**:
- Benchmark comparison
- Performance trend analysis
- Regression detection

## Test Infrastructure

### Performance Monitor Class

The `PerformanceMonitor` class provides comprehensive monitoring capabilities:

```python
class PerformanceMonitor:
    def start_monitoring(self):
        """Start performance monitoring with tracemalloc."""
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Return detailed performance metrics."""
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
```

### Context Manager

The `performance_monitor()` context manager simplifies test instrumentation:

```python
with performance_monitor() as monitor:
    # Test code here
    result = compute_distance_priors(params)

metrics = monitor.stop_monitoring()
assert metrics['elapsed_time'] < 10.0
assert metrics['memory_increase_mb'] < 50.0
```

## Test Execution

### Running All Tests

```bash
# Full test suite
python pipelines/data_preparation/tests/run_cmb_performance_tests.py

# Quick essential tests only
python pipelines/data_preparation/tests/run_cmb_performance_tests.py --quick

# Stress tests only
python pipelines/data_preparation/tests/run_cmb_performance_tests.py --stress-only

# With detailed report
python pipelines/data_preparation/tests/run_cmb_performance_tests.py --report-file performance_report.json
```

### Individual Test Modules

```bash
# Specific test class
pytest pipelines/data_preparation/tests/test_cmb_performance_stress.py::TestProcessingTimePerformance -v

# Specific test method
pytest pipelines/data_preparation/tests/test_cmb_performance_stress.py::TestMemoryUsageMonitoring::test_covariance_propagation_memory_scaling -v
```

## Performance Requirements

### Processing Time Limits (Requirement 8.1)

| Operation | Maximum Time | Notes |
|-----------|--------------|-------|
| Single parameter set | 10 seconds | Typical Planck dataset |
| Jacobian computation | 30 seconds | Numerical differentiation |
| Batch processing | 1 second/set | MCMC chain processing |
| Covariance propagation | 5 seconds | Large matrix operations |
| Parameter parsing | 0.1 seconds | File I/O operations |

### Memory Usage Limits (Requirement 8.2)

| Operation | Maximum Memory | Notes |
|-----------|----------------|-------|
| Single computation | 50 MB | Basic distance prior calculation |
| Covariance propagation | 1 GB | Standard dataset matrices |
| Batch processing | 5 MB/set | Memory per parameter set |
| Memory leaks | 100 MB | Over 100 iterations |

### Numerical Stability (Requirement 8.3)

| Test | Tolerance | Notes |
|------|-----------|-------|
| Extreme parameters | 10% relative error | At physical bounds |
| Precision limits | 1e-6 absolute | Near machine precision |
| Jacobian consistency | 80% elements | Across step sizes |
| Matrix conditioning | 1e12 condition number | Ill-conditioned matrices |

### Stress Conditions (Requirements 8.4, 8.5)

| Test | Requirement | Notes |
|------|-------------|-------|
| Concurrent processing | 120 seconds | 50 parameter sets |
| Repeated computation | 0.1 seconds/iteration | Stability test |
| Thread safety | No race conditions | Multi-threaded access |
| Error recovery | Graceful degradation | Various failure modes |

## Test Data and Mocking

### Mock Background Integrator

Tests use mocked background integrators for consistent, fast execution:

```python
with patch('pipelines.data_preparation.derivation.cmb_background.BackgroundIntegrator') as mock_integrator_class:
    mock_bg = Mock()
    mock_bg.comoving_distance.return_value = 14026.0
    mock_integrator_class.return_value = mock_bg
```

### Test Parameter Sets

- **Standard Planck 2018**: H₀=67.36, Ωₘ=0.3153, Ωᵦh²=0.02237, nₛ=0.9649, τ=0.0544
- **Extreme Valid**: Parameters at physical bounds
- **Random Variations**: Gaussian perturbations around Planck values
- **Precision Tests**: Tiny differences to test numerical stability

### Covariance Matrices

- **Diagonal**: Simple uncorrelated uncertainties
- **Full**: Realistic parameter correlations
- **Ill-conditioned**: High condition numbers for stability testing
- **Large**: Extended parameter spaces (up to 50×50)

## Monitoring and Reporting

### System Information

Tests capture comprehensive system information:
- Platform and Python version
- CPU count and memory capacity
- Current system load

### Performance Metrics

Detailed metrics are collected for each test:
- Execution time (total and per-operation)
- Memory usage (current, peak, increase)
- Resource utilization
- Error rates and recovery times

### Test Reports

The test runner generates detailed JSON reports containing:
- Individual test results
- Performance benchmarks
- System information
- Failure analysis
- Trend data for regression detection

### Continuous Integration

Performance tests are designed for CI/CD integration:
- Fast execution with mocked dependencies
- Clear pass/fail criteria
- Detailed logging for debugging
- Regression detection capabilities

## Troubleshooting

### Common Issues

1. **Memory Limit Exceeded**
   - Reduce batch sizes in tests
   - Check for memory leaks
   - Verify garbage collection

2. **Timeout Errors**
   - Check system load
   - Verify mock setup
   - Reduce iteration counts

3. **Numerical Instability**
   - Check parameter ranges
   - Verify step sizes
   - Review matrix conditioning

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Profiling

For detailed profiling:

```python
import cProfile
cProfile.run('compute_distance_priors(params)', 'profile_output')
```

## Maintenance

### Updating Requirements

Performance requirements are defined in `performance_requirements.json` and should be updated when:
- Hardware capabilities change
- Algorithm improvements are made
- New performance bottlenecks are identified

### Adding New Tests

When adding new performance tests:
1. Follow the existing naming convention
2. Use the `PerformanceMonitor` class
3. Include appropriate assertions
4. Document expected behavior
5. Update this documentation

### Benchmark Updates

Performance benchmarks should be updated:
- After significant algorithm changes
- When hardware is upgraded
- Quarterly for trend analysis
- Before major releases

## References

- Requirements 8.1-8.5 in the CMB integration specification
- PBUF background integrator documentation
- NumPy performance best practices
- Python memory profiling guidelines