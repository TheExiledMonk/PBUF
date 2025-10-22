# BAO Anisotropic Fitting - Test Suite Documentation

This document provides comprehensive documentation for the BAO anisotropic fitting test suite, covering all test modules, their purposes, and how to run them.

## Overview

The test suite for BAO anisotropic fitting consists of four main test modules that comprehensively validate the implementation:

1. **Unit Tests** (`test_bao_aniso_fit.py`) - Core functionality testing
2. **Parity Tests** (`test_bao_aniso_parity.py`) - Validation against existing implementation
3. **Performance Tests** (`test_bao_aniso_performance.py`) - Performance benchmarking
4. **Integration Tests** (`test_bao_aniso_integration.py`) - fit_core infrastructure integration

## Requirements Coverage

The test suite addresses the following requirements from the specification:

- **Requirement 5.1**: Comprehensive validation against known benchmarks
- **Requirement 5.2**: Statistical tolerance validation for result comparison
- Parameter loading and optimization integration testing
- fit_core infrastructure integration validation
- Performance benchmarking and execution time validation

## Test Modules

### 1. Unit Tests (`test_bao_aniso_fit.py`)

**Purpose**: Tests core functionality of the BAO anisotropic fitting implementation.

**Test Classes**:
- `TestParameterLoadingAndOptimization`: Parameter store integration and optimization
- `TestValidationAgainstExistingImplementation`: Basic validation against original implementation
- `TestFitCoreIntegration`: Integration with fit_core components
- `TestPerformanceBenchmarks`: Basic performance measurements
- `TestCommandLineInterface`: CLI argument parsing

**Key Test Cases**:
- Parameter store initialization (success and failure scenarios)
- Optimized parameter retrieval with metadata
- Parameter override validation and application
- CMB optimization priority handling
- Fallback to hardcoded defaults
- Command-line argument parsing

**Usage**:
```bash
# Run all unit tests
python -m pytest pipelines/fit_core/test_bao_aniso_fit.py -v

# Run specific test class
python -m pytest pipelines/fit_core/test_bao_aniso_fit.py::TestParameterLoadingAndOptimization -v

# Run specific test
python -m pytest pipelines/fit_core/test_bao_aniso_fit.py::TestParameterLoadingAndOptimization::test_parameter_store_initialization_success -v
```

### 2. Parity Tests (`test_bao_aniso_parity.py`)

**Purpose**: Validates that the enhanced implementation produces results consistent with the original `fit_aniso.py` within statistical tolerance.

**Test Classes**:
- `TestBaoAnisoParity`: Mock-based parity validation
- `TestParityWithRealExecution`: Real script execution comparison (when available)

**Key Test Cases**:
- LCDM model parity (default and custom parameters)
- PBUF model parity (default and custom parameters)
- Parameter comparison within tolerance (1e-3 relative error)
- Chi-squared comparison within tolerance (1e-2 relative error)
- Enhanced feature validation (parameter source information)

**Statistical Tolerances**:
- Parameter values: 1e-3 relative error
- Chi-squared values: 1e-2 relative error
- Array predictions: 1e-3 relative tolerance

**Usage**:
```bash
# Run parity tests
python -m pytest pipelines/fit_core/test_bao_aniso_parity.py -v

# Run specific parity test
python -m pytest pipelines/fit_core/test_bao_aniso_parity.py::TestBaoAnisoParity::test_lcdm_default_parameters_parity -v
```

### 3. Performance Tests (`test_bao_aniso_performance.py`)

**Purpose**: Measures and validates performance characteristics of the implementation.

**Test Classes**:
- `TestBaoAnisoPerformance`: Core performance benchmarks
- `TestConcurrentPerformance`: Concurrent execution performance

**Performance Thresholds**:
- Parameter loading: < 10ms
- Parameter override: < 1ms
- Integrity check: < 100ms
- Full fit (mocked): < 50ms
- Result formatting: < 10ms

**Benchmark Categories**:
- Parameter loading and optimization integration
- Parameter override application
- Integrity validation execution
- Full fitting pipeline (mocked)
- Result formatting and output
- Memory usage estimation
- Scalability with data size
- Concurrent operation performance

**Usage**:
```bash
# Run performance tests
python -m pytest pipelines/fit_core/test_bao_aniso_performance.py -v

# Run specific performance test
python -m pytest pipelines/fit_core/test_bao_aniso_performance.py::TestBaoAnisoPerformance::test_parameter_loading_performance -v
```

### 4. Integration Tests (`test_bao_aniso_integration.py`)

**Purpose**: Tests integration with all fit_core infrastructure components.

**Test Classes**:
- `TestEngineIntegration`: Integration with optimization engine
- `TestParameterStoreIntegration`: Integration with parameter storage system
- `TestIntegrityValidationIntegration`: Integration with integrity validation
- `TestDatasetIntegration`: Integration with dataset loading
- `TestEndToEndIntegration`: Complete integration flow testing

**Key Integration Points**:
- Engine parameter passing and mode specification
- Parameter store initialization and integrity checking
- Integrity validation suite integration
- Dataset specification (bao_ani vs bao separation)
- Error handling and recovery mechanisms

**Usage**:
```bash
# Run integration tests
python -m pytest pipelines/fit_core/test_bao_aniso_integration.py -v

# Run specific integration test
python -m pytest pipelines/fit_core/test_bao_aniso_integration.py::TestEngineIntegration::test_engine_call_with_correct_parameters -v
```

## Running the Complete Test Suite

### Using the Test Runner

The recommended way to run all tests is using the provided test runner:

```bash
# Run complete test suite with detailed reporting
python pipelines/fit_core/run_bao_aniso_tests.py
```

This provides:
- Comprehensive test execution across all modules
- Detailed performance reporting
- Success rate statistics by category
- Final status summary

### Using pytest

You can also run tests using pytest directly:

```bash
# Run all BAO anisotropic tests
python -m pytest pipelines/fit_core/test_bao_aniso_*.py -v

# Run with coverage reporting
python -m pytest pipelines/fit_core/test_bao_aniso_*.py --cov=pipelines.fit_bao_aniso --cov-report=html

# Run specific test pattern
python -m pytest pipelines/fit_core/test_bao_aniso_*.py -k "parameter_loading" -v
```

### Using unittest

Individual test modules can be run with unittest:

```bash
# Run specific test module
python -m unittest pipelines.fit_core.test_bao_aniso_fit -v

# Run specific test class
python -m unittest pipelines.fit_core.test_bao_aniso_fit.TestParameterLoadingAndOptimization -v
```

## Test Configuration

### Mock Configuration

The tests use extensive mocking to isolate functionality and avoid dependencies:

- **OptimizedParameterStore**: Mocked for consistent parameter loading tests
- **Engine**: Mocked for predictable fitting results
- **Integrity Suite**: Mocked for controlled validation testing
- **File System**: Temporary directories for safe testing

### Test Data

Standard test parameters are defined for consistent testing:

**LCDM Parameters**:
```python
{
    "H0": 67.4, "Om0": 0.315, "Obh2": 0.02237, "ns": 0.9649,
    "Neff": 3.046, "Tcmb": 2.7255, "recomb_method": "RECFAST"
}
```

**PBUF Parameters**:
```python
{
    # LCDM parameters plus:
    "alpha": 0.1, "Rmax": 100.0, "eps0": 0.01, 
    "n_eps": 2.0, "k_sat": 0.1
}
```

## Expected Test Results

### Success Criteria

For the test suite to pass completely:

1. **Unit Tests**: All core functionality tests pass
2. **Parity Tests**: Results match original implementation within tolerance
3. **Performance Tests**: All operations meet performance thresholds
4. **Integration Tests**: All fit_core integrations work correctly

### Performance Expectations

Expected performance characteristics:
- Parameter loading: ~1-5ms average
- Parameter overrides: ~0.1-0.5ms average
- Full fit execution: ~10-30ms (mocked)
- Memory usage: <50MB for typical datasets

### Common Issues and Solutions

**Import Errors**:
- Ensure `pipelines` directory is in Python path
- Check that `fit_bao_aniso.py` exists and is importable

**Mock Failures**:
- Verify mock patch paths match actual module structure
- Check that mocked functions exist in target modules

**Performance Failures**:
- Performance thresholds may need adjustment for different hardware
- Consider system load when running performance tests

**Parity Failures**:
- Check that test parameters include all required fields
- Verify statistical tolerances are appropriate for the comparison

## Continuous Integration

For CI/CD integration, use:

```bash
# Quick test run (unit tests only)
python -m pytest pipelines/fit_core/test_bao_aniso_fit.py --tb=short

# Full test run with XML output
python -m pytest pipelines/fit_core/test_bao_aniso_*.py --junitxml=test_results.xml

# Performance regression testing
python -m pytest pipelines/fit_core/test_bao_aniso_performance.py --benchmark-only
```

## Test Maintenance

### Adding New Tests

When adding new functionality:

1. Add unit tests to `test_bao_aniso_fit.py`
2. Add parity validation to `test_bao_aniso_parity.py`
3. Add performance benchmarks if applicable
4. Update integration tests for new fit_core interactions

### Updating Test Data

When changing parameter requirements:

1. Update test parameter dictionaries
2. Adjust validation tolerances if needed
3. Update mock return values to match new structure
4. Verify parity tests still pass with new parameters

### Performance Threshold Updates

Performance thresholds may need updates for:
- Different hardware configurations
- Algorithm optimizations
- Changed dependencies

Update thresholds in `TestBaoAnisoPerformance.performance_thresholds`.

## Troubleshooting

### Debug Mode

Run tests with debug output:

```bash
# Verbose output with print statements
python -m pytest pipelines/fit_core/test_bao_aniso_fit.py -v -s

# Drop into debugger on failure
python -m pytest pipelines/fit_core/test_bao_aniso_fit.py --pdb
```

### Logging

Enable detailed logging during tests:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Isolation

If tests interfere with each other:

```bash
# Run tests in separate processes
python -m pytest pipelines/fit_core/test_bao_aniso_*.py --forked

# Run with fresh imports
python -m pytest pipelines/fit_core/test_bao_aniso_*.py --import-mode=importlib
```

## Conclusion

This comprehensive test suite ensures the BAO anisotropic fitting implementation is:

- **Functionally Correct**: Unit tests validate core functionality
- **Backward Compatible**: Parity tests ensure consistency with original implementation
- **Performant**: Performance tests validate execution speed requirements
- **Well Integrated**: Integration tests ensure proper fit_core interaction

The test suite provides confidence that the enhanced BAO anisotropic fitting implementation meets all requirements and is ready for production use.