# Implementation Plan

- [x] 1. Set up core infrastructure and data models
  - [x] 1.1 Create parameter set and distance prior data classes
    - Define ParameterSet dataclass with H0, Omega_m, Omega_b_h2, n_s, tau, A_s fields
    - Implement DistancePriors dataclass with R, l_A, Omega_b_h2, theta_star fields
    - Add validation methods and conversion utilities to both classes
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 1.2 Implement CMB configuration management
    - Create CMBConfig dataclass with use_raw_parameters, z_recombination, jacobian_step_size settings
    - Add configuration validation and default value handling
    - Integrate with existing configuration system in data preparation framework
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 1.3 Define exception hierarchy for CMB processing
    - Create CMBProcessingError base class and specific exception types
    - Implement ParameterDetectionError, ParameterValidationError, DerivationError, CovarianceError
    - Add detailed error messages and diagnostic information
    - _Requirements: 6.4, 6.5_

- [x] 2. Implement parameter detection and parsing engine
  - [x] 2.1 Create parameter detection logic
    - Implement detect_raw_parameters function to scan registry entries for CMB parameter files
    - Add classify_parameter_format function for automatic format detection (CSV/JSON/NumPy)
    - Create validate_parameter_completeness function to check required parameters are present
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 2.2 Build flexible parameter parser
    - Implement parse_parameter_file function supporting multiple input formats
    - Create normalize_parameter_names function with fuzzy matching for parameter aliases
    - Add extract_covariance_matrix function for covariance data extraction
    - _Requirements: 1.2, 1.3_

  - [x] 2.3 Add comprehensive parameter validation
    - Implement validate_parameter_ranges function with physical bounds checking
    - Create check_numerical_stability function to detect NaN, inf, extreme values
    - Add validate_covariance_matrix function for matrix property validation
    - _Requirements: 1.4, 6.1, 6.2, 6.3_

- [x] 3. Implement distance prior derivation engine
  - [x] 3.1 Create core distance computation functions
    - Implement compute_shift_parameter function for R = √(Ωm H₀²) × r(z*)/c calculation
    - Add compute_acoustic_scale function for ℓ_A = π × r(z*)/r_s(z*) computation
    - Create compute_angular_scale function for θ* = r_s(z*)/r(z*) calculation
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Integrate with PBUF background integrators
    - Connect to existing BackgroundIntegrator class from core.pbuf_background
    - Use compute_sound_horizon function from core.pbuf_models for r_s calculations
    - Ensure consistent integration methods with BAO and SN background calculations
    - _Requirements: 2.4, 2.5_

  - [x] 3.3 Add recombination redshift handling
    - Implement configurable z_recombination with default value ≈ 1089.8
    - Add metadata reading for custom recombination redshift values
    - Create validation for recombination redshift physical range
    - _Requirements: 2.4_

- [x] 4. Implement Jacobian computation and covariance propagation
  - [x] 4.1 Create numerical Jacobian computer
    - Implement compute_jacobian function using central finite differences
    - Add finite_difference_derivative function with optimized step size selection
    - Create optimize_step_size function for automatic step size determination
    - _Requirements: 3.2, 3.3_

  - [x] 4.2 Build covariance propagation system
    - Implement propagate_covariance function using C_derived = J × C_params × J^T formula
    - Add validate_covariance_properties function for symmetry and positive-definiteness checks
    - Create compute_correlation_matrix function for analysis and diagnostics
    - _Requirements: 3.1, 3.4, 3.5_

  - [x] 4.3 Add numerical stability enhancements
    - Implement adaptive step size algorithms for improved derivative accuracy
    - Add condition number monitoring for covariance matrix stability
    - Create fallback methods for ill-conditioned covariance matrices
    - _Requirements: 3.4, 3.5_

- [ ] 5. Create StandardDataset output builder
  - [x] 5.1 Implement dataset construction logic
    - Create build_standard_dataset function to convert priors and covariance to StandardDataset
    - Add create_metadata function for comprehensive provenance tracking
    - Implement validate_output_format function to ensure schema compliance
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 5.2 Add metadata and provenance tracking
    - Include processing method, parameter sources, and computation details in metadata
    - Add timestamp, validation results, and configuration settings to output
    - Create citation and source attribution from registry entry metadata
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 5.3 Ensure fitting pipeline compatibility
    - Verify StandardDataset structure matches existing fitting code expectations
    - Test observable array ordering and covariance matrix format
    - Validate metadata fields used by downstream components
    - _Requirements: 4.5_

- [x] 6. Implement main processing workflow with error handling
  - [x] 6.1 Create main CMB processing function
    - Implement process_cmb_dataset function orchestrating the complete workflow
    - Add configuration-driven processing with raw parameter detection
    - Create automatic fallback to legacy distance-prior mode when needed
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 6.2 Add comprehensive error handling and recovery
    - Implement try-catch blocks for each processing stage with specific error types
    - Add graceful degradation when covariance matrix is unavailable
    - Create diagnostic logging for debugging failed parameter processing
    - _Requirements: 6.4, 6.5_

  - [x] 6.3 Integrate with existing CMB derivation module
    - Extend pipelines/data_preparation/derivation/cmb_derivation.py with new functionality
    - Maintain backward compatibility with existing derive_cmb_data function
    - Add configuration flag to enable/disable raw parameter processing
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7. Create comprehensive test suite
  - [x] 7.1 Write unit tests for parameter processing
    - Test parameter detection across different file formats and naming conventions
    - Verify parameter validation catches invalid values and ranges
    - Test covariance matrix validation for symmetry and positive-definiteness
    - _Requirements: 7.1, 7.2_

  - [x] 7.2 Write unit tests for distance prior derivation
    - Test R, ℓ_A, θ* computation accuracy against published Planck values
    - Verify numerical integration consistency with existing PBUF methods
    - Test parameter sensitivity and derivative computation accuracy
    - _Requirements: 7.2, 7.3_

  - [x] 7.3 Write integration tests for complete pipeline
    - Test full workflow from registry entry to StandardDataset output
    - Verify backward compatibility with legacy distance-prior processing
    - Test error handling and fallback mechanisms
    - _Requirements: 7.4, 7.5_

  - [x] 7.4 Write performance and stress tests
    - Test processing time with typical Planck parameter datasets
    - Monitor memory usage during covariance propagation
    - Test numerical stability with extreme parameter values
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 8. Add configuration and deployment support
  - [x] 8.1 Create configuration integration
    - Add CMB raw parameter settings to data preparation configuration
    - Implement environment variable support for deployment settings
    - Create configuration validation and documentation
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 8.2 Add logging and monitoring
    - Implement structured logging for parameter processing steps
    - Add performance metrics collection for monitoring
    - Create diagnostic output for troubleshooting processing failures
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 8.3 Create deployment validation
    - Add system checks for required PBUF background integrator dependencies
    - Test registry integration with mock Planck-style datasets
    - Verify fitting pipeline compatibility across different parameter sets
    - _Requirements: 7.4, 7.5_

- [x] 9. Update documentation and examples
  - [x] 9.1 Update CMB preparation module documentation
    - Document new raw parameter processing capabilities in API reference
    - Add configuration options and usage examples
    - Update data preparation design document with CMB derivation workflow
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 9.2 Create user guide and examples
    - Write tutorial for processing Planck-style parameter datasets
    - Add example registry entries for different parameter file formats
    - Create troubleshooting guide for common processing issues
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [x] 9.3 Add scientific validation documentation
    - Document mathematical formulations for distance prior computations
    - Add validation results comparing derived priors with published Planck values
    - Create covariance propagation methodology documentation
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5_