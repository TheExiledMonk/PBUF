# Implementation Plan

- [ ] 1. Extend parameter system for optimization metadata
  - Create optimization metadata structures in parameter.py
  - Add parameter optimization bounds validation
  - Implement parameter classification (optimizable vs fixed)
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [ ] 2. Implement core optimization engine components
  - [ ] 2.1 Create ParameterOptimizer class with bounds validation
    - Implement optimize_parameters method with SciPy integration
    - Add get_optimization_bounds for both ΛCDM and PBUF models
    - Create validate_optimization_request for parameter validation
    - _Requirements: 1.1, 1.2, 1.3, 1.6_

  - [ ] 2.2 Build CMB-specific optimization routine
    - Implement optimize_cmb_parameters function with χ² objective
    - Add convergence diagnostics and provenance logging
    - Create optimization result validation and bounds checking
    - Add dataset integrity validation hook before optimization runs
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ] 2.3 Write unit tests for optimization engine
    - Test parameter bounds validation for both models
    - Test optimization convergence with mock objectives
    - Test error handling for invalid parameter requests
    - _Requirements: 1.1, 1.2, 1.3, 1.6_

- [ ] 3. Create optimized parameter storage system
  - [ ] 3.1 Implement OptimizedParameterStore class
    - Create get_model_defaults with optimization metadata support
    - Implement update_model_defaults with non-destructive merge
    - Add get_optimization_history and warm-start support
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 3.2 Add cross-model consistency validation
    - Implement validate_cross_model_consistency method
    - Create shared parameter comparison logic
    - Add tolerance-based divergence detection and logging
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 3.3 Create parameter store persistence layer
    - Implement JSON-based storage with metadata preservation
    - Add per-model file locking (pbuf.lock, lcdm.lock) for concurrent access protection
    - Create backup and recovery mechanisms for corrupted files
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 3.4 Write unit tests for parameter store
    - Test round-trip persistence of optimized parameters
    - Test non-destructive parameter merging
    - Test cross-model consistency validation
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 4. Extend configuration system for optimization settings
  - [ ] 4.1 Add optimization configuration parsing
    - Extend config.py to support optimization section
    - Add validation for optimize_parameters lists
    - Implement covariance_scaling and dry_run options
    - Add frozen_parameters support for explicit parameter locking
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ] 4.2 Create command-line argument parsing
    - Add --optimize flag with parameter list parsing
    - Implement --cov-scale and --dry-run flags
    - Add --warm-start flag for reusing recent optimizations
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ] 4.3 Write configuration validation tests
    - Test optimization parameter parsing and validation
    - Test command-line precedence over configuration files
    - Test error handling for invalid optimization settings
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 5. Integrate optimization into unified engine
  - [ ] 5.1 Extend engine.py for optimization dispatch
    - Modify run_fit to support optimization parameters
    - Add optimization result processing and storage
    - Implement parameter propagation to subsequent fits
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 5.2 Create optimization workflow orchestration
    - Implement full CMB optimization workflow for both models
    - Add automatic parameter propagation after optimization with dataset tagging
    - Create optimization result validation and reporting
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1, 4.2, 4.3, 4.4, 4.5_

  - [ ] 5.3 Write integration tests for engine optimization
    - Test end-to-end optimization workflow for ΛCDM model
    - Test end-to-end optimization workflow for PBUF model
    - Test parameter propagation across multiple fitters
    - Create optimization summary snapshot (reports/optimization_summary.json)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 6. Update all fitter scripts for optimization support
  - [ ] 6.1 Modify fit_cmb.py for optimization flags
    - Add optimization argument parsing and validation
    - Integrate with ParameterOptimizer for CMB-specific optimization
    - Update result formatting to show optimized vs fixed parameters
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ] 6.2 Update fit_bao.py, fit_sn.py for optimized parameter usage
    - Ensure automatic use of optimized parameters from store
    - Add validation that optimized parameters are being used
    - Update result reporting to indicate parameter source
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 6.3 Update fit_joint.py for optimization integration
    - Ensure joint fitting uses optimized individual parameters
    - Add optimization metadata to joint fitting results
    - Create validation that joint fits benefit from optimization
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 6.4 Write fitter integration tests
    - Test that BAO/SN fitters automatically use optimized CMB parameters
    - Test joint fitting with pre-optimized individual parameters
    - Test backward compatibility with existing fitter workflows
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 7. Add advanced features and reporting
  - [ ] 7.1 Implement provenance logging system
    - Add optimizer method and library version tracking
    - Create comprehensive optimization metadata recording
    - Implement timestamp and convergence diagnostic logging
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ] 7.2 Create HTML report integration
    - Add optimization summary section to unified reports
    - Include χ² improvements and parameter changes
    - Create visual indicators for optimized vs fixed parameters
    - Add validation that HTML report matches optimization_summary.json
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ] 7.3 Implement dry-run and warm-start features
    - Add dry-run mode that computes without persisting results
    - Implement warm-start using recent optimization results
    - Create validation for warm-start parameter compatibility
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ] 7.4 Write advanced feature tests
    - Test dry-run mode produces results without side effects
    - Test warm-start functionality with recent optimizations
    - Test provenance logging completeness and accuracy
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [ ] 8. Create comprehensive validation and testing
  - [ ] 8.1 Implement critical round-trip persistence test
    - Create test_propagation_roundtrip with numerical precision validation
    - Test parameter store reliability under concurrent access
    - Validate optimization metadata preservation across cycles
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ] 8.2 Create cross-model consistency validation
    - Test that ΛCDM and PBUF optimizations don't diverge inappropriately
    - Validate shared parameter consistency after sequential optimizations
    - Create tolerance-based validation for parameter drift detection
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 8.3 Build backward compatibility test suite
    - Validate that existing workflows work unchanged without optimization flags
    - Test that legacy parameter overrides continue to function
    - Ensure existing configuration files remain compatible
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 8.4 Create performance and robustness tests
    - Test optimization convergence under various starting conditions
    - Validate bounds enforcement and constraint handling
    - Test error recovery and fallback mechanisms
    - Add quick regression mode: rerun optimizer with new defaults and confirm χ² identical within 1e-8
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [ ] 9. Documentation and deployment preparation
  - [ ] 9.1 Update API documentation
    - Document new optimization flags and configuration options
    - Create usage examples for both ΛCDM and PBUF optimization
    - Add troubleshooting guide for optimization convergence issues
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ] 9.2 Create user migration guide
    - Document how to transition from fixed to optimized parameters
    - Provide best practices for optimization parameter selection
    - Create workflow examples for different research scenarios
    - Add "Reproducibility and Provenance" section covering version control, optimizer info, and checksums
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 9.3 Prepare deployment validation
    - Create deployment checklist for optimization features
    - Build validation scripts for production deployment
    - Document rollback procedures if optimization issues arise
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 8.1, 8.2, 8.3, 8.4, 8.5_