# Implementation Plan

- [x] 1. Set up project structure and core interfaces
  - Create pipelines/fit_core/ directory structure with __init__.py files
  - Define base interfaces and type hints for parameter dictionaries and result structures
  - Set up module imports and basic scaffolding for all core components
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Implement centralized parameter management system
  - [x] 2.1 Create parameter.py with ΛCDM and PBUF default parameter definitions
    - Implement DEFAULTS dictionary with all cosmological parameters and their default values
    - Create build_params() function that constructs parameter dictionaries for both models
    - Add parameter validation logic and override application functionality
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 2.2 Integrate with existing cmb_priors.prepare_background_params()
    - Ensure parameter.py calls existing physics modules for derived parameter computation
    - Verify consistent parameter dictionary structure across ΛCDM and PBUF models
    - Add parameter validation for physical bounds and unit consistency
    - _Requirements: 1.1, 1.2, 1.3, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 2.3 Write unit tests for parameter construction and validation
    - Test default parameter retrieval for both ΛCDM and PBUF models
    - Verify override application and parameter validation logic
    - Test integration with prepare_background_params() function
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 3. Implement unified dataset loading and validation
  - [x] 3.1 Create datasets.py with unified dataset loader interface
    - Implement load_dataset() function wrapping existing dataio.loaders
    - Add dataset validation for CMB, BAO, BAO anisotropic, and supernova data
    - Ensure consistent data format and labeling across all observational blocks
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 3.2 Add covariance matrix validation and metadata extraction
    - Implement covariance matrix positive definiteness checks
    - Add dataset metadata extraction (redshift ranges, data point counts)
    - Create error handling for missing or malformed datasets
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 3.3 Write unit tests for dataset loading and validation
    - Test dataset loading for all supported observational blocks
    - Verify covariance matrix validation and error handling
    - Test dataset metadata extraction and format consistency
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 4. Implement centralized statistics and metrics computation
  - [x] 4.1 Create statistics.py with chi-squared and information criteria functions
    - Implement chi2_generic() function for consistent χ² computation across all blocks
    - Add compute_metrics() for AIC, BIC, degrees of freedom, and p-value calculations
    - Create delta_aic() and model comparison utilities
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 4.2 Add degrees of freedom computation and statistical validation
    - Implement compute_dof() function accounting for all datasets and parameters
    - Add statistical validation for χ² distribution assumptions
    - Create utilities for p-value computation and confidence intervals
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 4.3 Write unit tests for statistical computations
    - Test χ² computation against analytical cases with known results
    - Verify AIC, BIC, and degrees of freedom calculations
    - Test model comparison utilities and edge cases
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5. Implement likelihood functions for all observational blocks
  - [x] 5.1 Create likelihoods.py with CMB likelihood function
    - Implement likelihood_cmb() using existing cmb_priors.distance_priors()
    - Ensure proper recombination redshift calculation and CMB distance prior computation
    - Add χ² computation using centralized chi2_generic() function
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.2 Implement BAO likelihood functions (isotropic and anisotropic)
    - Create likelihood_bao() using bao_background.bao_distance_ratios()
    - Implement likelihood_bao_ani() using bao_background.bao_anisotropic_ratios()
    - Ensure proper drag epoch calculation and BAO distance ratio computation
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.3 Implement supernova likelihood function
    - Create likelihood_sn() using existing gr_models.mu() for distance modulus
    - Add proper handling of Pantheon+ dataset format and systematic uncertainties
    - Integrate with centralized χ² computation and error handling
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.4 Write unit tests for all likelihood functions
    - Test CMB likelihood against known parameter sets and expected χ² values
    - Verify BAO likelihood computations for both isotropic and anisotropic cases
    - Test supernova likelihood with reference datasets and parameter combinations
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 6. Implement unified optimization engine
  - [x] 6.1 Create engine.py with core run_fit() function
    - Implement main optimization orchestration logic using scipy.optimize
    - Add parameter building via parameter.build_params() integration
    - Create likelihood dispatching system for multiple datasets
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 6.2 Add χ² summation and optimization execution
    - Implement total χ² computation by summing contributions from all requested blocks
    - Add optimization method selection (minimize, differential evolution)
    - Create result dictionary construction with parameters, χ² breakdown, and metrics
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 6.3 Integrate logging and diagnostics output
    - Add standardized result logging using logging_utils functions
    - Implement diagnostic output for parameter values and fit quality metrics
    - Create structured return format for downstream analysis and reporting
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 6.4 Write integration tests for optimization engine
    - Test run_fit() with individual datasets (CMB, BAO, SN) and verify results
    - Test joint fitting mode with multiple datasets and parameter consistency
    - Verify optimization convergence and result reproducibility
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 7. Implement logging and diagnostics system
  - [x] 7.1 Create logging_utils.py with standardized output formatting
    - Implement log_run() function for consistent diagnostic output across all fitters
    - Add log_diagnostics() for physics consistency checks and parameter reporting
    - Create format_results_table() for human-readable and machine-parseable output
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 7.2 Add physics consistency diagnostic checks
    - Implement H(z) ratio verification between PBUF and ΛCDM models
    - Add recombination redshift validation against Planck 2018 reference values
    - Create covariance matrix property checks and numerical stability diagnostics
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 7.3 Write unit tests for logging and diagnostics
    - Test standardized log output format and content consistency
    - Verify physics consistency checks and diagnostic computations
    - Test error handling and edge cases in diagnostic functions
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 8. Implement integrity validation system
  - [x] 8.1 Create integrity.py with physics validation functions
    - Implement verify_h_ratios() for Hubble parameter consistency checks
    - Add verify_recombination() for z* computation validation against references
    - Create verify_covariance_matrices() for dataset covariance property checks
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 8.2 Add comprehensive integrity test suite
    - Implement run_integrity_suite() combining all validation checks
    - Add sound horizon verification against Eisenstein & Hu reference values
    - Create unit consistency checks and dimensional analysis validation
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 8.3 Write unit tests for integrity validation system
    - Test individual validation functions with known good and bad parameter sets
    - Verify integrity suite execution and comprehensive reporting
    - Test edge cases and numerical precision requirements
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 9. Create wrapper scripts for individual fitters
  - [x] 9.1 Implement fit_cmb.py wrapper script
    - Create thin wrapper calling engine.run_fit() with CMB dataset only
    - Add command-line argument parsing for parameter overrides and options
    - Ensure identical interface to legacy fit_cmb.py for backward compatibility
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.2 Implement BAO wrapper scripts (fit_bao.py and fit_aniso.py)
    - Create fit_bao.py wrapper for isotropic BAO fitting using engine
    - Implement fit_aniso.py wrapper for anisotropic BAO fitting
    - Add proper dataset selection and mode configuration for each wrapper
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.3 Implement supernova and joint fitting wrappers
    - Create fit_sn.py wrapper for supernova-only fitting
    - Implement fit_joint.py wrapper for multi-dataset joint fitting
    - Add comprehensive command-line interface and configuration options
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 9.4 Write integration tests for all wrapper scripts
    - Test each wrapper script execution and output format consistency
    - Verify command-line argument parsing and parameter override functionality
    - Test backward compatibility with legacy script interfaces
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 10. Implement numerical equivalence validation
  - [x] 10.1 Create parity testing framework for legacy comparison
    - Implement side-by-side execution of legacy and unified systems
    - Add numerical comparison functions with configurable tolerance (1e-6)
    - Create comprehensive parity reports for all metrics (χ², AIC, BIC, parameters)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 10.2 Execute comprehensive parity validation across all blocks
    - Run parity tests for CMB fitting comparing legacy vs unified results
    - Execute BAO (isotropic and anisotropic) parity validation
    - Perform supernova and joint fitting numerical equivalence verification
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 10.3 Generate validation reports and address discrepancies
    - Create detailed parity reports documenting all numerical comparisons
    - Investigate and resolve any discrepancies exceeding tolerance thresholds
    - Document validation results and system equivalence certification
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 11. Add command-line interface and configuration options
  - [x] 11.1 Implement --verify-integrity command-line flag
    - Add integrity validation mode to all wrapper scripts
    - Create comprehensive integrity reporting with pass/fail status
    - Add configurable tolerance settings for physics consistency checks
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 11.2 Add configuration file support and advanced options
    - Implement configuration file parsing for default parameters and settings
    - Add optimizer selection options (scipy minimize vs differential evolution)
    - Create output format options (JSON, CSV, human-readable tables)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 12. Final integration and documentation
  - [x] 12.1 Integrate all components and perform end-to-end testing
    - Execute complete system integration testing across all fitters
    - Verify seamless operation of unified architecture with all datasets
    - Test extensibility by adding a mock new cosmological model
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 12.2 Create comprehensive documentation and usage examples
    - Document all public APIs and configuration options
    - Create usage examples for individual and joint fitting scenarios
    - Add developer documentation for extending the system with new models
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 12.3 Perform final validation and prepare for deployment
    - Execute final numerical equivalence tests and generate certification report
    - Verify all integrity checks pass and system meets performance requirements
    - Create migration guide and deployment checklist for production use
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_