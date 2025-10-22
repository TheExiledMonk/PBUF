# Implementation Plan

- [x] 1. Create enhanced BAO anisotropic fitting script
  - Create `pipelines/fit_bao_aniso.py` as enhanced version of existing `fit_aniso.py`
  - Integrate with `OptimizedParameterStore` for CMB-optimized parameter usage
  - Add comprehensive command-line argument parsing matching other pipeline scripts
  - Implement parameter source tracking and validation metadata
  - _Requirements: 1.1, 3.1, 4.1_

- [x] 2. Implement optimized parameter integration
  - Add parameter store initialization and optimized parameter retrieval
  - Implement fallback to default parameters when optimization unavailable
  - Add parameter override handling on top of optimized/default values
  - Include parameter source metadata in results for transparency
  - _Requirements: 1.1, 3.1, 3.2_

- [x] 3. Enhance integrity validation for anisotropic BAO
  - Extend integrity checking with anisotropic BAO-specific validations
  - Add covariance matrix validation for 2N×2N structure (N redshift bins)
  - Implement physics consistency checks for transverse/radial BAO ratios
  - Add comprehensive integrity reporting with anisotropic-specific metrics
  - _Requirements: 2.1, 2.2, 5.1_

- [x] 4. Implement enhanced result formatting and reporting
  - Create detailed human-readable output with anisotropic BAO predictions
  - Add parameter source information display (optimized vs default)
  - Implement JSON output format for programmatic access
  - Include validation metadata and optimization status in results
  - _Requirements: 4.1, 5.1_

- [x] 5. Add joint fit separation validation
  - Implement checks to prevent simultaneous use of "bao" and "bao_ani" datasets
  - Add validation warnings when both isotropic and anisotropic BAO are requested
  - Create configuration validation for proper dataset separation
  - Document best practices for BAO dataset selection
  - _Requirements: 1.1, 1.2_

- [x] 6. Create comprehensive test suite
  - Write unit tests for parameter loading and optimization integration
  - Create validation tests comparing against existing `fit_aniso.py` results
  - Implement integration tests with fit_core infrastructure
  - Add performance benchmarks for fitting execution time
  - _Requirements: 5.1_

- [ ]* 7. Add parity validation against existing implementation
  - Create test cases comparing new implementation against `fit_aniso.py`
  - Validate that results match within statistical tolerance
  - Test parameter optimization benefits in anisotropic BAO context
  - Document any differences and their justifications
  - _Requirements: 5.1, 5.2_