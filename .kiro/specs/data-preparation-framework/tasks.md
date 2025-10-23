# Implementation Plan

- [x] 1. Set up core framework structure and interfaces
  - Create directory structure for data preparation framework components
  - Define abstract base classes and core interfaces for derivation modules
  - Implement standardized dataset schema and validation methods
  - _Requirements: 1.1, 2.2, 4.1_
  - **Deliverable checkpoint**: Core framework structure and interfaces implemented and passing unit tests

- [x] 2. Implement registry integration layer
  - [x] 2.1 Create registry integration module with existing RegistryManager interface
    - Write RegistryIntegration class that interfaces with existing RegistryManager
    - Implement methods to retrieve verified raw datasets and metadata
    - Add provenance tracking for derived datasets with source hash references
    - _Requirements: 3.1, 3.2, 6.1, 6.4_

  - [x] 2.2 Implement dataset metadata extraction and validation
    - Create functions to extract dataset type and processing requirements from registry metadata
    - Validate raw dataset integrity before processing
    - Handle registry entry status checking and error reporting
    - _Requirements: 4.1, 6.1, 7.2_
  - **Deliverable checkpoint**: Registry integration layer complete with provenance tracking

- [x] 3. Build validation engine with comprehensive checks
  - [x] 3.1 Implement schema validation system
    - Create schema validation functions for standardized dataset format
    - Validate required fields (z, observable, uncertainty, metadata) and data types
    - Implement field presence and format checking with detailed error messages
    - _Requirements: 1.1, 4.1, 8.1_

  - [x] 3.2 Create numerical integrity validation
    - Implement checks for NaN, infinity, and negative variance detection
    - Add redshift range consistency validation against known catalog limits
    - Create monotonicity and physical sanity checks for observables
    - _Requirements: 4.2, 8.2_

  - [x] 3.3 Build covariance matrix validation
    - Implement symmetry checking for covariance matrices
    - Add positive-definiteness validation using eigenvalue decomposition
    - Create correlation coefficient range validation (-1 to 1)
    - _Requirements: 4.2, 8.2_
  - **Deliverable checkpoint**: All validation functions implemented and passing unit tests

- [x] 4. Implement core preparation engine
  - [x] 4.1 Create main orchestration engine
    - Build DataPreparationFramework class that coordinates the workflow
    - Implement dataset type detection and module dispatching
    - Add error handling and recovery mechanisms with detailed logging
    - _Requirements: 2.1, 7.1, 7.2_

  - [x] 4.2 Implement processing pipeline coordination
    - Create workflow that loads raw data, applies transformations, and validates output
    - Add deterministic processing with environment hash tracking
    - Implement caching mechanism to avoid reprocessing identical inputs
    - _Requirements: 3.3, 9.3_
  - **Deliverable checkpoint**: Core preparation engine operational with deterministic processing

- [x] 5. Create dataset-specific derivation modules
  - [x] 5.1 Implement Supernova (SN) derivation module
    - Create SN-specific transformation logic for magnitude to distance modulus conversion
    - Implement duplicate removal by coordinate matching and calibration homogenization
    - Add systematic covariance matrix application and z-μ-σμ extraction
    - _Requirements: 5.1_

  - [x] 5.2 Implement BAO derivation module  
    - Create BAO processing for both isotropic (D_V/r_d) and anisotropic (D_M/r_d, D_H/r_d) measurements
    - Implement distance measure unit conversion to consistent Mpc units
    - Add correlation matrix validation and survey-specific systematic corrections
    - _Requirements: 5.2_

  - [x] 5.3 Implement CMB derivation module
    - Create CMB distance priors extraction from Planck files (R, l_A, θ_*)
    - Implement dimensionless consistency checking and covariance matrix application
    - Add cosmological constant validation and parameter extraction logic
    - _Requirements: 5.3_

  - [x] 5.4 Implement Cosmic Chronometers (CC) derivation module
    - Create H(z) data merging from multiple compilation sources
    - Implement overlapping redshift bin filtering and uncertainty propagation
    - Add H(z) sign convention validation and systematic error handling
    - _Requirements: 5.4_

  - [x] 5.5 Implement RSD derivation module
    - Create growth rate (fσ₈) processing with sign convention validation
    - Implement covariance homogenization from published sources
    - Add survey-specific correction application and error propagation
    - _Requirements: 5.5_
  - **Deliverable checkpoint**: All dataset-specific derivation modules implemented and tested
  - [x] 6. Build output manager and format conversion
  - [x] 6.1 Implement standardized output generation
    - Create output manager that generates analysis-ready datasets in standard format
    - Implement file I/O handling for CSV, Parquet, and NumPy formats
    - Add metadata serialization and provenance record creation
    - _Requirements: 6.2, 3.2_

  - [x] 6.2 Create compatibility layer with existing fit pipelines
    - Implement conversion from StandardDataset to existing DatasetDict format
    - Ensure seamless integration with pipelines/fit_core/datasets.py interface
    - Add backward compatibility testing with existing fitting workflows
    - _Requirements: 6.3_
  - **Deliverable checkpoint**: Output manager and compatibility layer complete

- [x] 7. Integrate with existing PBUF infrastructure
  - [x] 7.1 Enhance existing datasets.py with framework integration
    - Modify load_dataset() function to use preparation framework when available
    - Implement fallback mechanism to legacy loading during transition period
    - Add framework availability detection and graceful degradation
    - _Requirements: 6.1, 6.3_

  - [x] 7.2 Create provenance integration with existing registry
    - Implement derived dataset registration with complete provenance tracking
    - Add source registry entry hash references and environment snapshot links
    - Reference latest environment registry entry (data/registry/environment_*.json) in derived dataset provenance
    - Create audit trail integration with existing registry audit system
    - _Requirements: 3.1, 3.2, 3.4_
  - **Deliverable checkpoint**: Full PBUF infrastructure integration complete

- [x] 8. Implement comprehensive error handling and logging
  - [x] 8.1 Create detailed error reporting system
    - Implement ProcessingError class with comprehensive error context
    - Add stage-specific error handling (input validation, transformation, output validation)
    - Create human-readable error reports with suggested remediation actions
    - _Requirements: 7.2, 4.2_

  - [x] 8.2 Build transformation logging and audit trails
    - Implement detailed logging of all transformation steps with formula references
    - Add processing summaries suitable for publication materials
    - Create performance metrics tracking and system health monitoring
    - _Requirements: 7.1, 7.3, 7.4_
  - **Deliverable checkpoint**: Comprehensive error handling and logging system operational

- [x] 9. Create comprehensive testing suite
  - [x] 9.1 Implement unit tests for all components
    - Write unit tests for each derivation module with known input/output pairs
    - Create validation engine tests covering all validation rules and edge cases
    - Add schema compliance tests for standardized dataset format
    - _Requirements: 8.1, 8.2_

  - [x] 9.2 Build integration tests for system components
    - Create end-to-end workflow tests for complete preparation pipeline
    - Write registry integration tests for data retrieval and provenance recording
    - Add fit pipeline integration tests for compatibility verification
    - _Requirements: 8.4, 8.5_

  - [x] 9.3 Implement validation and performance tests
    - Create round-trip tests that verify deterministic behavior with identical inputs
    - Write cross-validation tests comparing outputs with existing legacy loaders
    - Add performance tests ensuring full Phase A prep ≤ 10 min on reference workstation
    - _Requirements: 8.3, 9.1_
  - **Deliverable checkpoint**: Comprehensive testing suite complete with all tests passing

- [x] 10. Deploy and validate framework with Phase A datasets
  - [x] 10.1 Process and validate all Phase A datasets
    - Run complete preparation pipeline on CMB, SN, and BAO (iso/aniso) datasets
    - Validate all derived datasets pass comprehensive validation checks
    - Generate quality assurance reports and processing summaries
    - _Requirements: 8.5_

  - [x] 10.2 Create deployment documentation and user guides
    - Write comprehensive API documentation for all framework components
    - Create user guides for adding new dataset types and derivation modules
    - Document integration procedures with existing PBUF infrastructure
    - _Requirements: 2.2, 7.1_

  - [x] 10.3 Perform final system certification
    - Execute complete test suite and validate 100% pass rate
    - Generate certification report demonstrating framework readiness
    - Create deployment checklist and operational procedures
    - _Requirements: 8.5, 9.1_
  - **Deliverable checkpoint**: Framework deployed and certified for production use- [ ]*
 11. Optional future work: Enhanced security and integrity
  - [ ]* 11.1 Implement cryptographic signing of provenance and output archives
    - Add cryptographic signing capabilities for long-term data authenticity
    - Implement digital signature verification for provenance records
    - Create secure archive generation with tamper-evident sealing
    - _Requirements: Future enhancement for long-term data integrity_
  - **Deliverable checkpoint**: Enhanced security features for long-term data authenticity

## Success Criteria Summary

Upon completion of this implementation plan, the framework shall demonstrate:

✅ **100% test pass rate** - All unit, integration, and validation tests passing
✅ **Deterministic re-runs** - Identical checksums for derived datasets with identical inputs and environment hashes  
✅ **Provenance registry entries complete** - All derived datasets registered with complete lineage information
✅ **Documentation and API published** - Comprehensive documentation and user guides available
✅ **Phase A dataset certification** - CMB, SN, and BAO (iso/aniso) datasets successfully processed and validated
✅ **Performance benchmarks met** - Full Phase A preparation completed within defined runtime thresholds
✅ **Integration compatibility** - Seamless operation with existing PBUF fitting pipelines