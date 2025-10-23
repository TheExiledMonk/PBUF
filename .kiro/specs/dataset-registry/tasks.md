# Implementation Plan

## Release Milestones

| Milestone | Criteria | Expected Tag |
|-----------|----------|--------------|
| Phase 1 completion | Downloader + verification pass | v1.0-alpha |
| Phase 2 completion | Registry + provenance integrated | v1.0-beta |
| Phase 3 completion | Full pipeline integration (CMB/SN/BAO real data) | v1.0 |

## Roles and Responsibilities

- **Kiro**: Implementation, testing, integration
- **Fabian**: Architectural oversight, scientific validation, data curation, final review
- **Automated agents**: Continuous integration & regression tests

- [x] 1. Set up core infrastructure and manifest management
  - Create directory structure for dataset registry components
  - Implement JSON schema validation for dataset manifest
  - Create manifest parser with error handling and validation
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 1.1 Create dataset registry module structure
  - Create `pipelines/dataset_registry/` directory with `__init__.py`
  - Create subdirectories for `core/`, `protocols/`, `verification/`, `integration/`
  - Set up module imports and basic package structure
  - _Requirements: 1.1_

- [x] 1.2 Implement dataset manifest schema and validation
  - Create `core/manifest_schema.py` with JSON schema definition
  - Implement `DatasetManifest` class with schema validation
  - Add manifest parsing with comprehensive error reporting
  - Create example manifest file with CMB, BAO, and SN datasets
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 1.3 Build manifest query and lookup functionality
  - Implement dataset metadata lookup by canonical name
  - Add manifest versioning support and migration utilities
  - Create manifest update and validation methods
  - _Requirements: 1.4_

- [x] 2. Implement download manager with multi-protocol support
  - Create base download interface and HTTP protocol implementation
  - Add retry logic with exponential backoff and mirror fallback
  - Implement progress reporting and cancellation support
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.1 Create download manager base classes
  - Implement `DownloadManager` base class with protocol interface
  - Create `HTTPDownloader` class with requests-based implementation
  - Add download progress tracking and cancellation support
  - _Requirements: 2.1, 2.3_

- [x] 2.2 Implement retry logic and fallback mechanisms
  - Add exponential backoff retry strategy with configurable limits
  - Implement automatic fallback to mirror sources on failure
  - Create comprehensive error handling with detailed diagnostics
  - _Requirements: 2.2, 2.4, 2.5_

- [x] 2.3 Add file caching and decompression support
  - Implement local file caching to avoid redundant downloads
  - Add ZIP and TAR extraction with target file selection
  - Create file integrity checks during download process
  - _Requirements: 2.2, 2.3_

- [x] 3. Build verification engine with comprehensive validation
  - Implement SHA256 checksum verification with detailed reporting
  - Add file size validation and schema structure checking
  - Create verification result tracking and error reporting
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.1 Create verification engine core
  - Implement `VerificationEngine` class with validation interface
  - Add SHA256 checksum calculation and comparison methods
  - Create file size validation with tolerance checking
  - _Requirements: 3.1, 3.2_

- [x] 3.2 Implement schema validation for dataset formats
  - Add ASCII table format validation for CMB and BAO data
  - Implement column structure and data type validation
  - Create schema validation for different dataset types
  - _Requirements: 3.3, 3.4_

- [x] 3.3 Build verification reporting and error handling
  - Create detailed verification result objects with error details
  - Implement comprehensive error reporting with resolution suggestions
  - Add verification failure recovery strategies
  - _Requirements: 3.5_

- [x] 4. Create registry manager with immutable provenance tracking
  - Implement registry entry creation with complete metadata
  - Add immutable audit trail maintenance with file locking
  - Create registry queries and export functionality
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 4.1 Implement registry storage and entry management
  - Create `RegistryManager` class with JSON file storage
  - Implement registry entry creation with mandatory metadata
  - Add file locking for concurrent write safety
  - _Requirements: 4.1, 4.2_

- [x] 4.2 Add provenance tracking with environment fingerprinting
  - Implement environment metadata collection (PBUF commit, Python version, platform)
  - Create immutable audit trail with append-only operations
  - Add timestamp and agent tracking for all registry operations
  - _Requirements: 4.2, 4.3, 4.4_

- [x] 4.3 Build registry query and export capabilities
  - Implement registry listing and filtering by dataset properties
  - Add provenance record export for publication materials
  - Create registry summary generation for proof bundles
  - Manual QA: Verify registry JSON readability and provenance completeness for at least one dataset (CMB Planck 2018)
  - _Requirements: 4.5_

**Phase 1 Deliverable Checkpoint**: Core infrastructure complete - all datasets fetchable and verifiable via manifest; basic registry operations functional.

- [x] 5. Integrate with existing PBUF pipeline infrastructure
  - Replace existing `load_dataset()` function with registry-based implementation
  - Add pre-run dataset verification to fitting pipelines
  - Implement backward compatibility during transition period
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5.1 Create drop-in replacement for existing dataset loading
  - Modify `pipelines/fit_core/datasets.py` to use registry when available
  - Implement `load_dataset()` wrapper that fetches and verifies datasets
  - Add fallback to existing loading logic during transition
  - _Requirements: 5.1, 5.3_

- [x] 5.2 Add pre-run verification to fitting pipelines
  - Implement `verify_all_datasets()` function for pipeline integration
  - Add dataset verification calls to `run_fit()` functions in all pipelines
  - Create clear error messages when datasets are missing or corrupted
  - _Requirements: 5.2_

- [x] 5.3 Integrate provenance tracking with fit results
  - Modify fit result objects to include dataset provenance information
  - Add registry entries to proof bundle generation
  - Update result serialization to include dataset metadata
  - _Requirements: 5.4_

- [x] 6. Add manual dataset registration for proprietary data
  - Implement manual dataset registration interface with checksum validation
  - Create consistent handling for manually registered datasets
  - Add manual registration validation and metadata collection
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 6.1 Create manual dataset registration interface
  - Implement `register_manual_dataset()` function with metadata form
  - Add checksum calculation and validation for manual files
  - Create manual dataset entry creation with proper provenance
  - _Requirements: 6.1, 6.3_

- [x] 6.2 Ensure consistent behavior for manual datasets
  - Integrate manual datasets with existing registry query functions
  - Add manual dataset verification and re-verification support
  - Create identical provenance tracking for manual and downloaded datasets
  - _Requirements: 6.2, 6.4_

**Phase 2 Deliverable Checkpoint**: Registry and provenance systems integrated - immutable audit trails operational; manual dataset support complete.

- [x] 7. Build administrative and management capabilities
  - Implement dataset listing and metadata display functionality
  - Add integrity re-verification and corruption detection
  - Create export utilities for publication materials
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 7.1 Create dataset management CLI interface
  - Implement command-line tool for dataset listing and status
  - Add dataset re-verification and integrity checking commands
  - Create dataset cleanup and orphan file detection
  - _Requirements: 7.1, 7.2, 7.5_

- [x] 7.2 Build export and summary generation tools
  - Implement manifest summary table export for publications
  - Add complete audit trail export functionality
  - Create provenance summary generation for papers and reports
  - _Requirements: 7.3, 7.4_

- [x] 8. Implement one-command reproducibility system
  - Create comprehensive dataset fetch and verification command
  - Add progress reporting and completion status for reproduction
  - Implement detailed diagnostics for reproduction failures
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 8.1 Build one-command dataset preparation
  - Implement `fetch_all_datasets()` function for complete workflow reproduction
  - Add parallel dataset downloading with progress reporting
  - Create comprehensive verification of all required datasets
  - _Requirements: 8.1, 8.2_

- [x] 8.2 Add reproduction diagnostics and error handling
  - Implement detailed progress feedback during reproduction process
  - Create comprehensive error diagnostics for troubleshooting failures
  - Add recovery suggestions for common reproduction issues
  - _Requirements: 8.3, 8.4_

- [x] 9. Add extensibility and version control integration
  - Implement versioned API interface for dataset access
  - Add support for future dataset types through manifest declarations
  - Create version control integration with commit hash tracking
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 9.1 Create extensible dataset interface
  - Implement versioned API for dataset requests by canonical names
  - Add plugin architecture for new dataset types and protocols
  - Create interface documentation for future extensions
  - _Requirements: 9.1, 9.2_

- [x] 9.2 Integrate with PBUF version control system
  - Add automatic PBUF commit hash recording in registry entries
  - Implement environment fingerprinting for complete reproducibility
  - Create version-aware dataset compatibility checking
  - _Requirements: 9.3, 9.4_

**Phase 3 Deliverable Checkpoint**: Full pipeline integration complete - all datasets fetchable and verified via manifest; registry entries reproducible and auditable with real CMB/BAO/SN data.

- [x] 10. Add structured logging and observability
  - Implement JSON Lines logging for all registry operations
  - Add integration with existing PBUF logging infrastructurei think 
  - Create audit trail export for external monitoring systems
  - _Requirements: Performance, Transparency_

- [x] 10.1 Create structured logging system
  - Implement JSON Lines event logging for all dataset operations
  - Add timestamp and correlation ID tracking for operation tracing
  - Create log level configuration and filtering capabilities
  - _Requirements: Transparency_

- [x] 10.2 Integrate with PBUF logging infrastructure
  - Connect registry logging with existing pipeline logging systems
  - Add registry events to proof bundle audit trails
  - Create log aggregation for comprehensive operation tracking
  - _Requirements: Transparency_

- [x] 11. Create comprehensive test suite
  - Write unit tests for all core components with mock data
  - Add integration tests for end-to-end dataset workflows
  - Create performance benchmarks for download and verification operations
  - _Requirements: All requirements validation_

- [x] 11.1 Write unit tests for core components (Priority: Critical test paths)
  - Create test suite for manifest schema validation (HIGH PRIORITY)
  - Add tests for multi-protocol download with mirror fallback (HIGH PRIORITY)
  - Write SHA256 and schema verification tests (HIGH PRIORITY)
  - Add registry write/read immutability tests (HIGH PRIORITY)
  - Create tests for download manager with mock HTTP responses
  - Write verification engine tests with various data formats
  - _Requirements: 1.1-1.4, 2.1-2.5, 3.1-3.5_

- [x] 11.2 Add integration and performance tests
  - Create end-to-end tests for complete dataset fetch and verification
  - Add performance benchmarks for large dataset operations
  - Write pipeline integration tests with existing PBUF workflows
  - _Requirements: 5.1-5.5, Performance requirements_

- [ ] 12. Create configuration and deployment support
  - Add configuration file support for registry settings
  - Create deployment documentation and setup instructions
  - Implement configuration migration from existing dataset settings
  - _Requirements: Portability, Integration_

- [ ] 12.1 Implement configuration management
  - Add registry configuration section to existing PBUF config system
  - Create configuration validation and default value handling
  - Implement environment-specific configuration overrides
  - _Requirements: Portability_

- [ ] 12.2 Create deployment and migration tools
  - Write deployment documentation with setup instructions
  - Create migration scripts for existing dataset configurations
  - Add backward compatibility validation during deployment
  - _Requirements: Integration with existing systems_

**Final Deliverable Checkpoint**: All configuration and migration tools operational; ready for real-data validation runs and production deployment.