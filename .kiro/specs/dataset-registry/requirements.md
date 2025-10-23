# Requirements Document

## Introduction

The Centralized Dataset Downloader & Verification Registry provides a unified system for acquiring, verifying, and documenting all external datasets used in PBUF cosmological analyses. This system ensures complete reproducibility by maintaining structured provenance records and eliminating scattered hard-coded URLs throughout the codebase. Every dataset used in fits, tests, or publications can be traced back to a verified source with full chain of custody documentation.

### Scope Statement

This system governs dataset acquisition and verification across all PBUF modules including pipelines, proofs, and reports. It does not define analytical logic or cosmological model code; rather, it provides verified data access and provenance integration to those modules.

### Data Provenance Model

The system implements the following data flow:
```
[Manifest] → [Downloader] → [Verification Engine] → [Registry JSON]
                                     ↓
                          [Audit Log + Proof Bundle]
```

### FAIR Principles Alignment

The design adheres to FAIR data principles by making all datasets findable (via manifest metadata), accessible (via verified sources), interoperable (standard JSON schema), and reusable (licensed and cited in provenance records).

## Requirements

### Requirement 1

**User Story:** As a cosmology researcher, I want all dataset definitions centralized in a single manifest file, so that I can easily understand and modify data sources without hunting through scattered pipeline code.

#### Acceptance Criteria

1. WHEN the system is initialized THEN it SHALL create a human-readable manifest file (JSON or YAML) containing all dataset definitions
2. WHEN a dataset is defined in the manifest THEN it SHALL include canonical name, description, citation reference, download sources, file structure expectations, and expected checksums
3. WHEN multiple download sources are specified THEN the system SHALL support primary and mirror URLs for redundancy
4. WHEN the manifest is updated THEN all subsequent operations SHALL use the updated definitions without code changes

### Requirement 2

**User Story:** As a PBUF pipeline developer, I want datasets to be automatically fetched and verified, so that I can focus on analysis rather than data management logistics.

#### Acceptance Criteria

1. WHEN a dataset is requested THEN the system SHALL automatically download it from configured sources
2. WHEN a dataset already exists locally and passes checksum validation THEN the system SHALL skip redundant downloads
3. WHEN multiple protocols are needed THEN the system SHALL support HTTPS, Zenodo, arXiv, local mirrors, and manual upload methods
4. WHEN a download fails from the primary source THEN the system SHALL attempt mirror sources automatically
5. WHEN all download attempts fail THEN the system SHALL provide clear diagnostic information

### Requirement 3

**User Story:** As a scientific researcher, I want every dataset to be cryptographically verified, so that I can trust the integrity of my analysis inputs.

#### Acceptance Criteria

1. WHEN a dataset is downloaded or detected THEN the system SHALL verify its SHA256 checksum
2. WHEN checksum validation is enabled THEN the system SHALL compare against expected values from the manifest
3. WHEN size validation is configured THEN the system SHALL check file sizes against optional thresholds
4. WHEN schema validation is enabled THEN the system SHALL verify expected columns, dimensions, or structure
5. WHEN any verification fails THEN the system SHALL halt processing and provide detailed failure reasons

### Requirement 4

**User Story:** As a reproducibility auditor, I want complete provenance records for every dataset, so that I can verify the scientific chain of custody.

#### Acceptance Criteria

1. WHEN a dataset is verified THEN the system SHALL create a structured registry entry with metadata
2. WHEN registry entries are created THEN they SHALL include dataset name, version, source URLs, download timestamp, checksum, file size, and verification results
3. WHEN registry entries are created THEN they SHALL include responsible agent identifier and citation metadata
4. WHEN registry entries exist THEN they SHALL be immutable and append-only for audit trails
5. WHEN proof bundles are generated THEN verified dataset entries SHALL be automatically included

### Requirement 5

**User Story:** As a PBUF pipeline user, I want seamless integration with existing fitting and verification workflows, so that dataset management becomes transparent.

#### Acceptance Criteria

1. WHEN a pipeline run starts THEN the system SHALL perform pre-run checks ensuring all required datasets are present and verified
2. WHEN required datasets are missing or corrupted THEN the system SHALL halt execution with clear diagnostics
3. WHEN datasets are referenced in pipelines THEN they SHALL use canonical names rather than file paths
4. WHEN proof runs execute THEN verified dataset entries SHALL be attached to provenance blocks automatically
5. WHEN integration is complete THEN existing pipeline code SHALL require minimal modifications

### Requirement 6

**User Story:** As a researcher with proprietary datasets, I want to manually register non-public data while maintaining the same provenance standards, so that all datasets are treated consistently.

#### Acceptance Criteria

1. WHEN manual datasets need registration THEN the system SHALL support checksum and metadata form input
2. WHEN manual datasets are registered THEN they SHALL behave identically to automatically downloaded datasets
3. WHEN manual registration occurs THEN the system SHALL validate provided checksums and metadata
4. WHEN manual datasets are used THEN they SHALL appear in provenance records with appropriate source attribution

### Requirement 7

**User Story:** As a system administrator, I want comprehensive dataset management capabilities, so that I can maintain and audit the data infrastructure.

#### Acceptance Criteria

1. WHEN queried THEN the system SHALL list all verified datasets with complete metadata
2. WHEN integrity checks are requested THEN the system SHALL re-verify checksums to detect data corruption
3. WHEN publication materials are needed THEN the system SHALL export manifest summary tables for inclusion in papers
4. WHEN audit trails are required THEN the system SHALL provide complete history of all dataset operations
5. WHEN cleanup is needed THEN the system SHALL identify and remove orphaned or outdated dataset files

### Requirement 8

**User Story:** As a collaborating researcher, I want one-command reproducibility, so that I can easily replicate the entire PBUF workflow.

#### Acceptance Criteria

1. WHEN full reproduction is requested THEN a single command SHALL fetch and verify all required datasets
2. WHEN the reproduction command completes THEN all fits, figures, and statistics SHALL be reproducible from verified data
3. WHEN reproduction is attempted THEN the system SHALL provide progress feedback and completion status
4. WHEN reproduction fails THEN the system SHALL provide detailed diagnostics for troubleshooting

### Requirement 9

**User Story:** As a system architect, I want extensible interfaces and version control integration, so that the system can evolve with future PBUF needs.

#### Acceptance Criteria

1. WHEN the downloader is accessed THEN it SHALL expose a minimal, versioned interface for requesting datasets by canonical names
2. WHEN new dataset types are needed THEN they SHALL be addable via manifest declaration without modifying core logic
3. WHEN a dataset registry entry is created THEN the system SHALL record the current PBUF software commit hash and environment fingerprint
4. WHEN future extensions are required THEN the interface SHALL support gravitational-wave, weak-lensing, or other cosmological datasets

## Non-Functional Requirements

### Performance Requirements
- Verification operations SHALL complete within reasonable time (<2 minutes for 100 MB datasets)
- Concurrent downloads SHALL be supported for multiple datasets
- Local caching SHALL minimize redundant network operations

### Security Requirements  
- Only HTTPS or verified mirror protocols SHALL be permitted by default
- Checksum verification SHALL be mandatory for all downloaded datasets
- Manual dataset registration SHALL require explicit user confirmation

### Reliability Requirements
- Retry logic SHALL ensure mirror fallback without data corruption
- Network failures SHALL not corrupt partially downloaded files
- System SHALL gracefully handle interrupted operations with resume capability

### Portability Requirements
- The system SHALL operate in both offline and cluster environments
- Configuration SHALL be environment-independent via relative paths
- Dependencies SHALL be minimal and well-documented

### Transparency Requirements
- All actions SHALL be logged in a human-readable audit trail
- Registry entries SHALL be human-readable JSON with clear structure
- Error messages SHALL provide actionable diagnostic information