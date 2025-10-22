# Requirements Document

## Introduction

The PBUF cosmology pipeline refactor aims to transform the current organically-grown fitting infrastructure into a unified, modular architecture. The current system suffers from code duplication, parameter drift, and inconsistent results between individual fitters (CMB, BAO, SN, Joint). This refactor will centralize parameter handling, create a unified optimization engine, and ensure physical consistency across all cosmological blocks while maintaining scientific accuracy and enabling easy extension for new models.

## Requirements

### Requirement 1

**User Story:** As a cosmology researcher, I want a unified parameter handling system, so that all fitters use the same source of truth and eliminate parameter drift between blocks.

#### Acceptance Criteria

1. WHEN any fitter is executed THEN the system SHALL use centralized parameter construction from a single source
2. WHEN ΛCDM parameters are requested THEN the system SHALL return identical default values across all blocks
3. WHEN PBUF parameters are requested THEN the system SHALL return consistent alpha, Rmax, eps0, n_eps, and k_sat values
4. WHEN derived parameters are computed THEN the system SHALL use cmb_priors.prepare_background_params consistently
5. IF parameter overrides are provided THEN the system SHALL apply them uniformly across all fitters

### Requirement 2

**User Story:** As a researcher, I want a unified optimization engine, so that all fitters share the same core logic and produce consistent results.

#### Acceptance Criteria

1. WHEN any individual fitter is called THEN the system SHALL use the same optimization engine as joint fitting
2. WHEN optimization is performed THEN the system SHALL dispatch appropriate likelihoods for each dataset
3. WHEN χ² is computed THEN the system SHALL sum contributions using identical logic across all blocks
4. WHEN optimization completes THEN the system SHALL compute AIC, BIC, and degrees of freedom using consistent formulas
5. WHEN results are logged THEN the system SHALL use standardized formatting across all fitters

### Requirement 3

**User Story:** As a researcher, I want physical consistency guarantees, so that recombination and drag epoch calculations are properly isolated to their respective blocks.

#### Acceptance Criteria

1. WHEN CMB fitting is performed THEN the system SHALL use recombination (z*) calculations exclusively
2. WHEN BAO fitting is performed THEN the system SHALL use drag epoch (z_d) calculations exclusively  
3. WHEN any fitting is performed THEN the system SHALL use identical physical constants (Tcmb=2.7255, Neff=3.046)
4. WHEN model equations are applied THEN the system SHALL reference documented derivations in documents/
5. IF physical constants differ between blocks THEN the system SHALL log the discrepancy

### Requirement 4

**User Story:** As a developer, I want eliminated code duplication, so that χ², AIC, BIC, and dataset handling logic exists in only one place.

#### Acceptance Criteria

1. WHEN χ² is calculated THEN the system SHALL use a single chi2_generic function across all blocks
2. WHEN AIC/BIC metrics are computed THEN the system SHALL use centralized statistics functions
3. WHEN datasets are loaded THEN the system SHALL use unified dataset loader abstraction
4. WHEN any statistical computation is performed THEN the system SHALL ensure consistent implementation
5. IF duplicate logic is detected THEN the system SHALL consolidate it into shared modules

### Requirement 5

**User Story:** As a researcher, I want reproducible results, so that identical inputs always produce identical outputs with deterministic behavior.

#### Acceptance Criteria

1. WHEN the same parameters are used THEN the system SHALL produce bit-for-bit identical results
2. WHEN random seeds are set THEN the system SHALL ensure deterministic optimization behavior
3. WHEN units are processed THEN the system SHALL maintain consistent unit handling across all calculations
4. WHEN parameter dictionaries are built THEN the system SHALL ensure identical structure across all blocks
5. WHEN numerical precision is required THEN the system SHALL maintain tolerance within 1e-6 for critical metrics

### Requirement 6

**User Story:** As a researcher, I want structured logging and diagnostics, so that I can verify system integrity and debug issues effectively.

#### Acceptance Criteria

1. WHEN any fitter runs THEN the system SHALL log standardized diagnostic information
2. WHEN integrity checks are requested THEN the system SHALL verify FRW vs model H(z) ratios
3. WHEN recombination is computed THEN the system SHALL verify r_s(z*) against Planck2018 reference
4. WHEN covariance matrices are used THEN the system SHALL verify positive definiteness
5. IF discrepancies exceed 1e-4 THEN the system SHALL output warnings with specific values

### Requirement 7

**User Story:** As a developer, I want extensible architecture, so that new cosmological models can be easily integrated without modifying core logic.

#### Acceptance Criteria

1. WHEN a new model is added THEN the system SHALL support it through parameter configuration only
2. WHEN new likelihood functions are needed THEN the system SHALL allow registration without core changes
3. WHEN new datasets are introduced THEN the system SHALL integrate them through the unified loader
4. WHEN model-specific equations are required THEN the system SHALL reference documented physics in documents/
5. IF new optimization methods are needed THEN the system SHALL support them through engine configuration

### Requirement 8

**User Story:** As a researcher, I want migration safety, so that the refactored system produces identical results to the legacy system during transition.

#### Acceptance Criteria

1. WHEN legacy and unified systems are compared THEN χ² values SHALL match within 1e-6 tolerance
2. WHEN physical parameters are computed THEN r_s(z*) and l_A values SHALL match legacy within 1e-6
3. WHEN statistical metrics are calculated THEN AIC, BIC, and p-values SHALL be identical between systems
4. WHEN validation tests are run THEN all numerical equivalence tests SHALL pass
5. IF discrepancies are found THEN the system SHALL report specific metric differences and locations