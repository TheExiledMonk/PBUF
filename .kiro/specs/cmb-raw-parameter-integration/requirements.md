# Requirements Document

## Introduction

The CMB Data Preparation Module Update enables processing of raw Planck-style cosmological parameter outputs containing background-level model parameters (H₀, Ωₘ, Ωᵦh², nₛ, τ) and internally computing distance priors such as R, ℓₐ, Ωᵦh², and θ*. This replaces the current assumption that distance priors arrive pre-computed, providing greater flexibility and scientific transparency in CMB data processing.

## Requirements

### Requirement 1: Raw Parameter Detection and Reading

**User Story:** As a cosmologist, I want the system to automatically detect and read raw Planck-style parameter files from the dataset registry, so that I can work with fundamental cosmological parameters rather than pre-computed distance priors.

#### Acceptance Criteria

1. WHEN the system processes a CMB dataset THEN it SHALL detect parameter files tagged as "dataset_type": "cmb" in the registry
2. WHEN parameter files are found THEN the system SHALL support both tabular (text/CSV) and structured (JSON/NumPy) formats
3. WHEN reading parameter names THEN the system SHALL auto-map case-insensitive variations including "H0", "Omega_m", "Omega_b_h2", "n_s", "tau", "As"
4. WHEN validating input parameters THEN the system SHALL reject data with NaN values, negative densities, or H₀ outside range [50, 80]
5. IF no raw parameters are detected THEN the system SHALL fall back to pre-computed distance-prior mode for backward compatibility

### Requirement 2: Distance Prior Derivation

**User Story:** As a researcher, I want the system to compute standard CMB distance priors (R, ℓₐ, θ*) from raw cosmological parameters using consistent background integrators, so that I have transparent and reproducible observables for fitting.

#### Acceptance Criteria

1. WHEN raw parameters are available THEN the system SHALL compute R = √(Ωₘ H₀²) × r(z*)/c using background integrators
2. WHEN computing acoustic scale THEN the system SHALL calculate ℓₐ = π × r(z*)/rₛ(z*) with consistent sound horizon integration
3. WHEN deriving angular scale THEN the system SHALL compute θ* = rₛ(z*)/r(z*) using the same numerical methods as BAO/SN calculations
4. WHEN z* is not specified THEN the system SHALL default to recombination redshift ≈ 1089.8
5. WHEN computations complete THEN the system SHALL validate derived quantities against published Planck tolerances (|Δ| < 1σ)

### Requirement 3: Covariance Matrix Propagation

**User Story:** As a data analyst, I want the system to properly propagate parameter uncertainties to derived distance priors through covariance matrix transformation, so that I maintain statistical rigor in my analysis.

#### Acceptance Criteria

1. WHEN parameter covariance is provided THEN the system SHALL load the covariance matrix from the same registry entry
2. WHEN propagating uncertainties THEN the system SHALL compute Jacobian J = ∂(R,ℓₐ,θ*)/∂(pᵢ) numerically around mean parameter values
3. WHEN transforming covariance THEN the system SHALL apply Cderived = J × Cparams × J^T
4. WHEN validating results THEN the system SHALL ensure covariance symmetry (|C - C^T| < 1e-8)
5. WHEN checking positive-definiteness THEN the system SHALL verify all eigenvalues > 0

### Requirement 4: Standardized Output Format

**User Story:** As a pipeline developer, I want the CMB module to output data in the same StandardDataset format as other preparation modules, so that downstream fitting code requires no modifications.

#### Acceptance Criteria

1. WHEN processing completes THEN the system SHALL produce StandardDataset with z=[z_recombination], observable=[R, ℓₐ, Ωᵦh², θ*]
2. WHEN creating metadata THEN the system SHALL include dataset_type="cmb", processing="derived from raw cosmological parameters"
3. WHEN documenting provenance THEN the system SHALL record parameters_used=["H0", "Omega_m", "Omega_b_h2", "n_s", "tau"]
4. WHEN storing results THEN the system SHALL include source citation and z_recombination value in metadata
5. WHEN interfacing with fitting code THEN the system SHALL maintain identical structure to pre-computed distance-prior datasets

### Requirement 5: Configuration and Backward Compatibility

**User Story:** As a system administrator, I want to configure whether the system uses raw parameters or pre-computed priors, so that I can maintain compatibility with existing workflows while enabling new capabilities.

#### Acceptance Criteria

1. WHEN configuring the system THEN it SHALL support use_raw_parameters: true/false flag
2. WHEN use_raw_parameters=false THEN the system SHALL operate in legacy pre-computed distance-prior mode
3. WHEN use_raw_parameters=true AND no raw parameters found THEN the system SHALL automatically fall back to legacy mode
4. WHEN switching modes THEN the system SHALL log the detection method and processing approach used
5. WHEN maintaining compatibility THEN existing fitting pipelines SHALL require no code changes

### Requirement 6: Validation and Error Handling

**User Story:** As a quality assurance engineer, I want comprehensive validation of parameter ranges, numerical stability, and physical consistency, so that I can trust the derived observables for scientific analysis.

#### Acceptance Criteria

1. WHEN validating parameters THEN the system SHALL check physical ranges: 50 < H₀ < 80, 0.1 < Ωₘ < 0.5, 0.01 < Ωᵦh² < 0.05
2. WHEN checking spectral index THEN the system SHALL validate 0.9 < nₛ < 1.1
3. WHEN validating optical depth THEN the system SHALL ensure 0.01 < τ < 0.15
4. WHEN numerical integration fails THEN the system SHALL provide detailed error messages with parameter values
5. WHEN covariance issues occur THEN the system SHALL report specific problems (non-positive-definite, asymmetric, etc.)

### Requirement 7: Testing and Verification

**User Story:** As a developer, I want comprehensive test coverage for parameter extraction, prior derivation, and covariance propagation, so that I can ensure the system works correctly across different input formats and parameter ranges.

#### Acceptance Criteria

1. WHEN testing parameter parsing THEN the system SHALL correctly detect H₀, Ωₘ, Ωᵦh² across different naming conventions
2. WHEN testing derivation accuracy THEN computed R, ℓₐ values SHALL match published Planck results within 1σ
3. WHEN testing covariance propagation THEN the system SHALL maintain matrix symmetry and positive-definiteness
4. WHEN running integration tests THEN the full pipeline SHALL produce StandardDataset compatible with existing fitting code
5. WHEN comparing with legacy mode THEN differences SHALL be < 0.5σ for identical input data

### Requirement 8: Performance and Scalability

**User Story:** As a computational scientist, I want the raw parameter processing to complete efficiently without significantly impacting pipeline runtime, so that I can process large parameter datasets in reasonable time.

#### Acceptance Criteria

1. WHEN processing typical Planck parameter sets THEN derivation SHALL complete within 10 seconds
2. WHEN handling large covariance matrices THEN memory usage SHALL remain under 1GB for standard datasets
3. WHEN computing Jacobians THEN numerical differentiation SHALL use optimized step sizes for accuracy and speed
4. WHEN caching is enabled THEN repeated computations with identical parameters SHALL use cached results
5. WHEN logging is active THEN performance metrics SHALL be recorded for monitoring and optimization

### Requirement 9: Documentation and Provenance

**User Story:** As a researcher reviewing analysis results, I want complete documentation of how distance priors were derived from raw parameters, so that I can understand and reproduce the analysis.

#### Acceptance Criteria

1. WHEN processing raw parameters THEN the system SHALL log all input parameter values and sources
2. WHEN computing derivatives THEN the system SHALL record integration methods and numerical settings used
3. WHEN propagating covariance THEN the system SHALL document Jacobian computation details
4. WHEN creating output THEN metadata SHALL include complete processing history and parameter provenance
5. WHEN errors occur THEN diagnostic information SHALL include parameter values and computation state for debugging