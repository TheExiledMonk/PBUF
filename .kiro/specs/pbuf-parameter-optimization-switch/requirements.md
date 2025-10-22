# Requirements Document

## Introduction

The PBUF cosmology pipeline currently uses fixed default values for all cosmological parameters in both ΛCDM and PBUF models. However, for optimal model fitting precision, researchers need the ability to optimize key parameters rather than using fixed values. This is particularly important for CMB fitting where parameters like `k_sat` in PBUF models and core parameters like `H0`, `Om0` in ΛCDM models should be optimizable to find the best match from default numbers for precision across fits. This feature will add a configurable optimization switch that allows researchers to specify which parameters should be optimized versus held fixed for both model types, with pre-computed optimal values stored for different cosmological contexts to improve fitting performance and scientific accuracy.

## Requirements

### Requirement 1

**User Story:** As a cosmology researcher, I want to control which parameters are optimized during fitting for both ΛCDM and PBUF models, so that I can choose between fixed values and optimization based on my research needs.

#### Acceptance Criteria

1. WHEN I run any model fit THEN the system SHALL support an `--optimize` flag to specify which parameters to optimize
2. WHEN I specify `--optimize k_sat` for PBUF THEN the system SHALL treat k_sat as a free parameter during optimization
3. WHEN I specify `--optimize H0,Om0` for ΛCDM THEN the system SHALL optimize both H0 and Om0 parameters
4. WHEN I specify `--optimize alpha,k_sat,H0` THEN the system SHALL optimize parameters appropriate to the selected model
5. WHEN no `--optimize` flag is provided THEN the system SHALL use fixed default values for all parameters
6. IF I specify an invalid parameter name for the selected model THEN the system SHALL raise a clear error message

### Requirement 2

**User Story:** As a researcher, I want pre-computed optimal parameter values for different model contexts, so that I can use scientifically validated starting points for optimization.

#### Acceptance Criteria

1. WHEN optimizing ΛCDM parameters THEN the system SHALL use ΛCDM-optimized starting values for H0, Om0, etc.
2. WHEN optimizing PBUF parameters THEN the system SHALL use PBUF-optimized starting values for k_sat, alpha, etc.
3. WHEN optimizing for CMB fitting specifically THEN the system SHALL use CMB-optimized parameter starting values
4. WHEN optimization completes successfully THEN the system SHALL optionally save optimized values for future use
5. WHEN saved optimal values exist THEN the system SHALL offer to use them as defaults in subsequent runs
6. IF no pre-computed values exist THEN the system SHALL use current hardcoded defaults as starting points

### Requirement 3

**User Story:** As a researcher, I want configuration file support for optimization settings, so that I can maintain consistent optimization strategies across research projects.

#### Acceptance Criteria

1. WHEN I create a configuration file THEN the system SHALL support an `optimization` section
2. WHEN I specify `optimize_parameters: ["k_sat", "alpha"]` for PBUF in config THEN the system SHALL optimize those parameters
3. WHEN I specify `optimize_parameters: ["H0", "Om0"]` for ΛCDM in config THEN the system SHALL optimize those parameters
4. WHEN I specify `use_precomputed: true` in config THEN the system SHALL load saved optimal values as starting points
5. WHEN command-line and config file both specify optimization THEN command-line SHALL take precedence
6. IF configuration contains invalid optimization settings THEN the system SHALL validate and report errors

### Requirement 4

**User Story:** As a developer, I want the optimization switch to integrate seamlessly with the existing unified parameter system, so that it doesn't disrupt the current refactored architecture.

#### Acceptance Criteria

1. WHEN optimization is enabled THEN the system SHALL still use the centralized parameter.py for base values
2. WHEN parameters are built THEN the system SHALL mark optimizable parameters with appropriate metadata
3. WHEN the optimization engine runs THEN it SHALL respect the optimization flags for each parameter
4. WHEN results are returned THEN optimized parameters SHALL be clearly distinguished from fixed parameters
5. IF optimization fails THEN the system SHALL fall back to fixed parameter values gracefully

### Requirement 5

**User Story:** As a researcher, I want stored optimization results to be model-specific and dataset-specific, so that I can maintain different optimal values for different fitting contexts.

#### Acceptance Criteria

1. WHEN optimization results are saved THEN the system SHALL store them by model type (lcdm vs pbuf)
2. WHEN optimization results are saved THEN the system SHALL store them by dataset combination (cmb, bao, sn, joint)
3. WHEN loading stored values THEN the system SHALL match the current fitting context exactly
4. WHEN multiple stored results exist THEN the system SHALL use the most recent for the matching context
5. IF no matching stored results exist THEN the system SHALL use default values without error

### Requirement 6

**User Story:** As a researcher, I want validation that optimized parameters remain within physical bounds, so that optimization doesn't produce unphysical results.

#### Acceptance Criteria

1. WHEN k_sat is optimized THEN the system SHALL enforce bounds [0.1, 2.0] during optimization
2. WHEN alpha is optimized THEN the system SHALL enforce bounds [1e-6, 1e-2] during optimization
3. WHEN H0 is optimized THEN the system SHALL enforce bounds [20.0, 150.0] during optimization
4. WHEN Om0 is optimized THEN the system SHALL enforce bounds [0.01, 0.99] during optimization
5. WHEN any parameter is optimized THEN the system SHALL validate against existing physical bounds in parameter.py
6. WHEN optimization reaches a bound THEN the system SHALL log a warning about boundary constraints
7. IF optimized values violate physics THEN the system SHALL reject the result and use fallback values

### Requirement 7

**User Story:** As a researcher, I want clear reporting of which parameters were optimized versus fixed, so that I can properly interpret and document my results.

#### Acceptance Criteria

1. WHEN fitting completes THEN the system SHALL report which parameters were optimized vs fixed
2. WHEN optimization was used THEN the system SHALL show starting values vs final optimized values
3. WHEN results are saved THEN the system SHALL include optimization metadata in output files
4. WHEN human-readable output is generated THEN it SHALL clearly distinguish optimized parameters
5. IF optimization improved the fit THEN the system SHALL report the χ² improvement achieved

### Requirement 8

**User Story:** As a researcher, I want backward compatibility with existing workflows, so that current scripts and analyses continue to work without modification.

#### Acceptance Criteria

1. WHEN no optimization flags are used THEN the system SHALL behave identically to current implementation
2. WHEN legacy parameter overrides are provided THEN they SHALL work exactly as before
3. WHEN existing configuration files are used THEN they SHALL continue to work without modification
4. WHEN validation tests are run THEN all existing functionality SHALL pass unchanged
5. IF new optimization features are used THEN they SHALL not affect non-optimization code paths