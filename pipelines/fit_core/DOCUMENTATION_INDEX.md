# PBUF Cosmology Pipeline - Documentation Index

This document provides a comprehensive index to all documentation for the PBUF cosmology pipeline, organized by user type and use case.

## Quick Navigation

### For New Users
1. **[README.md](README.md)** - Start here for system overview and basic usage
2. **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Practical examples from simple to advanced
3. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Solutions to common issues

### For Developers
1. **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Extending the system with new models and datasets
2. **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
3. **[CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)** - Advanced configuration patterns

### For System Administrators
1. **Integration Tests** - `test_end_to_end_integration.py`
2. **Parity Validation** - `../run_parity_tests.py`
3. **Performance Monitoring** - `../analyze_parity_discrepancies.py`

## Documentation by Topic

### System Architecture and Design

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](README.md) | System overview, architecture, and basic usage | All users |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Extensibility patterns and plugin architecture | Developers |
| [API_REFERENCE.md](API_REFERENCE.md) | Complete API documentation | Developers, Advanced users |

### Usage and Examples

| Document | Description | Use Case |
|----------|-------------|----------|
| [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) | Comprehensive usage examples | Learning, Reference |
| [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md) | Configuration file patterns | Complex analyses |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Problem solving and debugging | Issue resolution |

### Physics and Theory

| Document | Location | Description |
|----------|----------|-------------|
| PBUF Theory | `../../documents/PBUF-Math-Supplement-v9.md` | Mathematical foundations |
| Empirical Summary | `../../documents/Empirical_Summary_v9.md` | Observational constraints |
| Evolution Theory | `../../documents/evolution_theory.md` | Cosmological evolution |
| Equation Reference | `../../documents/equations_reference.mc` | Numerical validation |

### Testing and Validation

| Document | Location | Purpose |
|----------|----------|---------|
| Integration Tests | `test_end_to_end_integration.py` | System functionality |
| Parity Tests | `../run_parity_tests.py` | Legacy compatibility |
| Unit Tests | `test_*.py` | Module validation |
| Validation Report | `../../parity_results/validation_certification.md` | System certification |

## Documentation by User Journey

### 1. Getting Started (New Users)

**Goal**: Understand the system and run first analysis

**Path**:
1. Read [README.md](README.md) - System overview
2. Follow [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) sections 1-3 - Basic usage
3. Try command-line examples from [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
4. Consult [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if issues arise

**Key Concepts**:
- Unified parameter management
- Individual vs joint fitting
- Model comparison with AIC/BIC
- Physics consistency checks

### 2. Advanced Analysis (Experienced Users)

**Goal**: Perform complex multi-model analyses

**Path**:
1. Review [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) sections 9-12 - Advanced techniques
2. Study [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md) - Complex configurations
3. Reference [API_REFERENCE.md](API_REFERENCE.md) - Direct API usage
4. Check physics documentation in `../../documents/`

**Key Concepts**:
- Parameter grid searches
- Monte Carlo analysis
- Custom optimization methods
- Batch processing

### 3. System Extension (Developers)

**Goal**: Add new models, datasets, or analysis methods

**Path**:
1. Study [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) - Extension patterns
2. Review [API_REFERENCE.md](API_REFERENCE.md) - Extensibility APIs
3. Examine existing implementations in source code
4. Follow testing patterns from `test_*.py` files
5. Document physics in `../../documents/`

**Key Concepts**:
- Plugin architecture
- Configuration-driven extensions
- Physics documentation integration
- Automated testing

### 4. System Maintenance (Administrators)

**Goal**: Monitor system health and performance

**Path**:
1. Run integration tests regularly
2. Monitor parity test results
3. Review performance metrics
4. Update documentation as needed

**Key Tools**:
- `python test_end_to_end_integration.py`
- `python ../run_parity_tests.py`
- `python ../analyze_parity_discrepancies.py`

## Documentation Standards

### Code Documentation
- **Docstrings**: All public functions have comprehensive docstrings
- **Type Hints**: Function signatures include type annotations
- **Examples**: Docstrings include usage examples where appropriate
- **Physics References**: Link to equations in `documents/` directory

### User Documentation
- **Progressive Complexity**: Examples progress from simple to advanced
- **Complete Examples**: All examples are runnable and tested
- **Error Handling**: Common errors and solutions documented
- **Configuration**: All options explained with examples

### Developer Documentation
- **Extension Patterns**: Clear patterns for adding functionality
- **Testing Requirements**: All extensions must include tests
- **Physics Validation**: New models must reference documented physics
- **Backward Compatibility**: Changes must maintain API compatibility

## Maintenance and Updates

### Documentation Lifecycle
1. **Creation**: New features require documentation updates
2. **Review**: Documentation reviewed with code changes
3. **Testing**: Examples tested with integration tests
4. **Versioning**: Documentation versioned with code releases

### Update Procedures
1. **API Changes**: Update [API_REFERENCE.md](API_REFERENCE.md)
2. **New Features**: Add examples to [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)
3. **Configuration Changes**: Update [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)
4. **Bug Fixes**: Update [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Quality Assurance
- All examples must run without errors
- Configuration files must validate successfully
- Links between documents must remain valid
- Physics references must be accurate and current

## Getting Help

### Self-Service Resources
1. **Search this documentation** for keywords related to your issue
2. **Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for common problems
3. **Review [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** for similar use cases
4. **Run diagnostic tests** to identify system issues

### Escalation Path
1. **Check integration test results** for system status
2. **Review recent changes** in version control
3. **Consult physics documentation** for theoretical questions
4. **Contact development team** for unresolved issues

### Contributing to Documentation
1. **Identify gaps** in current documentation
2. **Follow existing patterns** for new content
3. **Include working examples** for all new features
4. **Test all examples** before submission
5. **Update this index** when adding new documents

## Document Relationships

```
README.md (Entry Point)
├── USAGE_EXAMPLES.md (Practical Examples)
│   ├── Basic Usage (Sections 1-8)
│   ├── Advanced Usage (Sections 9-12)
│   └── Extensibility (Sections 13-16)
├── API_REFERENCE.md (Technical Reference)
│   ├── Core Functions
│   ├── Data Structures
│   └── Extensibility APIs
├── DEVELOPER_GUIDE.md (Extension Guide)
│   ├── Adding Models
│   ├── Adding Datasets
│   └── Advanced Patterns
├── CONFIGURATION_EXAMPLES.md (Configuration Reference)
│   ├── Basic Configurations
│   ├── Advanced Configurations
│   └── Extension Configurations
├── TROUBLESHOOTING.md (Problem Solving)
│   ├── Common Issues
│   ├── Debugging Techniques
│   └── FAQ
└── Physics Documentation (../../documents/)
    ├── PBUF Theory
    ├── Empirical Constraints
    └── Numerical Validation
```

This documentation index provides a comprehensive guide to navigating and maintaining the PBUF cosmology pipeline documentation ecosystem.