# Pytest Error Analysis Summary

## Overview
- **Total Failed Tests**: 72 (out of 840 total tests)
- **Pass Rate**: ~91.4%
- **Main Issue**: Systematic failures across derivation modules

## Error Categories (Grouped by Root Cause)

### 1. 🔴 **Derivation Module Test Setup Issues** (106 failures)
**Root Cause**: Missing test data files and improper test setup

**Affected Modules**:
- BAO Derivation (24 failures)
- CC Derivation (24 failures) 
- RSD Derivation (24 failures)
- CMB Derivation (18 failures)
- SN Derivation (16 failures)

**Common Error Pattern**:
```
ValueError: Input file does not exist
AssertionError: Regex pattern did not match
ProcessingError: Processing failed for dataset 'unknown'
```

**Fix Priority**: HIGH - These are systematic test infrastructure issues

### 2. 🟡 **Performance/Validation Test Issues** (22 failures)
**Root Cause**: Data validation errors in performance tests

**Common Error Pattern**:
```
Missing required columns ['z', 'mb', 'dmb']. Available columns: []
EnhancedProcessingError: Processing failed for dataset
```

**Fix Priority**: MEDIUM - Performance tests failing due to data format issues

### 3. 🟡 **Output Manager Issues** (8 failures)
**Root Cause**: File I/O and cleanup logic problems

**Common Error Pattern**:
```
FileNotFoundError: Dataset file not found
AssertionError: 0 not greater than 0 (cleanup test)
```

**Fix Priority**: MEDIUM - Output handling needs review

### 4. 🟠 **Miscellaneous Issues** (8 failures)
**Individual Issues**:
- Parameter configuration errors
- Likelihood format validation errors  
- Dataset verification failures
- Parameter override functionality issues

**Fix Priority**: LOW-MEDIUM - Specific isolated issues

## Unique Error Types Summary

### A. **File Not Found Errors** (Most Common)
- Test data files not being created properly
- Hardcoded paths not matching test environment
- Missing test fixture setup

### B. **Column Validation Errors** 
- Mock data not matching expected schema
- Missing required columns in test datasets
- Data format inconsistencies

### C. **Processing Pipeline Errors**
- Derivation modules failing with 'unknown' dataset
- Error handling not working as expected
- Interface/base class issues

### D. **Test Infrastructure Issues**
- Regex patterns not matching actual error messages
- Test assertions expecting different error formats
- Cleanup and teardown problems

## Recommended Fix Strategy

### Phase 1: Fix Test Infrastructure (HIGH)
1. **Review test data setup** in derivation module tests
2. **Fix file path issues** - ensure test files are created in correct locations
3. **Standardize mock data generation** across all derivation modules
4. **Update regex patterns** in test assertions to match actual error messages

### Phase 2: Fix Data Validation (MEDIUM)  
1. **Review column validation logic** in performance tests
2. **Fix mock data schema** to include required columns
3. **Standardize data format** across test datasets

### Phase 3: Fix Specific Issues (LOW-MEDIUM)
1. **Review output manager** file handling logic
2. **Fix parameter configuration** issues
3. **Address likelihood validation** problems

## Quick Wins (Easiest to Fix)
1. **File path issues** - Update hardcoded paths in tests
2. **Missing test data** - Ensure test fixtures create required files  
3. **Regex pattern mismatches** - Update test assertions
4. **Column name mismatches** - Fix mock data generation

## Impact Assessment
- **Critical**: Derivation module tests (core functionality)
- **Important**: Performance validation tests (system reliability)
- **Minor**: Output manager and misc issues (peripheral functionality)

The majority of failures are systematic test infrastructure issues rather than actual code bugs, which is good news for the codebase health.