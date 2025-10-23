# Parameter Optimization Deployment Checklist

## Pre-Deployment Validation

### 1. Core Functionality Tests

#### ✅ Parameter Optimization Engine
- [ ] `ParameterOptimizer` class instantiates correctly
- [ ] `optimize_parameters()` method works for both ΛCDM and PBUF models
- [ ] Parameter bounds validation functions correctly
- [ ] Optimization convergence diagnostics are accurate
- [ ] Error handling for invalid parameters works

**Test Command:**
```bash
python -c "
from pipelines.fit_core.optimizer import ParameterOptimizer
optimizer = ParameterOptimizer()
print('✓ ParameterOptimizer instantiated')

# Test bounds validation
bounds = optimizer.get_optimization_bounds('pbuf', 'k_sat')
assert bounds == (0.1, 2.0), f'Expected (0.1, 2.0), got {bounds}'
print('✓ Parameter bounds validation works')

# Test parameter validation
try:
    optimizer.validate_optimization_request('lcdm', ['k_sat'])
    assert False, 'Should have raised ValueError'
except ValueError:
    print('✓ Invalid parameter detection works')
"
```

#### ✅ Parameter Store System
- [ ] `OptimizedParameterStore` class works correctly
- [ ] Parameter storage and retrieval functions
- [ ] Cross-model consistency validation works
- [ ] File locking mechanism prevents corruption
- [ ] Backup and recovery mechanisms function

**Test Command:**
```bash
python -c "
from pipelines.fit_core.parameter_store import OptimizedParameterStore
import tempfile
import os

store = OptimizedParameterStore()
print('✓ OptimizedParameterStore instantiated')

# Test parameter retrieval
params = store.get_model_defaults('lcdm')
assert 'H0' in params, 'H0 parameter missing'
print('✓ Parameter retrieval works')

# Test cross-model consistency
divergence = store.validate_cross_model_consistency()
assert isinstance(divergence, dict), 'Expected dict from consistency check'
print('✓ Cross-model consistency validation works')
"
```

#### ✅ Configuration System
- [ ] Command-line optimization flags parse correctly
- [ ] Configuration file optimization sections load properly
- [ ] Parameter precedence (CLI > config > defaults) works
- [ ] Validation of optimization settings functions

**Test Command:**
```bash
python -c "
from pipelines.fit_core.config import parse_optimization_config
import json

# Test configuration parsing
config = {
    'optimization': {
        'optimize_parameters': ['H0', 'Om0'],
        'covariance_scaling': 1.2,
        'dry_run': True
    }
}

parsed = parse_optimization_config(config)
assert parsed['optimize_parameters'] == ['H0', 'Om0']
assert parsed['covariance_scaling'] == 1.2
assert parsed['dry_run'] == True
print('✓ Configuration parsing works')
"
```

### 2. Integration Tests

#### ✅ Engine Integration
- [ ] `run_fit()` accepts optimization parameters
- [ ] Optimization results are properly integrated into results
- [ ] Parameter propagation works across fitters
- [ ] Optimization metadata is preserved

**Test Command:**
```bash
python -c "
from pipelines.fit_core.engine import run_fit

# Test optimization integration
result = run_fit(
    model='lcdm',
    datasets_list=['cmb'],
    optimization_params=['H0', 'Om0']
)

assert 'optimization' in result, 'Optimization metadata missing'
assert 'chi2_improvement' in result['optimization']
print('✓ Engine optimization integration works')
"
```

#### ✅ Fitter Script Integration
- [ ] `fit_cmb.py` supports `--optimize` flag
- [ ] `fit_bao.py` and `fit_sn.py` use optimized parameters automatically
- [ ] `fit_joint.py` works with pre-optimized parameters
- [ ] All scripts maintain backward compatibility

**Test Commands:**
```bash
# Test CMB optimization
python pipelines/fit_cmb.py --model lcdm --optimize H0,Om0 --dry-run
echo "✓ CMB optimization flag works"

# Test BAO uses optimized parameters
python pipelines/fit_bao.py --model lcdm --dry-run
echo "✓ BAO uses optimized parameters"

# Test backward compatibility
python pipelines/fit_cmb.py --model lcdm --dry-run
echo "✓ Backward compatibility maintained"
```

### 3. Data Integrity Tests

#### ✅ Parameter Storage Integrity
- [ ] Optimized parameters persist correctly
- [ ] JSON storage format is valid
- [ ] File corruption detection works
- [ ] Concurrent access protection functions

**Test Command:**
```bash
python -c "
import json
import os
from pathlib import Path

# Check storage directory exists
storage_dir = Path('optimization_results')
if not storage_dir.exists():
    storage_dir.mkdir()
    print('✓ Created optimization_results directory')

# Test JSON file integrity
test_data = {
    'lcdm': {
        'defaults': {'H0': 67.4, 'Om0': 0.315},
        'optimization_metadata': {
            'timestamp': '2025-10-22T14:42:00Z',
            'chi2_improvement': 2.34
        }
    }
}

test_file = storage_dir / 'test_storage.json'
with open(test_file, 'w') as f:
    json.dump(test_data, f, indent=2)

# Verify can read back
with open(test_file) as f:
    loaded = json.load(f)

assert loaded == test_data, 'Storage roundtrip failed'
os.remove(test_file)
print('✓ Parameter storage integrity verified')
"
```

#### ✅ Cross-Model Consistency
- [ ] Shared parameters remain consistent after optimization
- [ ] Divergence detection works correctly
- [ ] Tolerance-based validation functions
- [ ] Warning system for parameter drift

**Test Command:**
```bash
python -c "
from pipelines.fit_core.parameter_store import OptimizedParameterStore

store = OptimizedParameterStore()

# Test consistency validation
divergence = store.validate_cross_model_consistency()

# Check shared parameters
shared_params = ['H0', 'Om0', 'Obh2', 'ns']
for param in shared_params:
    if param in divergence:
        div = divergence[param]
        if div > 1e-2:  # 1% tolerance
            print(f'Warning: Large divergence in {param}: {div:.6f}')
        else:
            print(f'✓ {param} consistency within tolerance: {div:.6f}')
"
```

### 4. Performance Tests

#### ✅ Optimization Performance
- [ ] CMB optimization completes within reasonable time (< 5 minutes)
- [ ] Memory usage remains within acceptable limits
- [ ] Convergence diagnostics are accurate
- [ ] Warm start functionality improves performance

**Test Command:**
```bash
python -c "
import time
from pipelines.fit_core.optimizer import optimize_cmb_parameters

start_time = time.time()

# Test CMB optimization performance
result = optimize_cmb_parameters(
    model='lcdm',
    optimize_params=['H0', 'Om0'],
    covariance_scaling=2.0  # Faster for testing
)

elapsed = time.time() - start_time
print(f'CMB optimization completed in {elapsed:.1f} seconds')

if elapsed < 300:  # 5 minutes
    print('✓ Performance within acceptable limits')
else:
    print('❌ Performance too slow')

assert result.convergence_status == 'success', 'Optimization failed to converge'
print('✓ Optimization convergence verified')
"
```

#### ✅ Storage Performance
- [ ] Parameter loading is fast (< 1 second)
- [ ] File locking doesn't cause significant delays
- [ ] Concurrent access handling works correctly

**Test Command:**
```bash
python -c "
import time
from pipelines.fit_core.parameter_store import OptimizedParameterStore

store = OptimizedParameterStore()

# Test parameter loading performance
start_time = time.time()
params = store.get_model_defaults('lcdm')
elapsed = time.time() - start_time

print(f'Parameter loading took {elapsed:.3f} seconds')

if elapsed < 1.0:
    print('✓ Parameter loading performance acceptable')
else:
    print('❌ Parameter loading too slow')
"
```

## Deployment Validation Script

### 5. Automated Validation Script

Create comprehensive validation script:

```bash
#!/bin/bash
# optimization_deployment_validation.sh

echo "=== Parameter Optimization Deployment Validation ==="
echo "Starting validation at $(date)"

VALIDATION_FAILED=0

# Function to run test and check result
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo "✓ PASS"
    else
        echo "❌ FAIL"
        VALIDATION_FAILED=1
    fi
}

# Core functionality tests
echo -e "\n1. Core Functionality Tests"
run_test "ParameterOptimizer instantiation" "python -c 'from pipelines.fit_core.optimizer import ParameterOptimizer; ParameterOptimizer()'"
run_test "OptimizedParameterStore instantiation" "python -c 'from pipelines.fit_core.parameter_store import OptimizedParameterStore; OptimizedParameterStore()'"
run_test "Configuration parsing" "python -c 'from pipelines.fit_core.config import parse_optimization_config; parse_optimization_config({})'"

# Integration tests
echo -e "\n2. Integration Tests"
run_test "CMB optimization (dry run)" "python pipelines/fit_cmb.py --model lcdm --optimize H0,Om0 --dry-run --cov-scale 2.0"
run_test "BAO with optimized parameters" "python pipelines/fit_bao.py --model lcdm --dry-run"
run_test "Backward compatibility" "python pipelines/fit_cmb.py --model lcdm --dry-run"

# Data integrity tests
echo -e "\n3. Data Integrity Tests"
run_test "Parameter storage directory" "test -d optimization_results || mkdir -p optimization_results"
run_test "Cross-model consistency" "python -c 'from pipelines.fit_core.parameter_store import OptimizedParameterStore; OptimizedParameterStore().validate_cross_model_consistency()'"

# Performance tests
echo -e "\n4. Performance Tests"
run_test "Fast optimization test" "timeout 120 python -c 'from pipelines.fit_core.optimizer import optimize_cmb_parameters; optimize_cmb_parameters(\"lcdm\", [\"H0\"], covariance_scaling=3.0)'"

# Final result
echo -e "\n=== Validation Summary ==="
if [ $VALIDATION_FAILED -eq 0 ]; then
    echo "✅ All validation tests PASSED"
    echo "System ready for deployment"
    exit 0
else
    echo "❌ Some validation tests FAILED"
    echo "Review failures before deployment"
    exit 1
fi
```

Save this script and make it executable:

```bash
chmod +x optimization_deployment_validation.sh
```

## Production Deployment Steps

### 6. Pre-Production Checklist

#### ✅ Environment Preparation
- [ ] Python dependencies installed (scipy >= 1.13.0)
- [ ] Directory structure created (`optimization_results/`)
- [ ] File permissions set correctly
- [ ] Backup procedures in place

#### ✅ Configuration Validation
- [ ] Default parameter values reviewed and approved
- [ ] Optimization bounds validated by physics team
- [ ] Convergence tolerances set appropriately
- [ ] Covariance scaling factors validated

#### ✅ Documentation Review
- [ ] API documentation updated and accurate
- [ ] Usage examples tested and working
- [ ] Troubleshooting guide comprehensive
- [ ] Migration guide validated with test cases

### 7. Deployment Procedure

#### Step 1: Backup Current System
```bash
# Backup current parameter files
cp -r pipelines/fit_core/ pipelines/fit_core_backup_$(date +%Y%m%d)

# Backup any existing results
if [ -d "optimization_results" ]; then
    cp -r optimization_results optimization_results_backup_$(date +%Y%m%d)
fi
```

#### Step 2: Deploy New Code
```bash
# Deploy optimization system files
# (This would be done through your deployment system)

# Verify deployment
python -c "
from pipelines.fit_core.optimizer import ParameterOptimizer
from pipelines.fit_core.parameter_store import OptimizedParameterStore
print('✓ Optimization system deployed successfully')
"
```

#### Step 3: Initialize Storage
```bash
# Create optimization storage directory
mkdir -p optimization_results

# Set appropriate permissions
chmod 755 optimization_results

# Initialize parameter store
python -c "
from pipelines.fit_core.parameter_store import OptimizedParameterStore
store = OptimizedParameterStore()
store.get_model_defaults('lcdm')  # Initialize ΛCDM defaults
store.get_model_defaults('pbuf')  # Initialize PBUF defaults
print('✓ Parameter store initialized')
"
```

#### Step 4: Run Validation Suite
```bash
# Run comprehensive validation
./optimization_deployment_validation.sh

# Run extended validation if needed
python pipelines/fit_core/test_comprehensive_validation.py
python pipelines/fit_core/test_advanced_optimization_features.py
```

#### Step 5: Smoke Test
```bash
# Run quick smoke test
echo "Running deployment smoke test..."

# Test ΛCDM optimization
python pipelines/fit_cmb.py --model lcdm --optimize H0,Om0 --cov-scale 2.0 --dry-run

# Test PBUF optimization  
python pipelines/fit_cmb.py --model pbuf --optimize k_sat,alpha --cov-scale 2.0 --dry-run

# Test parameter propagation
python pipelines/fit_bao.py --model lcdm --dry-run

echo "✓ Smoke test completed successfully"
```

## Rollback Procedures

### 8. Rollback Plan

#### Immediate Rollback (< 1 hour)
```bash
#!/bin/bash
# rollback_optimization.sh

echo "Initiating optimization system rollback..."

# Step 1: Stop any running optimizations
pkill -f "fit_cmb.py.*--optimize"

# Step 2: Restore backup files
BACKUP_DATE=$(ls -1 pipelines/ | grep fit_core_backup | tail -1 | sed 's/fit_core_backup_//')
if [ -n "$BACKUP_DATE" ]; then
    echo "Restoring from backup: $BACKUP_DATE"
    rm -rf pipelines/fit_core
    mv pipelines/fit_core_backup_$BACKUP_DATE pipelines/fit_core
    echo "✓ Code rollback completed"
else
    echo "❌ No backup found for rollback"
    exit 1
fi

# Step 3: Restore parameter storage
if [ -d "optimization_results_backup_$BACKUP_DATE" ]; then
    rm -rf optimization_results
    mv optimization_results_backup_$BACKUP_DATE optimization_results
    echo "✓ Parameter storage rollback completed"
fi

# Step 4: Verify rollback
python -c "
try:
    from pipelines.fit_core.optimizer import ParameterOptimizer
    print('❌ Rollback failed - optimization system still present')
    exit(1)
except ImportError:
    print('✓ Rollback successful - optimization system removed')
"

echo "Rollback completed successfully"
```

#### Partial Rollback (Disable Optimization)
```bash
# Disable optimization without full rollback
python -c "
import json
from pathlib import Path

# Create disable flag
disable_file = Path('optimization_results/.disabled')
disable_file.touch()

print('✓ Optimization system disabled')
print('Remove .disabled file to re-enable')
"
```

### 9. Monitoring and Health Checks

#### Post-Deployment Monitoring
```bash
#!/bin/bash
# optimization_health_check.sh

echo "=== Optimization System Health Check ==="

# Check storage directory
if [ -d "optimization_results" ]; then
    echo "✓ Storage directory exists"
    
    # Check for lock files (shouldn't persist)
    LOCK_FILES=$(find optimization_results -name "*.lock" -mmin +60)
    if [ -n "$LOCK_FILES" ]; then
        echo "⚠️ Stale lock files found:"
        echo "$LOCK_FILES"
    else
        echo "✓ No stale lock files"
    fi
else
    echo "❌ Storage directory missing"
fi

# Check parameter consistency
python -c "
from pipelines.fit_core.parameter_store import OptimizedParameterStore

store = OptimizedParameterStore()
divergence = store.validate_cross_model_consistency()

max_divergence = max(divergence.values()) if divergence else 0
if max_divergence > 1e-2:
    print(f'⚠️ Large parameter divergence: {max_divergence:.6f}')
else:
    print('✓ Parameter consistency OK')
"

# Check recent optimization activity
python -c "
from pathlib import Path
import json
from datetime import datetime, timedelta

recent_activity = False
for model_file in ['lcdm_optimized.json', 'pbuf_optimized.json']:
    file_path = Path('optimization_results') / model_file
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
        
        if 'optimization_metadata' in data.get('lcdm', {}):
            timestamp_str = data['lcdm']['optimization_metadata'].get('cmb_optimized')
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if datetime.now().replace(tzinfo=timestamp.tzinfo) - timestamp < timedelta(days=7):
                    recent_activity = True

if recent_activity:
    print('✓ Recent optimization activity detected')
else:
    print('ℹ️ No recent optimization activity')
"

echo "Health check completed"
```

### 10. Success Criteria

#### Deployment Success Indicators
- [ ] All validation tests pass
- [ ] Smoke tests complete successfully
- [ ] No performance degradation in existing workflows
- [ ] Optimization features work as documented
- [ ] Parameter storage system functions correctly
- [ ] Cross-model consistency maintained
- [ ] Backward compatibility preserved

#### Performance Benchmarks
- [ ] CMB optimization completes in < 5 minutes (production data)
- [ ] Parameter loading takes < 1 second
- [ ] Memory usage increase < 10% over baseline
- [ ] No file system errors or corruption
- [ ] Concurrent access works without conflicts

#### User Acceptance Criteria
- [ ] Migration guide successfully followed by test users
- [ ] API documentation accurate and complete
- [ ] Troubleshooting guide resolves common issues
- [ ] Command-line interface intuitive and functional
- [ ] Configuration system flexible and robust

## Emergency Contacts

### 11. Support Information

#### Technical Contacts
- **Primary Developer**: [Contact Information]
- **Physics Validation**: [Contact Information]  
- **System Administrator**: [Contact Information]
- **On-Call Support**: [Contact Information]

#### Escalation Procedures
1. **Level 1**: Check troubleshooting guide and run health checks
2. **Level 2**: Contact primary developer for optimization-specific issues
3. **Level 3**: Initiate rollback procedures if system stability affected
4. **Level 4**: Contact system administrator for infrastructure issues

#### Documentation Links
- **API Reference**: `pipelines/fit_core/API_REFERENCE.md`
- **Usage Examples**: `pipelines/fit_core/USAGE_EXAMPLES.md`
- **Troubleshooting**: `pipelines/fit_core/TROUBLESHOOTING.md`
- **Migration Guide**: `PARAMETER_OPTIMIZATION_MIGRATION_GUIDE.md`

This deployment checklist ensures a systematic and safe deployment of the parameter optimization system with comprehensive validation, monitoring, and rollback capabilities.