# PBUF Cosmology Pipeline - Migration Guide

## Overview

This guide provides step-by-step instructions for migrating from the legacy PBUF cosmology fitting system to the new unified architecture. The migration is designed to be gradual and safe, with full backward compatibility during the transition period.

**Certification Status**: ✅ CERTIFIED FOR DEPLOYMENT (Task 12.3 Complete)  
**Validation Date**: 2025-10-22  
**System Version**: 1.0.0

## Pre-Migration Checklist

### System Requirements
- [ ] Python 3.8 or higher
- [ ] Required dependencies installed (see `requirements.txt`)
- [ ] Sufficient disk space for both legacy and unified systems
- [ ] Backup of existing analysis scripts and results

### Validation Requirements
- [x] Final validation report shows "CERTIFIED" status ✅
- [x] All integrity checks pass ✅
- [x] Performance benchmarks meet requirements ✅
- [x] Integration tests pass successfully ✅
- [x] Deployment artifacts created ✅

## Migration Strategy

### Phase 1: Parallel Installation (Week 1)

#### Step 1: Install Unified System
```bash
# Clone or update repository
git pull origin main

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from pipelines.fit_core.engine import run_fit; print('Installation successful')"
```

#### Step 2: Run Validation Tests
```bash
# Run Task 12.3 final validation (COMPLETED ✅)
python task_12_3_final_validation.py

# Check certification status (CERTIFIED ✅)
# Status: CERTIFIED - Ready for deployment
```

#### Step 3: Backup Legacy System
```bash
# Create backup directory
mkdir legacy_backup_$(date +%Y%m%d)

# Backup legacy scripts (adjust paths as needed)
cp -r legacy_fitters/ legacy_backup_$(date +%Y%m%d)/
cp -r analysis_scripts/ legacy_backup_$(date +%Y%m%d)/
cp -r results/ legacy_backup_$(date +%Y%m%d)/
```

### Phase 2: Parallel Testing (Week 2)

#### Step 4: Test Wrapper Scripts
```bash
# Test individual fitters
python pipelines/fit_cmb.py --model lcdm
python pipelines/fit_bao.py --model pbuf
python pipelines/fit_sn.py --model lcdm

# Test joint fitting
python pipelines/fit_joint.py --model pbuf --datasets cmb bao sn
```

#### Step 5: Compare Results with Legacy
```bash
# Run parity tests for critical analyses
python pipelines/run_parity_tests.py

# Compare specific results
python -c "
from pipelines.fit_core.engine import run_fit
result = run_fit('lcdm', ['cmb', 'bao', 'sn'])
print(f'Unified χ²: {result[\"metrics\"][\"total_chi2\"]:.6f}')
"

# Compare with legacy result
# Should match within 1e-6 tolerance
```

### Phase 3: Gradual Migration (Weeks 3-4)

#### Step 6: Update Analysis Scripts (Optional)

**Option A: Minimal Changes (Recommended)**
Keep existing scripts, just replace fitter calls:

```bash
# Old way
python legacy_fitters/fit_cmb_legacy.py --model lcdm

# New way (drop-in replacement)
python pipelines/fit_cmb.py --model lcdm
```

**Option B: Modernize to Use Engine API**
```python
# Old analysis script
import subprocess
result = subprocess.run(['python', 'legacy_fitters/fit_cmb_legacy.py', '--model', 'lcdm'])

# New analysis script
from pipelines.fit_core.engine import run_fit
result = run_fit(model="lcdm", datasets_list=["cmb"])
chi2 = result["metrics"]["total_chi2"]
```

#### Step 7: Migrate Configuration Files
```bash
# Convert legacy config to new format
python pipelines/fit_joint.py --create-config new_config.json

# Edit new_config.json with your parameters
# Use new config
python pipelines/fit_joint.py --config new_config.json
```

### Phase 4: Full Migration (Week 5)

#### Step 8: Update Production Scripts
Replace all legacy fitter calls in production scripts:

```bash
# Find all legacy calls
grep -r "legacy_fitters" analysis_scripts/

# Replace systematically
sed -i 's/legacy_fitters\/fit_cmb_legacy.py/pipelines\/fit_cmb.py/g' analysis_scripts/*.py
sed -i 's/legacy_fitters\/fit_bao_legacy.py/pipelines\/fit_bao.py/g' analysis_scripts/*.py
sed -i 's/legacy_fitters\/fit_sn_legacy.py/pipelines\/fit_sn.py/g' analysis_scripts/*.py
```

#### Step 9: Validate Production Results
```bash
# Run critical production analyses
python production_analysis_1.py
python production_analysis_2.py

# Compare results with legacy baseline
# Document any differences > 1e-6
```

### Phase 5: Cleanup (Week 6)

#### Step 10: Archive Legacy System
```bash
# Move legacy system to archive
mkdir archive/
mv legacy_fitters/ archive/
mv legacy_configs/ archive/

# Keep backup accessible but out of the way
```

## Detailed Migration Examples

### Example 1: Simple CMB Analysis

**Legacy Script:**
```python
#!/usr/bin/env python3
import subprocess
import json

# Run legacy CMB fitter
result = subprocess.run([
    'python', 'legacy_fitters/fit_cmb_legacy.py',
    '--model', 'lcdm',
    '--output', 'cmb_results.json'
], capture_output=True, text=True)

# Parse results
with open('cmb_results.json') as f:
    data = json.load(f)
    
print(f"H0 = {data['H0']:.2f}")
print(f"χ² = {data['chi2']:.3f}")
```

**Migrated Script (Option 1 - Minimal Change):**
```python
#!/usr/bin/env python3
import subprocess
import json

# Use new unified fitter (drop-in replacement)
result = subprocess.run([
    'python', 'pipelines/fit_cmb.py',
    '--model', 'lcdm',
    '--save-results', 'cmb_results.json',
    '--output-format', 'json'
], capture_output=True, text=True)

# Parse results (same format)
with open('cmb_results.json') as f:
    data = json.load(f)
    
print(f"H0 = {data['params']['H0']:.2f}")
print(f"χ² = {data['metrics']['total_chi2']:.3f}")
```

**Migrated Script (Option 2 - Modern API):**
```python
#!/usr/bin/env python3
from pipelines.fit_core.engine import run_fit

# Direct API call
result = run_fit(model="lcdm", datasets_list=["cmb"])

print(f"H0 = {result['params']['H0']:.2f}")
print(f"χ² = {result['metrics']['total_chi2']:.3f}")

# Save results if needed
import json
with open('cmb_results.json', 'w') as f:
    json.dump(result, f, indent=2)
```

### Example 2: Parameter Grid Search

**Legacy Script:**
```python
import itertools
import subprocess

h0_values = [65, 67, 70, 73, 75]
om0_values = [0.25, 0.30, 0.35]

results = []
for h0, om0 in itertools.product(h0_values, om0_values):
    # Legacy call
    cmd = [
        'python', 'legacy_fitters/fit_joint_legacy.py',
        '--model', 'lcdm',
        '--H0', str(h0),
        '--Om0', str(om0),
        '--datasets', 'cmb', 'bao'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse output...
```

**Migrated Script:**
```python
import itertools
from pipelines.fit_core.engine import run_fit

h0_values = [65, 67, 70, 73, 75]
om0_values = [0.25, 0.30, 0.35]

results = []
for h0, om0 in itertools.product(h0_values, om0_values):
    # Modern API call
    result = run_fit(
        model="lcdm",
        datasets_list=["cmb", "bao"],
        overrides={"H0": h0, "Om0": om0}
    )
    
    results.append({
        "H0": h0,
        "Om0": om0,
        "chi2": result["metrics"]["total_chi2"],
        "aic": result["metrics"]["aic"]
    })
```

### Example 3: Batch Processing Script

**Legacy Script:**
```bash
#!/bin/bash
# Legacy batch processing

for model in lcdm pbuf; do
    for dataset in cmb bao sn; do
        echo "Processing $model with $dataset"
        python legacy_fitters/fit_${dataset}_legacy.py \
            --model $model \
            --output results_${model}_${dataset}.json
    done
done
```

**Migrated Script:**
```bash
#!/bin/bash
# Unified batch processing

for model in lcdm pbuf; do
    for dataset in cmb bao sn; do
        echo "Processing $model with $dataset"
        python pipelines/fit_${dataset}.py \
            --model $model \
            --save-results results_${model}_${dataset}.json \
            --output-format json
    done
done
```

## Configuration Migration

### Legacy Configuration Format
```ini
[model]
type = lcdm
H0 = 67.4
Om0 = 0.315

[fitting]
datasets = cmb,bao,sn
optimizer = minimize
```

### New Configuration Format
```json
{
    "model": "lcdm",
    "datasets": ["cmb", "bao", "sn"],
    "parameters": {
        "H0": 67.4,
        "Om0": 0.315
    },
    "optimizer": {
        "method": "minimize",
        "options": {
            "maxiter": 1000
        }
    }
}
```

### Conversion Script
```python
#!/usr/bin/env python3
"""Convert legacy INI config to new JSON format."""

import configparser
import json
import sys

def convert_config(ini_file, json_file):
    config = configparser.ConfigParser()
    config.read(ini_file)
    
    # Convert to new format
    new_config = {
        "model": config.get("model", "type"),
        "datasets": config.get("fitting", "datasets").split(","),
        "parameters": {},
        "optimizer": {
            "method": config.get("fitting", "optimizer", fallback="minimize")
        }
    }
    
    # Extract parameters
    for key, value in config["model"].items():
        if key != "type":
            try:
                new_config["parameters"][key] = float(value)
            except ValueError:
                new_config["parameters"][key] = value
    
    # Save new config
    with open(json_file, 'w') as f:
        json.dump(new_config, f, indent=2)
    
    print(f"Converted {ini_file} -> {json_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_config.py input.ini output.json")
        sys.exit(1)
    
    convert_config(sys.argv[1], sys.argv[2])
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'pipelines'
```

**Solution:**
```bash
# Ensure you're in the correct directory
cd /path/to/pbuf/project

# Check Python path
python -c "import sys; print(sys.path)"

# Add project root to PYTHONPATH if needed
export PYTHONPATH=$PYTHONPATH:/path/to/pbuf/project
```

#### Issue 2: Numerical Differences
```
Results differ from legacy by more than tolerance
```

**Solution:**
```python
# Run detailed parity test
from pipelines.fit_core.parity_testing import run_parity_test

result = run_parity_test("lcdm", ["cmb"], tolerance=1e-6, verbose=True)
print(result["detailed_comparison"])

# Check if differences are in acceptable range
# May need to adjust tolerance for specific cases
```

#### Issue 3: Performance Issues
```
New system is slower than legacy
```

**Solution:**
```python
# Check if caching is enabled
# Profile specific bottlenecks
import cProfile

def profile_fit():
    from pipelines.fit_core.engine import run_fit
    return run_fit("lcdm", ["cmb", "bao", "sn"])

cProfile.run('profile_fit()', 'profile_output.txt')

# Analyze profile_output.txt for bottlenecks
```

#### Issue 4: Configuration Errors
```
Invalid configuration format
```

**Solution:**
```python
# Validate configuration
from pipelines.fit_core.config import validate_config

try:
    validate_config("my_config.json")
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
    
# Use config creation tool
python pipelines/fit_joint.py --create-config example.json
```

## Rollback Procedure

If issues arise during migration, follow this rollback procedure:

### Step 1: Stop Using Unified System
```bash
# Immediately stop all production jobs using unified system
pkill -f "pipelines/"

# Revert any modified scripts to use legacy system
git checkout HEAD~1 analysis_scripts/
```

### Step 2: Restore Legacy System
```bash
# Restore from backup
cp -r legacy_backup_YYYYMMDD/* ./

# Verify legacy system works
python legacy_fitters/fit_cmb_legacy.py --model lcdm --test
```

### Step 3: Document Issues
```bash
# Create issue report
cat > rollback_report.md << EOF
# Rollback Report - $(date)

## Issues Encountered:
- [Describe specific issues]

## Data Affected:
- [List any affected analyses]

## Next Steps:
- [Plan for fixing issues]
EOF
```

### Step 4: Plan Re-migration
- Investigate and fix identified issues
- Re-run validation tests
- Plan new migration timeline

## Post-Migration Validation

### Continuous Monitoring
```bash
# Set up automated validation
crontab -e

# Add daily validation check
0 2 * * * cd /path/to/pbuf && python daily_validation.py
```

### Performance Monitoring
```python
#!/usr/bin/env python3
"""Daily performance monitoring script."""

import time
from pipelines.fit_core.engine import run_fit

def monitor_performance():
    start_time = time.time()
    
    # Standard benchmark
    result = run_fit("lcdm", ["cmb", "bao", "sn"])
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Log performance
    with open("performance_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: "
                f"Duration={duration:.2f}s, "
                f"χ²={result['metrics']['total_chi2']:.6f}\n")
    
    # Alert if performance degrades
    if duration > 60:  # Alert if takes more than 1 minute
        print(f"ALERT: Performance degraded - {duration:.1f}s")

if __name__ == "__main__":
    monitor_performance()
```

## Support and Resources

### Documentation
- **API Reference**: `pipelines/fit_core/API_REFERENCE.md`
- **Usage Examples**: `pipelines/fit_core/USAGE_EXAMPLES.md`
- **Developer Guide**: `pipelines/fit_core/DEVELOPER_GUIDE.md`

### Testing
- **Integration Tests**: `python integration_test_report.py`
- **Parity Tests**: `python pipelines/run_parity_tests.py`
- **Unit Tests**: `python -m pytest pipelines/fit_core/test_*.py`

### Getting Help
1. Check documentation and examples first
2. Run diagnostic tests to identify issues
3. Review error messages and logs
4. Create detailed issue reports with:
   - System configuration
   - Error messages
   - Steps to reproduce
   - Expected vs actual behavior

This migration guide ensures a smooth transition from the legacy system to the unified PBUF cosmology pipeline while maintaining scientific accuracy and system reliability.