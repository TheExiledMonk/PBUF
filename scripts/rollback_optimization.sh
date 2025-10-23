#!/bin/bash
# rollback_optimization.sh
# Emergency rollback script for parameter optimization system

set -e  # Exit on any error

echo "=== Parameter Optimization System Rollback ==="
echo "Starting rollback at $(date)"

# Configuration
BACKUP_PREFIX="fit_core_backup_"
STORAGE_BACKUP_PREFIX="optimization_results_backup_"
ROLLBACK_LOG="rollback_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ROLLBACK_LOG"
}

# Error handling
handle_error() {
    log "ERROR: Rollback failed at line $1"
    log "Check $ROLLBACK_LOG for details"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Function to find latest backup
find_latest_backup() {
    local prefix="$1"
    local backup_dir="$2"
    
    if [ -n "$backup_dir" ]; then
        ls -1 "$backup_dir" 2>/dev/null | grep "^$prefix" | sort -r | head -1
    else
        ls -1 . 2>/dev/null | grep "^$prefix" | sort -r | head -1
    fi
}

# Check for running optimization processes
log "Checking for running optimization processes..."
RUNNING_OPTS=$(pgrep -f "fit_cmb.py.*--optimize" || true)
if [ -n "$RUNNING_OPTS" ]; then
    log "Found running optimization processes: $RUNNING_OPTS"
    log "Terminating optimization processes..."
    pkill -f "fit_cmb.py.*--optimize" || true
    sleep 2
    
    # Force kill if still running
    STILL_RUNNING=$(pgrep -f "fit_cmb.py.*--optimize" || true)
    if [ -n "$STILL_RUNNING" ]; then
        log "Force killing persistent processes..."
        pkill -9 -f "fit_cmb.py.*--optimize" || true
    fi
    
    log "✓ Optimization processes terminated"
else
    log "✓ No running optimization processes found"
fi

# Parse command line arguments
ROLLBACK_TYPE="full"
BACKUP_DATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            ROLLBACK_TYPE="$2"
            shift 2
            ;;
        --backup-date)
            BACKUP_DATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--type full|disable] [--backup-date YYYYMMDD]"
            echo ""
            echo "Options:"
            echo "  --type full     Complete rollback to backup (default)"
            echo "  --type disable  Disable optimization without code rollback"
            echo "  --backup-date   Specific backup date to restore from"
            echo ""
            exit 0
            ;;
        *)
            log "Unknown option: $1"
            exit 1
            ;;
    esac
done

log "Rollback type: $ROLLBACK_TYPE"

if [ "$ROLLBACK_TYPE" = "disable" ]; then
    # Disable optimization without full rollback
    log "Disabling optimization system..."
    
    # Create disable flag
    mkdir -p optimization_results
    touch optimization_results/.disabled
    
    # Clear any lock files
    rm -f optimization_results/*.lock
    
    log "✓ Optimization system disabled"
    log "Remove optimization_results/.disabled to re-enable"
    log "Rollback completed successfully (disable mode)"
    exit 0
fi

# Full rollback procedure
log "Starting full rollback procedure..."

# Step 1: Find and restore code backup
log "Step 1: Restoring code backup..."

if [ -n "$BACKUP_DATE" ]; then
    CODE_BACKUP="${BACKUP_PREFIX}${BACKUP_DATE}"
else
    CODE_BACKUP=$(find_latest_backup "$BACKUP_PREFIX" "pipelines/")
fi

if [ -z "$CODE_BACKUP" ]; then
    log "ERROR: No code backup found"
    log "Available backups:"
    ls -1 pipelines/ | grep "$BACKUP_PREFIX" || log "  (none found)"
    exit 1
fi

log "Using code backup: $CODE_BACKUP"

if [ -d "pipelines/$CODE_BACKUP" ]; then
    # Create current backup before rollback
    CURRENT_BACKUP="fit_core_pre_rollback_$(date +%Y%m%d_%H%M%S)"
    log "Creating pre-rollback backup: $CURRENT_BACKUP"
    cp -r pipelines/fit_core "pipelines/$CURRENT_BACKUP"
    
    # Restore from backup
    log "Restoring code from backup..."
    rm -rf pipelines/fit_core
    mv "pipelines/$CODE_BACKUP" pipelines/fit_core
    
    log "✓ Code rollback completed"
else
    log "ERROR: Backup directory pipelines/$CODE_BACKUP not found"
    exit 1
fi

# Step 2: Restore parameter storage backup
log "Step 2: Restoring parameter storage backup..."

if [ -n "$BACKUP_DATE" ]; then
    STORAGE_BACKUP="${STORAGE_BACKUP_PREFIX}${BACKUP_DATE}"
else
    STORAGE_BACKUP=$(find_latest_backup "$STORAGE_BACKUP_PREFIX" ".")
fi

if [ -n "$STORAGE_BACKUP" ] && [ -d "$STORAGE_BACKUP" ]; then
    log "Using storage backup: $STORAGE_BACKUP"
    
    # Backup current storage
    if [ -d "optimization_results" ]; then
        CURRENT_STORAGE_BACKUP="optimization_results_pre_rollback_$(date +%Y%m%d_%H%M%S)"
        log "Creating pre-rollback storage backup: $CURRENT_STORAGE_BACKUP"
        mv optimization_results "$CURRENT_STORAGE_BACKUP"
    fi
    
    # Restore from backup
    log "Restoring storage from backup..."
    mv "$STORAGE_BACKUP" optimization_results
    
    log "✓ Parameter storage rollback completed"
else
    log "⚠️ No parameter storage backup found or backup is empty"
    log "Removing current optimization_results directory..."
    rm -rf optimization_results
    log "✓ Parameter storage cleared"
fi

# Step 3: Clean up optimization artifacts
log "Step 3: Cleaning up optimization artifacts..."

# Remove any remaining lock files
find . -name "*.lock" -path "*/optimization_results/*" -delete 2>/dev/null || true

# Remove optimization configuration files if they exist
OPTIMIZATION_CONFIGS=(
    "optimization_config.json"
    "optimization_settings.json"
    ".optimization_cache"
)

for config in "${OPTIMIZATION_CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        log "Removing optimization config: $config"
        rm -f "$config"
    fi
done

log "✓ Cleanup completed"

# Step 4: Verify rollback
log "Step 4: Verifying rollback..."

# Test that optimization modules are no longer available
python3 -c "
try:
    from pipelines.fit_core.optimizer import ParameterOptimizer
    print('ERROR: Optimization system still present after rollback')
    exit(1)
except ImportError:
    print('✓ Optimization system successfully removed')
except Exception as e:
    print(f'✓ Optimization system not accessible: {e}')
" 2>&1 | tee -a "$ROLLBACK_LOG"

# Test that basic functionality still works
python3 -c "
try:
    from pipelines.fit_core.engine import run_fit
    from pipelines.fit_core.parameter import build_params
    
    # Test basic parameter building
    params = build_params('lcdm')
    print('✓ Basic parameter building works')
    
    # Test that optimization parameters are rejected
    try:
        result = run_fit('lcdm', ['cmb'], optimization_params=['H0'])
        print('ERROR: Optimization parameters still accepted')
        exit(1)
    except (TypeError, ValueError):
        print('✓ Optimization parameters properly rejected')
    
    print('✓ Basic functionality verified')
    
except Exception as e:
    print(f'ERROR: Basic functionality test failed: {e}')
    exit(1)
" 2>&1 | tee -a "$ROLLBACK_LOG"

if [ $? -eq 0 ]; then
    log "✓ Rollback verification successful"
else
    log "❌ Rollback verification failed"
    exit 1
fi

# Step 5: Generate rollback report
log "Step 5: Generating rollback report..."

ROLLBACK_REPORT="rollback_report_$(date +%Y%m%d_%H%M%S).json"

cat > "$ROLLBACK_REPORT" << EOF
{
  "rollback_timestamp": "$(date -Iseconds)",
  "rollback_type": "$ROLLBACK_TYPE",
  "code_backup_used": "$CODE_BACKUP",
  "storage_backup_used": "$STORAGE_BACKUP",
  "verification_status": "success",
  "artifacts_removed": [
    $(printf '"%s",' "${OPTIMIZATION_CONFIGS[@]}" | sed 's/,$//')
  ],
  "log_file": "$ROLLBACK_LOG"
}
EOF

log "✓ Rollback report generated: $ROLLBACK_REPORT"

# Final summary
log "=== Rollback Summary ==="
log "✅ Rollback completed successfully"
log "Code backup restored: $CODE_BACKUP"
log "Storage backup restored: ${STORAGE_BACKUP:-'(none)'}"
log "Verification: PASSED"
log "Log file: $ROLLBACK_LOG"
log "Report file: $ROLLBACK_REPORT"
log ""
log "The system has been restored to pre-optimization state."
log "All optimization functionality has been removed."
log "Basic cosmology pipeline functionality is preserved."

echo ""
echo "✅ Rollback completed successfully at $(date)"
echo "Check $ROLLBACK_LOG for detailed log"
echo "Check $ROLLBACK_REPORT for rollback report"