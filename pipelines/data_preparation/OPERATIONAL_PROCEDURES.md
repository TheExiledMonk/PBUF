# Data Preparation Framework - Operational Procedures

## Overview

This document provides operational procedures for running, maintaining, and monitoring the PBUF Data Preparation Framework in production environments.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [System Monitoring](#system-monitoring)
3. [Maintenance Procedures](#maintenance-procedures)
4. [Troubleshooting Guide](#troubleshooting-guide)
5. [Performance Optimization](#performance-optimization)
6. [Backup and Recovery](#backup-and-recovery)
7. [Security Procedures](#security-procedures)
8. [Emergency Procedures](#emergency-procedures)

## Daily Operations

### Morning Checklist

1. **System Health Check**
   ```bash
   # Check framework status
   python pipelines/data_preparation/system_certification.py --health-check
   
   # Verify log files
   tail -n 50 data/logs/processing.log
   tail -n 20 data/logs/errors.log
   ```

2. **Resource Monitoring**
   ```bash
   # Check disk space
   df -h /path/to/data
   
   # Check memory usage
   free -h
   
   # Check running processes
   ps aux | grep data_preparation
   ```

3. **Data Validation**
   ```bash
   # Verify recent processing
   ls -la data/derived/ | head -10
   
   # Check processing summaries
   ls -la data/logs/transformations/ | head -5
   ```

### Processing Workflow

#### Standard Dataset Processing

```python
from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
from pipelines.data_preparation.derivation import *

# Initialize framework
framework = DataPreparationFramework()

# Register modules
modules = [SNDerivationModule(), BAODerivationModule(), CMBDerivationModule()]
for module in modules:
    framework.register_derivation_module(module)

# Process dataset
try:
    dataset = framework.prepare_dataset("dataset_name")
    print(f"✅ Successfully processed {len(dataset.z)} data points")
except Exception as e:
    print(f"❌ Processing failed: {e}")
    # Log error and notify administrators
```

#### Batch Processing

```python
def process_multiple_datasets(dataset_names):
    """Process multiple datasets in batch."""
    framework = DataPreparationFramework()
    
    # Register all modules
    modules = [SNDerivationModule(), BAODerivationModule(), CMBDerivationModule(),
               CCDerivationModule(), RSDDerivationModule()]
    for module in modules:
        framework.register_derivation_module(module)
    
    results = {}
    for dataset_name in dataset_names:
        try:
            dataset = framework.prepare_dataset(dataset_name)
            results[dataset_name] = {
                'status': 'success',
                'data_points': len(dataset.z),
                'processing_time': time.time()
            }
        except Exception as e:
            results[dataset_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results

# Example usage
datasets_to_process = ["sn_pantheon_plus", "bao_boss_dr12", "cmb_planck2018"]
results = process_multiple_datasets(datasets_to_process)
```

## System Monitoring

### Key Metrics to Monitor

1. **Processing Performance**
   - Dataset processing time
   - Memory usage during processing
   - CPU utilization
   - Disk I/O rates

2. **Data Quality**
   - Validation pass rates
   - Error frequencies
   - Data completeness metrics
   - Processing success rates

3. **System Health**
   - Available disk space
   - Memory availability
   - Log file sizes
   - Process status

### Monitoring Scripts

#### Health Check Script

```python
#!/usr/bin/env python3
"""System health monitoring script."""

import psutil
import json
from datetime import datetime, timezone
from pathlib import Path

def check_system_health():
    """Perform comprehensive system health check."""
    
    # System resources
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/path/to/data')
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Framework status
    try:
        from pipelines.data_preparation.engine.preparation_engine import DataPreparationFramework
        framework = DataPreparationFramework()
        framework_status = "healthy"
    except Exception as e:
        framework_status = f"unhealthy: {e}"
    
    # Log file sizes
    log_dir = Path("data/logs")
    log_sizes = {}
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            log_sizes[log_file.name] = log_file.stat().st_size
    
    health_report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'system_resources': {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1e9,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / 1e9,
            'cpu_percent': cpu_percent
        },
        'framework_status': framework_status,
        'log_file_sizes': log_sizes,
        'alerts': []
    }
    
    # Generate alerts
    if memory.percent > 85:
        health_report['alerts'].append("High memory usage detected")
    if disk.percent > 90:
        health_report['alerts'].append("Low disk space warning")
    if cpu_percent > 90:
        health_report['alerts'].append("High CPU usage detected")
    
    return health_report

if __name__ == "__main__":
    health = check_system_health()
    print(json.dumps(health, indent=2))
```

#### Performance Monitoring

```python
def monitor_processing_performance(dataset_name):
    """Monitor performance during dataset processing."""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # Process dataset
    framework = DataPreparationFramework()
    dataset = framework.prepare_dataset(dataset_name)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    performance_metrics = {
        'dataset_name': dataset_name,
        'processing_time': end_time - start_time,
        'memory_delta_mb': (end_memory - start_memory) / 1e6,
        'data_points_processed': len(dataset.z),
        'processing_rate': len(dataset.z) / (end_time - start_time)
    }
    
    return performance_metrics
```

### Alerting Configuration

Set up alerts for critical conditions:

```bash
# Example cron job for health monitoring
# Add to crontab: crontab -e
*/15 * * * * /path/to/venv/bin/python /path/to/health_check.py >> /path/to/logs/health.log 2>&1

# Example alert script
#!/bin/bash
HEALTH_OUTPUT=$(python health_check.py)
ALERTS=$(echo "$HEALTH_OUTPUT" | jq -r '.alerts[]' 2>/dev/null)

if [ ! -z "$ALERTS" ]; then
    echo "ALERT: $ALERTS" | mail -s "PBUF Framework Alert" admin@example.com
fi
```

## Maintenance Procedures

### Weekly Maintenance

1. **Log Rotation and Cleanup**
   ```bash
   # Rotate logs
   logrotate /etc/logrotate.d/pbuf-data-preparation
   
   # Clean old derived datasets (older than 30 days)
   find data/derived -name "*.json" -mtime +30 -delete
   
   # Clean cache files (older than 7 days)
   find data/cache -name "*" -mtime +7 -delete
   ```

2. **Performance Analysis**
   ```python
   # Analyze processing performance trends
   def analyze_performance_trends():
       log_files = Path("data/logs/transformations").glob("*_processing_summary_*.json")
       
       processing_times = []
       for log_file in log_files:
           with open(log_file) as f:
               data = json.load(f)
               processing_times.append(data.get('processing_duration', 0))
       
       if processing_times:
           avg_time = sum(processing_times) / len(processing_times)
           max_time = max(processing_times)
           print(f"Average processing time: {avg_time:.2f}s")
           print(f"Maximum processing time: {max_time:.2f}s")
   ```

3. **Data Integrity Checks**
   ```python
   def verify_data_integrity():
       """Verify integrity of derived datasets."""
       from pipelines.data_preparation.core.validation import ValidationEngine
       
       validation_engine = ValidationEngine()
       derived_dir = Path("data/derived")
       
       for dataset_file in derived_dir.glob("*.json"):
           try:
               # Load and validate dataset
               with open(dataset_file) as f:
                   dataset_data = json.load(f)
               
               # Perform validation
               # (Implementation depends on dataset format)
               print(f"✅ {dataset_file.name} - integrity verified")
               
           except Exception as e:
               print(f"❌ {dataset_file.name} - integrity check failed: {e}")
   ```

### Monthly Maintenance

1. **System Certification**
   ```bash
   # Run full system certification
   python pipelines/data_preparation/system_certification.py
   
   # Review certification report
   cat data/logs/certification_report_*.json | jq '.certification_status'
   ```

2. **Performance Optimization Review**
   ```python
   def performance_optimization_review():
       """Review and optimize system performance."""
       
       # Analyze processing patterns
       # Identify bottlenecks
       # Recommend optimizations
       
       optimization_report = {
           'memory_usage_trends': analyze_memory_trends(),
           'processing_time_trends': analyze_processing_trends(),
           'disk_usage_patterns': analyze_disk_usage(),
           'recommendations': generate_optimization_recommendations()
       }
       
       return optimization_report
   ```

3. **Security Audit**
   ```bash
   # Check file permissions
   find data/ -type f -perm /o+w -ls
   
   # Review access logs
   grep "ERROR\|WARN" data/logs/*.log | tail -50
   
   # Verify input validation
   python -c "from pipelines.data_preparation.core.validation import ValidationEngine; print('Validation engine operational')"
   ```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Processing Failures

**Symptoms**: Dataset processing fails with errors

**Diagnosis**:
```python
# Check error logs
tail -50 data/logs/errors.log

# Verify input data
python -c "
from pathlib import Path
raw_file = Path('data/raw/dataset.csv')
print(f'File exists: {raw_file.exists()}')
print(f'File size: {raw_file.stat().st_size if raw_file.exists() else 0}')
"
```

**Solutions**:
1. Verify input data format and integrity
2. Check available system resources
3. Review derivation module configuration
4. Validate file permissions

#### 2. Memory Issues

**Symptoms**: Out of memory errors during processing

**Diagnosis**:
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head -10
```

**Solutions**:
1. Increase available memory
2. Process datasets in smaller chunks
3. Clear processing cache
4. Optimize derivation module memory usage

#### 3. Validation Failures

**Symptoms**: Dataset validation fails

**Diagnosis**:
```python
from pipelines.data_preparation.core.validation import ValidationEngine

validation_engine = ValidationEngine()
# Run validation with detailed output
results = validation_engine.validate_dataset(dataset, "dataset_name")
print(json.dumps(results, indent=2))
```

**Solutions**:
1. Review validation criteria
2. Check data quality and completeness
3. Verify covariance matrix properties
4. Validate numerical ranges

#### 4. Performance Issues

**Symptoms**: Slow processing times

**Diagnosis**:
```python
# Profile processing performance
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Process dataset
framework.prepare_dataset("dataset_name")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

**Solutions**:
1. Optimize derivation module algorithms
2. Enable caching
3. Use parallel processing (when deterministic)
4. Upgrade hardware resources

## Performance Optimization

### Optimization Strategies

1. **Caching Optimization**
   ```python
   # Configure intelligent caching
   framework = DataPreparationFramework()
   
   # Enable aggressive caching for stable datasets
   dataset = framework.prepare_dataset(
       "stable_dataset",
       force_reprocess=False  # Use cache when available
   )
   ```

2. **Memory Management**
   ```python
   # Process large datasets in chunks
   def process_large_dataset_chunked(dataset_path, chunk_size=1000):
       # Implementation for chunked processing
       pass
   ```

3. **I/O Optimization**
   ```bash
   # Use faster storage for cache and temporary files
   # Configure appropriate file system options
   mount -o noatime,nodiratime /dev/fast_ssd /path/to/data/cache
   ```

### Performance Monitoring

```python
def setup_performance_monitoring():
    """Set up continuous performance monitoring."""
    
    # Monitor key metrics
    metrics_to_track = [
        'processing_time_per_dataset',
        'memory_usage_peak',
        'disk_io_rates',
        'validation_time',
        'cache_hit_rate'
    ]
    
    # Set up metric collection
    # Configure alerting thresholds
    # Generate performance reports
```

## Backup and Recovery

### Backup Procedures

1. **Daily Backups**
   ```bash
   #!/bin/bash
   # daily_backup.sh
   
   DATE=$(date +%Y%m%d)
   BACKUP_DIR="/backup/pbuf/daily/$DATE"
   
   mkdir -p "$BACKUP_DIR"
   
   # Backup derived datasets
   rsync -av data/derived/ "$BACKUP_DIR/derived/"
   
   # Backup registry
   rsync -av data/registry/ "$BACKUP_DIR/registry/"
   
   # Backup configuration
   cp ~/.pbuf/data_preparation_config.json "$BACKUP_DIR/"
   ```

2. **Weekly Full Backups**
   ```bash
   #!/bin/bash
   # weekly_backup.sh
   
   DATE=$(date +%Y%m%d)
   BACKUP_DIR="/backup/pbuf/weekly/$DATE"
   
   mkdir -p "$BACKUP_DIR"
   
   # Full system backup
   tar -czf "$BACKUP_DIR/complete_backup.tar.gz" \
       data/ \
       pipelines/data_preparation/ \
       ~/.pbuf/
   ```

### Recovery Procedures

1. **Dataset Recovery**
   ```bash
   # Restore derived datasets
   rsync -av /backup/pbuf/daily/20231201/derived/ data/derived/
   
   # Verify restoration
   python -c "
   from pathlib import Path
   derived_dir = Path('data/derived')
   print(f'Restored {len(list(derived_dir.glob(\"*.json\")))} datasets')
   "
   ```

2. **Configuration Recovery**
   ```bash
   # Restore configuration
   cp /backup/pbuf/daily/20231201/data_preparation_config.json ~/.pbuf/
   
   # Verify configuration
   python -c "
   import json
   with open('~/.pbuf/data_preparation_config.json') as f:
       config = json.load(f)
   print('Configuration restored successfully')
   "
   ```

## Security Procedures

### Access Control

1. **File Permissions**
   ```bash
   # Set appropriate permissions
   chmod 755 data/
   chmod 644 data/derived/*.json
   chmod 600 ~/.pbuf/data_preparation_config.json
   ```

2. **User Management**
   ```bash
   # Create dedicated user for framework operations
   sudo useradd -m -s /bin/bash pbuf_operator
   sudo usermod -aG pbuf pbuf_operator
   ```

### Input Validation

```python
def validate_input_security(raw_data_path, metadata):
    """Perform security validation of input data."""
    
    # Check file path for directory traversal
    if ".." in str(raw_data_path):
        raise SecurityError("Invalid file path detected")
    
    # Validate file size
    if raw_data_path.stat().st_size > 1e9:  # 1GB limit
        raise SecurityError("File size exceeds limit")
    
    # Validate metadata
    if not isinstance(metadata, dict):
        raise SecurityError("Invalid metadata format")
    
    # Additional security checks...
```

### Audit Logging

```python
def setup_audit_logging():
    """Set up comprehensive audit logging."""
    
    audit_logger = logging.getLogger('audit')
    audit_handler = logging.FileHandler('data/logs/audit.log')
    audit_formatter = logging.Formatter(
        '%(asctime)s - AUDIT - %(message)s'
    )
    audit_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_handler)
    
    return audit_logger

# Usage
audit_logger = setup_audit_logging()
audit_logger.info(f"Dataset processed: {dataset_name} by user {user_id}")
```

## Emergency Procedures

### System Failure Response

1. **Immediate Actions**
   ```bash
   # Stop processing
   pkill -f data_preparation
   
   # Check system status
   systemctl status pbuf-data-preparation
   
   # Review recent logs
   tail -100 data/logs/errors.log
   ```

2. **Recovery Steps**
   ```bash
   # Restart framework service
   systemctl restart pbuf-data-preparation
   
   # Verify system health
   python pipelines/data_preparation/system_certification.py --health-check
   
   # Resume processing
   python resume_processing.py
   ```

### Data Corruption Response

1. **Detection**
   ```python
   def detect_data_corruption():
       """Detect potential data corruption."""
       
       validation_engine = ValidationEngine()
       corrupted_datasets = []
       
       for dataset_file in Path("data/derived").glob("*.json"):
           try:
               # Load and validate dataset
               dataset = load_dataset(dataset_file)
               results = validation_engine.validate_dataset(dataset, dataset_file.stem)
               
               if not results['validation_passed']:
                   corrupted_datasets.append(dataset_file)
                   
           except Exception as e:
               corrupted_datasets.append(dataset_file)
       
       return corrupted_datasets
   ```

2. **Recovery**
   ```bash
   # Restore from backup
   rsync -av /backup/pbuf/daily/latest/derived/ data/derived/
   
   # Reprocess affected datasets
   python reprocess_datasets.py --datasets corrupted_list.txt
   ```

### Contact Information

- **System Administrator**: admin@pbuf.org
- **Framework Developer**: dev@pbuf.org
- **Emergency Contact**: emergency@pbuf.org
- **On-call Phone**: +1-555-PBUF-911

This operational procedures document provides comprehensive guidance for running and maintaining the data preparation framework in production environments. Regular review and updates of these procedures ensure optimal system performance and reliability.