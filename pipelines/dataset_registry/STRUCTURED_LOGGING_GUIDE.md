# Dataset Registry Structured Logging Guide

This guide explains how to use the structured logging system implemented for the dataset registry, including integration with PBUF logging infrastructure.

## Overview

The structured logging system provides:

- **JSON Lines event logging** for all dataset operations
- **Correlation ID tracking** for operation tracing
- **Integration with PBUF logging** infrastructure
- **Audit trail export** for external monitoring systems
- **Configurable log levels** and filtering capabilities

## Architecture

The logging system consists of three main components:

1. **StructuredLogger** (`core/structured_logging.py`) - Core JSON Lines logging
2. **RegistryLoggingIntegration** (`core/logging_integration.py`) - PBUF integration
3. **Automatic integration** in registry components (RegistryManager, DownloadManager, VerificationEngine)

## Basic Usage

### Configuration

```python
from pipelines.dataset_registry.core.logging_integration import configure_logging_integration

# Configure logging for the registry
configure_logging_integration(
    registry_path="data/registry",
    pbuf_log_level="INFO",
    enable_structured_logging=True,
    enable_pbuf_integration=True
)
```

### Manual Event Logging

```python
from pipelines.dataset_registry.core.logging_integration import log_registry_operation

# Log a registry operation
log_registry_operation(
    operation_type="download",
    dataset_name="cmb_planck2018",
    status="success",
    metadata={
        "source": "https://pla.esac.esa.int/...",
        "size_bytes": 1024000,
        "duration_ms": 5000.0
    }
)
```

### Operation Context Tracking

```python
from pipelines.dataset_registry.core.logging_integration import get_logging_integration

integration = get_logging_integration()

# Track an operation with automatic timing
with integration.operation_logging_context("verification", "cmb_planck2018") as op_id:
    # Perform verification steps
    # All events in this context will share the same operation_id
    pass
```

## Automatic Integration

The logging system is automatically integrated into key registry components:

### Registry Manager

All registry operations are automatically logged:

```python
from pipelines.dataset_registry.core.registry_manager import RegistryManager

# Logging is enabled by default
registry = RegistryManager("data/registry")

# This will automatically log registry_create events
registry.create_registry_entry(dataset_info, verification_result, source_used, local_path)
```

### Download Manager

Download operations include structured logging:

```python
from pipelines.dataset_registry.protocols.download_manager import DownloadManager

manager = DownloadManager()

# Pass dataset_name to enable logging
result = manager.download_with_fallback(
    sources=["https://example.com/data.dat"],
    target_path=Path("data/dataset.dat"),
    dataset_name="my_dataset"  # Enables logging
)
```

### Verification Engine

Verification operations are automatically logged:

```python
from pipelines.dataset_registry.verification.verification_engine import VerificationEngine

engine = VerificationEngine()

# This will automatically log verification events
result = engine.verify_dataset("dataset_name", file_path, verification_config)
```

## Log Event Types

The system logs the following event types:

### Download Events
- `download_started` - Download operation begins
- `download_success` - Download completed successfully
- `download_failed` - Download failed after all retries

### Verification Events
- `verification_started` - Verification begins
- `verification_success` - Verification passed
- `verification_failed` - Verification failed

### Registry Events
- `registry_create` - Registry entry created
- `verification_update` - Registry verification updated

## Log Format

Events are stored in JSON Lines format:

```json
{
  "timestamp": "2025-10-23T12:26:46.123456Z",
  "event": "download_success",
  "correlation_id": "abc123ef",
  "dataset": "cmb_planck2018",
  "level": "INFO",
  "metadata": {
    "source": "https://pla.esac.esa.int/...",
    "size_bytes": 1024000,
    "duration_ms": 5000.0,
    "sha256_match": true
  }
}
```

## Querying Events

### Get Events by Dataset

```python
from pipelines.dataset_registry.core.structured_logging import get_logger

logger = get_logger()

# Get all events for a specific dataset
events = logger.get_events(dataset_name="cmb_planck2018")

# Get only download events
download_events = logger.get_events(
    dataset_name="cmb_planck2018",
    event_type="download_success"
)
```

### Get Operation Timeline

```python
from pipelines.dataset_registry.core.logging_integration import get_logging_integration

integration = get_logging_integration()

# Get chronological timeline for a dataset
timeline = integration.get_operation_timeline("cmb_planck2018")

for operation in timeline:
    print(f"{operation['operation_type']}: {operation['status']} ({operation['duration_ms']}ms)")
```

## Audit Trail Export

### Export for External Systems

```python
from pipelines.dataset_registry.core.logging_integration import get_logging_integration

integration = get_logging_integration()

# Export comprehensive audit report
result = integration.create_comprehensive_audit_report(
    output_path="audit_reports/registry_audit.json",
    dataset_names=["cmb_planck2018", "bao_compilation"],
    include_pbuf_logs=True
)

print(f"Exported {result['total_events']} events covering {result['datasets_covered']} datasets")
```

### Add to Proof Bundles

```python
# Add registry events to existing proof bundle
proof_bundle = {"analysis_results": {...}}

updated_bundle = integration.add_registry_events_to_proof_bundle(
    proof_bundle,
    dataset_names=["cmb_planck2018"],
    operation_types=["download", "verification"]
)
```

## Configuration Options

### Log Levels

- `DEBUG` - All events including detailed debugging
- `INFO` - Normal operations (default)
- `WARNING` - Warning conditions
- `ERROR` - Error conditions only

### File Rotation

Log files automatically rotate when they exceed 100MB by default:

```python
from pipelines.dataset_registry.core.structured_logging import configure_logger

configure_logger(
    log_file_path="data/registry/logs.jsonl",
    log_level="INFO",
    enable_console=True,
    max_file_size_mb=50  # Rotate at 50MB
)
```

### Disable Logging

```python
# Disable structured logging
configure_logging_integration(
    registry_path="data/registry",
    enable_structured_logging=False,
    enable_pbuf_integration=False
)

# Or disable for specific components
registry = RegistryManager("data/registry", enable_structured_logging=False)
```

## Integration with PBUF Pipelines

The logging system integrates seamlessly with existing PBUF logging:

### Console Output

Both structured and PBUF logs appear in console output:

```
2025-10-23 12:26:46 - REGISTRY - INFO - [download_success] dataset=cmb_planck2018 source=https://... size_bytes=1024000
INFO:pbuf.dataset_registry:[REGISTRY-DOWNLOAD] dataset=cmb_planck2018 status=success source=https://... size_bytes=1024000
```

### Log Aggregation

Registry events are automatically included in PBUF log aggregation systems when `enable_pbuf_integration=True`.

## Troubleshooting

### Common Issues

1. **Log file not created**
   - Check directory permissions
   - Verify registry_path exists and is writable

2. **Events not appearing in console**
   - Check log level configuration
   - Verify `enable_console=True` in configuration

3. **PBUF integration not working**
   - Ensure PBUF logging utilities are available
   - Check `enable_pbuf_integration=True` in configuration

### Debug Mode

Enable debug logging to see all events:

```python
configure_logging_integration(
    registry_path="data/registry",
    pbuf_log_level="DEBUG"
)
```

### Verify Configuration

```python
from pipelines.dataset_registry.test_structured_logging_integration import *

# Run integration tests
test_structured_logging()
test_logging_integration()
test_audit_trail_export()
```

## Performance Considerations

- Log files are written asynchronously to minimize performance impact
- File rotation prevents unbounded log growth
- Structured logging adds minimal overhead (~1-2ms per operation)
- Console output can be disabled in production for better performance

## Security and Privacy

- Log files contain dataset names and file paths but no sensitive data
- Checksums and file sizes are logged for verification purposes
- URLs may contain authentication tokens - review before sharing logs
- Audit trails support filtered exports to exclude sensitive information

## Best Practices

1. **Use operation contexts** for related operations
2. **Include relevant metadata** in log events
3. **Set appropriate log levels** for different environments
4. **Regularly export audit trails** for long-term storage
5. **Monitor log file sizes** and configure rotation appropriately
6. **Test logging configuration** in development environments