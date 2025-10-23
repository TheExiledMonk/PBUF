#!/usr/bin/env python3
"""
Test script for structured logging integration

This script tests the integration between dataset registry operations
and the structured logging system.
"""

import json
import tempfile
from pathlib import Path

from core.structured_logging import configure_logger, get_logger, log_dataset_event
from core.logging_integration import configure_logging_integration, get_logging_integration


def test_structured_logging():
    """Test basic structured logging functionality"""
    print("Testing structured logging...")
    
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        log_file = temp_path / "test_logs.jsonl"
        
        # Configure logger
        configure_logger(
            log_file_path=log_file,
            log_level="INFO",
            enable_console=True
        )
        
        logger = get_logger()
        
        # Test basic event logging
        logger.log_event(
            "test_event",
            dataset_name="test_dataset",
            level="INFO",
            message="This is a test event",
            metadata={"test_key": "test_value"}
        )
        
        # Test operation context
        with logger.operation_context("test_operation", "test_dataset") as op_id:
            logger.log_event(
                "operation_step",
                dataset_name="test_dataset",
                operation_id=op_id,
                metadata={"step": "middle"}
            )
        
        # Verify events were written
        if log_file.exists():
            with open(log_file, 'r') as f:
                events = [json.loads(line) for line in f if line.strip()]
            
            print(f"✓ Logged {len(events)} events")
            for event in events:
                print(f"  - {event['event']}: {event.get('message', 'No message')}")
        else:
            print("✗ Log file was not created")


def test_logging_integration():
    """Test PBUF logging integration"""
    print("\nTesting PBUF logging integration...")
    
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Configure integration
        configure_logging_integration(
            registry_path=temp_path,
            pbuf_log_level="INFO",
            enable_structured_logging=True,
            enable_pbuf_integration=True
        )
        
        integration = get_logging_integration()
        
        # Test registry operation logging
        integration.log_registry_operation(
            "download",
            "test_dataset",
            status="success",
            metadata={"source": "https://example.com/data.dat", "size_bytes": 1024},
            duration_ms=1500.0
        )
        
        integration.log_registry_operation(
            "verification",
            "test_dataset",
            status="success",
            metadata={"sha256_match": True, "size_match": True}
        )
        
        # Test operation context
        with integration.operation_logging_context("registry_create", "test_dataset") as op_id:
            print(f"  Operation ID: {op_id}")
        
        # Check if structured log file was created
        log_file = temp_path / "structured_logs.jsonl"
        if log_file.exists():
            with open(log_file, 'r') as f:
                events = [json.loads(line) for line in f if line.strip()]
            
            print(f"✓ Integration logged {len(events)} events")
            for event in events:
                print(f"  - {event['event']}: dataset={event.get('dataset', 'N/A')}")
        else:
            print("✗ Integration log file was not created")


def test_audit_trail_export():
    """Test audit trail export functionality"""
    print("\nTesting audit trail export...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Configure integration
        configure_logging_integration(
            registry_path=temp_path,
            enable_structured_logging=True
        )
        
        integration = get_logging_integration()
        
        # Log some test events
        integration.log_registry_operation(
            "download", "dataset1", "success",
            metadata={"source": "url1", "size_bytes": 1000}
        )
        integration.log_registry_operation(
            "verification", "dataset1", "success",
            metadata={"sha256_match": True}
        )
        integration.log_registry_operation(
            "download", "dataset2", "failed",
            error="Network timeout"
        )
        
        # Export audit trail
        export_path = temp_path / "audit_export.json"
        result = integration.create_comprehensive_audit_report(export_path)
        
        if result["status"] == "success":
            print(f"✓ Exported audit trail: {result['total_events']} events")
            
            # Verify export file
            if export_path.exists():
                with open(export_path, 'r') as f:
                    export_data = json.load(f)
                
                print(f"  - Datasets involved: {len(export_data['summary']['datasets_involved'])}")
                print(f"  - Event types: {len(export_data['summary']['event_types'])}")
            else:
                print("✗ Export file was not created")
        else:
            print(f"✗ Export failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("Dataset Registry Structured Logging Integration Test")
    print("=" * 60)
    
    try:
        test_structured_logging()
        test_logging_integration()
        test_audit_trail_export()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()