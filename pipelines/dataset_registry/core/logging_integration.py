"""
Integration between dataset registry structured logging and PBUF logging infrastructure

This module provides seamless integration between the dataset registry's
structured logging system and the existing PBUF pipeline logging infrastructure.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from contextlib import contextmanager

from .structured_logging import StructuredLogger, get_logger, configure_logger, log_dataset_event


class RegistryLoggingIntegration:
    """
    Integration layer between registry structured logging and PBUF logging
    
    Provides unified logging configuration and event correlation between
    dataset registry operations and pipeline execution.
    """
    
    def __init__(
        self,
        registry_path: Union[str, Path],
        pbuf_log_level: str = "INFO",
        enable_structured_logging: bool = True,
        enable_pbuf_integration: bool = True
    ):
        """
        Initialize logging integration
        
        Args:
            registry_path: Path to registry directory for log files
            pbuf_log_level: Log level for PBUF pipeline logging
            enable_structured_logging: Whether to enable structured JSON logging
            enable_pbuf_integration: Whether to integrate with PBUF logging
        """
        self.registry_path = Path(registry_path)
        self.pbuf_log_level = pbuf_log_level
        self.enable_structured_logging = enable_structured_logging
        self.enable_pbuf_integration = enable_pbuf_integration
        
        # Set up structured logging if enabled
        if self.enable_structured_logging:
            structured_log_path = self.registry_path / "structured_logs.jsonl"
            configure_logger(
                log_file_path=structured_log_path,
                log_level=pbuf_log_level,
                enable_console=True,
                max_file_size_mb=100
            )
        
        # Set up PBUF logging integration if enabled
        if self.enable_pbuf_integration:
            self._setup_pbuf_integration()
    
    def _setup_pbuf_integration(self):
        """Set up integration with PBUF logging infrastructure"""
        # Create a custom handler that forwards registry events to PBUF logging
        self.pbuf_logger = logging.getLogger("pbuf.dataset_registry")
        self.pbuf_logger.setLevel(getattr(logging, self.pbuf_log_level))
        
        # Ensure PBUF logging is configured
        try:
            # Try to import and use PBUF logging utilities
            from ...fit_core.logging_utils import setup_logging as setup_pbuf_logging
            setup_pbuf_logging(level=self.pbuf_log_level)
        except ImportError:
            # Fallback if PBUF logging utilities not available
            logging.basicConfig(level=getattr(logging, self.pbuf_log_level))
        except Exception:
            # Fallback if PBUF logging setup fails
            logging.basicConfig(level=getattr(logging, self.pbuf_log_level))
    
    def log_registry_operation(
        self,
        operation_type: str,
        dataset_name: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ):
        """
        Log registry operation to both structured and PBUF logging systems
        
        Args:
            operation_type: Type of operation (e.g., 'download', 'verification', 'registry_create')
            dataset_name: Name of dataset involved
            status: Operation status ('success', 'failed', 'warning')
            metadata: Additional operation metadata
            error: Error message if operation failed
            duration_ms: Operation duration in milliseconds
        """
        # Determine log level based on status
        if status == "failed":
            level = "ERROR"
        elif status == "warning":
            level = "WARNING"
        else:
            level = "INFO"
        
        # Log to structured logging system
        if self.enable_structured_logging:
            log_metadata = metadata.copy() if metadata else {}
            if duration_ms is not None:
                log_metadata["duration_ms"] = duration_ms
            log_metadata["status"] = status
            
            log_dataset_event(
                event_type=f"{operation_type}_{status}",
                dataset_name=dataset_name,
                level=level,
                metadata=log_metadata,
                error=error
            )
        
        # Log to PBUF logging system
        if self.enable_pbuf_integration:
            message = self._format_pbuf_message(operation_type, dataset_name, status, metadata, duration_ms)
            
            if level == "ERROR":
                self.pbuf_logger.error(message)
            elif level == "WARNING":
                self.pbuf_logger.warning(message)
            else:
                self.pbuf_logger.info(message)
    
    def _format_pbuf_message(
        self,
        operation_type: str,
        dataset_name: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None
    ) -> str:
        """Format message for PBUF logging system"""
        parts = [f"[REGISTRY-{operation_type.upper()}]", f"dataset={dataset_name}", f"status={status}"]
        
        if duration_ms is not None:
            parts.append(f"duration={duration_ms:.1f}ms")
        
        if metadata:
            # Include key metadata in PBUF log message
            for key, value in metadata.items():
                if key in ['source', 'sha256_match', 'size_bytes', 'verification_status']:
                    parts.append(f"{key}={value}")
        
        return " ".join(parts)
    
    @contextmanager
    def operation_logging_context(
        self,
        operation_type: str,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for logging operation start/completion with both systems
        
        Args:
            operation_type: Type of operation
            dataset_name: Name of dataset involved
            metadata: Additional operation metadata
        """
        # Use structured logging context if available
        if self.enable_structured_logging:
            logger = get_logger()
            with logger.operation_context(operation_type, dataset_name, metadata) as operation_id:
                try:
                    yield operation_id
                except Exception as e:
                    # Log failure to PBUF system as well
                    if self.enable_pbuf_integration:
                        self.pbuf_logger.error(
                            f"[REGISTRY-{operation_type.upper()}] dataset={dataset_name} status=failed error={str(e)}"
                        )
                    raise
        else:
            # Fallback to simple PBUF logging
            if self.enable_pbuf_integration:
                self.pbuf_logger.info(f"[REGISTRY-{operation_type.upper()}] dataset={dataset_name} status=started")
            
            try:
                yield None
                if self.enable_pbuf_integration:
                    self.pbuf_logger.info(f"[REGISTRY-{operation_type.upper()}] dataset={dataset_name} status=completed")
            except Exception as e:
                if self.enable_pbuf_integration:
                    self.pbuf_logger.error(f"[REGISTRY-{operation_type.upper()}] dataset={dataset_name} status=failed error={str(e)}")
                raise
    
    def add_registry_events_to_proof_bundle(
        self,
        proof_bundle: Dict[str, Any],
        dataset_names: Optional[List[str]] = None,
        operation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add registry events to proof bundle audit trails
        
        Args:
            proof_bundle: Existing proof bundle dictionary
            dataset_names: Filter events by dataset names (optional)
            operation_types: Filter events by operation types (optional)
            
        Returns:
            Updated proof bundle with registry events
        """
        if not self.enable_structured_logging:
            return proof_bundle
        
        logger = get_logger()
        
        # Collect relevant registry events
        registry_events = []
        
        if dataset_names:
            for dataset_name in dataset_names:
                events = logger.get_events(dataset_name=dataset_name)
                if operation_types:
                    events = [e for e in events if any(e.get('event', '').startswith(op) for op in operation_types)]
                registry_events.extend(events)
        else:
            events = logger.get_events()
            if operation_types:
                events = [e for e in events if any(e.get('event', '').startswith(op) for op in operation_types)]
            registry_events.extend(events)
        
        # Add registry events to proof bundle
        if "audit_trail" not in proof_bundle:
            proof_bundle["audit_trail"] = {}
        
        proof_bundle["audit_trail"]["dataset_registry_events"] = {
            "total_events": len(registry_events),
            "events": registry_events,
            "event_types": list(set(e.get('event', 'unknown') for e in registry_events)),
            "datasets_involved": list(set(e.get('dataset') for e in registry_events if e.get('dataset')))
        }
        
        return proof_bundle
    
    def create_comprehensive_audit_report(
        self,
        output_path: Union[str, Path],
        dataset_names: Optional[List[str]] = None,
        include_pbuf_logs: bool = True
    ) -> Dict[str, Any]:
        """
        Create comprehensive audit report combining registry and PBUF logs
        
        Args:
            output_path: Path to save audit report
            dataset_names: Filter by dataset names (optional)
            include_pbuf_logs: Whether to include PBUF log entries
            
        Returns:
            Summary of audit report creation
        """
        output_path = Path(output_path)
        
        audit_report = {
            "report_timestamp": get_logger().log_event.__defaults__[0] if hasattr(get_logger().log_event, '__defaults__') else None,
            "registry_events": [],
            "pbuf_log_entries": [],
            "summary": {}
        }
        
        # Collect registry events
        if self.enable_structured_logging:
            logger = get_logger()
            
            if dataset_names:
                for dataset_name in dataset_names:
                    events = logger.get_events(dataset_name=dataset_name)
                    audit_report["registry_events"].extend(events)
            else:
                audit_report["registry_events"] = logger.get_events()
        
        # Collect PBUF log entries (if available and requested)
        if include_pbuf_logs and self.enable_pbuf_integration:
            # This would require access to PBUF log files
            # For now, we'll include a placeholder for PBUF log integration
            audit_report["pbuf_log_entries"] = []
            audit_report["summary"]["pbuf_logs_note"] = "PBUF log integration requires additional configuration"
        
        # Create summary
        audit_report["summary"] = {
            "total_registry_events": len(audit_report["registry_events"]),
            "total_pbuf_entries": len(audit_report["pbuf_log_entries"]),
            "datasets_involved": list(set(
                e.get('dataset') for e in audit_report["registry_events"] 
                if e.get('dataset')
            )),
            "event_types": list(set(
                e.get('event') for e in audit_report["registry_events"] 
                if e.get('event')
            ))
        }
        
        # Save audit report
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(audit_report, f, indent=2, sort_keys=True)
            
            return {
                "status": "success",
                "output_path": str(output_path),
                "total_events": audit_report["summary"]["total_registry_events"],
                "datasets_covered": len(audit_report["summary"]["datasets_involved"])
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "total_events": 0,
                "datasets_covered": 0
            }
    
    def get_operation_timeline(
        self,
        dataset_name: str,
        operation_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of operations for a dataset
        
        Args:
            dataset_name: Name of dataset
            operation_types: Filter by operation types (optional)
            
        Returns:
            Chronologically ordered list of operations
        """
        if not self.enable_structured_logging:
            return []
        
        logger = get_logger()
        events = logger.get_events(dataset_name=dataset_name)
        
        # Filter by operation types if specified
        if operation_types:
            events = [
                e for e in events 
                if any(e.get('event', '').startswith(op) for op in operation_types)
            ]
        
        # Sort by timestamp
        events.sort(key=lambda e: e.get('timestamp', ''))
        
        # Group related events by operation_id
        timeline = []
        operations = {}
        
        for event in events:
            operation_id = event.get('operation_id')
            if operation_id:
                if operation_id not in operations:
                    operations[operation_id] = []
                operations[operation_id].append(event)
            else:
                # Standalone event
                timeline.append({
                    "operation_type": "standalone",
                    "events": [event],
                    "start_time": event.get('timestamp'),
                    "end_time": event.get('timestamp'),
                    "duration_ms": event.get('duration_ms'),
                    "status": self._determine_event_status(event)
                })
        
        # Process grouped operations
        for operation_id, op_events in operations.items():
            op_events.sort(key=lambda e: e.get('timestamp', ''))
            
            start_event = op_events[0]
            end_event = op_events[-1]
            
            operation_type = start_event.get('event', '').replace('_start', '').replace('_complete', '').replace('_failed', '')
            
            timeline.append({
                "operation_type": operation_type,
                "operation_id": operation_id,
                "events": op_events,
                "start_time": start_event.get('timestamp'),
                "end_time": end_event.get('timestamp'),
                "duration_ms": end_event.get('duration_ms'),
                "status": self._determine_operation_status(op_events)
            })
        
        # Sort timeline by start time
        timeline.sort(key=lambda op: op.get('start_time', ''))
        
        return timeline
    
    def _determine_event_status(self, event: Dict[str, Any]) -> str:
        """Determine status from a single event"""
        event_type = event.get('event', '')
        if 'failed' in event_type or event.get('error'):
            return 'failed'
        elif 'warning' in event_type or event.get('level') == 'WARNING':
            return 'warning'
        else:
            return 'success'
    
    def _determine_operation_status(self, events: List[Dict[str, Any]]) -> str:
        """Determine overall status from operation events"""
        for event in events:
            if 'failed' in event.get('event', '') or event.get('error'):
                return 'failed'
        
        for event in events:
            if 'warning' in event.get('event', '') or event.get('level') == 'WARNING':
                return 'warning'
        
        return 'success'


# Global integration instance
_global_integration: Optional[RegistryLoggingIntegration] = None


def get_logging_integration() -> RegistryLoggingIntegration:
    """Get the global logging integration instance"""
    global _global_integration
    if _global_integration is None:
        # Default configuration
        _global_integration = RegistryLoggingIntegration(
            registry_path="data/registry",
            pbuf_log_level="INFO",
            enable_structured_logging=True,
            enable_pbuf_integration=True
        )
    return _global_integration


def configure_logging_integration(
    registry_path: Union[str, Path],
    pbuf_log_level: str = "INFO",
    enable_structured_logging: bool = True,
    enable_pbuf_integration: bool = True
):
    """
    Configure the global logging integration
    
    Args:
        registry_path: Path to registry directory for log files
        pbuf_log_level: Log level for PBUF pipeline logging
        enable_structured_logging: Whether to enable structured JSON logging
        enable_pbuf_integration: Whether to integrate with PBUF logging
    """
    global _global_integration
    _global_integration = RegistryLoggingIntegration(
        registry_path=registry_path,
        pbuf_log_level=pbuf_log_level,
        enable_structured_logging=enable_structured_logging,
        enable_pbuf_integration=enable_pbuf_integration
    )


def log_registry_operation(
    operation_type: str,
    dataset_name: str,
    status: str = "success",
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    duration_ms: Optional[float] = None
):
    """
    Convenience function to log registry operations using global integration
    
    Args:
        operation_type: Type of operation
        dataset_name: Name of dataset involved
        status: Operation status
        metadata: Additional operation metadata
        error: Error message if operation failed
        duration_ms: Operation duration in milliseconds
    """
    integration = get_logging_integration()
    integration.log_registry_operation(
        operation_type=operation_type,
        dataset_name=dataset_name,
        status=status,
        metadata=metadata,
        error=error,
        duration_ms=duration_ms
    )