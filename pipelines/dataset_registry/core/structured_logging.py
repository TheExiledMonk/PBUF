"""
Structured logging system for dataset registry operations

This module provides JSON Lines event logging for all dataset operations
with timestamp and correlation ID tracking for operation tracing.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import threading


@dataclass
class LogEvent:
    """Structured log event for dataset operations"""
    timestamp: str
    event_type: str
    correlation_id: str
    dataset_name: Optional[str] = None
    operation_id: Optional[str] = None
    level: str = "INFO"
    message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "timestamp": self.timestamp,
            "event": self.event_type,
            "correlation_id": self.correlation_id,
            "level": self.level
        }
        
        # Add optional fields only if they have values
        if self.dataset_name:
            result["dataset"] = self.dataset_name
        if self.operation_id:
            result["operation_id"] = self.operation_id
        if self.message:
            result["message"] = self.message
        if self.metadata:
            result["metadata"] = self.metadata
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.error:
            result["error"] = self.error
            
        return result


class StructuredLogger:
    """
    Structured logger for dataset registry operations
    
    Provides JSON Lines event logging with correlation ID tracking,
    configurable log levels, and integration with standard Python logging.
    """
    
    def __init__(
        self,
        log_file_path: Union[str, Path],
        log_level: str = "INFO",
        enable_console: bool = True,
        max_file_size_mb: int = 100
    ):
        """
        Initialize structured logger
        
        Args:
            log_file_path: Path to JSON Lines log file
            log_level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR")
            enable_console: Whether to also log to console
            max_file_size_mb: Maximum log file size before rotation
        """
        self.log_file_path = Path(log_file_path)
        self.log_level = log_level.upper()
        self.enable_console = enable_console
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Thread-local storage for correlation IDs
        self._local = threading.local()
        
        # Ensure log directory exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up standard Python logger for console output
        if self.enable_console:
            self._setup_console_logger()
    
    def _setup_console_logger(self):
        """Set up console logger for structured events"""
        self.console_logger = logging.getLogger(f"dataset_registry.{id(self)}")
        self.console_logger.setLevel(getattr(logging, self.log_level))
        
        # Remove existing handlers to avoid duplicates
        for handler in self.console_logger.handlers[:]:
            self.console_logger.removeHandler(handler)
        
        # Add console handler with structured format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level))
        
        # Custom formatter for structured events
        formatter = logging.Formatter(
            fmt='%(asctime)s - REGISTRY - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.console_logger.addHandler(console_handler)
    
    def _get_correlation_id(self) -> str:
        """Get or create correlation ID for current thread"""
        if not hasattr(self._local, 'correlation_id'):
            self._local.correlation_id = str(uuid.uuid4())[:8]
        return self._local.correlation_id
    
    def _set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread"""
        self._local.correlation_id = correlation_id
    
    def _should_log(self, level: str) -> bool:
        """Check if event should be logged based on level"""
        level_values = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40
        }
        return level_values.get(level, 20) >= level_values.get(self.log_level, 20)
    
    def _write_event(self, event: LogEvent):
        """Write event to JSON Lines log file"""
        if not self._should_log(event.level):
            return
        
        # Check file size and rotate if necessary
        self._rotate_log_if_needed()
        
        try:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, separators=(',', ':'), sort_keys=True)
                f.write('\n')
        except IOError as e:
            # Fallback to console if file write fails
            if self.enable_console:
                self.console_logger.error(f"Failed to write to log file: {e}")
    
    def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds maximum size"""
        try:
            if self.log_file_path.exists() and self.log_file_path.stat().st_size > self.max_file_size_bytes:
                # Simple rotation: move current to .old and start fresh
                old_log_path = self.log_file_path.with_suffix('.old.jsonl')
                if old_log_path.exists():
                    old_log_path.unlink()  # Remove previous old log
                self.log_file_path.rename(old_log_path)
        except OSError:
            pass  # Continue if rotation fails
    
    def log_event(
        self,
        event_type: str,
        dataset_name: Optional[str] = None,
        level: str = "INFO",
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        operation_id: Optional[str] = None
    ):
        """
        Log a structured event
        
        Args:
            event_type: Type of event (e.g., 'download_start', 'verification_complete')
            dataset_name: Name of dataset involved in the event
            level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
            message: Human-readable message
            metadata: Additional structured metadata
            error: Error message if this is an error event
            operation_id: Optional operation identifier for grouping related events
        """
        event = LogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            correlation_id=self._get_correlation_id(),
            dataset_name=dataset_name,
            operation_id=operation_id,
            level=level.upper(),
            message=message,
            metadata=metadata,
            error=error
        )
        
        self._write_event(event)
        
        # Also log to console if enabled
        if self.enable_console and self._should_log(level.upper()):
            console_message = self._format_console_message(event)
            console_level = getattr(logging, level.upper(), logging.INFO)
            self.console_logger.log(console_level, console_message)
    
    def _format_console_message(self, event: LogEvent) -> str:
        """Format event for console output"""
        parts = [f"[{event.event_type}]"]
        
        if event.dataset_name:
            parts.append(f"dataset={event.dataset_name}")
        
        if event.operation_id:
            parts.append(f"op={event.operation_id}")
        
        if event.duration_ms is not None:
            parts.append(f"duration={event.duration_ms:.1f}ms")
        
        if event.message:
            parts.append(event.message)
        
        if event.error:
            parts.append(f"error={event.error}")
        
        if event.metadata:
            # Include key metadata in console output
            for key, value in event.metadata.items():
                if key in ['status', 'source', 'size_bytes', 'sha256_match']:
                    parts.append(f"{key}={value}")
        
        return " ".join(parts)
    
    @contextmanager
    def operation_context(
        self,
        operation_type: str,
        dataset_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracking operation duration
        
        Args:
            operation_type: Type of operation (e.g., 'download', 'verification')
            dataset_name: Name of dataset involved
            metadata: Additional metadata for the operation
            
        Yields:
            operation_id: Unique identifier for this operation
        """
        operation_id = str(uuid.uuid4())[:8]
        start_time = datetime.now(timezone.utc)
        
        # Log operation start
        self.log_event(
            f"{operation_type}_start",
            dataset_name=dataset_name,
            operation_id=operation_id,
            metadata=metadata
        )
        
        try:
            yield operation_id
            
            # Log successful completion
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            complete_metadata = metadata.copy() if metadata else {}
            complete_metadata["duration_ms"] = duration_ms
            
            self.log_event(
                f"{operation_type}_complete",
                dataset_name=dataset_name,
                operation_id=operation_id,
                metadata=complete_metadata
            )
            
        except Exception as e:
            # Log operation failure
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            failure_metadata = metadata.copy() if metadata else {}
            failure_metadata["duration_ms"] = duration_ms
            
            self.log_event(
                f"{operation_type}_failed",
                dataset_name=dataset_name,
                level="ERROR",
                operation_id=operation_id,
                error=str(e),
                metadata=failure_metadata
            )
            raise
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None):
        """
        Context manager for setting correlation ID
        
        Args:
            correlation_id: Correlation ID to use (generates new one if None)
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        
        # Save current correlation ID
        old_correlation_id = getattr(self._local, 'correlation_id', None)
        
        try:
            self._set_correlation_id(correlation_id)
            yield correlation_id
        finally:
            # Restore previous correlation ID
            if old_correlation_id:
                self._set_correlation_id(old_correlation_id)
            elif hasattr(self._local, 'correlation_id'):
                delattr(self._local, 'correlation_id')
    
    def get_events(
        self,
        event_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        correlation_id: Optional[str] = None,
        level: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events from log file with filtering
        
        Args:
            event_type: Filter by event type
            dataset_name: Filter by dataset name
            correlation_id: Filter by correlation ID
            level: Filter by log level
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        events = []
        
        if not self.log_file_path.exists():
            return events
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event = json.loads(line)
                        
                        # Apply filters
                        if event_type and event.get('event') != event_type:
                            continue
                        if dataset_name and event.get('dataset') != dataset_name:
                            continue
                        if correlation_id and event.get('correlation_id') != correlation_id:
                            continue
                        if level and event.get('level') != level.upper():
                            continue
                        
                        events.append(event)
                        
                        # Apply limit
                        if limit and len(events) >= limit:
                            break
                            
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
                        
        except IOError:
            pass  # Return empty list if file cannot be read
        
        return events
    
    def get_operation_events(self, operation_id: str) -> List[Dict[str, Any]]:
        """
        Get all events for a specific operation
        
        Args:
            operation_id: Operation identifier
            
        Returns:
            List of events for the operation
        """
        events = []
        
        if not self.log_file_path.exists():
            return events
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event = json.loads(line)
                        if event.get('operation_id') == operation_id:
                            events.append(event)
                    except json.JSONDecodeError:
                        continue
                        
        except IOError:
            pass
        
        return events
    
    def export_audit_trail(
        self,
        output_path: Union[str, Path],
        dataset_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export audit trail for external monitoring systems
        
        Args:
            output_path: Path to export audit trail
            dataset_name: Filter by dataset name (optional)
            start_time: Start time filter in ISO format (optional)
            end_time: End time filter in ISO format (optional)
            
        Returns:
            Summary of exported audit trail
        """
        events = []
        
        if not self.log_file_path.exists():
            return {"total_events": 0, "exported_events": 0, "error": "Log file not found"}
        
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event = json.loads(line)
                        
                        # Apply filters
                        if dataset_name and event.get('dataset') != dataset_name:
                            continue
                        
                        event_time = event.get('timestamp')
                        if start_time and event_time < start_time:
                            continue
                        if end_time and event_time > end_time:
                            continue
                        
                        events.append(event)
                        
                    except json.JSONDecodeError:
                        continue
            
            # Export filtered events
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "filter_criteria": {
                    "dataset_name": dataset_name,
                    "start_time": start_time,
                    "end_time": end_time
                },
                "total_events": len(events),
                "events": events
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, sort_keys=True)
            
            return {
                "total_events": len(events),
                "exported_events": len(events),
                "export_path": str(output_path)
            }
            
        except IOError as e:
            return {"total_events": 0, "exported_events": 0, "error": str(e)}


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Get the global structured logger instance"""
    global _global_logger
    if _global_logger is None:
        # Default configuration
        log_path = Path("data/registry/structured_logs.jsonl")
        _global_logger = StructuredLogger(log_path)
    return _global_logger


def configure_logger(
    log_file_path: Union[str, Path],
    log_level: str = "INFO",
    enable_console: bool = True,
    max_file_size_mb: int = 100
):
    """
    Configure the global structured logger
    
    Args:
        log_file_path: Path to JSON Lines log file
        log_level: Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR")
        enable_console: Whether to also log to console
        max_file_size_mb: Maximum log file size before rotation
    """
    global _global_logger
    _global_logger = StructuredLogger(
        log_file_path=log_file_path,
        log_level=log_level,
        enable_console=enable_console,
        max_file_size_mb=max_file_size_mb
    )


def log_dataset_event(
    event_type: str,
    dataset_name: Optional[str] = None,
    level: str = "INFO",
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
):
    """
    Convenience function to log dataset events using global logger
    
    Args:
        event_type: Type of event
        dataset_name: Name of dataset involved
        level: Log level
        message: Human-readable message
        metadata: Additional structured metadata
        error: Error message if this is an error event
    """
    logger = get_logger()
    logger.log_event(
        event_type=event_type,
        dataset_name=dataset_name,
        level=level,
        message=message,
        metadata=metadata,
        error=error
    )