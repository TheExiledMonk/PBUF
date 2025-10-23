"""
Comprehensive error handling and logging system for the data preparation framework.

This module provides detailed error reporting, stage-specific error handling,
and comprehensive logging capabilities for all framework operations.
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .interfaces import ProcessingError
from .schema import StandardDataset


class ProcessingStage(Enum):
    """Enumeration of processing stages for error categorization."""
    INITIALIZATION = "initialization"
    DATA_RETRIEVAL = "data_retrieval"
    INPUT_VALIDATION = "input_validation"
    TRANSFORMATION = "transformation"
    OUTPUT_VALIDATION = "output_validation"
    PROVENANCE_RECORDING = "provenance_recording"
    CACHE_MANAGEMENT = "cache_management"
    MODULE_SELECTION = "module_selection"
    REGISTRY_INTEGRATION = "registry_integration"


class ErrorSeverity(Enum):
    """Error severity levels for prioritization and handling."""
    CRITICAL = "critical"      # System cannot continue
    ERROR = "error"           # Processing failed but system stable
    WARNING = "warning"       # Issue detected but processing can continue
    INFO = "info"            # Informational message


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    dataset_name: str
    stage: ProcessingStage
    severity: ErrorSeverity
    error_type: str
    error_message: str
    timestamp: str
    processing_duration: Optional[float] = None
    system_info: Optional[Dict[str, Any]] = None
    dataset_info: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    suggested_actions: Optional[List[str]] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorContext':
        """Create ErrorContext from dictionary."""
        # Convert string enums back to enum objects
        data['stage'] = ProcessingStage(data['stage'])
        data['severity'] = ErrorSeverity(data['severity'])
        return cls(**data)


class EnhancedProcessingError(ProcessingError):
    """
    Enhanced processing error with comprehensive context and recovery information.
    
    Extends the base ProcessingError with additional context, severity levels,
    and recovery tracking capabilities.
    """
    
    def __init__(
        self,
        dataset_name: str,
        stage: Union[str, ProcessingStage],
        error_type: str,
        error_message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: Dict[str, Any] = None,
        suggested_actions: List[str] = None,
        original_exception: Exception = None,
        processing_duration: float = None
    ):
        """
        Initialize enhanced processing error.
        
        Args:
            dataset_name: Name of dataset being processed
            stage: Processing stage where error occurred
            error_type: Type of error
            error_message: Detailed error description
            severity: Error severity level
            context: Additional context information
            suggested_actions: List of suggested remediation actions
            original_exception: Original exception that caused this error
            processing_duration: Time spent processing before error
        """
        # Convert stage to ProcessingStage if string
        if isinstance(stage, str):
            try:
                stage = ProcessingStage(stage)
            except ValueError:
                stage = ProcessingStage.INITIALIZATION
        
        self.severity = severity
        self.original_exception = original_exception
        self.processing_duration = processing_duration
        
        # Create comprehensive error context
        self.error_context = ErrorContext(
            dataset_name=dataset_name,
            stage=stage,
            severity=severity,
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            processing_duration=processing_duration,
            system_info=self._collect_system_info(),
            dataset_info=context.get('dataset_info') if context else None,
            stack_trace=self._get_stack_trace(),
            suggested_actions=suggested_actions or [],
            recovery_attempted=False,
            recovery_successful=False
        )
        
        # Initialize base ProcessingError
        super().__init__(
            dataset_name=dataset_name,
            stage=stage.value,
            error_type=error_type,
            error_message=error_message,
            context=context or {},
            suggested_actions=suggested_actions or []
        )
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for error context."""
        import sys
        import platform
        import psutil
        
        try:
            return {
                "python_version": sys.version,
                "platform": platform.platform(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except Exception:
            return {"system_info": "unavailable"}
    
    def _get_stack_trace(self) -> str:
        """Get formatted stack trace."""
        if self.original_exception:
            return ''.join(traceback.format_exception(
                type(self.original_exception),
                self.original_exception,
                self.original_exception.__traceback__
            ))
        else:
            return traceback.format_exc()
    
    def mark_recovery_attempted(self, successful: bool = False):
        """Mark that recovery was attempted for this error."""
        self.error_context.recovery_attempted = True
        self.error_context.recovery_successful = successful
    
    def generate_detailed_report(self) -> str:
        """Generate comprehensive error report."""
        report = []
        report.append("=" * 80)
        report.append("DATA PREPARATION FRAMEWORK - ERROR REPORT")
        report.append("=" * 80)
        report.append(f"Dataset: {self.error_context.dataset_name}")
        report.append(f"Timestamp: {self.error_context.timestamp}")
        report.append(f"Stage: {self.error_context.stage.value}")
        report.append(f"Severity: {self.error_context.severity.value.upper()}")
        report.append(f"Error Type: {self.error_context.error_type}")
        
        if self.error_context.processing_duration:
            report.append(f"Processing Duration: {self.error_context.processing_duration:.2f}s")
        
        report.append("")
        report.append("ERROR DETAILS:")
        report.append("-" * 40)
        report.append(self.error_context.error_message)
        
        if self.context:
            report.append("")
            report.append("CONTEXT INFORMATION:")
            report.append("-" * 40)
            for key, value in self.context.items():
                report.append(f"  {key}: {value}")
        
        if self.error_context.system_info:
            report.append("")
            report.append("SYSTEM INFORMATION:")
            report.append("-" * 40)
            for key, value in self.error_context.system_info.items():
                report.append(f"  {key}: {value}")
        
        if self.error_context.suggested_actions:
            report.append("")
            report.append("SUGGESTED ACTIONS:")
            report.append("-" * 40)
            for i, action in enumerate(self.error_context.suggested_actions, 1):
                report.append(f"  {i}. {action}")
        
        if self.error_context.recovery_attempted:
            report.append("")
            report.append("RECOVERY INFORMATION:")
            report.append("-" * 40)
            report.append(f"  Recovery Attempted: Yes")
            report.append(f"  Recovery Successful: {self.error_context.recovery_successful}")
        
        if self.error_context.stack_trace:
            report.append("")
            report.append("STACK TRACE:")
            report.append("-" * 40)
            report.append(self.error_context.stack_trace)
        
        report.append("=" * 80)
        
        return "\n".join(report)


class ErrorRecoveryManager:
    """
    Manages error recovery strategies for different types of processing failures.
    
    Implements automatic recovery mechanisms and tracks recovery success rates
    for continuous improvement of error handling.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize error recovery manager.
        
        Args:
            logger: Logger instance for recovery operations
        """
        self.logger = logger
        self.recovery_strategies: Dict[str, callable] = {}
        self.recovery_stats: Dict[str, Dict[str, int]] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default recovery strategies for common error types."""
        self.recovery_strategies.update({
            "file_not_found": self._recover_file_not_found,
            "permission_denied": self._recover_permission_denied,
            "memory_error": self._recover_memory_error,
            "validation_failed": self._recover_validation_failed,
            "transformation_failed": self._recover_transformation_failed,
            "registry_access_error": self._recover_registry_access_error
        })
    
    def register_recovery_strategy(self, error_type: str, strategy_func: callable):
        """
        Register a custom recovery strategy for an error type.
        
        Args:
            error_type: Type of error to handle
            strategy_func: Function that attempts recovery
        """
        self.recovery_strategies[error_type] = strategy_func
        self.logger.info(f"Registered recovery strategy for error type: {error_type}")
    
    def attempt_recovery(self, error: EnhancedProcessingError) -> bool:
        """
        Attempt to recover from a processing error.
        
        Args:
            error: Enhanced processing error to recover from
            
        Returns:
            bool: True if recovery was successful
        """
        error_type = error.error_type
        
        if error_type not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy available for error type: {error_type}")
            return False
        
        self.logger.info(f"Attempting recovery for error type: {error_type}")
        
        try:
            strategy_func = self.recovery_strategies[error_type]
            success = strategy_func(error)
            
            # Update recovery statistics
            self._update_recovery_stats(error_type, success)
            
            # Mark recovery attempt on error
            error.mark_recovery_attempted(success)
            
            if success:
                self.logger.info(f"Recovery successful for error type: {error_type}")
            else:
                self.logger.warning(f"Recovery failed for error type: {error_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Recovery strategy failed with exception: {e}")
            error.mark_recovery_attempted(False)
            return False
    
    def _update_recovery_stats(self, error_type: str, success: bool):
        """Update recovery statistics for monitoring."""
        if error_type not in self.recovery_stats:
            self.recovery_stats[error_type] = {"attempts": 0, "successes": 0}
        
        self.recovery_stats[error_type]["attempts"] += 1
        if success:
            self.recovery_stats[error_type]["successes"] += 1
    
    def get_recovery_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get recovery statistics with success rates."""
        stats = {}
        for error_type, counts in self.recovery_stats.items():
            attempts = counts["attempts"]
            successes = counts["successes"]
            success_rate = (successes / attempts * 100) if attempts > 0 else 0
            
            stats[error_type] = {
                "attempts": attempts,
                "successes": successes,
                "success_rate": f"{success_rate:.1f}%"
            }
        
        return stats
    
    # Default recovery strategies
    def _recover_file_not_found(self, error: EnhancedProcessingError) -> bool:
        """Attempt to recover from file not found errors."""
        # Implementation would check alternative file locations,
        # suggest re-download, etc.
        return False
    
    def _recover_permission_denied(self, error: EnhancedProcessingError) -> bool:
        """Attempt to recover from permission denied errors."""
        # Implementation would check file permissions,
        # suggest permission fixes, etc.
        return False
    
    def _recover_memory_error(self, error: EnhancedProcessingError) -> bool:
        """Attempt to recover from memory errors."""
        # Implementation would suggest chunked processing,
        # memory cleanup, etc.
        return False
    
    def _recover_validation_failed(self, error: EnhancedProcessingError) -> bool:
        """Attempt to recover from validation failures."""
        # Implementation would try alternative validation rules,
        # data cleaning, etc.
        return False
    
    def _recover_transformation_failed(self, error: EnhancedProcessingError) -> bool:
        """Attempt to recover from transformation failures."""
        # Implementation would try alternative transformation methods,
        # partial processing, etc.
        return False
    
    def _recover_registry_access_error(self, error: EnhancedProcessingError) -> bool:
        """Attempt to recover from registry access errors."""
        # Implementation would retry registry access,
        # check network connectivity, etc.
        return False


class ProcessingLogger:
    """
    Enhanced logging system for data preparation framework operations.
    
    Provides structured logging with multiple output formats, performance
    tracking, and audit trail capabilities.
    """
    
    def __init__(
        self,
        name: str,
        log_level: int = logging.INFO,
        log_directory: Optional[Path] = None,
        enable_structured_logging: bool = True
    ):
        """
        Initialize processing logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_directory: Directory for log files
            enable_structured_logging: Enable structured JSON logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Set up log directory
        self.log_directory = log_directory or Path("logs")
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Set up handlers
        self._setup_console_handler()
        if enable_structured_logging:
            self._setup_file_handlers()
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
    
    def _setup_console_handler(self):
        """Set up console logging handler."""
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self):
        """Set up file logging handlers for different log types."""
        # General processing log
        processing_handler = logging.FileHandler(
            self.log_directory / "processing.log"
        )
        processing_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        processing_handler.setFormatter(processing_formatter)
        self.logger.addHandler(processing_handler)
        
        # Error log (errors and above only)
        error_handler = logging.FileHandler(
            self.log_directory / "errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
    
    def log_processing_start(self, dataset_name: str, stage: ProcessingStage, **kwargs):
        """Log the start of a processing stage."""
        message = f"Starting {stage.value} for dataset: {dataset_name}"
        if kwargs:
            message += f" - {kwargs}"
        
        self.logger.info(message)
        
        # Add to audit trail
        self.audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "processing_start",
            "dataset_name": dataset_name,
            "stage": stage.value,
            "details": kwargs
        })
    
    def log_processing_end(
        self,
        dataset_name: str,
        stage: ProcessingStage,
        success: bool,
        duration: float,
        **kwargs
    ):
        """Log the end of a processing stage."""
        status = "completed" if success else "failed"
        message = f"Processing {status} for dataset: {dataset_name}, stage: {stage.value}, duration: {duration:.2f}s"
        
        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)
        
        # Track performance metrics
        stage_key = f"{dataset_name}_{stage.value}"
        if stage_key not in self.performance_metrics:
            self.performance_metrics[stage_key] = []
        self.performance_metrics[stage_key].append(duration)
        
        # Add to audit trail
        self.audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "processing_end",
            "dataset_name": dataset_name,
            "stage": stage.value,
            "success": success,
            "duration": duration,
            "details": kwargs
        })
    
    def log_transformation_step(
        self,
        dataset_name: str,
        step_name: str,
        formula: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        output_shape: Optional[tuple] = None,
        **kwargs
    ):
        """Log a specific transformation step with formula reference."""
        message = f"Transformation step '{step_name}' for dataset: {dataset_name}"
        
        details = {}
        if formula:
            details["formula"] = formula
        if input_shape:
            details["input_shape"] = input_shape
        if output_shape:
            details["output_shape"] = output_shape
        details.update(kwargs)
        
        if details:
            message += f" - {details}"
        
        self.logger.info(message)
        
        # Add to audit trail with detailed transformation info
        self.audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "transformation_step",
            "dataset_name": dataset_name,
            "step_name": step_name,
            "formula": formula,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "details": kwargs
        })
    
    def log_validation_result(
        self,
        dataset_name: str,
        validation_type: str,
        passed: bool,
        details: Dict[str, Any] = None
    ):
        """Log validation results."""
        status = "PASSED" if passed else "FAILED"
        message = f"Validation {status}: {validation_type} for dataset: {dataset_name}"
        
        if details:
            message += f" - {details}"
        
        if passed:
            self.logger.info(message)
        else:
            self.logger.warning(message)
        
        # Add to audit trail
        self.audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "validation_result",
            "dataset_name": dataset_name,
            "validation_type": validation_type,
            "passed": passed,
            "details": details or {}
        })
    
    def log_error(self, error):
        """Log a processing error (either ProcessingError or EnhancedProcessingError)."""
        if isinstance(error, EnhancedProcessingError):
            self.logger.error(f"Processing error: {error.generate_message()}")
            
            # Save detailed error report
            error_file = self.log_directory / f"error_{error.error_context.dataset_name}_{error.error_context.timestamp.replace(':', '-')}.txt"
            
            try:
                with open(error_file, 'w') as f:
                    f.write(error.generate_detailed_report())
            except Exception as e:
                self.logger.warning(f"Could not save detailed error report: {e}")
            
            self.logger.info(f"Detailed error report saved to: {error_file}")
            
            # Add to audit trail
            self.audit_trail.append({
                "timestamp": error.error_context.timestamp,
                "event": "error",
                "dataset_name": error.error_context.dataset_name,
                "stage": error.error_context.stage.value,
                "error_type": error.error_context.error_type,
                "severity": error.error_context.severity.value,
                "error_context": error.error_context.to_dict()
            })
        else:
            # Handle regular ProcessingError
            self.logger.error(f"Processing error: {error}")
            
            # Add basic audit trail entry
            from datetime import datetime, timezone
            self.audit_trail.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "error",
                "dataset_name": error.dataset_name,
                "stage": error.stage,
                "error_type": error.error_type,
                "error_message": error.error_message
            })
    
    def generate_processing_summary(self, dataset_name: str) -> Dict[str, Any]:
        """
        Generate processing summary suitable for publication materials.
        
        Args:
            dataset_name: Name of dataset to summarize
            
        Returns:
            Dictionary containing processing summary
        """
        # Filter audit trail for this dataset
        dataset_events = [
            event for event in self.audit_trail
            if event.get("dataset_name") == dataset_name
        ]
        
        if not dataset_events:
            return {"error": f"No processing events found for dataset: {dataset_name}"}
        
        # Analyze processing events
        start_time = dataset_events[0]["timestamp"]
        end_time = dataset_events[-1]["timestamp"]
        
        # Count events by type
        event_counts = {}
        transformation_steps = []
        validation_results = []
        errors = []
        
        for event in dataset_events:
            event_type = event["event"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            if event_type == "transformation_step":
                transformation_steps.append({
                    "step_name": event["step_name"],
                    "formula": event.get("formula"),
                    "input_shape": event.get("input_shape"),
                    "output_shape": event.get("output_shape")
                })
            elif event_type == "validation_result":
                validation_results.append({
                    "validation_type": event["validation_type"],
                    "passed": event["passed"],
                    "details": event.get("details", {})
                })
            elif event_type == "error":
                errors.append({
                    "stage": event["stage"],
                    "error_type": event["error_type"],
                    "severity": event["severity"]
                })
        
        # Calculate performance metrics
        performance_summary = {}
        for key, durations in self.performance_metrics.items():
            if dataset_name in key:
                stage = key.replace(f"{dataset_name}_", "")
                performance_summary[stage] = {
                    "count": len(durations),
                    "total_time": sum(durations),
                    "average_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations)
                }
        
        return {
            "dataset_name": dataset_name,
            "processing_period": {
                "start_time": start_time,
                "end_time": end_time
            },
            "event_summary": event_counts,
            "transformation_steps": transformation_steps,
            "validation_results": validation_results,
            "errors": errors,
            "performance_metrics": performance_summary,
            "total_events": len(dataset_events)
        }
    
    def save_audit_trail(self, filename: Optional[str] = None):
        """Save complete audit trail to file."""
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"audit_trail_{timestamp}.json"
        
        audit_file = self.log_directory / filename
        
        try:
            with open(audit_file, 'w') as f:
                json.dump(self.audit_trail, f, indent=2)
            
            self.logger.info(f"Audit trail saved to: {audit_file}")
        except Exception as e:
            self.logger.error(f"Failed to save audit trail: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "total_operations": sum(len(durations) for durations in self.performance_metrics.values()),
            "stage_performance": {}
        }
        
        for stage_key, durations in self.performance_metrics.items():
            if durations:
                report["stage_performance"][stage_key] = {
                    "operations": len(durations),
                    "total_time": f"{sum(durations):.2f}s",
                    "average_time": f"{sum(durations) / len(durations):.2f}s",
                    "min_time": f"{min(durations):.2f}s",
                    "max_time": f"{max(durations):.2f}s"
                }
        
        return report