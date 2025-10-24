"""
Structured logging and monitoring for CMB raw parameter processing.

This module provides comprehensive logging, performance monitoring, and diagnostic
capabilities for the CMB derivation pipeline, including structured log output,
performance metrics collection, and troubleshooting support.
"""

import json
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np

from .cmb_config_integration import DataPreparationConfig


@dataclass
class PerformanceMetrics:
    """Performance metrics for CMB processing operations."""
    
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Operation-specific metrics
    parameters_processed: Optional[int] = None
    matrix_size: Optional[int] = None
    numerical_iterations: Optional[int] = None
    convergence_achieved: Optional[bool] = None
    
    # Error information
    errors_encountered: List[str] = field(default_factory=list)
    warnings_generated: List[str] = field(default_factory=list)
    
    def finish(self, end_time: Optional[float] = None):
        """Mark operation as finished and calculate duration."""
        if end_time is None:
            end_time = time.time()
        
        self.end_time = end_time
        self.duration_seconds = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "parameters_processed": self.parameters_processed,
            "matrix_size": self.matrix_size,
            "numerical_iterations": self.numerical_iterations,
            "convergence_achieved": self.convergence_achieved,
            "errors_count": len(self.errors_encountered),
            "warnings_count": len(self.warnings_generated)
        }


class CMBStructuredLogger:
    """Structured logger for CMB processing with performance monitoring."""
    
    def __init__(self, config: Optional[DataPreparationConfig] = None):
        """
        Initialize structured logger.
        
        Args:
            config: Data preparation configuration
        """
        if config is None:
            from .cmb_config_integration import get_data_preparation_config
            config = get_data_preparation_config()
        
        self.config = config
        self.logger = self._setup_logger()
        self.performance_monitoring = config.performance_monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Try to import optional performance monitoring dependencies
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            self.psutil = None
            self.process = None
    
    def _setup_logger(self) -> logging.Logger:
        """Set up structured logger with appropriate handlers."""
        logger = logging.getLogger("cmb_processing")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter for structured logging
        if self.config.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_processing_start(self, dataset_name: str, registry_entry: Dict[str, Any]):
        """Log the start of CMB dataset processing."""
        self.logger.info(
            "CMB processing started",
            extra={
                "event_type": "processing_start",
                "dataset_name": dataset_name,
                "dataset_type": registry_entry.get("metadata", {}).get("dataset_type"),
                "source": registry_entry.get("metadata", {}).get("source"),
                "config": {
                    "use_raw_parameters": self.config.cmb.use_raw_parameters,
                    "z_recombination": self.config.cmb.z_recombination,
                    "fallback_to_legacy": self.config.cmb.fallback_to_legacy
                }
            }
        )
    
    def log_processing_complete(self, dataset_name: str, output_info: Dict[str, Any]):
        """Log successful completion of CMB processing."""
        self.logger.info(
            "CMB processing completed successfully",
            extra={
                "event_type": "processing_complete",
                "dataset_name": dataset_name,
                "output_info": output_info,
                "performance_summary": self._get_performance_summary()
            }
        )
    
    def log_processing_error(self, dataset_name: str, error: Exception, context: Dict[str, Any]):
        """Log processing error with detailed context."""
        self.logger.error(
            f"CMB processing failed: {str(error)}",
            extra={
                "event_type": "processing_error",
                "dataset_name": dataset_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "traceback": traceback.format_exc()
            }
        )
    
    def log_parameter_detection(self, detection_result: Dict[str, Any]):
        """Log parameter detection results."""
        self.logger.info(
            "Parameter detection completed",
            extra={
                "event_type": "parameter_detection",
                "detection_result": detection_result
            }
        )
    
    def log_parameter_validation(self, validation_result: Dict[str, Any]):
        """Log parameter validation results."""
        if validation_result.get("valid", False):
            self.logger.info(
                "Parameter validation passed",
                extra={
                    "event_type": "parameter_validation",
                    "validation_result": validation_result
                }
            )
        else:
            self.logger.warning(
                "Parameter validation issues found",
                extra={
                    "event_type": "parameter_validation",
                    "validation_result": validation_result
                }
            )
    
    def log_distance_derivation(self, parameters: Dict[str, Any], derived_priors: Dict[str, Any]):
        """Log distance prior derivation results."""
        self.logger.info(
            "Distance priors derived from parameters",
            extra={
                "event_type": "distance_derivation",
                "input_parameters": parameters,
                "derived_priors": derived_priors
            }
        )
    
    def log_covariance_propagation(self, jacobian_info: Dict[str, Any], covariance_info: Dict[str, Any]):
        """Log covariance matrix propagation results."""
        self.logger.info(
            "Covariance matrix propagated",
            extra={
                "event_type": "covariance_propagation",
                "jacobian_info": jacobian_info,
                "covariance_info": covariance_info
            }
        )
    
    def log_fallback_to_legacy(self, reason: str, context: Dict[str, Any]):
        """Log fallback to legacy processing mode."""
        self.logger.warning(
            f"Falling back to legacy processing: {reason}",
            extra={
                "event_type": "fallback_to_legacy",
                "reason": reason,
                "context": context
            }
        )
    
    def log_performance_warning(self, operation: str, duration: float, threshold: float):
        """Log performance warning for slow operations."""
        self.logger.warning(
            f"Performance warning: {operation} took {duration:.2f}s (threshold: {threshold:.2f}s)",
            extra={
                "event_type": "performance_warning",
                "operation": operation,
                "duration_seconds": duration,
                "threshold_seconds": threshold
            }
        )
    
    def log_numerical_instability(self, operation: str, details: Dict[str, Any]):
        """Log numerical instability issues."""
        self.logger.error(
            f"Numerical instability detected in {operation}",
            extra={
                "event_type": "numerical_instability",
                "operation": operation,
                "details": details
            }
        )
    
    @contextmanager
    def performance_monitor(self, operation_name: str, **kwargs):
        """
        Context manager for monitoring operation performance.
        
        Args:
            operation_name: Name of the operation being monitored
            **kwargs: Additional operation-specific parameters
        """
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time()
        )
        
        # Set operation-specific parameters
        for key, value in kwargs.items():
            if hasattr(metrics, key):
                setattr(metrics, key, value)
        
        # Get initial memory usage if monitoring enabled
        if self.performance_monitoring and self.process:
            try:
                initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                initial_memory = None
        else:
            initial_memory = None
        
        try:
            yield metrics
            
        except Exception as e:
            metrics.errors_encountered.append(str(e))
            raise
            
        finally:
            # Finish timing
            metrics.finish()
            
            # Get final memory usage if monitoring enabled
            if self.performance_monitoring and self.process:
                try:
                    final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                    if initial_memory is not None:
                        metrics.memory_usage_mb = final_memory - initial_memory
                    
                    # Get CPU usage (approximate)
                    metrics.cpu_usage_percent = self.process.cpu_percent()
                except Exception:
                    pass
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Log performance metrics if monitoring enabled
            if self.performance_monitoring:
                self.logger.debug(
                    f"Performance metrics for {operation_name}",
                    extra={
                        "event_type": "performance_metrics",
                        "metrics": metrics.to_dict()
                    }
                )
                
                # Check for performance warnings
                if metrics.duration_seconds and metrics.duration_seconds > 30.0:  # 30 second threshold
                    self.log_performance_warning(operation_name, metrics.duration_seconds, 30.0)
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics_history:
            return {}
        
        total_duration = sum(m.duration_seconds or 0 for m in self.metrics_history)
        total_errors = sum(len(m.errors_encountered) for m in self.metrics_history)
        total_warnings = sum(len(m.warnings_generated) for m in self.metrics_history)
        
        return {
            "total_operations": len(self.metrics_history),
            "total_duration_seconds": total_duration,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "operations": [m.operation_name for m in self.metrics_history]
        }
    
    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        return {
            "configuration": {
                "log_level": self.config.log_level,
                "structured_logging": self.config.structured_logging,
                "performance_monitoring": self.config.performance_monitoring,
                "cmb_config": self.config.cmb.to_dict()
            },
            "performance_summary": self._get_performance_summary(),
            "detailed_metrics": [m.to_dict() for m in self.metrics_history],
            "system_info": self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for diagnostics."""
        info = {
            "python_version": None,
            "numpy_version": None,
            "memory_available": None,
            "cpu_count": None
        }
        
        try:
            import sys
            info["python_version"] = sys.version
        except Exception:
            pass
        
        try:
            info["numpy_version"] = np.__version__
        except Exception:
            pass
        
        if self.psutil:
            try:
                info["memory_available"] = self.psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
                info["cpu_count"] = self.psutil.cpu_count()
            except Exception:
                pass
        
        return info
    
    def save_diagnostic_report(self, output_file: str):
        """Save diagnostic report to file."""
        report = self.get_diagnostic_report()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'event_type'):
            log_entry["event_type"] = record.event_type
        
        # Add all extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'message', 'exc_info', 'exc_text',
                          'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


# Global logger instance
_global_logger: Optional[CMBStructuredLogger] = None


def get_cmb_logger(config: Optional[DataPreparationConfig] = None) -> CMBStructuredLogger:
    """
    Get global CMB structured logger instance.
    
    Args:
        config: Optional configuration (uses default if None)
        
    Returns:
        CMBStructuredLogger instance
    """
    global _global_logger
    
    if _global_logger is None or config is not None:
        _global_logger = CMBStructuredLogger(config)
    
    return _global_logger


def log_cmb_processing_step(step_name: str, **kwargs):
    """
    Decorator for logging CMB processing steps.
    
    Args:
        step_name: Name of the processing step
        **kwargs: Additional logging parameters
    """
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger = get_cmb_logger()
            
            with logger.performance_monitor(step_name, **kwargs) as metrics:
                logger.logger.debug(
                    f"Starting {step_name}",
                    extra={"event_type": "step_start", "step_name": step_name}
                )
                
                try:
                    result = func(*args, **func_kwargs)
                    
                    logger.logger.debug(
                        f"Completed {step_name}",
                        extra={"event_type": "step_complete", "step_name": step_name}
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.logger.error(
                        f"Error in {step_name}: {str(e)}",
                        extra={
                            "event_type": "step_error",
                            "step_name": step_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    )
                    raise
        
        return wrapper
    return decorator


def create_troubleshooting_guide(error_log_file: str, output_file: str):
    """
    Create troubleshooting guide from error logs.
    
    Args:
        error_log_file: Path to error log file
        output_file: Path to output troubleshooting guide
    """
    common_issues = {
        "ParameterDetectionError": {
            "description": "Raw parameters could not be detected or parsed",
            "common_causes": [
                "Parameter file format not supported",
                "Parameter names don't match expected conventions",
                "File corruption or access issues"
            ],
            "solutions": [
                "Check parameter file format (CSV, JSON, NumPy)",
                "Verify parameter names match expected conventions",
                "Enable fallback to legacy mode",
                "Check file permissions and integrity"
            ]
        },
        "ParameterValidationError": {
            "description": "Parameters failed validation checks",
            "common_causes": [
                "Parameter values outside physical bounds",
                "NaN or infinite values in parameters",
                "Inconsistent parameter relationships"
            ],
            "solutions": [
                "Check parameter values against physical bounds",
                "Verify data quality and completeness",
                "Review parameter extraction process",
                "Check for data corruption"
            ]
        },
        "DerivationError": {
            "description": "Distance prior computation failed",
            "common_causes": [
                "Numerical integration issues",
                "Invalid cosmological parameters",
                "Background integrator failures"
            ],
            "solutions": [
                "Check parameter values for physical consistency",
                "Verify background integrator dependencies",
                "Adjust numerical integration settings",
                "Enable diagnostic logging for details"
            ]
        },
        "CovarianceError": {
            "description": "Covariance matrix propagation failed",
            "common_causes": [
                "Non-positive-definite covariance matrix",
                "Jacobian computation issues",
                "Matrix dimension mismatches"
            ],
            "solutions": [
                "Validate input covariance matrix properties",
                "Check Jacobian computation settings",
                "Verify parameter-observable mapping",
                "Consider using diagonal uncertainties as fallback"
            ]
        }
    }
    
    guide = {
        "troubleshooting_guide": {
            "overview": "This guide helps diagnose and resolve common CMB processing issues",
            "common_issues": common_issues,
            "diagnostic_steps": [
                "Check configuration validation",
                "Review structured logs for error patterns",
                "Verify input data quality",
                "Test with simplified configuration",
                "Enable detailed performance monitoring"
            ],
            "configuration_tips": [
                "Use fallback_to_legacy=true for compatibility",
                "Enable performance_monitoring for diagnostics",
                "Set appropriate log_level (DEBUG for troubleshooting)",
                "Validate configuration before deployment"
            ]
        }
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(guide, f, indent=2)