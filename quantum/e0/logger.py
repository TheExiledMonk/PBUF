"""
Logging module for rigidity analysis.

This module provides a centralized logging system with multiple severity levels
(INFO, WARN, ERROR, DEBUG) and writes to separate log files with timestamps.
Thread-safe logging is supported for parallel processing.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import traceback
import threading


# Global debug mode flag
_DEBUG_MODE = False

# Thread lock for thread-safe logging
_LOG_LOCK = threading.Lock()


def set_debug_mode(debug: bool) -> None:
    """
    Set global debug mode flag.
    
    Parameters
    ----------
    debug : bool
        Enable or disable debug mode
    
    Requirements: 11.4, 11.5
    """
    global _DEBUG_MODE
    _DEBUG_MODE = debug


def _ensure_logs_directory() -> Path:
    """
    Create logs directory if it doesn't exist.
    
    Returns
    -------
    Path
        Path to logs directory
    
    Requirements: 6.6
    """
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def _format_log_entry(level: str, message: str) -> str:
    """
    Format log entry with timestamp and level.
    
    Parameters
    ----------
    level : str
        Log level (INFO, WARN, ERROR, DEBUG)
    message : str
        Log message
    
    Returns
    -------
    str
        Formatted log entry with timestamp
    
    Requirements: 6.10, 13.1
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"[{timestamp}] {level}: {message}\n"


def log_info(message: str) -> None:
    """
    Log info message to logs/run.log with timestamp.
    
    Info messages track normal operation progress and milestones.
    Thread-safe for parallel processing.
    
    Parameters
    ----------
    message : str
        Info message to log
    
    Requirements: 13.1, 8.5
    """
    logs_dir = _ensure_logs_directory()
    log_entry = _format_log_entry('INFO', message)
    
    log_file = logs_dir / 'run.log'
    try:
        with _LOG_LOCK:
            with open(log_file, 'a') as f:
                f.write(log_entry)
    except Exception:
        # Silently fail if logging fails
        pass


def log_warn(message: str) -> None:
    """
    Log warning message to logs/run.log with timestamp.
    
    Warning messages indicate recoverable issues that don't prevent execution.
    Thread-safe for parallel processing.
    
    Parameters
    ----------
    message : str
        Warning message to log
    
    Requirements: 13.1, 8.5
    """
    logs_dir = _ensure_logs_directory()
    log_entry = _format_log_entry('WARN', message)
    
    log_file = logs_dir / 'run.log'
    try:
        with _LOG_LOCK:
            with open(log_file, 'a') as f:
                f.write(log_entry)
    except Exception:
        pass


def log_error(message: str, exception: Optional[Exception] = None) -> None:
    """
    Log error message to logs/errors.txt with timestamp.
    
    Error messages indicate failures that prevent processing specific events
    or operations but allow the program to continue.
    Thread-safe for parallel processing.
    
    Parameters
    ----------
    message : str
        Error message to log
    exception : Exception or None, optional
        Exception object to include traceback in debug mode
    
    Requirements: 6.9, 11.2, 11.4, 11.5, 8.5
    """
    logs_dir = _ensure_logs_directory()
    log_entry = _format_log_entry('ERROR', message)
    
    # Add traceback if in debug mode and exception provided
    if _DEBUG_MODE and exception is not None:
        tb_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        log_entry += f"Traceback:\n{tb_str}\n"
    
    log_file = logs_dir / 'errors.txt'
    try:
        with _LOG_LOCK:
            with open(log_file, 'a') as f:
                f.write(log_entry)
    except Exception:
        pass


def log_debug(message: str) -> None:
    """
    Log debug message to logs/run.log with timestamp.
    
    Debug messages provide detailed internal information for troubleshooting.
    Only written when debug mode is enabled.
    Thread-safe for parallel processing.
    
    Parameters
    ----------
    message : str
        Debug message to log
    
    Requirements: 11.5, 8.5
    """
    if not _DEBUG_MODE:
        return
    
    logs_dir = _ensure_logs_directory()
    log_entry = _format_log_entry('DEBUG', message)
    
    log_file = logs_dir / 'run.log'
    try:
        with _LOG_LOCK:
            with open(log_file, 'a') as f:
                f.write(log_entry)
    except Exception:
        pass


def log_runtime_summary(n_events: int, runtime: float) -> None:
    """
    Log runtime summary with total events processed and execution time.
    
    Parameters
    ----------
    n_events : int
        Number of events processed
    runtime : float
        Total runtime in seconds
    
    Requirements: 13.2
    """
    message = f"Analysis complete. Runtime: {runtime:.1f}s, Events processed: {n_events}"
    log_info(message)
