"""Error handling and exceptions for HPC communication."""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Any, Optional


class HPCError(Exception):
    """Base HPC communication error."""
    
    def __init__(
        self, 
        error_code: str, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        retry_after: Optional[timedelta] = None
    ):
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.retry_after = retry_after
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "retry_after": self.retry_after.total_seconds() if self.retry_after else None
        }


class ConnectionError(HPCError):
    """Network connection failure."""
    
    def __init__(
        self, 
        message: str, 
        endpoint: Optional[str] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {"endpoint": endpoint} if endpoint else {}
        super().__init__("CONNECTION_ERROR", message, details, retry_after)


class AuthenticationError(HPCError):
    """Authentication/authorization failure."""
    
    def __init__(
        self, 
        message: str, 
        node_id: Optional[str] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {"node_id": node_id} if node_id else {}
        super().__init__("AUTHENTICATION_ERROR", message, details, retry_after)


class SerializationError(HPCError):
    """Message serialization/deserialization error."""
    
    def __init__(
        self, 
        message: str, 
        message_type: Optional[str] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {"message_type": message_type} if message_type else {}
        super().__init__("SERIALIZATION_ERROR", message, details, retry_after)


class TimeoutError(HPCError):
    """Operation timeout."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        super().__init__("TIMEOUT_ERROR", message, details, retry_after)


class ResourceError(HPCError):
    """Insufficient resources."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        required: Optional[Any] = None,
        available: Optional[Any] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if required is not None:
            details["required"] = required
        if available is not None:
            details["available"] = available
        super().__init__("RESOURCE_ERROR", message, details, retry_after)


class ValidationError(HPCError):
    """Message validation error."""
    
    def __init__(
        self, 
        message: str, 
        field: Optional[str] = None,
        value: Optional[Any] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__("VALIDATION_ERROR", message, details, retry_after)


class NodeError(HPCError):
    """Node-related errors."""
    
    def __init__(
        self, 
        message: str, 
        node_id: Optional[str] = None,
        node_status: Optional[str] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {}
        if node_id:
            details["node_id"] = node_id
        if node_status:
            details["node_status"] = node_status
        super().__init__("NODE_ERROR", message, details, retry_after)


class WorkError(HPCError):
    """Work execution errors."""
    
    def __init__(
        self, 
        message: str, 
        work_id: Optional[str] = None,
        node_id: Optional[str] = None,
        retry_after: Optional[timedelta] = None
    ):
        details = {}
        if work_id:
            details["work_id"] = work_id
        if node_id:
            details["node_id"] = node_id
        super().__init__("WORK_ERROR", message, details, retry_after)


# Error mapping for HTTP status codes
HTTP_ERROR_MAPPING = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: ConnectionError,
    408: TimeoutError,
    409: ResourceError,
    429: ResourceError,
    500: HPCError,
    502: ConnectionError,
    503: ResourceError,
    504: TimeoutError,
}


def error_from_http_status(
    status_code: int, 
    message: str, 
    details: Optional[Dict[str, Any]] = None
) -> HPCError:
    """Create appropriate error from HTTP status code."""
    error_class = HTTP_ERROR_MAPPING.get(status_code, HPCError)
    return error_class(
        error_code=f"HTTP_{status_code}",
        message=message,
        details=details
    )


def is_retryable_error(error: HPCError) -> bool:
    """Check if error is retryable."""
    retryable_codes = {
        "CONNECTION_ERROR",
        "TIMEOUT_ERROR", 
        "RESOURCE_ERROR"
    }
    return error.error_code in retryable_codes or error.retry_after is not None
