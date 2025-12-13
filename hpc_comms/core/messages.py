"""Core message protocol and data structures for HPC communication."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class WorkStatus(str, Enum):
    """Status of work execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    """Status of compute nodes."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


class PerformanceMetrics(BaseModel):
    """Performance metrics for work execution."""
    execution_time_ms: float
    backend_used: str
    memory_peak_mb: float
    cpu_utilization: float
    gpu_utilization: Optional[float] = None
    operations_per_second: Optional[float] = None
    cache_hit_rate: Optional[float] = None


class ResourceRequirements(BaseModel):
    """Resource requirements for work execution."""
    min_memory_mb: float
    min_cpu_cores: int
    preferred_backend: Optional[str] = None
    max_execution_time: timedelta
    requires_gpu: bool = False


class NodeCapabilities(BaseModel):
    """Hardware and software capabilities of a node."""
    backend_type: str
    device_count: int = Field(ge=0)
    memory_gb: float = Field(gt=0)
    supported_operations: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = Field(default=1, ge=1)
    performance_profile: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('backend_type')
    @classmethod
    def validate_backend_type(cls, v):
        allowed = {'rocm', 'numba', 'cpu', 'cuda'}
        if v not in allowed:
            raise ValueError(f'backend_type must be one of {allowed}')
        return v


class Message(BaseModel):
    """Base message with routing and metadata."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_node: str
    target_node: Optional[str] = None
    message_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[timedelta] = None
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        return datetime.utcnow() > self.timestamp + self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = self.model_dump(exclude_none=True)
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = data['timestamp'].isoformat()
        if 'ttl' in data and data['ttl']:
            data['ttl'] = data['ttl'].total_seconds()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create message from dictionary."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'ttl' in data and isinstance(data['ttl'], (int, float)):
            data['ttl'] = timedelta(seconds=data['ttl'])
        return cls(**data)


class WorkRequest(Message):
    """Request for compute work."""
    work_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_configuration: Dict[str, Any]
    parameters: List[Dict[str, Any]]
    requirements: ResourceRequirements
    timeout: timedelta = Field(default=timedelta(minutes=30))
    
    def __init__(self, **data):
        data['message_type'] = "WORK_REQUEST"
        super().__init__(**data)


class WorkResponse(Message):
    """Response containing computation results."""
    work_id: str
    results: List[Dict[str, Any]]
    performance_metrics: PerformanceMetrics
    execution_time: timedelta
    status: WorkStatus
    error_message: Optional[str] = None
    
    def __init__(self, **data):
        data['message_type'] = "WORK_RESPONSE"
        super().__init__(**data)


class NodeInfo(BaseModel):
    """Node capabilities and status information."""
    node_id: str
    endpoint: str
    capabilities: NodeCapabilities
    status: NodeStatus = Field(default=NodeStatus.OFFLINE)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def is_healthy(self, timeout: timedelta = timedelta(minutes=2)) -> bool:
        """Check if node is healthy based on heartbeat."""
        return (self.status == NodeStatus.ONLINE and 
                datetime.utcnow() - self.last_heartbeat < timeout)
    
    def can_handle_work(self, requirements: ResourceRequirements) -> bool:
        """Check if node can handle given work requirements."""
        if self.status != NodeStatus.ONLINE:
            return False
        
        # Check memory requirements
        if self.capabilities.memory_gb * 1024 < requirements.min_memory_mb:
            return False
        
        # Check CPU requirements
        if self.capabilities.max_concurrent_tasks < requirements.min_cpu_cores:
            return False
        
        # Check GPU requirements
        if requirements.requires_gpu:
            gpu_backends = {'rocm', 'cuda'}
            if (self.capabilities.backend_type not in gpu_backends or 
                self.capabilities.device_count == 0):
                return False
        
        # Check backend preference
        if (requirements.preferred_backend and 
            requirements.preferred_backend != self.capabilities.backend_type):
            return False
        
        return True


class WorkerHello(Message):
    """Handshake message from worker providing static status."""
    worker_id: str
    cores: int
    datasets: Dict[str, str] = Field(default_factory=dict)
    local_node: bool = False
    MESSAGE_TYPE: ClassVar[str] = "WORKER_HELLO"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class RequestWork(Message):
    """Worker requests slices."""
    worker_id: str
    current_load: int = 0
    MESSAGE_TYPE: ClassVar[str] = "REQUEST_WORK"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class SliceProgress(Message):
    """Update of slice progress."""
    execution_id: str
    slice_id: str
    progress: float
    MESSAGE_TYPE: ClassVar[str] = "SLICE_PROGRESS"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class SliceCompletion(Message):
    """Results from a completed slice."""
    execution_id: str
    slice_id: str
    success: bool
    logs: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    data: Dict[str, Any] = Field(default_factory=dict)
    progress: float = 1.0
    MESSAGE_TYPE: ClassVar[str] = "SLICE_COMPLETION"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class DatasetHashSummary(Message):
    """Worker reports cached dataset hashes."""
    worker_id: str
    datasets: Dict[str, str] = Field(default_factory=dict)
    MESSAGE_TYPE: ClassVar[str] = "DATASET_HASH_SUMMARY"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class WorkerError(Message):
    """Worker reports unexpected error."""
    worker_id: str
    error: str
    context: Dict[str, Any] = Field(default_factory=dict)
    MESSAGE_TYPE: ClassVar[str] = "WORKER_ERROR"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class DatasetUpdate(Message):
    """Controller pushes updated dataset payload."""
    dataset_id: str
    hash: str
    payload: str
    MESSAGE_TYPE: ClassVar[str] = "DATASET_UPDATE"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class JobAssignment(Message):
    """Controller assigns slices to a worker."""
    assignments: List[Dict[str, Any]] = Field(default_factory=list)
    MESSAGE_TYPE: ClassVar[str] = "JOB_ASSIGNMENT"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class CancelSlice(Message):
    """Controller asks worker to cancel a single slice."""
    execution_id: str
    slice_id: str
    MESSAGE_TYPE: ClassVar[str] = "CANCEL_SLICE"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class CancelAll(Message):
    """Controller asks worker to cancel all slices for a job."""
    execution_id: str
    MESSAGE_TYPE: ClassVar[str] = "CANCEL_ALL"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


class NoWork(Message):
    """Controller indicates no available work."""
    reason: str | None = None
    MESSAGE_TYPE: ClassVar[str] = "NO_WORK"

    def __init__(self, **data: Any):
        data.setdefault("message_type", self.MESSAGE_TYPE)
        super().__init__(**data)


# Message type registry for serialization/deserialization
MESSAGE_TYPES = {
    "WORK_REQUEST": WorkRequest,
    "WORK_RESPONSE": WorkResponse,
    "NODE_REGISTER": NodeInfo,
    "NODE_HEARTBEAT": NodeInfo,
    "NODE_DEREGISTER": NodeInfo,
    "HEALTH_CHECK": Message,
    "ERROR_REPORT": Message,
    "WORKER_HELLO": WorkerHello,
    "REQUEST_WORK": RequestWork,
    "SLICE_PROGRESS": SliceProgress,
    "SLICE_COMPLETION": SliceCompletion,
    "DATASET_HASH_SUMMARY": DatasetHashSummary,
    "WORKER_ERROR": WorkerError,
    "DATASET_UPDATE": DatasetUpdate,
    "JOB_ASSIGNMENT": JobAssignment,
    "CANCEL_SLICE": CancelSlice,
    "CANCEL_ALL": CancelAll,
    "NO_WORK": NoWork,
}


def serialize_message(message: Message) -> str:
    """Serialize message to JSON string."""
    data = message.to_dict()
    return json.dumps(data, default=str)


def deserialize_message(data: str) -> Message:
    """Deserialize message from JSON string."""
    parsed = json.loads(data)
    message_type = parsed.get("message_type")
    
    if message_type in MESSAGE_TYPES:
        message_class = MESSAGE_TYPES[message_type]
        return message_class.from_dict(parsed)
    else:
        return Message.from_dict(parsed)


def create_work_request(
    source_node: str,
    target_node: str,
    model_config: Dict[str, Any],
    parameters: List[Dict[str, Any]],
    requirements: ResourceRequirements,
    timeout: Optional[timedelta] = None
) -> WorkRequest:
    """Create a work request message."""
    return WorkRequest(
        source_node=source_node,
        target_node=target_node,
        model_configuration=model_config,
        parameters=parameters,
        requirements=requirements,
        timeout=timeout or timedelta(minutes=30)
    )


def create_work_response(
    source_node: str,
    target_node: str,
    work_id: str,
    results: List[Dict[str, Any]],
    performance_metrics: PerformanceMetrics,
    execution_time: timedelta,
    status: WorkStatus,
    error_message: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> WorkResponse:
    """Create a work response message."""
    return WorkResponse(
        source_node=source_node,
        target_node=target_node,
        work_id=work_id,
        results=results,
        performance_metrics=performance_metrics,
        execution_time=execution_time,
        status=status,
        error_message=error_message,
        correlation_id=correlation_id
    )
