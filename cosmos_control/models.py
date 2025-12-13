"""Data models for the controller/worker architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def timestamp() -> datetime:
    return datetime.utcnow()


class JobStatus(str):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class SliceStatus(str):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class SliceDescriptor:
    slice_id: str
    kind: str
    index: int
    total_slices: int
    dataset_id: Optional[str] = None
    range: Optional[Dict[str, float]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SliceRecord(SliceDescriptor):
    status: SliceStatus = SliceStatus.PENDING
    assigned_node: Optional[str] = None
    progress: float = 0.0
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)

    def to_descriptor(self) -> SliceDescriptor:
        return SliceDescriptor(
            slice_id=self.slice_id,
            kind=self.kind,
            index=self.index,
            total_slices=self.total_slices,
            dataset_id=self.dataset_id,
            range=self.range,
            parameters=dict(self.parameters),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "kind": self.kind,
            "index": self.index,
            "total_slices": self.total_slices,
            "dataset_id": self.dataset_id,
            "range": self.range,
            "parameters": self.parameters,
            "status": self.status,
            "assigned_node": self.assigned_node,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


@dataclass
class JobRecord:
    execution_id: str
    run_id: str
    config: Dict[str, Any]
    slices: Dict[str, SliceRecord]
    created_at: datetime = field(default_factory=timestamp)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: JobStatus = JobStatus.QUEUED
    aggregate_progress: float = 0.0
    report_written: bool = False
    report_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.report_dir = Path("data/science_runs") / self.execution_id / "report"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "status": self.status,
            "aggregate_progress": self.aggregate_progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "report_written": self.report_written,
            "slices": {sid: slice_rec.to_dict() for sid, slice_rec in self.slices.items()},
        }


@dataclass
class WorkerSlotInfo:
    worker_id: str
    total_slots: int
    busy_slots: int = 0
    local_node: bool = False
    dataset_hashes: Dict[str, str] = field(default_factory=dict)
    cores: int = 0
    last_seen: datetime = field(default_factory=timestamp)
    transport_status: Dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    retry_count: int = 0

    @property
    def free_slots(self) -> int:
        return max(0, self.total_slots - self.busy_slots)

    def update_heartbeat(self, *, transport_status: Dict[str, Any] | None = None) -> None:
        self.last_seen = datetime.utcnow()
        if transport_status:
            self.transport_status = dict(transport_status)
            if "retry_count" in transport_status:
                try:
                    self.retry_count = int(transport_status.get("retry_count") or 0)
                except Exception:
                    pass
            if "last_error" in transport_status:
                self.last_error = transport_status.get("last_error")


@dataclass
class JobAssignment:
    execution_id: str
    run_id: str
    config: Dict[str, Any]
    slice: SliceDescriptor


@dataclass
class SliceResult:
    execution_id: str
    slice_id: str
    success: bool
    logs: str
    metrics: Dict[str, Any]
    data: Dict[str, Any]
    progress: float


@dataclass
class DatasetUpdatePayload:
    dataset_id: str
    hash: str
    payload: str
