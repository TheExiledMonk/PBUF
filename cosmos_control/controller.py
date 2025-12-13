"""Core controller responsible for job orchestration, slicing, and aggregation."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cosmos2.science_runner.config import ScienceRunConfig

from .dataset import DatasetManager
from .models import (
    DatasetUpdatePayload,
    JobAssignment,
    JobRecord,
    JobStatus,
    SliceDescriptor,
    SliceRecord,
    SliceResult,
    SliceStatus,
    WorkerSlotInfo,
)


logger = logging.getLogger("cosmos_control.controller")
STALE_THRESHOLD = timedelta(seconds=90)


class Controller:
    """Controller handling job intake, slicing, assignment, aggregation, and reporting."""

    def __init__(self, base_dir: Path | str = Path("data/science_runs")) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: Dict[str, JobRecord] = {}
        self.pending_slices: deque[str] = deque()
        self.slice_lookup: Dict[str, str] = {}
        self.workers: Dict[str, WorkerSlotInfo] = {}
        self.dataset_manager = DatasetManager()
        self.started_at = datetime.utcnow()
        self.log_history: deque[str] = deque(maxlen=20)

    def submit_job(
        self,
        *,
        config_path: Path | str | None = None,
        config_payload: Dict[str, Any] | None = None,
        slice_count: int | None = None,
        dataset_id: str | None = None,
        execution_id: str | None = None,
    ) -> JobRecord:
        """Validate and register a new job."""
        config = self._load_config(config_path=config_path, payload=config_payload)
        run_id = str(config.run_name)
        execution_id = execution_id or uuid.uuid4().hex
        job = JobRecord(
            execution_id=execution_id,
            run_id=run_id,
            config=config.raw,
            slices={},
        )

        job_dir = self.base_dir / execution_id
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "slices").mkdir(exist_ok=True)
        (job_dir / "report").mkdir(exist_ok=True)

        total_slices = self._determine_slice_count(config.raw, override=slice_count)
        self._create_slices(job, total_slices=total_slices, dataset_id=dataset_id or config.raw.get("dataset_id"))
        self.jobs[execution_id] = job
        self._write_job_config(job_dir, config.raw)
        self._persist_state(job)
        self._log(job, f"Job submitted with {total_slices} slices.")
        return job

    def _load_config(
        self,
        *,
        config_path: Path | str | None = None,
        payload: Dict[str, Any] | None = None,
    ) -> ScienceRunConfig:
        if config_path:
            return ScienceRunConfig.from_path(config_path)
        if payload is not None:
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
                json.dump(payload, tmp)
                tmp.flush()
                temp_path = tmp.name
            try:
                return ScienceRunConfig.from_path(temp_path)
            finally:
                os.remove(temp_path)
        raise ValueError("Either config_path or config_payload must be provided.")

    def _write_job_config(self, job_dir: Path, config: Dict[str, Any]) -> None:
        path = job_dir / "config.json"
        path.write_text(json.dumps(config, indent=2, default=str))

    def _create_slices(self, job: JobRecord, *, total_slices: int, dataset_id: str | None = None) -> None:
        total_slices = max(1, total_slices)
        kind = job.config.get("run_type", "science")
        for idx in range(total_slices):
            slice_id = f"{job.execution_id}_slice_{idx + 1:03d}"
            descriptor = SliceDescriptor(
                slice_id=slice_id,
                kind=kind,
                index=idx + 1,
                total_slices=total_slices,
                dataset_id=dataset_id,
                range={"index": idx + 1},
                parameters={"chunk_index": idx, "total_chunks": total_slices},
            )
            record = SliceRecord(**descriptor.__dict__)
            job.slices[slice_id] = record
            self.slice_lookup[slice_id] = job.execution_id
            self.pending_slices.append(slice_id)
            slice_dir = self.base_dir / job.execution_id / "slices" / slice_id
            slice_dir.mkdir(parents=True, exist_ok=True)

    def _determine_slice_count(self, config: Dict[str, Any], override: int | None = None) -> int:
        if override and override >= 1:
            return override
        candidate = config.get("slice_count")
        if isinstance(candidate, int) and candidate >= 1:
            return candidate
        engine_workers = config.get("engine_settings", {}).get("workers")
        if isinstance(engine_workers, int) and engine_workers >= 1:
            return engine_workers
        return 1

    def register_worker(
        self,
        worker_id: str,
        cores: int,
        datasets: Dict[str, str],
        local_node: bool = False,
    ) -> Tuple[WorkerSlotInfo, List[DatasetUpdatePayload]]:
        """Register a worker and return its slot summary + dataset updates."""
        ratio = 0.5 if local_node else 0.8
        total_slots = max(1, math.floor(cores * ratio))
        slot_info = WorkerSlotInfo(
            worker_id=worker_id,
            total_slots=total_slots,
            local_node=local_node,
            dataset_hashes=dict(datasets),
            cores=cores,
        )
        slot_info.update_heartbeat()
        self.workers[worker_id] = slot_info
        self._log_general(f"Registered worker {worker_id} with {total_slots} slots (local={local_node}).")
        updates = self.collect_dataset_updates(datasets)
        return slot_info, updates

    def touch_worker(
        self,
        worker_id: str,
        *,
        transport_status: Dict[str, Any] | None = None,
    ) -> None:
        """Update worker heartbeat and transport metadata."""
        slot = self.workers.get(worker_id)
        if not slot:
            return
        slot.update_heartbeat(transport_status=transport_status)

    def collect_dataset_updates(
        self,
        datasets: Dict[str, str],
    ) -> List[DatasetUpdatePayload]:
        updates: List[DatasetUpdatePayload] = []
        for dataset_id, worker_hash in datasets.items():
            if self.dataset_manager.needs_update(dataset_id, worker_hash):
                canonical = self.dataset_manager.get_hash(dataset_id)
                payload = self.dataset_manager.encode_payload(dataset_id)
                if canonical and payload:
                    updates.append(DatasetUpdatePayload(dataset_id=dataset_id, hash=canonical, payload=payload))
        return updates

    def handle_dataset_summary(
        self,
        worker_id: str,
        datasets: Dict[str, str],
    ) -> List[DatasetUpdatePayload]:
        slot = self.workers.get(worker_id)
        if slot:
            slot.update_heartbeat()
            slot.dataset_hashes = dict(datasets)
        updates = self.collect_dataset_updates(datasets)
        return updates

    def assign_slices(self, worker_id: str, current_load: int = 0) -> List[JobAssignment]:
        self._enforce_worker_health()
        worker = self.workers.get(worker_id)
        if not worker:
            return []

        running = self._count_worker_running(worker_id)
        worker.busy_slots = min(worker.total_slots, max(running, current_load))
        available_slots = worker.total_slots - worker.busy_slots
        assignments: List[JobAssignment] = []
        updated_jobs: set[str] = set()

        while available_slots > 0 and self.pending_slices:
            slice_id = self.pending_slices.popleft()
            job = self.jobs[self.slice_lookup[slice_id]]
            slice_record = job.slices[slice_id]
            slice_record.status = SliceStatus.ASSIGNED
            slice_record.assigned_node = worker_id
            slice_record.started_at = slice_record.started_at or datetime.utcnow()
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.RUNNING
                job.started_at = job.started_at or datetime.utcnow()
            assignments.append(
                JobAssignment(
                    execution_id=job.execution_id,
                    run_id=job.run_id,
                    config=job.config,
                    slice=slice_record.to_descriptor(),
                )
            )
            worker.busy_slots += 1
            available_slots -= 1
            updated_jobs.add(job.execution_id)

        for execution_id in updated_jobs:
            self._persist_state(self.jobs[execution_id])
        return assignments

    def _count_worker_running(self, worker_id: str) -> int:
        count = 0
        for job in self.jobs.values():
            for slice_record in job.slices.values():
                if (
                    slice_record.assigned_node == worker_id
                    and slice_record.status in {SliceStatus.ASSIGNED, SliceStatus.RUNNING}
                ):
                    count += 1
        return count

    def process_slice_result(self, worker_id: str, result: SliceResult) -> None:
        job = self.jobs.get(result.execution_id)
        if not job:
            return
        slice_record = job.slices.get(result.slice_id)
        if not slice_record:
            return

        slice_record.progress = max(0.0, min(1.0, result.progress))
        slice_record.metrics = result.metrics
        slice_record.result = result.data
        slice_record.logs.append(result.logs)
        slice_record.status = SliceStatus.COMPLETED if result.success else SliceStatus.FAILED
        slice_record.ended_at = slice_record.ended_at or datetime.utcnow()
        slice_record.assigned_node = worker_id
        self._write_slice_artifacts(job.execution_id, slice_record)

        slot = self.workers.get(worker_id)
        if slot:
            slot.busy_slots = max(0, slot.busy_slots - 1)

        self._update_job_state(job)
        self._persist_state(job)
        if job.status == JobStatus.COMPLETED and not job.report_written:
            self._write_report(job)

    def _write_slice_artifacts(self, execution_id: str, slice_record: SliceRecord) -> None:
        slice_dir = self.base_dir / execution_id / "slices" / slice_record.slice_id
        slice_dir.mkdir(parents=True, exist_ok=True)
        (slice_dir / "logs.txt").write_text("\n".join(slice_record.logs) + "\n")
        (slice_dir / "metrics.json").write_text(json.dumps(slice_record.metrics, indent=2, default=str))
        (slice_dir / "result.json").write_text(json.dumps(slice_record.result, indent=2, default=str))

    def _update_job_state(self, job: JobRecord) -> None:
        progresses = [slice_record.progress for slice_record in job.slices.values()]
        job.aggregate_progress = sum(progresses) / len(progresses) if progresses else 0.0
        statuses = {slice_record.status for slice_record in job.slices.values()}
        now = datetime.utcnow()

        if job.status != JobStatus.CANCELED:
            if SliceStatus.FAILED in statuses:
                job.status = JobStatus.FAILED
                job.ended_at = job.ended_at or now
            elif all(
                slice_record.status == SliceStatus.COMPLETED
                for slice_record in job.slices.values()
            ):
                job.status = JobStatus.COMPLETED
                job.ended_at = job.ended_at or now
            elif any(
                slice_record.status in {SliceStatus.ASSIGNED, SliceStatus.RUNNING}
                for slice_record in job.slices.values()
            ):
                job.status = JobStatus.RUNNING
                job.started_at = job.started_at or now

    def _persist_state(self, job: JobRecord) -> None:
        state_path = self.base_dir / job.execution_id / "state.json"
        state_path.write_text(json.dumps(job.to_dict(), indent=2, default=str))

    def _log(self, job: JobRecord, message: str) -> None:
        self._log_general(f"[{job.execution_id}] {message}")
        log_path = self.base_dir / job.execution_id / "logs.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_history.append(f"{datetime.utcnow().isoformat()} {message}")
        with log_path.open("a") as stream:
            stream.write(f"{datetime.utcnow().isoformat()} {message}\n")

    def _log_general(self, message: str) -> None:
        logger.info(message)

    def _write_report(self, job: JobRecord) -> None:
        report_dir = self.base_dir / job.execution_id / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "execution_id": job.execution_id,
            "run_id": job.run_id,
            "status": job.status,
            "progress": job.aggregate_progress,
            "completed_at": job.ended_at.isoformat() if job.ended_at else None,
        }
        (report_dir / "report.json").write_text(json.dumps(summary, indent=2))
        html = f"""<html><body><h1>Job {job.execution_id}</h1><p>Status: {job.status}</p></body></html>"""
        (report_dir / "index.html").write_text(html)
        job.report_written = True

    def cancel_job(self, execution_id: str) -> None:
        job = self.jobs.get(execution_id)
        if not job:
            return
        job.status = JobStatus.CANCELED
        job.ended_at = job.ended_at or datetime.utcnow()
        for slice_record in job.slices.values():
            if slice_record.status in {SliceStatus.PENDING, SliceStatus.ASSIGNED, SliceStatus.RUNNING}:
                slice_record.status = SliceStatus.CANCELED
        self._persist_state(job)

    def report_slice_progress(self, execution_id: str, slice_id: str, progress: float) -> None:
        job = self.jobs.get(execution_id)
        if not job:
            return
        slice_record = job.slices.get(slice_id)
        if not slice_record or slice_record.status not in {SliceStatus.ASSIGNED, SliceStatus.RUNNING, SliceStatus.PENDING}:
            return
        slice_record.progress = max(0.0, min(1.0, progress))
        self._update_job_state(job)
        self._persist_state(job)

    def system_status(self) -> Dict[str, Any]:
        now = datetime.utcnow()
        self._enforce_worker_health(now=now)
        job_counts = {
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "canceled": 0,
        }
        for job in self.jobs.values():
            if job.status == JobStatus.QUEUED:
                job_counts["queued"] += 1
            elif job.status == JobStatus.RUNNING:
                job_counts["running"] += 1
            elif job.status == JobStatus.COMPLETED:
                job_counts["completed"] += 1
            elif job.status == JobStatus.FAILED:
                job_counts["failed"] += 1
            elif job.status == JobStatus.CANCELED:
                job_counts["canceled"] += 1

        workers_info = []
        worker_alerts: list[Dict[str, Any]] = []
        total_cores = 0
        total_slots = 0
        active_counts = self._worker_active_counts()
        for worker_id, slot in self.workers.items():
            total_cores += slot.cores
            total_slots += slot.total_slots
            is_stale = now - slot.last_seen > STALE_THRESHOLD
            active = active_counts.get(worker_id, 0)
            state = "stale" if is_stale and active == 0 else "connected"
            worker_payload = {
                "worker_id": worker_id,
                "cores": slot.cores,
                "allocated_slots": slot.total_slots,
                "active_slices": active,
                "state": state,
                "datasets": dict(slot.dataset_hashes),
                "last_seen": slot.last_seen.isoformat(),
                "retry_count": slot.retry_count,
                "last_error": slot.last_error,
                "transport_status": dict(slot.transport_status),
            }
            workers_info.append(worker_payload)
            if slot.last_error:
                worker_alerts.append(
                    {
                        "worker_id": worker_id,
                        "severity": "warning",
                        "message": f"Transport error: {slot.last_error}",
                        "timestamp": slot.last_seen.isoformat(),
                    }
                )
            if is_stale and active == 0:
                delta = now - slot.last_seen
                worker_alerts.append(
                    {
                        "worker_id": worker_id,
                        "severity": "critical",
                        "message": f"No heartbeat for {int(delta.total_seconds())}s",
                        "timestamp": slot.last_seen.isoformat(),
                    }
                )

        recent_jobs = []
        if self.jobs:
            sorted_jobs = sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)
            for job in sorted_jobs[:8]:
                metadata_raw = job.config.get("metadata") if isinstance(job.config, dict) else {}
                metadata = metadata_raw or {}
                created_at = job.created_at.isoformat() if job.created_at else None
                recent_jobs.append(
                    {
                        "execution_id": job.execution_id,
                        "run_id": job.run_id,
                        "status": job.status,
                        "created_at": created_at,
                        "package_type": metadata.get("package_type"),
                        "package_id": metadata.get("package_id"),
                        "metadata": metadata,
                    }
                )

        return {
            "controller_ready": True,
            "controller_uptime": self._format_duration(now - self.started_at),
            "last_updated": now.isoformat(),
            "jobs": {
                "queued": job_counts["queued"],
                "running": job_counts["running"],
                "completed": job_counts["completed"],
                "failed": job_counts["failed"],
                "canceled": job_counts["canceled"],
            },
            "worker_summary": {
                "total_workers": len(self.workers),
                "total_cores": total_cores,
                "allocated_slots": total_slots,
            },
            "workers": workers_info,
            "worker_alerts": worker_alerts,
            "slices_active": self._count_active_slices(),
            "last_logs": list(self.log_history),
            "recent_jobs": recent_jobs,
        }

    def _format_duration(self, delta: timedelta) -> str:
        seconds = int(delta.total_seconds())
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h{minutes}m{seconds}s"

    def _count_active_slices(self) -> int:
        return sum(
            1
            for job in self.jobs.values()
            for slice_record in job.slices.values()
            if slice_record.status in {SliceStatus.ASSIGNED, SliceStatus.RUNNING}
        )

    def _worker_active_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        active_statuses = {SliceStatus.ASSIGNED, SliceStatus.RUNNING}
        for job in self.jobs.values():
            for slice_record in job.slices.values():
                if slice_record.status in active_statuses and slice_record.assigned_node:
                    counts[slice_record.assigned_node] += 1
        return counts

    def _enforce_worker_health(self, *, now: datetime | None = None) -> None:
        """Requeue slices assigned to workers that have timed out."""
        now = now or datetime.utcnow()
        for worker_id in list(self.workers.keys()):
            slot = self.workers.get(worker_id)
            if not slot:
                continue
            if now - slot.last_seen <= STALE_THRESHOLD:
                continue
            self._requeue_worker_slices(worker_id, now)

    def _requeue_worker_slices(self, worker_id: str, now: datetime) -> None:
        slot = self.workers.get(worker_id)
        if not slot:
            return
        requeued: list[str] = []
        affected_jobs: set[str] = set()
        for job in self.jobs.values():
            for slice_record in job.slices.values():
                if (
                    slice_record.assigned_node != worker_id
                    or slice_record.status not in {SliceStatus.ASSIGNED, SliceStatus.RUNNING}
                ):
                    continue
                slice_record.status = SliceStatus.PENDING
                slice_record.assigned_node = None
                slice_record.progress = 0.0
                slice_record.started_at = None
                slice_record.ended_at = None
                slice_record.logs.append(f"Requeued after worker {worker_id} timeout at {now.isoformat()}")
                self.pending_slices.appendleft(slice_record.slice_id)
                affected_jobs.add(job.execution_id)
                requeued.append(slice_record.slice_id)
        if not requeued:
            slot.busy_slots = 0
            return
        slot.busy_slots = 0
        self._log_general(f"Requeued {len(requeued)} slices from stale worker {worker_id}")
        for execution_id in affected_jobs:
            self._persist_state(self.jobs[execution_id])

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [job.to_dict() for job in self.jobs.values()]

    def get_job(self, execution_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(execution_id)
        return job.to_dict() if job else None

    def get_slice(self, execution_id: str, slice_id: str) -> Optional[Dict[str, Any]]:
        job = self.jobs.get(execution_id)
        if not job:
            return None
        slice_record = job.slices.get(slice_id)
        return slice_record.to_dict() if slice_record else None

    def get_job_logs(self, execution_id: str) -> Optional[str]:
        path = self.base_dir / execution_id / "logs.txt"
        return path.read_text() if path.exists() else None

    def get_slice_logs(self, execution_id: str, slice_id: str) -> Optional[str]:
        path = self.base_dir / execution_id / "slices" / slice_id / "logs.txt"
        return path.read_text() if path.exists() else None
