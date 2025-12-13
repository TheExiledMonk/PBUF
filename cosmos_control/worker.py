"""Lightweight worker implementation that drives the controller via HTTP transport."""

from __future__ import annotations

import base64
import hashlib
import logging
import threading
import time
from pathlib import Path
import json
from typing import Any, Sequence

from hpc_comms.core.messages import (
    DatasetHashSummary,
    DatasetUpdate,
    JobAssignment as JobAssignmentMessage,
    Message,
    NoWork,
    RequestWork,
    SliceCompletion as SliceCompletionMessage,
    SliceProgress,
    WorkerHello,
)

from .compute_plugins import compute_slice
from .models import JobAssignment as JobAssignmentRecord, SliceDescriptor
from .transports.worker_http import WorkerHTTPTransport

logger = logging.getLogger("cosmos_control.worker")


class WorkerClient:
    """Worker loop that requests slices, executes them, and reports results."""

    def __init__(
        self,
        transport: WorkerHTTPTransport,
        worker_id: str,
        cores: int,
        *,
        local_node: bool = False,
        dataset_cache: Path | None = None,
        datasets: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        self.transport = transport
        self.worker_id = worker_id
        self.cores = cores
        self.local_node = local_node
        self.dataset_cache = (dataset_cache or Path(".worker_datasets")).expanduser()
        self.dataset_cache.mkdir(parents=True, exist_ok=True)
        self.datasets: dict[str, str] = {dataset_id: hash_value for dataset_id, hash_value in (datasets or [])}

    def _apply_update(self, update: DatasetUpdate) -> None:
        payload = base64.b64decode(update.payload.encode("utf-8"))
        path = self.dataset_cache / update.dataset_id
        path.write_bytes(payload)
        self.datasets[update.dataset_id] = hashlib.sha256(payload).hexdigest()
        logger.info("Worker %s cached dataset %s", self.worker_id, update.dataset_id)

    def _send_hello(self) -> Sequence[Message]:
        hello = WorkerHello(
            source_node=self.worker_id,
            target_node="controller",
            worker_id=self.worker_id,
            cores=self.cores,
            datasets=self.datasets,
            local_node=self.local_node,
        )
        return self._dispatch(hello)

    def _report_dataset_summary(self) -> Sequence[Message]:
        summary = DatasetHashSummary(
            source_node=self.worker_id,
            target_node="controller",
            worker_id=self.worker_id,
            datasets=self.datasets,
        )
        return self._dispatch(summary)

    def _execute_slice(self, assignment: JobAssignmentRecord) -> None:
        slice_desc = assignment.slice
        metadata = assignment.config.get("metadata")
        if metadata:
            logger.info(
                "Worker %s slice %s metadata=%s",
                self.worker_id,
                slice_desc.slice_id,
                json.dumps(metadata),
            )
        logs = [f"Worker {self.worker_id} starting slice {slice_desc.slice_id}"]
        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._progress_heartbeat,
            args=(stop_event, assignment.execution_id, slice_desc.slice_id),
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            compute_result = compute_slice(assignment.config, slice_desc, self.datasets)
            success = bool(compute_result.get("success", True))
            metrics = compute_result.get("metrics", {})
            data = compute_result.get("data", {})
            log_entry = compute_result.get("logs")
            if log_entry:
                logs.append(log_entry)
            completion = SliceCompletionMessage(
                source_node=self.worker_id,
                target_node="controller",
                execution_id=assignment.execution_id,
                slice_id=slice_desc.slice_id,
                success=success,
                logs="\n".join([entry for entry in logs if entry]),
                metrics=metrics,
                data=data,
                progress=1.0 if success else 0.0,
            )
        except Exception as exc:
            completion = SliceCompletionMessage(
                source_node=self.worker_id,
                target_node="controller",
                execution_id=assignment.execution_id,
                slice_id=slice_desc.slice_id,
                success=False,
                logs=f"Exception: {exc}",
                metrics={},
                data={},
                progress=0.0,
            )
        finally:
            stop_event.set()
            heartbeat_thread.join(timeout=1.0)
            if heartbeat_thread.is_alive():
                logger.warning(
                    "Worker %s heartbeat thread for slice %s failed to exit promptly",
                    self.worker_id,
                    slice_desc.slice_id,
                )
        self._dispatch(completion)

    def _send_slice_progress(self, execution_id: str, slice_id: str, progress: float) -> None:
        progress_msg = SliceProgress(
            source_node=self.worker_id,
            target_node="controller",
            worker_id=self.worker_id,
            execution_id=execution_id,
            slice_id=slice_id,
            progress=progress,
        )
        self._dispatch(progress_msg)

    def _progress_heartbeat(
        self,
        stop_event: threading.Event,
        execution_id: str,
        slice_id: str,
    ) -> None:
        interval = 10.0
        try:
            while not stop_event.wait(interval):
                self._send_slice_progress(execution_id, slice_id, 0.0)
        except Exception:
            return

    def run(self, *, max_rounds: int | None = None) -> None:
        self._send_hello()

        rounds = 0
        while True:
            request = RequestWork(
                source_node=self.worker_id,
                target_node="controller",
                worker_id=self.worker_id,
                current_load=0,
            )
            responses = self._dispatch(request)
            assignments_payload: list[dict[str, Any]] = []
            no_work = False

            for msg in responses:
                if isinstance(msg, JobAssignmentMessage):
                    assignments_payload.extend(msg.assignments)
                elif isinstance(msg, NoWork):
                    no_work = True

            if not assignments_payload and no_work:
                logger.info("Worker %s no work available", self.worker_id)
                break

            if not assignments_payload:
                time.sleep(0.1)
                continue

            for assignment_payload in assignments_payload:
                assignment = JobAssignmentRecord(
                    execution_id=assignment_payload["execution_id"],
                    run_id=assignment_payload["run_id"],
                    config=assignment_payload.get("config", {}),
                    slice=SliceDescriptor(**assignment_payload["slice"]),
                )
                self._execute_slice(assignment)

            self._apply_updates_from_summary()
            rounds += 1
            if max_rounds and rounds >= max_rounds:
                break

    def _apply_updates_from_summary(self) -> None:
        self._report_dataset_summary()

    def _dispatch(self, message: Message) -> list[Message]:
        message.payload["transport_status"] = self.transport.transport_status()
        try:
            responses = self.transport.send(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Worker %s transport failure while sending %s: %s",
                self.worker_id,
                message.message_type,
                exc,
            )
            time.sleep(2.0)
            return []
        non_dataset: list[Message] = []
        for response in responses:
            if isinstance(response, DatasetUpdate):
                self._apply_update(response)
            else:
                non_dataset.append(response)
        return non_dataset
