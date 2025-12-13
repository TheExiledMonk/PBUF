"""Bridge between HPC messages and the controller runtime."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List

from hpc_comms.core.messages import (
    DatasetHashSummary,
    DatasetUpdate,
    JobAssignment as JobAssignmentMessage,
    Message,
    NoWork,
    RequestWork,
    SliceCompletion as SliceCompletionMessage,
    SliceProgress,
    WorkerError,
    WorkerHello,
)

from .controller import Controller
from .models import DatasetUpdatePayload, SliceResult

logger = logging.getLogger("cosmos_control.hpc_bridge")


class HPCControllerBridge:
    """Convert HPC messages into controller actions and responses."""

    def __init__(self, controller: Controller, controller_id: str = "controller") -> None:
        self.controller = controller
        self.controller_id = controller_id

    def _transport_status_from(self, message: Message) -> Dict[str, Any] | None:
        payload = getattr(message, "payload", None)
        if isinstance(payload, dict):
            status = payload.get("transport_status")
            if isinstance(status, dict):
                return status
        return None

    def _update_worker_heartbeat(
        self,
        message: Message,
        worker_id: str | None = None,
    ) -> None:
        worker_id = worker_id or self._worker_id(message)
        if not worker_id:
            return
        transport_status = self._transport_status_from(message)
        self.controller.touch_worker(worker_id, transport_status=transport_status)

    def _worker_id(self, message) -> str:
        if message.source_node:
            return message.source_node
        payload = getattr(message, "payload", None)
        if isinstance(payload, dict):
            return payload.get("worker_id", "")
        return ""

    def _dataset_update_message(self, payload: DatasetUpdatePayload) -> DatasetUpdate:
        return DatasetUpdate(
            source_node=self.controller_id,
            target_node=None,
            dataset_id=payload.dataset_id,
            hash=payload.hash,
            payload=payload.payload,
        )

    def handle_worker_hello(self, message: WorkerHello) -> List[DatasetUpdate]:
        self._update_worker_heartbeat(message, worker_id=message.worker_id)
        _, updates = self.controller.register_worker(
            worker_id=message.worker_id,
            cores=message.cores,
            datasets=message.datasets,
            local_node=message.local_node,
        )
        return [self._dataset_update_message(payload) for payload in updates]

    def handle_dataset_summary(self, message: DatasetHashSummary) -> List[DatasetUpdate]:
        self._update_worker_heartbeat(message, worker_id=message.worker_id)
        updates = self.controller.handle_dataset_summary(message.worker_id, message.datasets)
        return [self._dataset_update_message(payload) for payload in updates]

    def handle_request_work(self, message: RequestWork) -> JobAssignmentMessage | NoWork:
        self._update_worker_heartbeat(message, worker_id=message.worker_id)
        assignments = self.controller.assign_slices(message.worker_id, message.current_load)
        if not assignments:
            return NoWork(
                source_node=self.controller_id,
                target_node=message.source_node,
                reason="no pending slices",
            )

        payloads: List[Dict[str, object]] = []
        for assignment in assignments:
            payloads.append(
                {
                    "execution_id": assignment.execution_id,
                    "run_id": assignment.run_id,
                    "config": assignment.config,
                    "slice": dataclasses.asdict(assignment.slice),
                }
            )

        return JobAssignmentMessage(
            source_node=self.controller_id,
            target_node=message.source_node,
            assignments=payloads,
        )

    def handle_slice_progress(self, message: SliceProgress) -> None:
        self._update_worker_heartbeat(message)
        self.controller.report_slice_progress(message.execution_id, message.slice_id, message.progress)

    def handle_slice_completion(self, message: SliceCompletionMessage) -> None:
        worker_id = self._worker_id(message) or "unknown"
        self._update_worker_heartbeat(message, worker_id=worker_id)
        result = SliceResult(
            execution_id=message.execution_id,
            slice_id=message.slice_id,
            success=message.success,
            logs=message.logs,
            metrics=message.metrics,
            data=message.data,
            progress=message.progress,
        )
        self.controller.process_slice_result(worker_id, result)

    def handle_worker_error(self, message: WorkerError) -> NoWork:
        self._update_worker_heartbeat(message, worker_id=message.worker_id)
        logger.error("Worker reported error: %s %s", message.worker_id, message.error)
        return NoWork(
            source_node=self.controller_id,
            target_node=message.source_node,
            reason=message.error,
        )

    def handle_message(self, message: Message) -> List[Message]:
        if isinstance(message, WorkerHello):
            return self.handle_worker_hello(message)
        if isinstance(message, DatasetHashSummary):
            return self.handle_dataset_summary(message)
        if isinstance(message, RequestWork):
            assignment = self.handle_request_work(message)
            return [assignment] if assignment else []
        if isinstance(message, SliceProgress):
            self.handle_slice_progress(message)
            return []
        if isinstance(message, SliceCompletionMessage):
            self.handle_slice_completion(message)
            return []
        if isinstance(message, WorkerError):
            return [self.handle_worker_error(message)]
        return []
