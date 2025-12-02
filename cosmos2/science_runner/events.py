"""Eventing primitives for the unified science runner."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Dict, List


class RunEvent:
    """Base event emitted during a science runner execution."""

    def __init__(self, name: str, payload: Dict[str, Any] | None = None) -> None:
        self.name = name
        self.payload = payload or {}


class RunStartedEvent(RunEvent):
    def __init__(self, timestamp: str, config_path: str) -> None:
        super().__init__("RunStarted", {"timestamp": timestamp, "config": config_path})


class ModelPreparedEvent(RunEvent):
    def __init__(self, model_name: str) -> None:
        super().__init__("ModelPrepared", {"model": model_name})


class EngineProgressEvent(RunEvent):
    def __init__(self, model_name: str, progress: float) -> None:
        super().__init__("EngineProgress", {"model": model_name, "progress": progress})


class JackknifeDrawStartedEvent(RunEvent):
    def __init__(self, draw_index: int, total_draws: int, removed_datasets: dict[str, int], seed: int | None) -> None:
        payload = {
            "draw_index": draw_index,
            "total_draws": total_draws,
            "removed_datasets": removed_datasets,
            "random_seed": seed,
        }
        super().__init__("JackknifeDrawStarted", payload)


class JackknifeDrawFinishedEvent(RunEvent):
    def __init__(self, draw_index: int, success: bool, best_model_full: str, best_model_jackknife: str, error_message: str | None = None) -> None:
        payload = {
            "draw_index": draw_index,
            "success": success,
            "best_model_full": best_model_full,
            "best_model_jackknife": best_model_jackknife,
            "error_message": error_message,
        }
        super().__init__("JackknifeDrawFinished", payload)


class JackknifeAnalysisReadyEvent(RunEvent):
    def __init__(self, analysis: dict[str, Any]) -> None:
        super().__init__("JackknifeAnalysisReady", {"analysis": analysis})


class RunFinishedEvent(RunEvent):
    def __init__(self, run_dir: Path, success: bool) -> None:
        super().__init__("RunFinished", {"run_dir": str(run_dir), "success": success})


class MonitorSnapshotEvent(RunEvent):
    def __init__(self, snapshot: dict[str, Any]) -> None:
        super().__init__("MonitorSnapshot", {"snapshot": snapshot})


class EventBus:
    """Thread-safe event bus for monitoring science runner executions."""

    def __init__(self) -> None:
        self._subscribers: List[Callable[[RunEvent], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[RunEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def emit(self, event: RunEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:
                # Monitoring should never fail the run.
                continue
