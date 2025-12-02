"""Shared context data structures for unified runner modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.recorder import RunRecorder
from cosmos2.science_runner.events import EventBus


@dataclass
class RunContext:
    config: ScienceRunConfig
    recorder: RunRecorder
    run_dir: Path
    timestamp: str
    joint_payload: dict[str, Any]
    dataset_manifest: dict[str, Any]
    hashes: dict[str, str]
    event_bus: EventBus
    model_summaries: dict[str, Any] = field(default_factory=dict)
    history_entries: list[dict[str, Any]] = field(default_factory=list)
    chi2_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModeResult:
    success: bool
    history_entries: list[dict[str, Any]] = field(default_factory=list)
    chi2_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
