"""Unified science runner infrastructure for Cosmos2."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.context import ModeResult, RunContext
from cosmos2.science_runner.environment import gather_run_environment
from cosmos2.science_runner.events import EventBus, RunEvent, RunFinishedEvent, RunStartedEvent
from cosmos2.science_runner.modes import get_mode
from cosmos2.science_runner.recorder import RunRecorder
from cosmos2.science_runner.utils import hash_payload


class UnifiedScienceRunner:
    """Core orchestrator for science runs with interchangeable modes."""

    def __init__(
        self,
        config: ScienceRunConfig,
        *,
        dry_run: bool = False,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.recorder = RunRecorder(self.config.output.base_dir)
        self.event_bus = event_bus or EventBus()

    def execute(self, *, progress_callback: Callable[[RunEvent], None] | None = None) -> Path:
        if progress_callback:
            self.event_bus.subscribe(progress_callback)

        primary_mode = (self.config.auto_mode or "joint").strip().lower()
        plugin_names: list[str] = [primary_mode]
        if self.config.jackknife_enabled and primary_mode == "joint" and "jackknife" not in plugin_names:
            plugin_names.append("jackknife")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        self.event_bus.emit(RunStartedEvent(timestamp, str(self.config.path)))

        context = self.prepare_run(timestamp=timestamp)
        executed_results: list[tuple[str, ModeResult]] = []

        for plugin_name in plugin_names:
            mode_cls = get_mode(plugin_name)
            plugin = mode_cls(self.config, self.event_bus)
            plugin.prepare(context)
            if self.dry_run:
                continue
            result = plugin.execute(context)
            plugin.finalize(context, result)
            executed_results.append((plugin_name, result))

        aggregated = self._aggregate_results(context, executed_results)
        self.finalize_run(context, aggregated, timestamp)
        self.event_bus.emit(RunFinishedEvent(context.run_dir, success=aggregated.success))
        return context.run_dir

    # Internal helpers
    def prepare_run(self, *, timestamp: str) -> RunContext:
        run_dir = self.recorder.prepare_run_directory(self.config.run_name, timestamp)
        self.recorder.write_config(run_dir, self.config.to_dict())
        dataset_manifest = {
            "fits": list(self.config.fits_list),
            "fit_weights": dict(self.config.fit_weights),
        }
        self.recorder.write_json(run_dir, "datasets_used.json", dataset_manifest)

        joint_payload = self.config.joint_config_payload
        self.recorder.write_json(run_dir, "joint_config_used.json", joint_payload)

        hashes = {
            "joint": hash_payload(joint_payload),
            "parameters": hash_payload(self.config.parameter_bounds_payload),
            "dataset_manifest": hash_payload(dataset_manifest),
            "config": self._hash_file(self.config.path),
        }

        return RunContext(
            config=self.config,
            recorder=self.recorder,
            run_dir=run_dir,
            timestamp=timestamp,
            joint_payload=joint_payload,
            dataset_manifest=dataset_manifest,
            hashes=hashes,
            event_bus=self.event_bus,
        )

    def finalize_run(self, context: RunContext, result: ModeResult, start_timestamp: str) -> None:
        env_snapshot = gather_run_environment()
        meta = {
            "run_name": self.config.run_name,
            "timestamp": context.timestamp,
            "mode": self.config.mode,
            "engine": self.config.engine,
            "run_hashes": context.hashes,
            "success": result.success,
            "model_count": len(self.config.models),
            "history_entries": len(result.history_entries),
            "start_timestamp": start_timestamp,
            "end_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if env_snapshot:
            meta["environment"] = env_snapshot
            env_cli = env_snapshot.get("cli_command")
            if env_cli:
                meta["cli_command"] = env_cli
            git_info = env_snapshot.get("git")
            if git_info:
                meta["git_commit"] = git_info.get("commit")
                meta["git_dirty"] = git_info.get("dirty")
        self.recorder.write_meta(context.run_dir, meta)

        if result.history_entries:
            self.recorder.write_history_entry(context.run_dir, result.history_entries)
            self.recorder.append_history(result.history_entries)
        if result.chi2_history:
            self.recorder.write_json(context.run_dir, "chi2_history.json", result.chi2_history)

    @staticmethod
    def _hash_file(path: Path) -> str:
        payload = path.read_text(encoding="utf-8")
        return hash_payload({"content": payload})

    def _aggregate_results(
        self, context: RunContext, executed_results: list[tuple[str, ModeResult]]
    ) -> ModeResult:
        success = all(result.success for _, result in executed_results) if executed_results else True
        plugin_metadata: dict[str, dict[str, Any]] = {}
        for plugin_name, result in executed_results:
            if result.metadata:
                plugin_metadata[plugin_name] = result.metadata
        metadata: dict[str, Any] = {}
        if plugin_metadata:
            metadata["plugins"] = plugin_metadata

        return ModeResult(
            success=success,
            history_entries=list(context.history_entries),
            chi2_history=list(context.chi2_history),
            metadata=metadata,
        )
