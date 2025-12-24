"""Unified science runner infrastructure for Cosmos2."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from cosmos2.models.model_factory import create_model
from cosmos2.predictions import PredictionManager
from cosmos2.predictions.io import write_prediction_json, write_prediction_table
from cosmos2.predictions.structures import PredictionResult
from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.context import ModeResult, RunContext
from cosmos2.science_runner.environment import gather_run_environment
from cosmos2.science_runner.events import EventBus, RunEvent, RunFinishedEvent, RunStartedEvent
from cosmos2.science_runner.modes import get_mode
from cosmos2.science_runner.recorder import RunRecorder
from cosmos2.science_runner.utils import hash_payload

logger = logging.getLogger(__name__)


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

        controller_endpoint = self._controller_endpoint()
        primary_mode = (self.config.auto_mode or "joint").strip().lower()
        plugin_names: list[str] = [primary_mode]
        if (
            controller_endpoint is None
            and self.config.jackknife_enabled
            and primary_mode == "joint"
            and "jackknife" not in plugin_names
        ):
            plugin_names.append("jackknife")

        run_start = datetime.now(timezone.utc)
        timestamp = run_start.strftime("%Y-%m-%dT%H%M%S")
        start_iso = run_start.isoformat()
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
        self.finalize_run(context, aggregated, start_iso)
        self.event_bus.emit(RunFinishedEvent(context.run_dir, success=aggregated.success))
        return context.run_dir

    def _controller_endpoint(self) -> str | None:
        endpoint = self.config.engine_settings.get("controller_endpoint")
        if endpoint:
            return str(endpoint)
        return os.environ.get("COSMOS_CONTROLLER_ENDPOINT")

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

    def finalize_run(self, context: RunContext, result: ModeResult, start_timestamp_iso: str) -> None:
        controller_endpoint = self._controller_endpoint()
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
            "start_timestamp": start_timestamp_iso,
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
        self._maybe_run_predictions(context, controller_run=bool(controller_endpoint))

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

    def _maybe_run_predictions(self, context: RunContext, *, controller_run: bool = False) -> None:
        predictions_cfg = context.config.predictions
        if not (predictions_cfg.enabled and predictions_cfg.modules):
            return
        if controller_run:
            return

        summary_entries: list[dict[str, Any]] = []
        manager = PredictionManager(modules=predictions_cfg.modules)
        predictions_dir = context.run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for model_name in context.config.models:
            best_params = self._load_model_parameters(context.run_dir / model_name)
            if not best_params:
                logger.info("Skipping predictions for %s (no best-fit parameters)", model_name)
                continue
            try:
                model_obj = create_model(model_name, **best_params)
            except Exception as exc:
                logger.exception("Failed to instantiate %s for predictions: %s", model_name, exc)
                continue
            results = manager.run_for_model(model_name, model_obj, predictions_cfg.module_configs)
            for result in results:
                self._persist_prediction_to_disk(result, predictions_dir / result.name / model_name)
            summary_entries.append(manager.as_summary(model_name, results))

        if not summary_entries:
            return

        summary_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "science_run",
            "modules": list(predictions_cfg.modules),
            "models": summary_entries,
        }
        summary_path = predictions_dir / "predictions_summary.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        context.metadata.setdefault("predictions", {})["summary_path"] = str(summary_path)

    def _persist_prediction_to_disk(self, result: PredictionResult, target_dir: Path) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        write_prediction_json(result, target_dir / "result.json")
        tables_dir = target_dir / "tables"
        for table in result.tables:
            write_prediction_table(table, tables_dir / f"{table.name}.csv")
        plots_dir = target_dir / "plots"
        for plot in result.plots:
            plot_path = plots_dir / f"{plot.name}.json"
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plot_path.write_text(json.dumps(plot.to_dict(), indent=2), encoding="utf-8")

    def _load_model_parameters(self, model_dir: Path) -> dict[str, float]:
        source = model_dir / "best_fit.json"
        if not source.exists():
            return {}
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception:
            return {}
        params = payload.get("parameters") or {}
        normalized: dict[str, float] = {}
        for key, value in params.items():
            try:
                normalized[key] = float(value)
            except Exception:
                continue
        return normalized
