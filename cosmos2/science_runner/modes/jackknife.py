"""Jackknife mode plugin for the unified science runner."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Set

import numpy as np
from cosmos2.data.registry import _LOADERS, get_dataset
from data_interface.standardize import ensure_standard_dataset

from cosmos2.api.engine import clear_jackknife_masked_datasets, run_optimisation, set_jackknife_masked_datasets
from cosmos2.science_runner.context import ModeResult, RunContext
from cosmos2.science_runner.events import (
    JackknifeAnalysisReadyEvent,
    JackknifeDrawFinishedEvent,
    JackknifeDrawStartedEvent,
    MonitorSnapshotEvent,
    RunEvent,
)
from cosmos2.science_runner.jackknife import (
    JackknifeConfig,
    JackknifeDraw,
    JackknifeResampler,
    ModelResult,
    apply_mask_to_dataset,
    analyze_jackknife_results,
)
from cosmos2.science_runner.modes.base import BaseModePlugin, register_mode
from cosmos2.science_runner.modes.joint import _build_model_configs

try:
    from cosmos2.threads.enhanced_monitoring import get_integration
except ImportError:  # pragma: no cover - guard for monitoring-free builds
    get_integration = None


def _dataset_aliases(dataset_name: str) -> Set[str]:
    normalized = dataset_name.strip().lower()
    loader = _LOADERS.get(normalized)
    if loader is None:
        return {normalized}
    return {name for name, fn in _LOADERS.items() if fn is loader}


def _load_standardized_dataset(dataset_name: str) -> dict[str, Any]:
    dataset = get_dataset(dataset_name)
    payload = {key: dataset[key] for key in dataset}
    dtype = payload.get("type", dataset_name)
    return ensure_standard_dataset(payload, dtype)


def _get_dataset_size(dataset: dict[str, Any]) -> int:
    for key in ("data", "z", "mu", "distances"):
        value = dataset.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            return len(value)
    max_size = 0
    for value in dataset.values():
        if isinstance(value, (list, tuple, np.ndarray)):
            max_size = max(max_size, len(value))
    return max_size


def _best_model_name(models: dict[str, ModelResult]) -> str:
    candidates = [res for res in models.values() if np.isfinite(res.chi_squared)]
    if not candidates:
        return next(iter(models.keys()), "unknown") if models else "unknown"
    best = min(candidates, key=lambda res: res.chi_squared)
    return best.model_name


@register_mode
class JackknifeMode(BaseModePlugin):
    """Jackknife mode that uses the shared optimisation helpers with masked datasets."""

    name = "jackknife"

    def _monitor_state(self):
        if get_integration is None:
            return None
        integration = get_integration()
        return integration.monitor_state if integration else None

    def _clear_monitor_jackknife(self) -> None:
        monitor_state = self._monitor_state()
        if monitor_state:
            monitor_state.clear_jackknife_history()

    def _record_monitor_draw(self, label: str, chi2: float) -> None:
        monitor_state = self._monitor_state()
        if monitor_state:
            monitor_state.update_jackknife_trace(label, chi2)


    def prepare(self, context: RunContext) -> None:
        if not context.config.jackknife_enabled:
            raise ValueError("Jackknife mode requires 'jackknife.enabled' in the science config.")
        context.metadata.setdefault("jackknife", {})["prepared"] = True

    def execute(self, context: RunContext) -> ModeResult:
        config = context.config
        jackknife_config = config.jackknife
        assert jackknife_config is not None

        datasets = self._load_datasets(jackknife_config.datasets_to_test)
        if not datasets:
            raise ValueError("Jackknife configuration did not load any standardized datasets to mask.")

        joint_config_path = config.get_joint_config_path()
        model_configs = _build_model_configs(config, joint_config_path)
        engine_name = self._resolve_engine_name(config.engine)
        progress_callback = self._make_progress_callback(context)
        monitor_option = config.engine_settings.get("monitor")
        grid_points = config.engine_settings.get("grid_points")
        workers = config.engine_settings.get("workers")

        self._clear_monitor_jackknife()

        # Baseline run to capture reference model results.
        clear_jackknife_masked_datasets()
        baseline_result = self._run_optimisation(
            model_configs,
            engine_name,
            progress_callback,
            monitor_option,
            grid_points,
            workers,
            mode_label="Jackknife baseline",
        )
        baseline_models = self._parse_model_results(baseline_result)
        baseline_best = _best_model_name(baseline_models)
        baseline_best_result = baseline_models.get(baseline_best)
        baseline_best_chi2 = float(baseline_best_result.chi_squared) if baseline_best_result else float("nan")
        self._record_monitor_draw("baseline", baseline_best_chi2)

        # Generate masks for selected datasets.
        resampler = JackknifeResampler(jackknife_config, list(datasets.keys()))
        for name, dataset in datasets.items():
            resampler.set_dataset_size(name, _get_dataset_size(dataset))
        masks = resampler.generate_masks()

        draws: list[JackknifeDraw] = []
        for mask in masks:
            total_draws = len(masks)
            removed_counts = {
                name: int(np.count_nonzero(~mask.dataset_masks[name]))
                for name in mask.dataset_masks
            }
            masked_datasets: dict[str, dict[str, Any]] = {}
            for name, mask_array in mask.dataset_masks.items():
                if name not in datasets:
                    continue
                masked = apply_mask_to_dataset(datasets[name], mask_array)
                for alias in _dataset_aliases(name):
                    masked_datasets[alias] = masked
            set_jackknife_masked_datasets(masked_datasets)
            context.event_bus.emit(
                JackknifeDrawStartedEvent(
                    draw_index=mask.draw_index,
                    total_draws=total_draws,
                    removed_datasets=removed_counts,
                    seed=mask.random_seed,
                )
            )
            draw_models: dict[str, ModelResult] = {}
            draw_success = True
            error_message: str | None = None
            try:
                draw_result = self._run_optimisation(
                    model_configs,
                    engine_name,
                    progress_callback,
                    monitor_option,
                    grid_points,
                    workers,
                    mode_label=f"Jackknife draw {mask.draw_index + 1}/{len(masks)}",
                )
                draw_models = self._parse_model_results(draw_result)
                if not draw_models:
                    draw_success = False
            except Exception as exc:  # noqa: BLE001
                draw_success = False
                error_message = str(exc)
            finally:
                clear_jackknife_masked_datasets()

            best_draw_model = _best_model_name(draw_models)
            best_draw_result = draw_models.get(best_draw_model)
            best_draw_chi2 = float(best_draw_result.chi_squared) if best_draw_result else float("nan")
            self._record_monitor_draw(f"draw-{mask.draw_index + 1}", best_draw_chi2)
            draw = JackknifeDraw(
                draw_index=mask.draw_index,
                removed_datasets=removed_counts,
                original_models=baseline_models,
                jackknife_models=draw_models,
                best_model_full=baseline_best,
                best_model_jackknife=best_draw_model,
                success=draw_success,
                error_message=error_message,
                random_seed=mask.random_seed,
            )
            draws.append(draw)
            context.event_bus.emit(
                JackknifeDrawFinishedEvent(
                    draw_index=mask.draw_index,
                    success=draw_success,
                    best_model_full=baseline_best,
                    best_model_jackknife=best_draw_model,
                    error_message=error_message,
                )
            )

        analysis = analyze_jackknife_results(draws, jackknife_config)
        context.recorder.write_json(
            context.run_dir,
            "jackknife_results.json",
            {
                "config": jackknife_config.to_dict(),
                "baseline_models": {name: model.to_dict() for name, model in baseline_models.items()},
                "draws": [draw.to_dict() for draw in draws],
                "analysis": analysis,
            },
        )
        context.recorder.write_json(context.run_dir, "jackknife_summary.json", analysis)
        context.event_bus.emit(JackknifeAnalysisReadyEvent(analysis))

        overall_success = bool(draws) and all(draw.success for draw in draws)
        summary_entry = {
            "run_name": config.run_name,
            "timestamp": context.timestamp,
            "model": "jackknife",
            "best_chi2": float(analysis.get("chi2_changes", {}).get("mean", 0.0)),
            "fits_used": list(config.fits_list),
            "engine": engine_name,
            "comment": "Jackknife stability analysis",
            "success": overall_success,
        }

        metadata = {
            "jackknife_summary": analysis,
            "draw_count": len(draws),
        }

        return ModeResult(
            success=overall_success,
            history_entries=[summary_entry],
            metadata=metadata,
        )

    def finalize(self, context: RunContext, result: ModeResult) -> None:
        context.history_entries.extend(result.history_entries)
        context.metadata.setdefault("jackknife", {}).update(result.metadata)

    def _load_datasets(self, dataset_names: list[str]) -> dict[str, dict[str, Any]]:
        loaded: dict[str, dict[str, Any]] = {}
        for name in dataset_names:
            try:
                loaded[name] = _load_standardized_dataset(name)
            except FileNotFoundError:
                continue
        return loaded

    def _resolve_engine_name(self, engine: str | None) -> str:
        engine_name = (engine or "cosmos2_basin").strip().lower()
        allowed = {"cosmos2", "cosmos2_basin", "basin", "threaded"}
        if engine_name not in allowed:
            raise ValueError(f"Unsupported cosmos2 engine '{engine_name}'. Supported: {sorted(allowed)}")
        return engine_name

    def _run_optimisation(
        self,
        model_configs: list[Dict[str, Any]],
        engine_name: str,
        progress_callback: Callable[[Dict[str, Any]], None],
        monitor_value: str | bool | None,
        grid_points: int | None,
        workers: int | None,
        *,
        mode_label: str | None = None,
    ) -> dict[str, Any]:
        return run_optimisation(
            model_configs,
            monitor=monitor_value,
            grid_points=grid_points,
            workers=workers,
            progress_callback=progress_callback,
            engine=engine_name,
            mode_label=mode_label,
        )

    def _parse_model_results(self, result: dict[str, Any]) -> dict[str, ModelResult]:
        models: dict[str, ModelResult] = {}
        for summary in result.get("models", []):
            model_name = summary.get("name", "model")
            best_params = {k: float(v) for k, v in (summary.get("best_params") or {}).items()}
            best_chi2 = float(summary.get("best_chi2", float("inf")))
            models[model_name] = ModelResult(
                model_name=model_name,
                parameters=best_params,
                chi_squared=best_chi2,
                aic=summary.get("aic"),
                bic=summary.get("bic"),
                dof=summary.get("dof"),
                convergence_status=summary.get("convergence_status", "unknown"),
            )
        return models

    def _make_progress_callback(self, context: RunContext) -> Callable[[Dict[str, Any]], None]:
        def _callback(event: Dict[str, Any]) -> None:
            event_type = event.get("type", "EngineProgress")
            if event_type == "monitor_snapshot":
                context.event_bus.emit(MonitorSnapshotEvent(event.get("snapshot", {})))
            else:
                context.event_bus.emit(RunEvent(event_type, event))

        return _callback
