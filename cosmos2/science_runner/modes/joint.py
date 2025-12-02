"""Joint mode plugin implementation for the unified science runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence
import numpy as np
 
from cosmos2.api.engine import build_lcdm_model_config, build_pbuf_model_config, run_optimisation
from cosmos2.fits.joint import build_joint_chi2_evaluator
from cosmos2.models.pbuf import build_pbuf_joint_chi2
from cosmos2.models.model_factory import create_model
from cosmos2.parameters import ModelState, get_parameter_snapshot

from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.context import ModeResult, RunContext
from cosmos2.science_runner.events import RunEvent, ModelPreparedEvent, MonitorSnapshotEvent
from cosmos2.science_runner.modes.base import BaseModePlugin, register_mode
from cosmos2.science_runner.utils import serialize_value
 

def _compute_profile_likelihood(
    profile: dict[str, Any] | None,
    chi2_fn: Any,
    best_params: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, Any] | None:
    if not profile:
        return None
    parameters = profile.get("parameters") or []
    if not parameters:
        return None
    resolution = max(5, min(int(profile.get("resolution", 20)), 80))
    if len(parameters) == 1:
        name = parameters[0]
        if name not in bounds:
            return None
        lower, upper = bounds[name]
        values = (
            np.linspace(lower, upper, resolution, dtype=float)
            if lower != upper
            else np.asarray([lower], dtype=float)
        )
        points = []
        for value in values:
            candidate = {**best_params, name: float(value)}
            points.append({"value": float(value), "chi2": float(chi2_fn(candidate))})
        return {"type": "1d", "parameter": name, "points": points}
    if len(parameters) >= 2:
        x_name, y_name = parameters[0], parameters[1]
        if x_name not in bounds or y_name not in bounds:
            return None
        x_lower, x_upper = bounds[x_name]
        y_lower, y_upper = bounds[y_name]
        x_values = np.linspace(x_lower, x_upper, min(resolution, 40), dtype=float)
        y_values = np.linspace(y_lower, y_upper, min(resolution, 40), dtype=float)
        grid: list[dict[str, Any]] = []
        for x_val in x_values:
            for y_val in y_values:
                candidate = {**best_params, x_name: float(x_val), y_name: float(y_val)}
                grid.append({"x": float(x_val), "y": float(y_val), "chi2": float(chi2_fn(candidate))})
        return {"type": "2d", "x_parameter": x_name, "y_parameter": y_name, "grid": grid}
    return None


def _make_predictions_for_model(model_name: str, params: dict[str, float]):
    model = create_model(model_name, **params)
    snapshot = get_parameter_snapshot(
        ModelState(
            model_name=model_name,
            model=model,
            fitted_params=params,
            thermal_table=getattr(model, "_thermal", None),
        )
    )
    return snapshot, model


def _build_joint_chi2(model_name: str, joint_config_path: Path):
    normalized = model_name.strip().lower()
    if normalized == "pbuf":
        factory = lambda params: create_model("pbuf", **params)
        return build_pbuf_joint_chi2(factory, joint_config_path, skip_valid=False)
    factory = lambda params: create_model("lcdm", **params)
    return build_joint_chi2_evaluator(factory, joint_config_path, skip_valid=False)


def _build_model_configs(config: ScienceRunConfig, joint_config_path: Path) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    engine_settings = config.engine_settings or {}
    grid_points = engine_settings.get("grid_points")
    n_batches = engine_settings.get("n_batches") or engine_settings.get("n_seeds")
    batch_size = engine_settings.get("batch_size") or engine_settings.get("n_refine")
    rng_seed = engine_settings.get("rng_seed") if "rng_seed" in engine_settings else engine_settings.get("seed")
    for model_name in config.models:
        bounds = config.parameter_bounds_for_model(model_name)
        normalized = model_name.strip().lower()
        if normalized == "pbuf":
            cfg = build_pbuf_model_config(bounds, None, joint_config_path, grid_points=grid_points)
        else:
            cfg = build_lcdm_model_config(bounds, joint_config_path, grid_points=grid_points)
        if n_batches:
            cfg["n_batches"] = int(n_batches)
        if batch_size:
            cfg["batch_size"] = int(batch_size)
        if rng_seed is not None:
            cfg["rng_seed"] = int(rng_seed)
        for key in ("n_scatter", "scatter_scale", "island_fraction"):
            if key in engine_settings:
                cfg[key] = engine_settings[key]
        configs.append(cfg)
    return configs


@register_mode
class JointMode(BaseModePlugin):
    """Joint optimization mode that mirrors the legacy runner."""

    name = "joint"

    def prepare(self, context: RunContext) -> None:
        context.metadata.setdefault("joint", {})["prepared"] = True

    def execute(self, context: RunContext) -> ModeResult:
        engine_name = (self.config.engine or "cosmos2_basin").strip().lower()
        allowed_engines = {"cosmos2", "cosmos2_basin", "basin", "threaded"}
        if engine_name not in allowed_engines:
            raise ValueError(f"Unsupported cosmos2 engine '{engine_name}'. Supported: {sorted(allowed_engines)}")

        joint_config_path = self.config.get_joint_config_path()
        model_configs = _build_model_configs(self.config, joint_config_path)
        for cfg in model_configs:
            self.event_bus.emit(ModelPreparedEvent(cfg.get("name", "model")))

        grid_points = self.config.engine_settings.get("grid_points")
        monitor_option = self.config.engine_settings.get("monitor")
        resume_flag = bool(self.config.engine_settings.get("resume", False))
        checkpoint_path = context.run_dir / "checkpoint.json"

        result = None
        if resume_flag and checkpoint_path.exists():
            try:
                payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("complete"):
                result = {
                    "models": payload.get("models") or [],
                    "best_overall": payload.get("best_overall"),
                    "chi2_history": payload.get("events"),
                }

        progress_callback = self._make_progress_callback(context)

        if result is None:
            result = run_optimisation(
                model_configs,
                monitor=monitor_option,
                grid_points=grid_points,
                workers=self.config.engine_settings.get("workers"),
                mode_label="Joint",
                progress_callback=progress_callback,
                checkpoint_file=checkpoint_path,
                engine=engine_name,
            )

        return self._record_results(context, result, joint_config_path, engine_name)

    def finalize(self, context: RunContext, result: ModeResult) -> None:
        context.history_entries.extend(result.history_entries)
        context.chi2_history.extend(result.chi2_history)
        context.metadata.setdefault("joint", {}).update(result.metadata)

    def _make_progress_callback(self, context: RunContext) -> Callable[[Dict[str, Any]], None]:
        def _callback(event: Dict[str, Any]) -> None:
            event_type = event.get("type", "EngineProgress")
            if event_type == "monitor_snapshot":
                context.event_bus.emit(MonitorSnapshotEvent(event.get("snapshot", {})))
            else:
                context.event_bus.emit(RunEvent(event_type, event))

        return _callback

    def _record_results(
        self,
        context: RunContext,
        result: dict[str, Any],
        joint_config_path: Path,
        engine_name: str,
    ) -> ModeResult:
        history_entries: list[dict[str, Any]] = []
        chi2_history: list[dict[str, Any]] = []
        success = True
        model_failures: list[dict[str, str]] = []
        models = result.get("models", [])

        for summary in models:
            model_name = summary.get("name", "model")
            model_dir = context.run_dir / str(model_name)
            best_params = {k: float(v) for k, v in (summary.get("best_params") or {}).items()}
            best_chi2 = float(summary.get("best_chi2", float("inf")))
            fit_results = summary.get("fit_results") or {}
            chi2_breakdown = summary.get("chi2_breakdown") or {}
            weighted_chi2 = summary.get("weighted_chi2", best_chi2)

            if not np.isfinite(best_chi2):
                success = False
                model_failures.append({"model": model_name, "reason": "non-finite chi2"})
                context.recorder.record_model_failure(model_dir, "non-finite chi2")
                continue

            snapshot, model_obj = _make_predictions_for_model(model_name, best_params)
            engine_trace = summary.get("results")
            trace_meta = {"iterations": len(engine_trace) if isinstance(engine_trace, Sequence) else 0}
            predictions = snapshot.to_predictions()
            derived_params = {
                k: v for k, v in snapshot.derived.items() if k not in {"plot_data", "growth_curve"}
            }
            parameters_payload = {
                "base": serialize_value(best_params),
                "model_parameters": serialize_value(getattr(model_obj, "parameters", {})),
                "derived": serialize_value(derived_params),
                "parameter_snapshot": serialize_value(snapshot.to_dict()),
            }
            context.recorder.write_json(model_dir, "parameters.json", parameters_payload)

            profile_data = None
            if self.config.profile_likelihood:
                joint_chi2 = _build_joint_chi2(model_name, joint_config_path)
                bounds = self.config.parameter_bounds_for_model(model_name)
                profile_data = _compute_profile_likelihood(
                    self.config.profile_likelihood,
                    lambda p: joint_chi2({**best_params, **{k: float(v) for k, v in p.items()}}),
                    best_params,
                    bounds,
                )

            context.recorder.record_model_results(
                model_dir,
                best_params=best_params,
                best_chi2=weighted_chi2 if np.isfinite(weighted_chi2) else best_chi2,
                chi2_breakdown={k: float(v.get("chi2", v)) for k, v in chi2_breakdown.items()},
                fit_outputs=serialize_value(fit_results),
                predictions=predictions,
                engine_result={
                    "results": engine_trace,
                    "best_chi2": best_chi2,
                    "performance": summary.get("performance"),
                },
                parameter_snapshot=snapshot.to_dict(),
                profile_likelihood=profile_data,
                save_space=self.config.output.save_space,
            )
            if isinstance(fit_results, dict):
                for fit_name, payload in fit_results.items():
                    if isinstance(payload, dict):
                        context.recorder.save_fit_output(
                            model_dir,
                            fit_name,
                            chi2=payload.get("chi2", float("inf")),
                            extras=payload.get("extras"),
                        )
            context.recorder.save_engine_trace(
                model_dir,
                engine_name=engine_name,
                trace=engine_trace,
                trace_meta=trace_meta,
                save_space=self.config.output.save_space,
            )

            history_entries.append(
                {
                    "run_name": self.config.run_name,
                    "timestamp": context.timestamp,
                    "model": model_name,
                    "best_chi2": weighted_chi2 if np.isfinite(weighted_chi2) else best_chi2,
                    "fits_used": list(self.config.fits_list),
                    "engine": engine_name,
                    "comment": self.config.description,
                    "success": True,
                }
            )
            chi2_history.append({"model": model_name, "best_chi2": best_chi2, "weighted_chi2": weighted_chi2})

        result_history = result.get("chi2_history")
        if result_history:
            chi2_history = result_history
        return ModeResult(
            success=success and not model_failures,
            history_entries=history_entries,
            chi2_history=chi2_history,
            metadata={"model_failures": model_failures},
        )
