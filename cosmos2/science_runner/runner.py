"""Simplified science runner that mirrors cosmos/science_runner but uses cosmos2 threads/engine."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

import numpy as np

from cosmos2.api.engine import build_lcdm_model_config, build_pbuf_model_config, run_optimisation
from cosmos2.fits.joint import build_joint_chi2_evaluator
from cosmos2.models.model_factory import create_model
from cosmos2.models.pbuf import build_pbuf_joint_chi2
from cosmos2.parameters import ModelState, get_parameter_snapshot
from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.environment import gather_run_environment
from cosmos2.science_runner.recorder import RunRecorder
from cosmos2.science_runner.utils import hash_payload, serialize_value


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
        values = np.linspace(lower, upper, resolution, dtype=float) if lower != upper else np.asarray([lower], dtype=float)
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


def _hash_file(path: Path) -> str:
    payload = path.read_text(encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cosmos2ScienceRunner:
    """Science-run orchestrator that records outputs compatible with the legacy reports."""

    def __init__(self, config: ScienceRunConfig, *, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self.recorder = RunRecorder(self.config.output.base_dir)

    def execute(self, *, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> Path:
        import platform
        import sys
        import time
        import multiprocessing

        start_wall = time.time()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        run_dir = self.recorder.prepare_run_directory(self.config.run_name, timestamp)
        self.recorder.write_config(run_dir, self.config.to_dict())
        joint_payload = self.config.joint_config_payload
        self.recorder.write_json(run_dir, "joint_config_used.json", joint_payload)

        dataset_manifest = {
            "fits": list(self.config.fits_list),
            "fit_weights": dict(self.config.fit_weights),
        }
        self.recorder.write_json(run_dir, "datasets_used.json", dataset_manifest)
        self.recorder.write_json(run_dir, "engine_settings.json", self.config.engine_settings)

        joint_config_path = self.config.get_joint_config_path()
        joint_hash = hash_payload(joint_payload)
        parameter_bounds_payload = self.config.parameter_bounds_payload
        parameter_bounds_hash = hash_payload(parameter_bounds_payload)
        dataset_manifest_hash = hash_payload(dataset_manifest)
        config_hash = _hash_file(self.config.path)

        if self.dry_run:
            return run_dir

        engine_name = (self.config.engine or "cosmos2_basin").strip().lower()
        allowed_engines = {"cosmos2", "cosmos2_basin", "basin", "threaded"}
        if engine_name not in allowed_engines:
            raise ValueError(f"Unsupported cosmos2 engine '{engine_name}'. Supported: {sorted(allowed_engines)}")

        model_configs = self._build_model_configs(joint_config_path)
        grid_points = self.config.engine_settings.get("grid_points")
        monitor_option = self.config.engine_settings.get("monitor")
        resume_flag = bool(self.config.engine_settings.get("resume", False))
        checkpoint_path = run_dir / "checkpoint.json"

        result = None
        if resume_flag and checkpoint_path.exists():
            try:
                checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except Exception:
                checkpoint_payload = None
            if isinstance(checkpoint_payload, dict) and checkpoint_payload.get("complete"):
                cached_models = checkpoint_payload.get("models")
                if isinstance(cached_models, list):
                    print(f"[cosmos2] Resuming from checkpoint at {checkpoint_path}")
                    result = {
                        "models": cached_models,
                        "best_overall": checkpoint_payload.get("best_overall"),
                        "chi2_history": checkpoint_payload.get("events"),
                    }

        if result is None:
            result = run_optimisation(
                model_configs,
                monitor=monitor_option,
                grid_points=grid_points,
                workers=self.config.engine_settings.get("workers"),
                mode_label="legacy joint",
                progress_callback=progress_callback,
                checkpoint_file=checkpoint_path,
            )

        history_entries: list[dict[str, Any]] = []
        model_failures: list[dict[str, str]] = []
        chi2_history: list[dict[str, Any]] = []
        success = True
        for summary in result.get("models", []):
            model_name = summary.get("name") or "model"
            model_dir = run_dir / str(model_name)
            best_params = {k: float(v) for k, v in (summary.get("best_params") or {}).items()}
            best_chi2 = float(summary.get("best_chi2", float("inf")))
            fit_results = summary.get("fit_results") or {}
            chi2_breakdown = summary.get("chi2_breakdown") or {}
            weighted_chi2 = summary.get("weighted_chi2", best_chi2)

            if not np.isfinite(best_chi2):
                success = False
                model_failures.append({"model": model_name, "reason": "non-finite chi2"})
                self.recorder.record_model_failure(model_dir, "non-finite chi2")
                continue

            snapshot, model_obj = self._make_predictions_for_model(model_name, best_params)
            engine_trace = summary.get("results")
            trace_meta = {"iterations": len(engine_trace) if isinstance(engine_trace, Sequence) else 0}
            predictions = snapshot.to_predictions()
            # Save a flat parameters dump with derived quantities for quick inspection.
            derived_params = {
                k: v for k, v in snapshot.derived.items() if k not in {"plot_data", "growth_curve"}
            }
            parameters_payload = {
                "base": serialize_value(best_params),
                "model_parameters": serialize_value(getattr(model_obj, "parameters", {})),
                "derived": serialize_value(derived_params),
                "parameter_snapshot": serialize_value(snapshot.to_dict()),
            }
            self.recorder.write_json(model_dir, "parameters.json", parameters_payload)

            profile_data = None
            if self.config.profile_likelihood:
                joint_chi2 = self._build_joint_chi2(model_name, joint_config_path)
                bounds = self.config.parameter_bounds_for_model(model_name)
                profile_data = _compute_profile_likelihood(
                    self.config.profile_likelihood,
                    lambda p: joint_chi2({**best_params, **{k: float(v) for k, v in p.items()}}),
                    best_params,
                    bounds,
                )

            # Record outputs for reports.
            self.recorder.record_model_results(
                model_dir,
                best_params=best_params,
                best_chi2=weighted_chi2 if np.isfinite(weighted_chi2) else best_chi2,
                chi2_breakdown={k: float(v.get("chi2", v)) for k, v in chi2_breakdown.items()},
                fit_outputs=serialize_value(fit_results),
                predictions=predictions,
                engine_result={"results": engine_trace, "best_chi2": best_chi2},
                parameter_snapshot=snapshot.to_dict(),
                profile_likelihood=profile_data,
                save_space=self.config.output.save_space,
            )
            if isinstance(fit_results, dict):
                for fit_name, payload in fit_results.items():
                    if not isinstance(payload, dict):
                        continue
                    self.recorder.save_fit_output(
                        model_dir,
                        fit_name,
                        chi2=payload.get("chi2", float("inf")),
                        extras=payload.get("extras"),
                    )
            self.recorder.save_engine_trace(
                model_dir,
                engine_name=engine_name,
                trace=engine_trace,
                trace_meta=trace_meta,
                save_space=self.config.output.save_space,
            )

            history_entries.append(
                {
                    "run_name": self.config.run_name,
                    "timestamp": timestamp,
                    "model": model_name,
                    "best_chi2": weighted_chi2 if np.isfinite(weighted_chi2) else best_chi2,
                    "fits_used": list(self.config.fits_list),
                    "engine": engine_name,
                    "comment": self.config.description,
                    "success": True,
                }
            )
            chi2_history.append({"model": model_name, "best_chi2": best_chi2, "weighted_chi2": weighted_chi2})
        env_snapshot = gather_run_environment()

        run_meta = {
            "run_name": self.config.run_name,
            "timestamp": timestamp,
            "start_time": datetime.fromtimestamp(start_wall, tz=timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "total_runtime": time.time() - start_wall,
            "mode": self.config.mode,
            "engine": engine_name,
            "machine": {
                "node": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
                "cpus": multiprocessing.cpu_count(),
                "python": sys.version.split()[0],
            },
            "fits_used": list(self.config.fits_list),
            "joint_hash": joint_hash,
            "joint_config_hash": joint_hash,
            "parameter_bounds_hash": parameter_bounds_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "config_hash": config_hash,
            "success": success and not model_failures,
            "model_failures": model_failures,
        }
        if env_snapshot:
            run_meta["environment"] = env_snapshot
            env_cli = env_snapshot.get("cli_command")
            if env_cli:
                run_meta["cli_command"] = env_cli
            git_info = env_snapshot.get("git")
            if git_info:
                run_meta["git_commit"] = git_info.get("commit")
                run_meta["git_dirty"] = git_info.get("dirty")
        self.recorder.write_meta(run_dir, run_meta)
        if history_entries:
            self.recorder.write_history_entry(run_dir, history_entries)
            self.recorder.append_history(history_entries)
        result_history = result.get("chi2_history") if isinstance(result, dict) else None
        if result_history:
            self.recorder.write_json(run_dir, "chi2_history.json", result_history)
        elif chi2_history:
            self.recorder.write_json(run_dir, "chi2_history.json", chi2_history)
        return run_dir

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _build_model_configs(self, joint_config_path: Path) -> list[dict[str, Any]]:
        configs: list[dict[str, Any]] = []
        engine_settings = self.config.engine_settings or {}
        grid_points = engine_settings.get("grid_points")
        # Normalise engine knobs to the names expected by the basin walker.
        n_batches = engine_settings.get("n_batches") or engine_settings.get("n_seeds")
        batch_size = engine_settings.get("batch_size") or engine_settings.get("n_refine")
        rng_seed = engine_settings.get("rng_seed") if "rng_seed" in engine_settings else engine_settings.get("seed")
        for model_name in self.config.models:
            bounds = self.config.parameter_bounds_for_model(model_name)
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

    def _build_joint_chi2(self, model_name: str, joint_config_path: Path):
        normalized = model_name.strip().lower()
        if normalized == "pbuf":
            factory = lambda params: create_model("pbuf", **params)  # noqa: E731
            return build_pbuf_joint_chi2(factory, joint_config_path, skip_valid=False)
        factory = lambda params: create_model("lcdm", **params)  # noqa: E731
        skip_valid = False
        return build_joint_chi2_evaluator(factory, joint_config_path, skip_valid=skip_valid)

    def _make_predictions_for_model(self, model_name: str, params: dict[str, float]):
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



def run_science_run(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    """Convenience wrapper to execute a science run from a config file."""

    config = ScienceRunConfig.from_path(config_path)
    runner = Cosmos2ScienceRunner(config, dry_run=dry_run)
    return runner.execute(progress_callback=progress_callback)


__all__ = ["Cosmos2ScienceRunner", "run_science_run"]
