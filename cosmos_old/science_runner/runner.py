"""Core orchestration logic for the configurable science runner."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from cosmos.fits.joint import build_joint_chi2_evaluator
from cosmos.fits.registry import FIT_REGISTRY

from .config import ScienceRunConfig
from .factories import make_engine, make_model_factory
from .plots import SciencePlotter
from .recorder import RunRecorder
from .reports import ReportGenerator
from .utils import hash_payload, serialize_value


class _RunHistoryEntry:
    def __init__(
        self,
        run_name: str,
        timestamp: str,
        model: str,
        best_chi2: float,
        fits_used: Sequence[str],
        engine: str,
        comment: str,
        success: bool = True,
        failure_reason: str | None = None,
    ) -> None:
        self.run_name = run_name
        self.timestamp = timestamp
        self.model = model
        self.best_chi2 = best_chi2
        self.fits_used = list(fits_used)
        self.engine = engine
        self.comment = comment
        self.success = success
        self.failure_reason = failure_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "timestamp": self.timestamp,
            "model": self.model,
            "best_chi2": self.best_chi2,
            "fits_used": self.fits_used,
            "engine": self.engine,
            "comment": self.comment,
            "success": self.success,
            "failure_reason": self.failure_reason,
        }


class ScienceRunner:
    def __init__(
        self,
        config: ScienceRunConfig,
        *,
        dry_run: bool = False,
        plotter: SciencePlotter | None = None,
        reporter: ReportGenerator | None = None,
        recorder: RunRecorder | None = None,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.plotter = plotter or SciencePlotter()
        self.reporter = reporter or ReportGenerator()
        self.recorder = recorder or RunRecorder(self.config.output.base_dir)
        self.chi2_history: list[dict[str, Any]] = []
        self._phase7a_summary: dict[str, int] = {"calls": 0, "passes": 0, "fails": 0}
        self._phase6a_summary = self._phase7a_summary
        self._pbuf_alpha: float | None = None

    def execute(self) -> None:
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

        if self.dry_run:
            print(f"[dry run] Created run directory {run_dir}")
            self.config.cleanup()
            return

        joint_config_path = self.config.get_joint_config_path()
        joint_hash = hash_payload(joint_payload)
        parameter_bounds_payload = self.config.parameter_bounds_payload
        parameter_bounds_hash = hash_payload(parameter_bounds_payload)
        dataset_manifest_hash = hash_payload(dataset_manifest)
        config_hash = self._hash_file(self.config.path)

        history_entries: list[Dict[str, Any]] = []
        model_failures: list[Dict[str, str]] = []
        run_success = True
        error_messages: list[str] = []
        self.chi2_history = []
        self._phase7a_summary = {"calls": 0, "passes": 0, "fails": 0}
        self._phase6a_summary = self._phase7a_summary
        start_wall = datetime.now(timezone.utc)
        start_time = time.monotonic()
        models = list(self.config.models)
        fits_per_model = len(self.config.fits_list)
        total_joint_fits = len(models) * fits_per_model
        joint_fit_bar: Optional[tqdm] = None
        if total_joint_fits > 0:
            joint_fit_bar = tqdm(total=total_joint_fits, desc="Joint-fit evaluations", unit="eval")
        try:
            for model_name in tqdm(models, desc="Science models", unit="model"):
                model_bounds = self.config.parameter_bounds_for_model(model_name)
                entry = self._run_model(
                    model_name,
                    run_dir,
                    joint_config_path,
                    model_bounds,
                    list(self.config.fits_list),
                    timestamp=timestamp,
                    joint_fit_bar=joint_fit_bar,
                )
                history_entries.append(entry.to_dict())
                if not entry.success:
                    run_success = False
                    reason = entry.failure_reason or "unknown failure"
                    model_failures.append({"model": model_name, "reason": reason})
                    error_messages.append(reason)
        except Exception as exc:  # pragma: no cover - bubble up after logging
            run_success = False
            error_messages.append(str(exc))
            raise
        finally:
            if joint_fit_bar is not None:
                joint_fit_bar.close()
            total_runtime = time.monotonic() - start_time
            end_wall = datetime.now(timezone.utc)
            error_message = "; ".join(error_messages) if error_messages else None
            run_meta = self._build_run_meta(
                timestamp=timestamp,
                start_time=start_wall.isoformat(),
                end_time=end_wall.isoformat(),
                runtime=total_runtime,
                fits_used=list(self.config.fits_list),
                joint_hash=joint_hash,
                parameter_bounds_hash=parameter_bounds_hash,
                dataset_manifest_hash=dataset_manifest_hash,
                config_hash=config_hash,
                success=run_success,
                error_message=error_message,
                model_failures=model_failures,
            )
            self.recorder.write_meta(run_dir, run_meta)
            if history_entries:
                self.recorder.write_history_entry(run_dir, history_entries)
                self.recorder.append_history(history_entries)
            self.recorder.write_json(run_dir, "chi2_history.json", self.chi2_history)
            self.config.cleanup()

    def _run_model(
        self,
        model_name: str,
        run_dir: Path,
        joint_config_path: Path,
        parameter_bounds: dict[str, tuple[float, float]],
        fits_used: Sequence[str],
        timestamp: str,
        joint_fit_bar: Optional[tqdm] = None,
    ) -> _RunHistoryEntry:
        model_dir = run_dir / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        model_key = model_name.strip().lower()
        model_factory = make_model_factory(model_name, datasets=fits_used)
        joint_chi2 = build_joint_chi2_evaluator(model_factory, joint_config_path)
        fixed = self.config.fixed_parameters
        derived_fixed = {"Omega_r0": 9e-5}
        if model_key == "pbuf":
            alpha = self._ensure_pbuf_alpha()
            if alpha is not None:
                derived_fixed["Omega_k0"] = alpha
        fixed = {**fixed, **derived_fixed}
        initial = self.config.initial_parameters
        derived_keys = {"Omega_r0"}
        if model_key == "pbuf":
            derived_keys.add("Omega_k0")
        free_bounds = {
            key: bounds
            for key, bounds in parameter_bounds.items()
            if key not in fixed and key not in derived_keys
        }

        if self.config.mode == "scout" or not free_bounds:
            best_params = self._assemble_initial_parameters(parameter_bounds, fixed, initial)
            best_chi2 = float(joint_chi2(best_params))
            engine_result: dict[str, Any] | None = None
        else:
            engine = make_engine(self.config.engine, self.config.engine_settings)
            objective = lambda free_params: float(
                joint_chi2({**fixed, **{k: float(v) for k, v in free_params.items()}})
            )
            phase_guard = self._phase7a(model_factory, fixed)
            kwargs = {
                "objective": objective,
                "bounds": free_bounds,
                "fixed_parameters": fixed,
                "initial_parameters": self._prepare_initial_free(free_bounds, initial),
                "phase7a": phase_guard,
                "phase6a": phase_guard,
            }
            sig = inspect.signature(engine.optimise)
            filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
            result = engine.optimise(**filtered)
            free_best = {
                key: float(value) for key, value in result.get("best_params", {}).items()
            }
            best_params = {**fixed, **free_best}
            best_chi2 = float(joint_chi2(best_params))
            engine_result = result
            trace = engine_result.get("trace")
            if isinstance(trace, list):
                sanitized: list[dict[str, Any]] = []
                for entry in trace:
                    if not isinstance(entry, dict):
                        continue
                    normalized = dict(fixed)
                    for key, value in entry.items():
                        if isinstance(value, (int, float)):
                            normalized[key] = float(value)
                            continue
                        if isinstance(value, str):
                            try:
                                normalized[key] = float(value)
                                continue
                            except ValueError:
                                pass
                        normalized[key] = value
                    sanitized.append(normalized)
                engine_result["trace"] = sanitized

        full_best_params = {k: float(v) for k, v in best_params.items()}
        if engine_result is not None:
            engine_result["best_params_full"] = full_best_params
            trace = engine_result.get("trace")
            trace_meta = {
                "iterations": len(trace) if isinstance(trace, Sequence) else 0,
                "final_step": serialize_value(trace[-1]) if isinstance(trace, Sequence) and trace else None,
                "converged": bool(engine_result.get("converged")),
            }
            self.recorder.save_engine_trace(
                model_dir=model_dir,
                engine_name=self.config.engine,
                trace=trace if isinstance(trace, Sequence) else None,
                trace_meta=trace_meta,
                save_space=self.config.output.save_space,
            )
        model = model_factory(full_best_params)
        prediction_data = self._derive_predictions(model)
        best_params = {k: float(v) for k, v in model.parameters.items()}
        derived_for_output = {"Omega_r0"}
        if model_key == "pbuf":
            derived_for_output.add("Omega_k0")
        best_params = {k: v for k, v in best_params.items() if k not in derived_for_output}
        success = True
        failure_reason: str | None = None
        chi2_breakdown: dict[str, float] = {}
        fit_results: dict[str, Any] = {}
        fit_iter = tqdm(list(fits_used), desc=f"{model_name} fits", unit="fit", leave=False)
        for fit_name in fit_iter:
            fit_fn = FIT_REGISTRY.get(fit_name)
            if fit_fn is None:
                if joint_fit_bar is not None:
                    joint_fit_bar.update(1)
                continue
            try:
                fit_result = fit_fn(model)
            except Exception as exc:
                failure_reason = f"Fit '{fit_name}' failed: {exc}"
                self.recorder.record_model_failure(model_dir, failure_reason)
                success = False
                break
            else:
                if isinstance(fit_result, tuple):
                    chi2_value = fit_result[0]
                    extras = fit_result[1] if len(fit_result) > 1 else {}
                else:
                    chi2_value = fit_result
                    extras = {}
                chi2_value = float(chi2_value)
                extras_payload = extras or {}
                chi2_breakdown[fit_name] = chi2_value
                fit_results[fit_name] = {"chi2": chi2_value, "extras": extras_payload}
                self.recorder.save_fit_output(model_dir, fit_name, chi2_value, extras_payload)
                self.chi2_history.append(
                    {
                        "model": model_name,
                        "fit": fit_name,
                        "chi2": chi2_value,
                        "extras": serialize_value(extras_payload),
                    }
                )
            finally:
                if joint_fit_bar is not None:
                    joint_fit_bar.update(1)

        if not success:
            return _RunHistoryEntry(
                run_name=self.config.run_name,
                timestamp=timestamp,
                model=model_name,
                best_chi2=best_chi2,
                fits_used=list(fits_used),
                engine=self.config.engine,
                comment=f"{self.config.mode} run",
                success=False,
                failure_reason=failure_reason,
            )

        profile_data = None
        if self.config.profile_likelihood:
            profile_data = self._compute_profile_likelihood(
                self.config.profile_likelihood,
                joint_chi2,
                best_params,
                parameter_bounds,
            )

        if self.config.output.generate_plots:
            self.plotter.generate(predictions=prediction_data, model_dir=model_dir)

        if self.config.output.generate_reports:
            self.reporter.generate(
                run_dir=run_dir,
                model_dir=model_dir,
                model_name=model_name,
                run_meta={
                    "run_name": self.config.run_name,
                    "timestamp": timestamp,
                    "engine": self.config.engine,
                    "mode": self.config.mode,
                    "fits_used": list(fits_used),
                },
                best_params=best_params,
                best_chi2=best_chi2,
                chi2_breakdown=chi2_breakdown,
                fit_outputs=fit_results,
                predictions=prediction_data,
                report_formats=self.config.output.report_formats,
            )

        self.recorder.record_model_results(
            model_dir,
            best_params=best_params,
            best_chi2=best_chi2,
            chi2_breakdown=chi2_breakdown,
            fit_outputs={
                name: {
                    "chi2": frag["chi2"],
                    "extras": serialize_value(frag.get("extras")),
                }
                for name, frag in fit_results.items()
            },
            predictions=prediction_data,
            engine_result=engine_result,
            profile_likelihood=profile_data,
            save_space=self.config.output.save_space,
        )

        return _RunHistoryEntry(
            run_name=self.config.run_name,
            timestamp=timestamp,
            model=model_name,
            best_chi2=best_chi2,
            fits_used=list(fits_used),
            engine=self.config.engine,
            comment=f"{self.config.mode} run",
        )

    def _assemble_initial_parameters(
        self,
        bounds: dict[str, tuple[float, float]],
        fixed: dict[str, float],
        initial: dict[str, float],
    ) -> dict[str, float]:
        keys = set(bounds) | set(fixed) | set(initial)
        params: dict[str, float] = {}
        for key in keys:
            if key in fixed:
                params[key] = fixed[key]
                continue
            if key in initial:
                params[key] = initial[key]
                continue
            if key in bounds:
                lower, upper = bounds[key]
                params[key] = float((lower + upper) / 2.0)
                continue
            params[key] = 0.0
        return {key: float(value) for key, value in params.items()}

    def _prepare_initial_free(
        self, free_bounds: dict[str, tuple[float, float]], initial: dict[str, float]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, (lower, upper) in free_bounds.items():
            if key in initial:
                result[key] = float(initial[key])
            else:
                result[key] = float((lower + upper) / 2.0)
        return result

    def _phase7a(self, model_factory: Any, fixed: dict[str, float]) -> Any:
        def inner(candidate: dict[str, float]) -> tuple[bool, str | None]:
            params = {**fixed, **{k: float(v) for k, v in candidate.items()}}
            model = model_factory(params)
            is_valid = bool(model.is_valid())
            self._phase7a_summary["calls"] += 1
            if is_valid:
                self._phase7a_summary["passes"] += 1
            else:
                self._phase7a_summary["fails"] += 1
            return (is_valid, None)

        return inner

    _phase6a = _phase7a
    def _derive_predictions(self, model: Any) -> dict[str, Any]:
        z = np.linspace(0.0, 2.0, 101, dtype=float)
        try:
            h = np.asarray(model.Hubble(z), dtype=float)
        except Exception:
            h = np.zeros_like(z)
        try:
            dm = np.asarray(model.DM(z), dtype=float)
        except Exception:
            dm = np.zeros_like(z)
        try:
            fs8 = np.asarray(model.fs8(z), dtype=float)
        except Exception:
            fs8 = np.zeros_like(z)
        try:
            s8 = float(model.S8())
        except Exception:
            s8 = float(model.sigma8()) if hasattr(model, "sigma8") else 0.0
        try:
            omega_m0 = float(model.omega_m0())
        except Exception:
            omega_m0 = float(model.parameters.get("Omega_m0", 0.0))
        try:
            sigma8 = float(model.sigma8())
        except Exception:
            sigma8 = 0.0
        try:
            rd = float(model.sound_horizon())
        except Exception:
            rd = 0.0
        predictions = {
            "H0": float(model.parameters.get("H0", 0.0)),
            "Omega_m0": omega_m0,
            "Omega_k0": float(model.parameters.get("Omega_k0", 0.0)),
            "S8": s8,
            "sigma8": sigma8,
            "r_d": rd,
            "plot_data": {
                "z": z.tolist(),
                "H_z": h.tolist(),
                "DM_z": dm.tolist(),
                "fs8_z": fs8.tolist(),
            },
        }
        return predictions

    def _ensure_pbuf_alpha(self) -> float | None:
        if self._pbuf_alpha is not None:
            return self._pbuf_alpha
        try:
            from cosmos.models.pbuf.microphysics import ensure_thermal_table, get_last_bootstrap_metadata
        except ImportError:
            return None
        ensure_thermal_table()
        metadata = get_last_bootstrap_metadata()
        if not metadata:
            return None
        alpha = metadata.get("alpha_qm") or metadata.get("alpha")
        if alpha is None:
            return None
        self._pbuf_alpha = float(alpha)
        return self._pbuf_alpha

    def _compute_profile_likelihood(
        self,
        profile: dict[str, Any],
        chi2_fn: Any,
        best_params: dict[str, float],
        bounds: dict[str, tuple[float, float]],
    ) -> dict[str, Any] | None:
        parameters = profile.get("parameters") or []
        if not parameters:
            return None
        resolution = max(5, min(int(profile.get("resolution", 20)), 80))
        if len(parameters) == 1:
            name = parameters[0]
            if name not in bounds:
                return None
            lower, upper = bounds[name]
            if lower == upper:
                values = [lower]
            else:
                values = np.linspace(lower, upper, resolution, dtype=float)
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
            x_count = min(resolution, 40)
            y_count = min(resolution, 40)
            x_values = np.linspace(x_lower, x_upper, x_count, dtype=float)
            y_values = np.linspace(y_lower, y_upper, y_count, dtype=float)
            grid: list[dict[str, Any]] = []
            for x_val in x_values:
                for y_val in y_values:
                    candidate = {**best_params, x_name: float(x_val), y_name: float(y_val)}
                    grid.append({"x": float(x_val), "y": float(y_val), "chi2": float(chi2_fn(candidate))})
            return {
                "type": "2d",
                "x_parameter": x_name,
                "y_parameter": y_name,
                "grid": grid,
            }
        return None

    def _hash_file(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_run_meta(
        self,
        *,
        timestamp: str,
        start_time: str,
        end_time: str,
        runtime: float,
        fits_used: Sequence[str],
        joint_hash: str,
        parameter_bounds_hash: str,
        dataset_manifest_hash: str,
        config_hash: str,
        success: bool,
        error_message: str | None,
        model_failures: Sequence[dict[str, str]],
    ) -> dict[str, Any]:
        git_commit = self._git_commit()
        cosmos_version = self._git_version()
        quantum_version, lut_version = self._quantum_metadata()
        machine_info = {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "cpus": os.cpu_count(),
        }
        return {
            "run_name": self.config.run_name,
            "timestamp": timestamp,
            "models": self.config.models,
            "mode": self.config.mode,
            "engine": self.config.engine,
            "engine_settings": self.config.engine_settings,
            "fits_used": list(fits_used),
            "parameter_bounds_hash": parameter_bounds_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "config_hash": config_hash,
            "joint_config_hash": joint_hash,
            "git_commit": git_commit,
            "cosmos_version": cosmos_version,
            "quantum_version": quantum_version,
            "lut_version": lut_version,
            "machine": machine_info,
            "success": success,
            "error_message": error_message,
            "start_time": start_time,
            "end_time": end_time,
            "model_failures": list(model_failures),
            "chi2_history_entries": len(self.chi2_history),
            "phase7a_summary": dict(self._phase7a_summary),
            "phase6a_summary": dict(self._phase7a_summary),
            "total_runtime": runtime,
        }

    def _git_commit(self) -> str | None:
        try:
            output = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2])
            return output.decode("utf-8").strip()
        except Exception:
            return None

    def _git_version(self) -> str | None:
        try:
            output = subprocess.check_output(
                ["git", "describe", "--tags", "--dirty", "--always"],
                cwd=Path(__file__).resolve().parents[2],
            )
            return output.decode("utf-8").strip()
        except Exception:
            return None

    def _quantum_metadata(self) -> tuple[str | None, str | None]:
        version = None
        try:
            from quantum.e0 import __version__ as qv

            version = qv
        except Exception:
            version = None
        lut = None
        try:
            root = Path(__file__).resolve().parents[2]
            config_path = root / "configs" / "quantum" / "config.json"
            if config_path.exists():
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                table_version = payload.get("table_version")
                method_version = payload.get("method_version")
                if table_version is not None or method_version is not None:
                    lut = f"table:{table_version},method:{method_version}"
        except Exception:
            lut = None
        return version, lut
