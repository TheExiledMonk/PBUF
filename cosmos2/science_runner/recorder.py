"""Recording helpers for cosmos2 science runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from cosmos2.science_runner.utils import ensure_dir, safe_write_json, sanitize_run_name, serialize_value


class RunRecorder:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        ensure_dir(self.base_dir)
        self.history_file = self.base_dir / "history.json"

    def prepare_run_directory(self, run_name: str, timestamp: str) -> Path:
        sanitized = sanitize_run_name(run_name)
        run_dir = self.base_dir / f"{timestamp}_{sanitized}"
        ensure_dir(run_dir)
        return run_dir

    def write_config(self, run_dir: Path, config_payload: dict[str, Any]) -> None:
        safe_write_json(run_dir / "config_used.json", config_payload)

    def write_meta(self, run_dir: Path, meta_payload: dict[str, Any]) -> None:
        safe_write_json(run_dir / "run_meta.json", meta_payload)

    def write_json(self, run_dir: Path, filename: str, obj: Any) -> None:
        ensure_dir(run_dir)
        safe_write_json(run_dir / filename, obj)

    def write_text(self, run_dir: Path, filename: str, text: str) -> None:
        ensure_dir(run_dir)
        (run_dir / filename).write_text(text, encoding="utf-8")

    def save_metadata(self, run_dir: Path, data: dict[str, Any]) -> None:
        self.write_json(run_dir, "metadata.json", data)

    def write_history_entry(self, run_dir: Path, entries: Sequence[dict[str, Any]]) -> None:
        safe_write_json(run_dir / "history_entry.json", entries)

    def append_history(self, entries: Sequence[dict[str, Any]]) -> None:
        history: list[dict[str, Any]] = []
        if self.history_file.exists():
            try:
                loaded = json.loads(self.history_file.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    history = loaded
            except Exception:
                history = []
        history.extend([serialize_value(entry) for entry in entries])
        safe_write_json(self.history_file, history)

    def record_model_results(
        self,
        model_dir: Path,
        *,
        best_params: dict[str, float],
        best_chi2: float,
        chi2_breakdown: dict[str, float],
        fit_outputs: dict[str, Any],
        predictions: dict[str, Any],
        engine_result: dict[str, Any] | None,
        profile_likelihood: dict[str, Any] | None,
        save_space: bool,
    ) -> None:
        ensure_dir(model_dir)
        best_fit_payload = {
            "parameters": best_params,
            "chi2": best_chi2,
            "predictions": serialize_value(predictions),
        }
        if engine_result:
            best_fit_payload["engine_result"] = serialize_value(engine_result)
        if fit_outputs:
            best_fit_payload["fit_outputs"] = fit_outputs
        safe_write_json(model_dir / "best_fit.json", best_fit_payload)

        breakdown_payload = {
            "fits": {fit: float(value) for fit, value in chi2_breakdown.items()},
            "total": sum(float(value) for value in chi2_breakdown.values()),
        }
        safe_write_json(model_dir / "chi2_breakdown.json", breakdown_payload)

        if engine_result and engine_result.get("trace") and not save_space:
            safe_write_json(model_dir / "parameters_trace.json", engine_result.get("trace"))

        if profile_likelihood:
            safe_write_json(model_dir / "profile_likelihood.json", profile_likelihood)

    def save_fit_output(
        self,
        model_dir: Path,
        fit_name: str,
        chi2: float,
        extras: Any,
    ) -> None:
        fits_dir = model_dir / "fits"
        ensure_dir(fits_dir)
        payload = {"chi2": float(chi2), "extras": serialize_value(extras)}
        safe_write_json(fits_dir / f"{fit_name}.json", payload)

    def save_engine_trace(
        self,
        model_dir: Path,
        engine_name: str,
        trace: Sequence[Any] | None,
        trace_meta: dict[str, Any],
        save_space: bool,
    ) -> None:
        ensure_dir(model_dir)
        payload: dict[str, Any] = {"engine": engine_name, "trace_meta": trace_meta}
        if not save_space and trace is not None:
            payload["trace"] = [serialize_value(entry) for entry in trace]
        safe_write_json(model_dir / "engine_trace.json", payload)

    def record_model_failure(self, model_dir: Path, reason: str) -> None:
        ensure_dir(model_dir)
        failure_payload = {
            "failure_reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        safe_write_json(model_dir / "failure.json", failure_payload)


__all__ = ["RunRecorder"]
