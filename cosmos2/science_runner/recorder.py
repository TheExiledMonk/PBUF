"""Recording helpers for cosmos2 science runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from cosmos2.science_runner.utils import ensure_dir, safe_write_json, sanitize_run_name, serialize_value
from cosmos2.science_runner.model_summary import create_model_summary
from cosmos2.science_runner.table_generator import ScientificTableGenerator


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
        parameter_snapshot: dict[str, Any] | None = None,
        profile_likelihood: dict[str, Any] | None,
        save_space: bool,
    ) -> None:
        ensure_dir(model_dir)
        params_to_write = dict(best_params)
        if model_dir.name.lower() == "pbuf":
            params_to_write.pop("Omega_b0", None)
        best_fit_payload = {
            "parameters": params_to_write,
            "chi2": best_chi2,
            "predictions": serialize_value(predictions),
        }
        n_params = len(params_to_write)
        n_points = self._count_fit_data_points(fit_outputs)
        if "aic" not in best_fit_payload:
            best_fit_payload["aic"] = best_chi2 + 2 * n_params
        if "bic" not in best_fit_payload and n_points > 0:
            best_fit_payload["bic"] = best_chi2 + n_params * math.log(max(1, n_points))
        if engine_result:
            best_fit_payload["engine_result"] = serialize_value(engine_result)
        if parameter_snapshot:
            best_fit_payload["parameter_snapshot"] = serialize_value(parameter_snapshot)
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

    def _count_fit_data_points(self, fit_outputs: dict[str, Any]) -> int:
        total = 0
        for payload in fit_outputs.values():
            extras = payload.get("extras", {})
            observed = extras.get("observed", [])
            if observed is None:
                continue
            try:
                total += len(observed)
            except TypeError:
                continue
        return total

    def save_model_summary(
        self,
        model_dir: Path,
        model_name: str,
        model: Any,
        best_params: dict[str, float],
        chi2_total: float,
        chi2_breakdown: dict[str, float],
        config: dict[str, Any],
        runtime_metadata: dict[str, Any] | None = None,
        fit_outputs: dict[str, Any] | None = None,
        engine_result: dict[str, Any] | None = None,
    ) -> None:
        """Save comprehensive model summary JSON with all required scientific information."""
        ensure_dir(model_dir)
        
        summary = create_model_summary(
            model_name=model_name,
            model=model,
            best_params=best_params,
            chi2_total=chi2_total,
            chi2_breakdown=chi2_breakdown,
            config=config,
            runtime_metadata=runtime_metadata,
            fit_outputs=fit_outputs,
            engine_result=engine_result,
        )
        
        safe_write_json(model_dir / "model_summary.json", summary)
        return summary

    def generate_paper_tables(self, run_dir: Path, model_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Path]]:
        """Generate publication-ready tables from all model summaries."""
        tables_dir = run_dir / "tables"
        generator = ScientificTableGenerator(tables_dir)
        
        saved_tables = {}
        
        # Generate Best-Fit Parameter Table
        if len(model_summaries) > 0:
            best_fit_table = generator.generate_best_fit_parameter_table(model_summaries)
            saved_files = generator.save_table(best_fit_table, "best_fit_parameters")
            saved_tables["best_fit_parameters"] = saved_files
            
            # Generate χ² Breakdown Table
            chi2_table = generator.generate_chi2_breakdown_table(model_summaries, include_details=True)
            saved_files = generator.save_table(chi2_table, "chi2_breakdown_detailed")
            saved_tables["chi2_breakdown"] = saved_files
            
            # Generate Full Data Summary Table
            datasets_used = []
            for summary in model_summaries.values():
                chi2_breakdown = summary.get("chi_squared", {}).get("per_dataset", {})
                datasets_used.extend(chi2_breakdown.keys())
            datasets_used = list(set(datasets_used))  # Remove duplicates
            
            data_summary_table = generator.generate_full_data_summary_table(datasets_used)
            saved_files = generator.save_table(data_summary_table, "full_data_summary")
            saved_tables["full_data_summary"] = saved_files
            
            # Generate Model Comparison Table
            model_comparison_table = generator.generate_model_comparison_table(model_summaries, baseline_model="lcdm")
            saved_files = generator.save_table(model_comparison_table, "model_comparison")
            saved_tables["model_comparison"] = saved_files
            
            # Generate Quantum Engine Input Table
            quantum_engine_table = generator.generate_quantum_engine_input_table(model_summaries)
            saved_files = generator.save_table(quantum_engine_table, "quantum_engine_input")
            saved_tables["quantum_engine_input"] = saved_files
            
            # Generate Quantum Engine Output Table (LUT sample)
            quantum_output_table = generator.generate_quantum_engine_output_table(model_summaries, n_samples=15)
            saved_files = generator.save_table(quantum_output_table, "quantum_engine_output")
            saved_tables["quantum_engine_output"] = saved_files
        
        return saved_tables


__all__ = ["RunRecorder"]
