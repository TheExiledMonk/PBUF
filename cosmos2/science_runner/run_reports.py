"""High-level report generator that summarizes everything produced by a science run."""

from __future__ import annotations

from html import escape as _html_escape
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence, Mapping

from cosmos2.data.registry import get_dataset

from cosmos2.science_runner.utils import ensure_dir, load_json_or_yaml, safe_write_json, serialize_value


class ScienceRunReportGenerator:
    def __init__(
        self,
        run_dir: Path,
        *,
        output_dir: Path | None = None,
        formats: Sequence[str] = ("json", "html"),
    ) -> None:
        self.run_dir = run_dir
        self._user_output_dir = output_dir is not None
        self.output_dir = output_dir or run_dir / "science_reports"
        self.formats = {fmt.strip().lower() for fmt in formats if fmt.strip()}
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: E402

            self._plt = plt
        except ImportError:
            self._plt = None
        self._plot_dir = self.output_dir / "plots"
        self._thermal_plot_cache: dict[str, Path] = {}
        self._thermal_plot_path: Path | None = None
        self._thermal_metadata: dict[str, Any] | None = None

    def generate(self) -> dict[str, Any]:
        self._maybe_resolve_run_dir()
        self._build_thermal_plot()
        summary = self._collect_summary()
        self._dump_summary(summary)
        return summary

    def _maybe_resolve_run_dir(self) -> None:
        """If run_dir lacks run_meta.json, descend into a single timestamped child that has it."""

        meta_path = self.run_dir / "run_meta.json"
        if meta_path.exists():
            return
        candidates = []
        for child in self.run_dir.iterdir():
            if child.is_dir() and (child / "run_meta.json").exists():
                candidates.append(child)
        if not candidates:
            return
        # Prefer the most recently modified candidate.
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        self.run_dir = candidates[0]
        if not self._user_output_dir:
            self.output_dir = self.run_dir / "science_reports"
        self._plot_dir = self.output_dir / "plots"

    def _build_thermal_plot(self) -> None:
        """Regenerate a temperature-vs-a plot from the cached PBUF thermal table."""

        if self._plt is None:
            return
        try:
            from cosmos2.pbuf.microphysics import THERMAL_CACHE_PATH, ensure_thermal_table
            if not THERMAL_CACHE_PATH.exists():
                return
            table = ensure_thermal_table()
            a_vals = table.a
            T_vals = table.T
            self._thermal_metadata = table.metadata_summary()
            ensure_dir(self._plot_dir)
            fig, ax = self._plt.subplots(figsize=(6, 4))
            ax.semilogy(a_vals, T_vals, label="T(a)")
            ax.set_xlabel("scale factor a")
            ax.set_ylabel("Temperature [K]")
            ax.grid(True, which="both", ls=":")
            ax.legend()
            path = self._plot_dir / "thermal_table_T.png"
            fig.tight_layout()
            fig.savefig(path)
            self._plt.close(fig)
            self._thermal_plot_path = path
        except Exception:
            self._thermal_plot_path = None
            self._thermal_metadata = self._thermal_metadata or None

    def _collect_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "run_meta": self._load_json("run_meta.json"),
            "joint_config": self._load_json("joint_config_used.json"),
            "datasets": self._load_json("datasets_used.json"),
            "engine_settings": self._load_json("engine_settings.json"),
            "config_used": self._load_json("config_used.json"),
            "history_entry": self._load_json("history_entry.json"),
            "chi2_history": self._load_json("chi2_history.json"),
            "models": [self._collect_model_summary(path) for path in self._model_directories()],
        }
        datasets_obj = summary.get("datasets") or {}
        fits_list = datasets_obj.get("fits") if isinstance(datasets_obj, Mapping) else []
        summary["num_data_points"] = self._count_total_data_points(fits_list or [])
        if self._thermal_metadata:
            summary["thermal_metadata"] = self._thermal_metadata
        if self._thermal_plot_path:
            summary["thermal_plot"] = str(self._thermal_plot_path)
        summary["engine_quantum"] = self._build_engine_quantum_info(summary)
        summary["model_metrics"] = self._compute_model_metrics(summary)
        return summary

    def _model_directories(self) -> list[Path]:
        return [
            child
            for child in sorted(self.run_dir.iterdir())
            if child.is_dir() and (child / "best_fit.json").exists()
        ]

    def _collect_model_summary(self, model_dir: Path) -> dict[str, Any]:
        best_fit = self._load_json(model_dir / "best_fit.json")
        breakdown = self._load_json(model_dir / "chi2_breakdown.json")
        report = self._load_json(model_dir / "reports" / "summary.json")
        fits_dir = model_dir / "fits"
        fits = {}
        if fits_dir.exists():
            for fit_file in sorted(fits_dir.glob("*.json")):
                data = self._load_json(fit_file)
                if data is not None:
                    fits[fit_file.stem] = data
        predictions = best_fit.get("predictions") if isinstance(best_fit, dict) else None
        plot_paths = self._create_model_plots(model_dir.name, predictions or {}, fits)
        if model_dir.name.strip().lower() == "pbuf" and self._thermal_plot_path is not None:
            plot_paths.append({"label": "Thermal table T(a)", "path": self._relative_path(self._thermal_plot_path)})

        return {
            "name": model_dir.name,
            "best_fit": best_fit,
            "chi2_breakdown": breakdown,
            "fits": fits,
            "existing_report": report,
            "predictions": predictions,
            "plots": plot_paths,
        }

    def _count_total_data_points(self, dataset_names: Sequence[str]) -> int:
        total = 0
        seen: set[str] = set()
        for name in dataset_names:
            normalized = str(name).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            dataset = self._load_dataset_by_name(normalized)
            total += self._dataset_length(dataset)
        return total

    def _load_dataset_by_name(self, name: str) -> Any:
        try:
            return get_dataset(name)
        except Exception:
            return None

    def _dataset_length(self, dataset: Any) -> int:
        candidate = None
        if dataset is None:
            return 0
        if hasattr(dataset, "observed"):
            candidate = getattr(dataset, "observed")
        elif isinstance(dataset, dict):
            for key in ("obs", "observed", "data", "y"):
                if key in dataset:
                    candidate = dataset[key]
                    break
            if candidate is None:
                for alt in ("H_obs", "fs8", "fsigma8", "fs8_obs"):
                    if alt in dataset:
                        candidate = dataset[alt]
                        break
        if candidate is None and hasattr(dataset, "get"):
            candidate = dataset.get("obs") or dataset.get("observed")
        if candidate is None:
            return 0
        if isinstance(candidate, (str, bytes)):
            return 0
        try:
            return len(candidate)
        except Exception:
            return 0

    def _compute_model_metrics(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        models = summary.get("models") or []
        n_data = max(1, int(summary.get("num_data_points") or 0))
        metrics: list[dict[str, Any]] = []
        for model in models:
            name = model.get("name") or "unknown"
            best_fit = model.get("best_fit") or {}
            parameters = best_fit.get("parameters") or {}
            k = len(parameters)
            try:
                chi2 = float(best_fit.get("chi2", math.inf))
            except Exception:
                chi2 = math.inf
            aic = chi2 + 2.0 * k
            bic = chi2 + k * math.log(n_data)
            metrics.append(
                {
                    "model": name,
                    "best_chi2": chi2,
                    "k": k,
                    "aic": aic,
                    "bic": bic,
                }
            )
        if not metrics:
            return metrics
        min_chi2 = min(entry["best_chi2"] for entry in metrics)
        min_aic = min(entry["aic"] for entry in metrics)
        min_bic = min(entry["bic"] for entry in metrics)
        for entry in metrics:
            entry["delta_chi2"] = entry["best_chi2"] - min_chi2
            entry["delta_aic"] = entry["aic"] - min_aic
            entry["delta_bic"] = entry["bic"] - min_bic
        return metrics

    def _load_json(self, relative: Path | str) -> Any:
        target = Path(relative) if isinstance(relative, Path) else Path(self.run_dir / relative)
        try:
            return load_json_or_yaml(target)
        except Exception:
            return None

    def _create_model_plots(
        self,
        model_name: str,
        predictions: dict[str, Any],
        fits: dict[str, Any],
    ) -> list[dict[str, str]]:
        plots: list[dict[str, str]] = []
        if self._plt is None:
            return plots
        plot_data = predictions.get("plot_data") if isinstance(predictions, dict) else None
        if plot_data:
            z = plot_data.get("z")
            h = plot_data.get("H_z")
            dm = plot_data.get("DM_z")
            fs8 = plot_data.get("fs8_z")
            if z and h and dm and fs8:
                ensure_dir(self._plot_dir)
                fig, axes = self._plt.subplots(nrows=3, sharex=True, figsize=(6, 8))
                axes[0].plot(z, h, label="H(z)")
                axes[0].set_ylabel("H (km/s/Mpc)")
                axes[0].legend()
                axes[1].plot(z, dm, label="D_M(z)", color="tab:orange")
                axes[1].set_ylabel("D_M (Mpc)")
                axes[1].legend()
                axes[2].plot(z, fs8, label="fσ₈(z)", color="tab:green")
                axes[2].set_ylabel("fσ₈")
                axes[2].set_xlabel("z")
                axes[2].legend()
                fig.suptitle(f"{model_name} predictions")
                filename = self._plot_dir / f"{model_name}_predictions.png"
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(filename)
                self._plt.close(fig)
                plots.append({"label": f"{model_name} prediction curves", "path": self._relative_path(filename)})
        thermal_plot = self._maybe_generate_thermal_plot(model_name, fits)
        if thermal_plot:
            plots.extend(thermal_plot)
        return plots

    def _build_engine_quantum_info(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Collect engine/quantum metadata for the report header."""

        info: dict[str, Any] = {}
        run_meta = summary.get("run_meta") or {}
        engine_settings = summary.get("engine_settings") or {}

        if isinstance(run_meta, Mapping):
            info["engine_name"] = run_meta.get("engine") or engine_settings.get("engine")
        elif isinstance(engine_settings, Mapping):
            info["engine_name"] = engine_settings.get("engine")

        meta = self._thermal_metadata or {}
        meta_block = meta.get("metadata") if isinstance(meta, Mapping) else {}
        if isinstance(meta, Mapping):
            info["quantum_version"] = meta_block.get("micro_source") or meta.get("micro_source")
            table_version = meta_block.get("table_version") or meta.get("table_version")
            method_version = meta_block.get("method_version") or meta.get("method_version")
            if table_version is not None and method_version is not None:
                info["lut_version"] = f"v{table_version} (method {method_version})"
            elif table_version is not None:
                info["lut_version"] = f"v{table_version}"

        commit = self._git_commit()
        if commit:
            info["cosmos_commit"] = commit
        return info

    def _git_commit(self) -> str | None:
        try:
            repo_root = Path(__file__).resolve().parents[2]
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    def _maybe_generate_thermal_plot(
        self, model_name: str, fits: Mapping[str, Any]
    ) -> list[dict[str, str]]:
        if not fits or self._plt is None:
            return []
        metadata = self._locate_thermal_metadata(fits)
        if metadata is None:
            return []
        signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        if signature in self._thermal_plot_cache:
            cached = self._thermal_plot_cache[signature]
            if cached.exists():
                return [{"label": "thermal lookup", "path": self._relative_path(cached)}]
        json_path = self.output_dir / "thermal" / f"thermal_{signature}.json"
        ensure_dir(json_path.parent)
        if not json_path.exists():
            if not self._run_export_command(metadata, json_path):
                return []
        image_path = self._render_thermal_plot_from_json(model_name, json_path, signature)
        if image_path:
            self._thermal_plot_cache[signature] = image_path
            return [{"label": "thermal lookup", "path": self._relative_path(image_path)}]
        return []

    def _run_export_command(self, metadata: Mapping[str, Any], output_path: Path) -> bool:
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "quantum" / "tools" / "export_thermal_table.py"
        if not script.exists():
            return False
        python_bin = (repo_root / ".venv" / "bin" / "python")
        if not python_bin.exists():
            python_bin = Path(sys.executable)
        args = [str(python_bin), str(script)]

        def _add_arg(option: str, key_names: tuple[str, ...]) -> None:
            for key in key_names:
                if key in metadata:
                    value = metadata[key]
                    if value is not None:
                        args.extend([option, str(value)])
                        return

        _add_arg("--mode", ("mode",))
        _add_arg("--beta", ("beta",))
        _add_arg("--t-star", ("t_star",))
        _add_arg("--power", ("power",))
        _add_arg("--alpha-qm", ("alpha_qm", "alpha_QM"))
        _add_arg("--eps-min", ("eps_min", "eps_min_T"))
        _add_arg("--t-min", ("t_min",))
        _add_arg("--t-max", ("t_max",))
        _add_arg("--points", ("num_points", "points"))
        _add_arg("--dense-points", ("dense_points",))
        _add_arg("--table-version", ("table_version",))
        _add_arg("--method-version", ("method_version",))
        _add_arg("--regulator", ("regulator",))
        _add_arg("--field-content", ("field_content",))
        _add_arg("--f-cut", ("f_cut_T", "f_cut"))
        _add_arg("--f-coup", ("f_coup_T", "f_coup"))
        _add_arg("--notes", ("notes",))

        args.extend(["--output", str(output_path), "--overwrite"])

        try:
            subprocess.run(args, cwd=repo_root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False

    def _render_thermal_plot_from_json(self, model_name: str, json_path: Path, signature: str) -> Path | None:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        rows = payload.get("rows") or []
        if not rows:
            return None
        temperatures: list[float] = []
        epsilons: list[float] = []
        alphas: list[float] = []
        for entry in rows:
            try:
                temp = float(entry.get("T_K", entry.get("temperature", 0.0)))
            except Exception:
                temp = 0.0
            try:
                eps = float(entry.get("epsilon0_T", entry.get("epsilon0", 0.0)))
            except Exception:
                eps = 0.0
            try:
                alpha = float(entry.get("alpha_T", entry.get("alpha", 0.0)))
            except Exception:
                alpha = 0.0
            temperatures.append(temp)
            epsilons.append(eps)
            alphas.append(alpha)
        if not temperatures:
            return None
        ensure_dir(self._plot_dir)
        fig, ax = self._plt.subplots(figsize=(6, 4))
        ax.plot(temperatures, epsilons, label="ε₀(T)", color="tab:purple")
        ax.plot(temperatures, alphas, label="α(T)", color="tab:teal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Lookup")
        ax.set_title(f"{model_name} thermal lookup")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        image_path = self._plot_dir / f"thermal_lookup_{signature}.png"
        fig.savefig(image_path)
        self._plt.close(fig)
        return image_path

    def _locate_thermal_metadata(self, payload: Mapping[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, Mapping):
            return None

        if "thermal_metadata" in payload:
            meta = payload["thermal_metadata"]
            if isinstance(meta, Mapping):
                return meta.get("metadata", meta)

        for value in payload.values():
            if isinstance(value, Mapping):
                found = self._locate_thermal_metadata(value)
                if found:
                    return found
        return None

    def _relative_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.output_dir))
        except Exception:
            return str(path)

    def _dump_summary(self, summary: dict[str, Any]) -> None:
        ensure_dir(self.output_dir)
        if "json" in self.formats:
            safe_write_json(self.output_dir / "run_summary.json", summary)
        if "html" in self.formats:
            (self.output_dir / "run_summary.html").write_text(
                self._render_html(summary),
                encoding="utf-8",
            )

    def _render_html(self, summary: dict[str, Any]) -> str:
        meta = summary.get("run_meta") or {}
        dataset_manifest = summary.get("datasets") or {}
        engine_settings = summary.get("engine_settings") or {}
        config_used = summary.get("config_used") or {}
        models = summary.get("models") or []
        metrics = summary.get("model_metrics") or []
        fits_used = dataset_manifest.get("fits") or []
        fit_weights = dataset_manifest.get("fit_weights") or {}
        machine = meta.get("machine") or {}
        phase7a = meta.get("phase7a_summary") or {}
        phase6a = meta.get("phase6a_summary") or {}
        engine_quantum = summary.get("engine_quantum") or {}

        def _format_value(value: Any, precision: int = 4) -> str:
            if value is None:
                return "—"
            if isinstance(value, bool):
                return "Yes" if value else "No"
            if isinstance(value, int):
                return str(value)
            if isinstance(value, float):
                return f"{value:.{precision}g}"
            if isinstance(value, (list, tuple)):
                return ", ".join(_format_value(item, precision) for item in value)
            if isinstance(value, dict):
                try:
                    return json.dumps(serialize_value(value), indent=2)
                except Exception:
                    return str(value)
            return str(value)

        def _format_duration(value: Any) -> str:
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                return "—"
            if seconds < 0:
                return "—"
            hours = int(seconds) // 3600
            minutes = (int(seconds) % 3600) // 60
            secs = int(seconds) % 60
            if hours:
                return f"{hours}h {minutes}m"
            if minutes:
                return f"{minutes}m {secs}s"
            return f"{secs}s"

        def _render_stats_grid(stats: Sequence[tuple[str, str]]) -> str:
            cells = []
            for label, value in stats:
                cells.append(
                    "<div class='stat-card'>"
                    f"<div class='stat-label'>{_html_escape(label)}</div>"
                    f"<div class='stat-value'>{_html_escape(value)}</div>"
                    "</div>"
                )
            return "<div class='stats-grid'>" + "".join(cells) + "</div>"

        def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], table_class: str = "details") -> str:
            body = []
            for row in rows:
                cols = "".join(f"<td>{_html_escape('' if col is None else str(col))}</td>" for col in row)
                body.append(f"<tr>{cols}</tr>")
            header_cells = "".join(f"<th>{_html_escape('' if col is None else str(col))}</th>" for col in headers)
            return (
                f"<table class='{table_class}'>"
                f"<thead><tr>{header_cells}</tr></thead>"
                "<tbody>"
                + "".join(body)
                + "</tbody></table>"
            )

        def _render_fits_table(fit_details: dict[str, Any]) -> str:
            if not fit_details:
                return "<p>No fit outputs recorded.</p>"
            rows = []
            for fit_name, fit_payload in sorted(fit_details.items()):
                chi2 = _format_value(fit_payload.get("chi2"))
                extras = fit_payload.get("extras")
                if extras:
                    pretty = _html_escape(json.dumps(serialize_value(extras), indent=2))
                    extras_html = (
                        "<details>"
                        f"<summary>Extras</summary>"
                        f"<pre>{pretty}</pre>"
                        "</details>"
                    )
                else:
                    extras_html = "—"
                rows.append(
                    "<tr>"
                    f"<td>{_html_escape(fit_name)}</td>"
                    f"<td>{_html_escape(chi2)}</td>"
                    f"<td>{extras_html}</td>"
                    "</tr>"
                )
            return (
                "<table class='details'>"
                "<thead><tr><th>Fit</th><th>χ²</th><th>Extras</th></tr></thead>"
                "<tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )

        def _render_phase_card(label: str, payload: dict[str, Any]) -> str:
            if not payload:
                return ""
            return (
                "<div class='phase-card'>"
                f"<div class='phase-label'>{_html_escape(label)}</div>"
                f"<div class='phase-value'>{_html_escape(str(payload.get('calls') or 0))} calls</div>"
                "<div class='phase-meta'>"
                f"<span>passes {payload.get('passes', 0)}</span>"
                f"<span>fails {payload.get('fails', 0)}</span>"
                "</div>"
                "</div>"
            )

        status_badge = "success" if meta.get("success") else "failure"
        status_text = "Success" if meta.get("success") else "Failure"

        hero_stats = [
            ("Timestamp", meta.get("timestamp") or meta.get("start_time") or "—"),
            ("Mode", meta.get("mode") or "—"),
            ("Engine", meta.get("engine") or "—"),
            ("Fits", f"{len(fits_used)} ({', '.join(fits_used)})" if fits_used else "—"),
            ("Run status", status_text),
            ("LUT version", meta.get("lut_version") or "—"),
        ]

        summary_cards = [
            ("Duration", _format_duration(meta.get("total_runtime"))),
            ("Data points", _format_value(summary.get("num_data_points") or 0)),
            ("Models", str(len(models))),
            ("Threads", str(engine_settings.get("n_threads") or "—")),
            ("Phase-7a calls", str(phase7a.get("calls") or 0)),
            ("Phase-6a calls", str(phase6a.get("calls") or 0)),
        ]

        lines = [
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
            "<title>Science Run Report</title>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            "<style>",
            """
            :root {
                color-scheme: light;
                font-family: "Inter", "Segoe UI", system-ui, sans-serif;
            }
            body {
                margin: 0;
                background: #f4f6fb;
                color: #0f172a;
            }
            header.hero {
                padding: 2.5rem 1.5rem 2rem;
                background: linear-gradient(135deg, #020617, #0f172a);
                color: #f8fafc;
            }
            .hero h1 {
                margin: 0;
                font-size: clamp(1.9rem, 3vw, 2.6rem);
            }
            .hero p {
                margin: 0.25rem 0;
                color: rgba(248, 250, 252, 0.8);
            }
            .eyebrow {
                letter-spacing: 0.4em;
                text-transform: uppercase;
                font-size: 0.75rem;
                color: #94a3b8;
            }
            .hero-grid {
                margin-top: 1.5rem;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1rem;
            }
            .hero-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 0.8rem;
                padding: 0.9rem 1rem;
                border: 1px solid rgba(255, 255, 255, 0.15);
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.4);
            }
            .hero-card strong {
                display: block;
                font-size: 1rem;
                margin-bottom: 0.2rem;
                color: #cbd5f5;
            }
            .hero-card span {
                font-size: 0.9rem;
            }
            main.content {
                max-width: 1200px;
                margin: -2rem auto 3rem;
                padding: 0 1.5rem 2.5rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            .panel {
                background: #fff;
                border-radius: 1.2rem;
                padding: 1.5rem 1.75rem;
                box-shadow: 0 20px 45px rgba(15, 23, 42, 0.08);
                border: 1px solid #e2e8f0;
            }
            .panel h2 {
                margin-top: 0;
                font-size: 1.3rem;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .stat-card {
                border-radius: 0.8rem;
                padding: 0.9rem 1rem;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
            }
            .stat-label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #64748b;
            }
            .stat-value {
                margin-top: 0.25rem;
                font-weight: 600;
                font-size: 1.1rem;
                color: #0f172a;
            }
            .summary-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 0.5rem;
            }
            .summary-table th,
            .summary-table td {
                padding: 0.5rem 0.75rem;
                border-bottom: 1px solid #e2e8f0;
                text-align: left;
                font-size: 0.95rem;
            }
            .summary-table th {
                background: #f1f5f9;
                font-weight: 600;
            }
            .details {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.95rem;
                margin-top: 1rem;
            }
            .details th,
            .details td {
                border-bottom: 1px solid #e2e8f0;
                padding: 0.5rem 0.75rem;
            }
            .details th {
                text-transform: uppercase;
                font-size: 0.75rem;
                letter-spacing: 0.1em;
            }
            .phase-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .phase-card {
                padding: 1rem;
                border: 1px solid #e2e8f0;
                border-radius: 0.8rem;
                background: #f8fafc;
            }
            .phase-label {
                font-size: 0.85rem;
                color: #475569;
            }
            .phase-value {
                font-size: 1.2rem;
                font-weight: 600;
                margin: 0.2rem 0;
            }
            .phase-meta {
                display: flex;
                gap: 0.6rem;
                font-size: 0.8rem;
                color: #64748b;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1rem;
            }
            .grid-item {
                padding: 0.5rem 0;
            }
            .grid-item label {
                font-size: 0.75rem;
                color: #475569;
                display: block;
            }
            .grid-item span {
                font-weight: 600;
            }
            .model-panel {
                border-radius: 1rem;
                padding-bottom: 1.5rem;
            }
            .model-panel h3 {
                margin-bottom: 0.5rem;
            }
            .model-meta {
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
                font-size: 0.9rem;
                color: #475569;
            }
            .plot-column {
                display: flex;
                flex-direction: column;
                gap: 1rem;
                margin-top: 1rem;
            }
            .plot-figure {
                margin: 0;
                border: 1px solid #e2e8f0;
                border-radius: 0.8rem;
                padding: 0.5rem;
                background: #f8fafc;
            }
            .plot-image {
                width: 100%;
                display: block;
                border-radius: 0.6rem;
            }
            .plot-figure figcaption {
                text-align: center;
                margin-top: 0.4rem;
                font-size: 0.9rem;
                color: #475569;
            }
            details summary {
                cursor: pointer;
                font-weight: 500;
            }
            details pre {
                background: #0f172a;
                color: #f1f5f9;
                padding: 0.7rem;
                border-radius: 0.6rem;
                overflow: auto;
                max-height: 320px;
                margin-top: 0.5rem;
            }
            .status-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.3rem 0.75rem;
                border-radius: 999px;
                font-size: 0.85rem;
                font-weight: 600;
            }
            .status-chip.success {
                background: rgba(34, 197, 94, 0.1);
                color: #059669;
                border: 1px solid rgba(34, 197, 94, 0.3);
            }
            .status-chip.failure {
                background: rgba(239, 68, 68, 0.18);
                color: #b91c1c;
                border: 1px solid rgba(239, 68, 68, 0.4);
            }
            .note {
                font-size: 0.9rem;
                color: #475569;
                margin-top: 0.75rem;
            }
            .model-extra {
                margin-top: 0.5rem;
                font-size: 0.9rem;
                color: #475569;
            }
            @media (max-width: 640px) {
                .hero-grid {
                    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                }
                .details th,
                .details td,
                .summary-table th,
                .summary-table td {
                    font-size: 0.8rem;
                }
            }
            """,
            "</style></head><body>",
            "<header class='hero'>",
            "<p class='eyebrow'>Science Run Overview</p>",
            f"<h1>{_html_escape(meta.get('run_name') or 'Untitled run')}</h1>",
            f"<p>{_html_escape(meta.get('description') or 'Aggregated performance summary')}</p>",
            "<div class='hero-grid'>",
        ]

        for label, value in hero_stats:
            lines.append(
                "<div class='hero-card'>"
                f"<strong>{_html_escape(label)}</strong>"
                f"<span>{_html_escape(str(value or '—'))}</span>"
                "</div>"
            )

        lines.extend(
            [
                "</div>",
                f"<div class='status-chip {status_badge}'>{_html_escape(status_text)}</div>",
                "</header>",
                "<main class='content'>",
                "<section class='panel'>",
                "<h2>Run at a glance</h2>",
                _render_stats_grid(summary_cards),
                "<div class='grid'>",
                "<div class='grid-item'>",
                "<label>Machine</label>",
                "<span>"
                f"{_html_escape(machine.get('node') or '—')} ({_html_escape(machine.get('system') or '—')})"
                "</span>",
                "<p class='note'>CPUs: "
                f"{_html_escape(str(machine.get('cpus') or '—'))} · Python { _html_escape(machine.get('python') or '—') }</p>",
                "</div>",
                "<div class='grid-item'>",
                "<label>Timeline</label>",
                "<span>"
                f"{_html_escape(str(meta.get('start_time') or '—'))}"
                "</span>",
                "<p class='note'>Ended: "
                f"{_html_escape(str(meta.get('end_time') or '—'))}</p>",
                "</div>",
                "<div class='grid-item'>",
                "<label>Datasets in play</label>",
                "<span>"
                f"{_html_escape(', '.join(fits_used) or '—')}"
                "</span>",
                "<p class='note'>Total data points: "
                f"{_html_escape(str(summary.get('num_data_points') or 0))}</p>",
                "</div>",
                "</div>",
                "<div class='grid'>",
                _render_phase_card("Phase 7a", phase7a),
                _render_phase_card("Phase 6a", phase6a),
                "<div class='grid-item'>",
                "<label>Joint config hash</label>",
                "<span>"
                f"{_html_escape(meta.get('joint_config_hash') or '—')}"
                "</span>",
                "</div>",
                "</div>",
                "</section>",
            ]
        )

        if metrics:
            lines.extend(
                [
                    "<section class='panel'>",
                    "<h2>Model comparison</h2>",
                    "<table class='summary-table'>",
                    "<thead><tr><th>Model</th><th>χ²</th><th>Δχ²</th><th>AIC</th><th>ΔAIC</th><th>BIC</th><th>ΔBIC</th></tr></thead>",
                    "<tbody>",
                ]
            )
            min_delta = min((entry.get("delta_chi2") or 0) for entry in metrics) if metrics else None
            for entry in metrics:
                highlight = " class='highlight-row'" if entry.get("delta_chi2") == min_delta else ""
                lines.append(
                    "<tr" + highlight + ">"
                    f"<td>{_html_escape(entry.get('model') or 'unknown')}</td>"
                    f"<td>{_format_value(entry.get('best_chi2'))}</td>"
                    f"<td>{_format_value(entry.get('delta_chi2'))}</td>"
                    f"<td>{_format_value(entry.get('aic'))}</td>"
                    f"<td>{_format_value(entry.get('delta_aic'))}</td>"
                    f"<td>{_format_value(entry.get('bic'))}</td>"
                    f"<td>{_format_value(entry.get('delta_bic'))}</td>"
                    "</tr>"
                )
            lines.extend(["</tbody></table></section>"])

        engine_meta_rows = [
            ("Engine name", engine_quantum.get("engine_name") or meta.get("engine")),
            ("Quantum version", engine_quantum.get("quantum_version")),
            ("LUT version", engine_quantum.get("lut_version")),
            ("Cosmos commit", engine_quantum.get("cosmos_commit")),
        ]
        engine_setting_rows = [
            (key, _format_value(value))
            for key, value in sorted(engine_settings.items())
        ]
        lines.extend(
            [
                "<section class='panel'>",
                "<h2>Engine & quantum metadata</h2>",
                _render_table(["Setting", "Value"], engine_meta_rows, "details"),
            ]
        )
        if engine_setting_rows:
            lines.extend(
                [
                    "<div style='margin-top: 1rem;'>",
                    "<h3>Engine settings</h3>",
                    _render_table(["Parameter", "Value"], engine_setting_rows, "details"),
                    "</div>",
                ]
            )
        if config_used:
            lines.extend(
                [
                    "<div style='margin-top: 1rem;'>",
                    "<h3>Science config</h3>",
                    "<details><summary>config_used.json</summary>",
                    "<pre>",
                    _html_escape(json.dumps(serialize_value(config_used), indent=2)),
                    "</pre>",
                    "</details>",
                    "</div>",
                ]
            )
        lines.extend(
            [
                "<div class='note'>",
                "Thermal lookup table metadata is captured by the PBUF fits; "
                "the temperature-vs-LUT figure is regenerated automatically via "
                "<code>quantum/tools/export_thermal_table.py</code> and attached below the model section.",
                "</div>",
                "</section>",
            ]
        )

        if fits_used:
            dataset_rows = [
                (
                    dataset,
                    str(fit_weights.get(dataset, "1.0")),
                )
                for dataset in fits_used
            ]
            lines.extend(
                [
                    "<section class='panel'>",
                    "<h2>Dataset coverage</h2>",
                    "<table class='summary-table'>",
                    "<thead><tr><th>Dataset</th><th>Weight</th></tr></thead>",
                    "<tbody>",
                ]
            )
            for dataset, weight in dataset_rows:
                lines.append(
                    "<tr>"
                    f"<td>{_html_escape(dataset)}</td>"
                    f"<td>{_html_escape(weight)}</td>"
                    "</tr>"
                )
            lines.extend(["</tbody></table></section>"])

        for model in models:
            name = model.get("name") or "unknown"
            best = model.get("best_fit") or {}
            best_params = best.get("parameters") or {}
            predictions = best.get("predictions") or {}
            breakdown = model.get("chi2_breakdown") or {}
            fit_details = model.get("fits") or {}
            plots = model.get("plots") or []
            metric_entry = next((entry for entry in metrics if entry.get("model") == name), {})

            param_rows = [
                [key, _format_value(value)]
                for key, value in sorted(best_params.items())
            ]
            def _build_chi2_rows(payload: dict[str, Any]) -> list[list[str]]:
                rows: list[list[str]] = []
                for key in sorted(payload):
                    value = payload[key]
                    if isinstance(value, Mapping):
                        rows.append([f"{key} (details)", ""])
                        for subkey in sorted(value):
                            rows.append([f"  {subkey}", _format_value(value[subkey])])
                    else:
                        rows.append([key, _format_value(value)])
                return rows
            chi2_rows = _build_chi2_rows(breakdown)

            lines.extend(
                [
                    "<section class='panel model-panel'>",
                    f"<h2>{_html_escape(name.capitalize())}</h2>",
                    "<div class='model-meta'>",
                    f"<span>χ² { _format_value(best.get('chi2')) }</span>",
                    f"<span>k parameters { _format_value(metric_entry.get('k')) }</span>",
                    f"<span>AIC { _format_value(metric_entry.get('aic')) }</span>",
                    f"<span>BIC { _format_value(metric_entry.get('bic')) }</span>",
                    "</div>",
                    "<div class='grid'>",
                    "<div class='grid-item'>",
                    "<label>Best-fit parameters</label>",
                    _render_table(["Parameter", "Value"], param_rows),
                    "</div>",
                    "<div class='grid-item'>",
                    "<label>χ² breakdown</label>",
                    _render_table(["Fit", "χ²"], chi2_rows),
                    "</div>",
                    "<div class='grid-item'>",
                    "<label>Predictions</label>",
                    "<ul class='model-extra'>",
                ]
            )
            for key, value in sorted(predictions.items()):
                if key == "plot_data":
                    continue
                lines.append(f"<li>{_html_escape(str(key))}: {_html_escape(str(_format_value(value)))}</li>")
            lines.extend(
                [
                    "</ul>",
                    "</div>",
                    "</div>",
            ]
            )

            if fit_details:
                lines.extend(
                    [
                        "<div style='margin-top: 1rem;'>",
                        "<h3>Fit outputs</h3>",
                        _render_fits_table(fit_details),
                        "</div>",
                    ]
                )

            if plots:
                lines.append("<div class='plot-column'>")
                for plot in plots:
                    label = plot.get("label") or "Prediction"
                    path = plot.get("path") or ""
                    lines.append(
                        "<figure class='plot-figure'>"
                        f"<img class='plot-image' src='{_html_escape(path)}' alt='{_html_escape(label)}' loading='lazy'>"
                        f"<figcaption>{_html_escape(label)}</figcaption>"
                        "</figure>"
                    )
                lines.append("</div>")

            lines.append("</section>")

        lines.append("</main></body></html>")
        return "".join(lines)


__all__ = ["ScienceRunReportGenerator"]
