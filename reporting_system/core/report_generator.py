"""
Standalone report generator that produces beautiful, scientifically accurate reports
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import hashlib
import html
import json
import logging
import numbers

from ..data.data_loader import DataLoader
from .panel_builders import (
    generate_hero_section,
    generate_overview_panel,
    generate_model_comparison_panel,
    generate_metadata_panel,
    generate_model_panel,
    generate_figures_panels,
    generate_prediction_comparison_section,
    _to_relative_path,
)
from cosmos2.branding import COSMOS2_GENERATED_FOOTER
from cosmos2.data.registry import get_dataset
from .template_engine import ReportTemplateEngine

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Standalone report generator for cosmos2 science runs."""
    
    def __init__(
        self,
        run_directory: Path,
        template_path: Path | str | None = None,
        css_path: Path | str | None = None,
    ):
        """Initialize report generator."""
        self.run_dir = Path(run_directory)
        self.data_loader = DataLoader(self.run_dir)
        self.logger = logging.getLogger(f"{__name__}.ReportGenerator")
        self.template_engine = ReportTemplateEngine(Path(template_path) if template_path else None)
        self.css_path = (
            Path(css_path)
            if css_path
            else Path(__file__).parent.parent / "themes" / "beautiful.css"
        )
    
    def generate_report(self, output_path: Path = None) -> str:
        """Generate a comprehensive report."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.run_dir / f"science_report_{timestamp}.html"
        
        # Load all data
        models = self.data_loader.get_available_models()
        model_data = {}
        for model in models:
            model_data[model] = self.data_loader.load_model_data(model)
        
        run_metadata = self.data_loader.load_run_metadata()
        figures = self.data_loader.get_figures()
        predictions_details = self.data_loader.load_predictions_details()
        
        # Generate report content
        html_content = self._generate_html(
            models,
            model_data,
            run_metadata,
            figures,
            predictions_details,
        )
        
        # Write report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Report generated: {output_path}")
        return str(output_path)
    
    def _generate_html(
        self,
        models: List[str],
        model_data: Dict[str, Any],
        run_metadata: Dict[str, Any],
        figures: List[Dict[str, Any]],
        predictions_details: Dict[str, Any] | None,
    ) -> str:
        """Generate HTML report content."""

        css_content = self._load_css()
        run_name = self.run_dir.name

        hero_html = generate_hero_section(self.run_dir, models, run_metadata)
        overview_html = generate_overview_panel(models, run_metadata)
        comparison_html = generate_model_comparison_panel(models, model_data)
        model_figures_map, general_figures = self._partition_figures(figures, models)
        jackknife_comparison_html = self._build_jackknife_comparison_section(general_figures)
        metadata_html = generate_metadata_panel(run_metadata)
        predictions_html = generate_prediction_comparison_section(predictions_details, self.run_dir)

        best_model = None
        best_chi2 = float("inf")
        for model in models:
            chi2 = model_data[model].get("best_fit", {}).get("chi_squared", float("inf"))
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_model = model

        model_sections = []
        for model in models:
            model_sections.append(
                generate_model_panel(
                    model,
                    model_data[model],
                    best_model,
                    best_chi2,
                    self.run_dir,
                    model_figures=model_figures_map.get(model, []),
                )
            )
        model_sections_html = "".join(model_sections)

        figures_html = generate_figures_panels(general_figures, self.run_dir)
        reproducibility_section_html = self._build_repro_section(run_metadata, figures)

        context = {
            "title": f"Science Run Report - {run_name}",
            "styles": css_content,
            "hero_section": hero_html,
            "overview_section": overview_html,
            "comparison_section": comparison_html,
            "predictions_section": predictions_html,
            "jackknife_comparison_section": jackknife_comparison_html,
            "metadata_section": metadata_html,
            "model_sections": model_sections_html,
            "figures_section": figures_html,
            "reproducibility_section": reproducibility_section_html,
            "footer_text": COSMOS2_GENERATED_FOOTER,
        }

        return self.template_engine.render(context)

    def _partition_figures(
        self,
        figures: List[Dict[str, Any]],
        models: List[str],
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Group figures by model while keeping leftover figures general."""

        model_figures: Dict[str, List[Dict[str, Any]]] = {model: [] for model in models}
        general_figures: List[Dict[str, Any]] = []

        for fig in figures:
            name = (fig.get("name") or "").lower()
            model_assigned = False
            for model in models:
                model_key = model.lower()
                if model_key and name.startswith(model_key):
                    fig["model_group"], fig["model_subgroup"] = self._extract_model_group(name, model_key)
                    model_figures.setdefault(model, []).append(fig)
                    model_assigned = True
                    break
            if not model_assigned:
                general_figures.append(fig)

        return model_figures, general_figures

    def _extract_model_group(self, name: str, model_key: str) -> Tuple[str, str]:
        """Parse the group/subgroup for a model-specific figure."""

        remainder = name[len(model_key) :].lstrip("_")
        if not remainder:
            return "", ""
        parts = remainder.split("_", 1)
        group = parts[0] if parts else ""
        subgroup = parts[1] if len(parts) > 1 else ""
        return group.lower(), subgroup.lower()

    def _build_figure_section(self, title: str, figures: List[Dict[str, Any]]) -> str:
        """Wrap a gallery of figures inside a titled block."""

        gallery = self._render_figure_gallery(figures)
        if not gallery:
            return ""

        return f"""
    <div class='figure-section'>
        <h3>{title}</h3>
        <div class='figure-gallery'>
            {gallery}
        </div>
    </div>"""

    def _render_figure_gallery(self, figures: List[Dict[str, Any]]) -> str:
        """Produce inline figure cards for the provided figure list."""

        rows = []
        for fig in figures:
            fig_path = Path(fig.get("file_path", ""))
            if not fig_path.exists():
                continue
            rel_path = _to_relative_path(self.run_dir, fig_path)
            caption = self._format_label(fig.get("name") or fig_path.stem) or fig_path.stem
            rows.append(f"""
            <figure class='figure-card'>
                <img src="{rel_path.as_posix()}" alt="{caption}" loading="lazy">
                <figcaption>{caption}</figcaption>
            </figure>""")
        return "".join(rows)

    def _format_label(self, value: str | None) -> str:
        """Convert snake_case identifiers into readable labels."""

        if not value:
            return ""
        normalized = " ".join(value.replace("_", " ").split())
        return normalized.title()

    def _load_css(self) -> str:
        """Read the CSS bundle for the report."""

        if not self.css_path.exists():
            raise FileNotFoundError(f"CSS file missing: {self.css_path}")
        css_content = self.css_path.read_text()
        css_content += """
            .footer {
                text-align: center;
                color: #64748b;
                font-size: 0.9rem;
                padding: 1rem 0;
                border-top: 1px solid #e2e8f0;
                margin-top: 2rem;
            }
            .repro-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 1rem;
            }
            .repro-block {
                border: 1px solid #e2e8f0;
                border-radius: 0.5rem;
                padding: 1rem;
                background: #fff;
                box-shadow: 0 0.15rem 0.5rem rgba(15, 23, 42, 0.08);
            }
            .repro-block h3 {
                margin-top: 0;
            }
            .package-list {
                margin: 0.5rem 0 0;
                padding-left: 1rem;
            }
            .hash-table code {
                font-size: 0.75rem;
                word-break: break-all;
            }
            .dataset-note td {
                font-size: 0.85rem;
                color: #475569;
                padding-top: 0.25rem;
            }
        """
        return css_content

    def _build_jackknife_comparison_section(self, general_figures: List[Dict[str, Any]]) -> str:
        """Render a dedicated jackknife comparison section using run-level plots."""

        target_names = [
            "run_jackknife_chi2",
            "run_datasets_chi2_contributions",
            "run_jackknife_h0_convergence",
        ]
        selected_figures = self._pop_figures_by_names(general_figures, target_names)
        if not selected_figures:
            return ""

        figure_blocks = []
        for fig in selected_figures:
            img_path = Path(fig.get("file_path", ""))
            if not img_path.exists():
                continue
            rel_path = _to_relative_path(self.run_dir, img_path)
            caption = self._format_caption(fig.get("name") or img_path.stem)
            figure_blocks.append(f"""
        <div class='figure-block'>
            <h4>{caption}</h4>
            <img src="{rel_path.as_posix()}" alt="{caption}" style="max-width: 100%; height: auto; margin-top: 0.5rem;">
        </div>""")

        if not figure_blocks:
            return ""

        return f"""
<section class='panel'>
    <h2>Jackknife comparison</h2>
    {''.join(figure_blocks)}
</section>"""

    def _pop_figures_by_names(
        self,
        figures: List[Dict[str, Any]],
        names: List[str],
    ) -> List[Dict[str, Any]]:
        """Extract and remove figures matching the specified stems."""

        extracted: List[Dict[str, Any]] = []
        for target in names:
            target_lower = target.lower()
            for idx, fig in enumerate(figures):
                if self._figure_matches_name(fig, target_lower):
                    extracted.append(figures.pop(idx))
                    break
        return extracted

    def _figure_matches_name(self, fig: Dict[str, Any], target_name: str) -> bool:
        """Check if a figure matches the target name (stem or explicit name)."""

        name = (fig.get("name") or "").lower()
        stem = Path(fig.get("file_path", "")).stem.lower()
        return name == target_name or stem == target_name

    def _format_caption(self, value: str | None) -> str:
        """Normalize figure identifiers into display captions."""

        if not value:
            return ""
        normalized = " ".join(value.replace("_", " ").split())
        return normalized.title()

    def _build_repro_section(self, run_metadata: Dict[str, Any], figures: List[Dict[str, Any]]) -> str:
        """Render the reproducibility panel summarizing environment and assets."""

        run_meta = run_metadata.get("run_meta") or {}
        environment = run_meta.get("environment") or {}
        environment_block = self._render_environment_block(environment, run_meta)
        dataset_block = self._render_dataset_manifest(run_metadata)
        config_block = self._render_config_cli(run_metadata, run_meta)
        asset_block = self._render_asset_hashes(figures)

        return f"""
<section class='panel reproducibility-panel'>
    <h2>Reproducibility</h2>
    <div class='repro-grid'>
        <div class='repro-block'>
            {environment_block}
        </div>
        <div class='repro-block'>
            {dataset_block}
        </div>
        <div class='repro-block'>
            {config_block}
        </div>
        <div class='repro-block'>
            {asset_block}
        </div>
    </div>
</section>"""

    def _render_environment_block(self, environment: Dict[str, Any], run_meta: Dict[str, Any]) -> str:
        lines: List[str] = []
        git_info = environment.get("git") or {}
        commit = git_info.get("commit") or run_meta.get("git_commit")
        dirty = git_info.get("dirty") or run_meta.get("git_dirty")
        if commit:
            dirty_marker = " (dirty)" if dirty else ""
            lines.append(f"<li>Git commit <code>{html.escape(commit)}</code>{dirty_marker}</li>")
        python_info = environment.get("python") or {}
        python_version = python_info.get("version")
        if python_version:
            executable = python_info.get("executable", "")
            exec_name = Path(executable).name or executable
            lines.append(f"<li>Python {python_version} @ {html.escape(exec_name)}</li>")
        blas = environment.get("blas_backend")
        if blas:
            lines.append(f"<li>BLAS backend: {html.escape(str(blas))}</li>")
        cpu = environment.get("cpu") or {}
        cpu_parts: List[str] = []
        if cpu.get("node"):
            cpu_parts.append(cpu["node"])
        if cpu.get("platform"):
            cpu_parts.append(cpu["platform"])
        if cpu.get("processor"):
            cpu_parts.append(cpu["processor"])
        max_cores = cpu.get("cores_logical") or cpu.get("cores_physical")
        if max_cores:
            cpu_parts.append(f"{max_cores} cores")
        if cpu_parts:
            lines.append(f"<li>CPU: {', '.join(html.escape(str(part)) for part in cpu_parts)}</li>")
        gpu = environment.get("gpu")
        if gpu:
            if gpu.get("available"):
                devices = gpu.get("devices") or []
                device_lines = "<br>".join(html.escape(str(device)) for device in devices)
                lines.append(f"<li>GPU(s): {device_lines}</li>")
            else:
                reason = gpu.get("reason", "not available")
                lines.append(f"<li>GPU: unavailable ({html.escape(reason)})</li>")
        if not lines:
            lines.append("<li>Environment metadata unavailable.</li>")
        package_block = ""
        packages = environment.get("packages", {})
        if packages:
            package_rows = "".join(
                f"<li><code>{name}</code>: {html.escape(version)}</li>"
                for name, version in sorted(packages.items())
            )
            package_block = f"<h4>Packages</h4><ul class='package-list'>{package_rows}</ul>"
        else:
            package_block = "<p>No package metadata recorded.</p>"
        return f"""
        <h3>Environment snapshot</h3>
        <ul>
            {''.join(lines)}
        </ul>
        {package_block}"""

    def _render_dataset_manifest(self, run_metadata: Dict[str, Any]) -> str:
        manifest = self._collect_dataset_manifest(run_metadata)
        if not manifest:
            return "<h3>Dataset manifest</h3><p>Dataset metadata is unavailable.</p>"
        rows = []
        for entry in manifest:
            dataset_label = html.escape(str(entry.get("name", "unknown")).upper())
            reference = html.escape(entry.get("reference") or entry.get("error") or "n/a")
            doi = html.escape(entry.get("doi") or "n/a")
            version = html.escape(entry.get("version") or "n/a")
            count = entry.get("count")
            count_display = str(count) if count is not None else "n/a"
            rows.append(
                "<tr>"
                f"<td>{dataset_label}</td>"
                f"<td>{reference}</td>"
                f"<td>{doi}</td>"
                f"<td>{version}</td>"
                f"<td>{count_display}</td>"
                "</tr>"
            )
            note = entry.get("note")
            if note:
                rows.append(
                    f"<tr class='dataset-note'><td colspan='5'>{html.escape(str(note))}</td></tr>"
                )
        return f"""
        <h3>Dataset manifest</h3>
        <table class='summary-table'>
            <thead>
                <tr><th>Dataset</th><th>Reference</th><th>DOI</th><th>Version/Tag</th><th>Points used</th></tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>"""

    def _collect_dataset_manifest(self, run_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        config = run_metadata.get("config") or {}
        run_meta = run_metadata.get("run_meta") or {}
        names: List[str] = []
        seen: set[str] = set()
        candidates = (
            config.get("fits_list")
            or config.get("fits")
            or run_meta.get("fits_used")
            or []
        )
        for candidate in candidates:
            if not candidate:
                continue
            normalized = str(candidate).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            names.append(candidate)
        datasets: List[Dict[str, Any]] = []
        for name in names:
            datasets.append(self._describe_dataset(name))
        return datasets

    def _describe_dataset(self, name: str) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"name": name}
        try:
            dataset = get_dataset(name)
        except Exception as exc:
            entry["error"] = f"Failed to load dataset: {exc}"
            return entry
        meta = self._extract_dataset_meta(dataset)
        reference = meta.get("reference")
        doi = meta.get("doi")
        if reference:
            entry["reference"] = reference
        if doi:
            entry["doi"] = doi
        version = meta.get("version") or meta.get("created_at") or meta.get("dataset_version")
        if version:
            entry["version"] = version
        note_parts: List[str] = []
        for key in ("note", "notes", "description", "dataset_type", "survey"):
            value = meta.get(key)
            if value:
                note_parts.append(str(value))
        if note_parts:
            entry["note"] = " / ".join(note_parts)
        entry["count"] = self._dataset_length(dataset)
        return entry

    def _dataset_length(self, dataset: Any) -> int:
        candidate = None
        if isinstance(dataset, dict):
            for key in (
                "n_data",
                "obs",
                "observed",
                "data",
                "y",
                "H_obs",
                "fs8",
                "fsigma8",
                "fs8_obs",
            ):
                candidate = dataset.get(key)
                if candidate is not None:
                    break
        elif hasattr(dataset, "__len__") and not isinstance(dataset, (str, bytes)):
            try:
                return len(dataset)
            except Exception:
                return 0
        if candidate is None:
            return 0
        candidate = self._normalize_dataset_value(candidate)
        if candidate is None:
            return 0
        if isinstance(candidate, numbers.Number):
            try:
                return max(int(candidate), 0)
            except Exception:
                return 0
        if isinstance(candidate, (list, tuple, dict, set)):
            return len(candidate)
        if hasattr(candidate, "__len__") and not isinstance(candidate, (str, bytes)):
            try:
                return len(candidate)
            except Exception:
                return 0
        return 0

    def _normalize_dataset_value(self, value: Any) -> Any:
        """Convert numpy scalars or wrappers into plain Python objects."""

        if value is None:
            return None
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _extract_dataset_meta(self, dataset: Any) -> Dict[str, Any]:
        if not isinstance(dataset, dict):
            return {}
        raw_meta = dataset.get("meta") or dataset.get("metadata")
        normalized = self._normalize_dataset_value(raw_meta)
        if isinstance(normalized, dict):
            return normalized
        return {}

    def _render_config_cli(self, run_metadata: Dict[str, Any], run_meta: Dict[str, Any]) -> str:
        config = run_metadata.get("config") or {}
        engine_settings = config.get("engine_settings") or {}
        jackknife = config.get("jackknife") or {}
        seed_parts: List[str] = []
        for key in ("seed", "rng_seed", "grid_seed"):
            if key in engine_settings:
                seed_parts.append(f"{key}={engine_settings[key]}")
        jk_seed = jackknife.get("random_seed") or jackknife.get("seed")
        if jk_seed is not None:
            seed_parts.append(f"jackknife.random_seed={jk_seed}")
        seed_summary = ", ".join(seed_parts) if seed_parts else "not recorded"
        environment = run_meta.get("environment") or {}
        cli_command = run_meta.get("cli_command") or environment.get("cli_command")
        cli_line = html.escape(cli_command) if cli_command else "not recorded"
        config_payload = {
            "engine_settings": engine_settings,
            "jackknife": jackknife,
        }
        config_dump = json.dumps(config_payload, indent=2, sort_keys=True)
        return f"""
        <h3>Config & CLI</h3>
        <p><strong>CLI:</strong> <code>{cli_line}</code></p>
        <p><strong>Seed summary:</strong> {html.escape(seed_summary)}</p>
        <details>
            <summary>Engine / jackknife settings</summary>
            <pre>{html.escape(config_dump)}</pre>
        </details>"""

    def _render_asset_hashes(self, figures: List[Dict[str, Any]]) -> str:
        figure_hashes, table_hashes = self._collect_asset_hashes(figures)
        if not figure_hashes and not table_hashes:
            return "<p>No figures or tables were generated yet.</p>"
        parts: List[str] = []
        if figure_hashes:
            figure_rows = "".join(
                f"<tr><td>{html.escape(entry['label'])}</td><td>{html.escape(entry['path'])}</td><td><code>{entry['hash']}</code></td></tr>"
                for entry in figure_hashes
            )
            parts.append(
                f"""
        <h4>Figures</h4>
        <table class='summary-table hash-table'>
            <thead>
                <tr><th>Figure</th><th>Path</th><th>SHA256</th></tr>
            </thead>
            <tbody>
                {figure_rows}
            </tbody>
        </table>"""
            )
        if table_hashes:
            table_rows = "".join(
                f"<tr><td>{html.escape(entry['label'])}</td><td>{html.escape(entry['path'])}</td><td><code>{entry['hash']}</code></td></tr>"
                for entry in table_hashes
            )
            parts.append(
                f"""
        <h4>Tables</h4>
        <table class='summary-table hash-table'>
            <thead>
                <tr><th>Table</th><th>Path</th><th>SHA256</th></tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>"""
            )
        return "".join(parts)

    def _collect_asset_hashes(self, figures: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        figure_hashes: List[Dict[str, str]] = []
        for fig in figures:
            path = Path(fig.get("file_path", ""))
            if not path.exists():
                continue
            rel = _to_relative_path(self.run_dir, path)
            figure_hashes.append(
                {"label": fig.get("name") or path.stem, "path": rel.as_posix(), "hash": self._file_hash(path)}
            )
        tables_dir = self.run_dir / "tables"
        table_hashes: List[Dict[str, str]] = []
        if tables_dir.exists():
            for table_path in sorted(tables_dir.rglob("*")):
                if not table_path.is_file():
                    continue
                rel = _to_relative_path(self.run_dir, table_path)
                table_hashes.append(
                    {"label": rel.name, "path": rel.as_posix(), "hash": self._file_hash(table_path)}
                )
        return sorted(figure_hashes, key=lambda entry: entry["label"]), sorted(table_hashes, key=lambda entry: entry["path"])

    def _file_hash(self, path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()
