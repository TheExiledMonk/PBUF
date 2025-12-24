"""Jackknife aggregation report generator.

This generator produces a combined report across multiple runs by pooling only
jackknife-dependent outputs (per-fold chi² and model-comparison Δχ²).

Deterministic outputs are taken from a single reference run and are not combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..data.data_loader import DataLoader
from ..data.jackknife_aggregation import (
    AggregatedJackknife,
    aggregate_jackknife_runs,
    discover_candidate_runs,
    distribution_stats,
    pool_draw_series,
    select_latest_per_run_name,
)
from .panel_builders import (
    generate_figures_panels,
    generate_hero_section,
    generate_metadata_panel,
    generate_model_comparison_panel,
    generate_model_panel,
    generate_overview_panel,
    generate_prediction_comparison_section,
    _to_relative_path,
)
from .template_engine import ReportTemplateEngine
from cosmos2.branding import COSMOS2_GENERATED_FOOTER


@dataclass(frozen=True)
class AggregationSelection:
    run_dirs: list[Path]
    reference_run: Path


class JackknifeAggregatedReportGenerator:
    """Generate a combined report by aggregating jackknife fold outputs across runs."""

    def __init__(
        self,
        suite_root: Path,
        *,
        run_dirs: list[Path] | None = None,
        select_latest: bool = True,
        template_path: Path | str | None = None,
        css_path: Path | str | None = None,
    ) -> None:
        self.suite_root = Path(suite_root)
        self.select_latest = bool(select_latest)
        self._explicit_run_dirs = [Path(p) for p in (run_dirs or [])]
        self.template_engine = ReportTemplateEngine(Path(template_path) if template_path else None)
        self.css_path = (
            Path(css_path)
            if css_path
            else Path(__file__).parent.parent / "themes" / "beautiful.css"
        )

    def _select_runs(self) -> AggregationSelection:
        if self._explicit_run_dirs:
            candidates = list(self._explicit_run_dirs)
        else:
            candidates = discover_candidate_runs(self.suite_root)

        if self.select_latest:
            candidates = select_latest_per_run_name(candidates)

        if not candidates:
            raise FileNotFoundError(f"No run directories found under '{self.suite_root}'.")

        # Reference run: latest timestamp by directory name (ISO prefix) as a stable heuristic.
        reference = max(candidates, key=lambda p: p.name)
        return AggregationSelection(run_dirs=candidates, reference_run=reference)

    def generate_report(self, output_path: Path | None = None) -> str:
        selection = self._select_runs()
        output_path = Path(output_path) if output_path else (self.suite_root / "science_report_aggregated.html")
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        reference_dir = selection.reference_run
        data_loader = DataLoader(reference_dir, output_directory=output_dir)

        models = data_loader.get_available_models()
        model_data: dict[str, dict[str, Any]] = {}
        for model in models:
            model_data[model] = data_loader.load_model_data(model, include_jackknife=False)

        run_metadata = data_loader.load_run_metadata()

        # Deterministic plots only (exclude per-run jackknife plots).
        figures = data_loader.get_figures(include_jackknife=False)
        predictions_details = data_loader.load_predictions_details()

        aggregated = aggregate_jackknife_runs(selection.run_dirs)
        pooled_series = pool_draw_series(selection.run_dirs, models=models)
        jackknife_section_html = self._build_jackknife_aggregation_section(
            aggregated,
            pooled_series=pooled_series,
            baseline_by_model={
                model: (
                    float(model_data[model].get("best_fit", {}).get("chi_squared"))
                    if isinstance(model_data[model].get("best_fit", {}).get("chi_squared"), (int, float))
                    else None
                )
                for model in models
            },
            baseline_h0_by_model={
                model: (
                    float((model_data[model].get("best_fit", {}).get("parameters") or {}).get("H0"))
                    if isinstance((model_data[model].get("best_fit", {}).get("parameters") or {}).get("H0"), (int, float))
                    else None
                )
                for model in models
            },
            output_dir=output_dir,
            suite_name=self.suite_root.name,
        )

        css_content = self._load_css()
        hero_html = generate_hero_section(reference_dir, models, run_metadata)
        overview_html = self._prepend_panel(
            self._build_aggregation_notice_panel(aggregated, selection),
            generate_overview_panel(models, run_metadata),
        )
        comparison_html = generate_model_comparison_panel(models, model_data)
        metadata_html = generate_metadata_panel(run_metadata)
        predictions_html = generate_prediction_comparison_section(predictions_details, output_dir)
        figures_html = generate_figures_panels(figures, output_dir)

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
                    output_dir,
                    model_figures=[],
                )
            )
        model_sections_html = "".join(model_sections)

        context = {
            "title": f"Science Run Report (Jackknife aggregated) - {self.suite_root.name}",
            "styles": css_content,
            "hero_section": hero_html,
            "overview_section": overview_html,
            "comparison_section": comparison_html,
            "jackknife_comparison_section": jackknife_section_html,
            "metadata_section": metadata_html,
            "model_sections": model_sections_html,
            "figures_section": figures_html,
            "predictions_section": predictions_html,
            "reproducibility_section": "",
            "footer_text": COSMOS2_GENERATED_FOOTER,
        }

        html_content = self.template_engine.render(context)
        output_path.write_text(html_content, encoding="utf-8")
        return str(output_path)

    def _load_css(self) -> str:
        if not self.css_path.exists():
            raise FileNotFoundError(f"CSS file missing: {self.css_path}")
        return self.css_path.read_text(encoding="utf-8")

    @staticmethod
    def _prepend_panel(panel_html: str, existing_html: str) -> str:
        if not panel_html:
            return existing_html
        return f"{panel_html}\n{existing_html}"

    def _build_aggregation_notice_panel(self, aggregated: AggregatedJackknife, selection: AggregationSelection) -> str:
        runs = aggregated.runs
        n_runs = len(runs)
        seeds = [str(r.jackknife_seed) for r in runs if r.jackknife_seed is not None]
        pooled_folds = 0
        if aggregated.pooled_delta_chi2:
            pooled_folds = int(next(iter(aggregated.pooled_delta_chi2.values())).size)
        elif aggregated.pooled_chi2:
            pooled_folds = int(next(iter(aggregated.pooled_chi2.values())).size)

        run_rows = ""
        for info in runs:
            seed = info.jackknife_seed if info.jackknife_seed is not None else "n/a"
            stamp = info.timestamp or ""
            run_rows += f"<tr><td>{info.run_name}</td><td>{seed}</td><td>{stamp}</td><td><code>{info.run_dir}</code></td></tr>"

        seeds_label = ", ".join(seeds) if seeds else "n/a"
        return f"""
<section class='panel'>
  <h2>Jackknife aggregation mode</h2>
  <p><strong>Aggregated results reflect jackknife resampling only. Model parameters and deterministic predictions are identical across all runs.</strong></p>
  <p>Reference run: <code>{selection.reference_run}</code></p>
  <p>Aggregating {n_runs} run(s), pooled folds: <strong>{pooled_folds}</strong>, run seeds: {seeds_label}</p>
  <table class='details'>
    <thead><tr><th>Run name</th><th>Jackknife seed</th><th>Timestamp</th><th>Directory</th></tr></thead>
    <tbody>{run_rows}</tbody>
  </table>
</section>"""

    def _build_jackknife_aggregation_section(
        self,
        aggregated: AggregatedJackknife,
        *,
        pooled_series: Any,
        baseline_by_model: dict[str, float | None],
        baseline_h0_by_model: dict[str, float | None],
        output_dir: Path,
        suite_name: str,
    ) -> str:
        if not aggregated.runs:
            return ""

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Match the single-run reporting style (draw-index plots), but pooled across runs.
        run_chi2_plot = self._plot_runstyle_jackknife_chi2(
            pooled_series,
            baseline_by_model,
            figures_dir / "run_jackknife_chi2.png",
        )
        run_h0_plot = self._plot_runstyle_h0_convergence(
            pooled_series,
            baseline_h0_by_model,
            figures_dir / "run_jackknife_h0_convergence.png",
        )

        # Keep the robustness diagnostics introduced for aggregation mode.
        delta_plot = self._plot_delta_chi2(aggregated, figures_dir / "aggregated_jackknife_delta_chi2.png")
        seed_plot = self._plot_delta_by_seed(aggregated, figures_dir / "aggregated_jackknife_delta_by_seed.png")

        delta_stats_block = ""
        favor_block = ""
        pair = None
        if aggregated.pooled_delta_chi2:
            pair, deltas = next(iter(aggregated.pooled_delta_chi2.items()))
            stats = distribution_stats(deltas)
            a, b = pair
            n = int(stats.get("n", 0) or 0)
            if n:
                n_a = int(np.sum(deltas < 0))
                n_b = int(np.sum(deltas > 0))
                n_tie = n - n_a - n_b
                favor_block = (
                    f"<p>Favors <strong>{a.upper()}</strong>: {n_a} • "
                    f"favors <strong>{b.upper()}</strong>: {n_b} • ties: {n_tie}</p>"
                )
            delta_stats_block = f"""
            <table class='details'>
              <thead><tr><th>Statistic</th><th>Value</th></tr></thead>
              <tbody>
                <tr><td>n</td><td>{stats.get('n', 0)}</td></tr>
                <tr><td>median</td><td>{stats.get('median', 'n/a')}</td></tr>
                <tr><td>mean</td><td>{stats.get('mean', 'n/a')}</td></tr>
                <tr><td>std</td><td>{stats.get('std', 'n/a')}</td></tr>
                <tr><td>16–84%</td><td>{stats.get('q16', 'n/a')} … {stats.get('q84', 'n/a')}</td></tr>
              </tbody>
            </table>"""

        figures_html = ""
        for title, path in (
            ("Jackknife χ² comparison (pooled draws)", run_chi2_plot),
            ("H₀ convergence across jackknife draws (pooled draws)", run_h0_plot),
            ("Δχ² distribution (pooled folds)", delta_plot),
            ("Δχ² by jackknife seed (stability)", seed_plot),
        ):
            if not path or not Path(path).exists():
                continue
            rel = _to_relative_path(output_dir, Path(path))
            figures_html += f"""
        <div class='figure-block'>
          <h4>{title}</h4>
          <img src="{rel.as_posix()}" alt="{title}" style="max-width: 100%; height: auto; margin-top: 0.5rem;">
        </div>"""

        subtitle = ""
        if pair:
            a, b = pair
            subtitle = f"<p>Δχ² is computed as χ²({a.upper()}) − χ²({b.upper()}).</p>"

        return f"""
<section class='panel'>
  <h2>Jackknife summary (aggregated)</h2>
  <p>Suite: <code>{suite_name}</code></p>
  {subtitle}
  {favor_block}
  {delta_stats_block}
  {figures_html}
</section>"""

    @staticmethod
    def _finite(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    def _plot_delta_chi2(self, aggregated: AggregatedJackknife, out_path: Path) -> str | None:
        if not aggregated.pooled_delta_chi2:
            return None
        (a, b), deltas = next(iter(aggregated.pooled_delta_chi2.items()))
        deltas = self._finite(deltas)
        if deltas.size == 0:
            return None
        median = float(np.median(deltas))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(deltas, bins=30, color="#2563eb", alpha=0.75, edgecolor="white")
        ax.axvline(0.0, color="#0f172a", linewidth=1.2, linestyle="--", label="Δχ² = 0")
        ax.axvline(median, color="#dc2626", linewidth=1.5, label=f"median = {median:.2f}")
        ax.set_title(f"Δχ² distribution (pooled folds): {a.upper()} − {b.upper()}")
        ax.set_xlabel("Δχ²")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    def _plot_chi2_distributions(self, aggregated: AggregatedJackknife, out_path: Path) -> str | None:
        models = [m for m in aggregated.models if self._finite(aggregated.pooled_chi2.get(m, np.asarray([]))).size]
        if not models:
            return None

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, model in enumerate(models):
            values = self._finite(aggregated.pooled_chi2.get(model, np.asarray([])))
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=30,
                alpha=0.4,
                color=colors[idx % len(colors)],
                label=f"{model.upper()}",
            )
            ax.axvline(float(np.median(values)), color=colors[idx % len(colors)], linewidth=1.3)

        ax.set_title("Jackknife χ² distributions (pooled folds)")
        ax.set_xlabel("χ²")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    def _plot_delta_by_seed(self, aggregated: AggregatedJackknife, out_path: Path) -> str | None:
        if not aggregated.delta_by_seed:
            return None
        labels = sorted(aggregated.delta_by_seed.keys(), key=lambda k: str(k))
        series = [self._finite(aggregated.delta_by_seed[label]) for label in labels]
        series = [s for s in series if s.size]
        if not series:
            return None

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.5))
        ax.boxplot(series, labels=labels, showfliers=False)
        ax.axhline(0.0, color="#0f172a", linestyle="--", linewidth=1.0)
        ax.set_title("Δχ² grouped by jackknife seed (pooled folds)")
        ax.set_xlabel("Jackknife seed")
        ax.set_ylabel("Δχ²")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    def _plot_runstyle_jackknife_chi2(
        self,
        pooled_series: Any,
        baseline_by_model: dict[str, float | None],
        out_path: Path,
    ) -> str | None:
        models = list(getattr(pooled_series, "models", []) or [])
        chi2_by_model = getattr(pooled_series, "chi2_by_model", {}) or {}
        if not models or not isinstance(chi2_by_model, dict):
            return None

        n_draws = 0
        for model in models:
            series = chi2_by_model.get(model)
            if isinstance(series, np.ndarray):
                n_draws = max(n_draws, int(series.size))
        if n_draws <= 0:
            return None

        draw_indices = np.arange(1, n_draws + 1, dtype=int)
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, model in enumerate(models):
            series = chi2_by_model.get(model)
            if not isinstance(series, np.ndarray) or series.size == 0:
                continue
            color = colors[idx % len(colors)]
            baseline = baseline_by_model.get(model)
            if baseline is not None and np.isfinite(baseline):
                ax.hlines(
                    float(baseline),
                    float(draw_indices[0]),
                    float(draw_indices[-1]),
                    colors=color,
                    linestyles="--",
                    label=f"{model.upper()} best-fit",
                )
            ax.scatter(
                draw_indices[: series.size],
                series,
                color=color,
                alpha=0.6,
                label=f"{model.upper()} draws",
                edgecolors="none",
            )

        ax.set_title("Jackknife χ² comparison (pooled draws)")
        ax.set_xlabel("Draw index (pooled)")
        ax.set_ylabel("χ²")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.7)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)

    def _plot_runstyle_h0_convergence(
        self,
        pooled_series: Any,
        baseline_h0_by_model: dict[str, float | None],
        out_path: Path,
    ) -> str | None:
        models = list(getattr(pooled_series, "models", []) or [])
        h0_by_model = getattr(pooled_series, "h0_by_model", {}) or {}
        if not models or not isinstance(h0_by_model, dict):
            return None

        n_draws = 0
        for model in models:
            series = h0_by_model.get(model)
            if isinstance(series, np.ndarray):
                n_draws = max(n_draws, int(series.size))
        if n_draws <= 0:
            return None

        draw_indices = np.arange(1, n_draws + 1, dtype=int)
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, model in enumerate(models):
            series = h0_by_model.get(model)
            if not isinstance(series, np.ndarray) or series.size == 0:
                continue
            color = colors[idx % len(colors)]
            baseline = baseline_h0_by_model.get(model)
            if baseline is not None and np.isfinite(baseline):
                ax.hlines(
                    float(baseline),
                    float(draw_indices[0]),
                    float(draw_indices[-1]),
                    colors=color,
                    linestyles="--",
                    linewidth=1.2,
                    label=f"{model.upper()} baseline",
                )
            ax.scatter(
                draw_indices[: series.size],
                series,
                color=color,
                alpha=0.75,
                label=f"{model.upper()} draws",
                edgecolors="none",
            )

        ax.set_title("H₀ convergence across jackknife draws (pooled draws)")
        ax.set_xlabel("Draw index (pooled)")
        ax.set_ylabel("H₀ (km/s/Mpc)")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return str(out_path)
