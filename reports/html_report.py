"""
HTML Report Generator — Science Run Dashboard
=============================================

Produces a comprehensive HTML report summarising aggregated statistics
and per-run diagnostics for the PBUF science workflow. The document
combines:

  * Dataset-level χ² / AIC / BIC comparisons
  * Global model preference metrics (ΔAIC, ΔBIC)
  * Best-fit parameter snapshots aggregated across runs
  * Automatically generated plots (aggregated + per run)
  * Per-run metadata, scenario tables, and provenance

Usage
-----
    from reports.html_report import build_html_report
    build_html_report(stats, plot_dir, \"reports/output/report.html\")
"""

from pathlib import Path
from datetime import datetime
import json
import math
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------

def _fmt(value: Any, precision: int = 4) -> str:
    """Consistent numeric formatting with graceful fallbacks."""
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "—"
    if abs(numeric) >= 1e5 or (abs(numeric) > 0 and abs(numeric) < 1e-3):
        return f"{numeric:.2e}"
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return f"{numeric:.{precision}f}"


def _html_escape(text: Any) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _section_header(title: str) -> str:
    return f"""
    <div class="section">
      <h2>{_html_escape(title)}</h2>
    </div>
    """


# ----------------------------------------------------------------------
# Aggregated overview sections
# ----------------------------------------------------------------------

def _build_dataset_table(aggregated: Dict[str, Any], model_a: str, model_b: str) -> str:
    datasets = aggregated.get("datasets", {})
    rows = []
    for dataset, entry in sorted(datasets.items()):
        da = entry.get(model_a, {})
        db = entry.get(model_b, {})
        rows.append(
            f"""
            <tr>
              <td>{_html_escape(dataset.upper())}</td>
              <td>{_fmt(da.get("chi2"))}</td>
              <td>{_fmt(db.get("chi2"))}</td>
              <td>{_fmt(db.get("chi2") and db.get("chi2", 0) - da.get("chi2", 0))}</td>
              <td>{_fmt(da.get("AIC"))}</td>
              <td>{_fmt(db.get("AIC"))}</td>
              <td>{_fmt(db.get("AIC") and db.get("AIC", 0) - da.get("AIC", 0))}</td>
              <td>{_fmt(da.get("BIC"))}</td>
              <td>{_fmt(db.get("BIC"))}</td>
              <td>{_fmt(db.get("BIC") and db.get("BIC", 0) - da.get("BIC", 0))}</td>
            </tr>
            """
        )

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Dataset</th>
          <th>χ² {model_a.upper()}</th>
          <th>χ² {model_b.upper()}</th>
          <th>Δχ² ({model_b}-{model_a})</th>
          <th>AIC {model_a.upper()}</th>
          <th>AIC {model_b.upper()}</th>
          <th>ΔAIC</th>
          <th>BIC {model_a.upper()}</th>
          <th>BIC {model_b.upper()}</th>
          <th>ΔBIC</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def _build_global_summary(aggregated: Dict[str, Any]) -> str:
    rows = []
    for model_name, stats in aggregated.get("models", {}).items():
        rows.append(
            f"""
            <tr>
              <td>{_html_escape(model_name.upper())}</td>
              <td>{_fmt(stats.get("chi2_total"), 6)}</td>
              <td>{_fmt(stats.get("AIC_total"), 6)}</td>
              <td>{_fmt(stats.get("BIC_total"), 6)}</td>
              <td>{_fmt(stats.get("chi2_reduced_total"), 6)}</td>
              <td>{_fmt(stats.get("n_data_total"), 0)}</td>
              <td>{_fmt(stats.get("n_params_total"), 0)}</td>
              <td>{_fmt(stats.get("run_count"), 0)}</td>
            </tr>
            """
        )

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>χ² total</th>
          <th>AIC total</th>
          <th>BIC total</th>
          <th>Reduced χ²</th>
          <th># Data</th>
          <th># Params</th>
          <th># Runs</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def _build_model_preference(aggregated: Dict[str, Any], model_a: str, model_b: str) -> str:
    comparison = aggregated.get("global", {}).get("comparison", {})
    if not comparison:
        return "<p>No model comparison metrics available.</p>"

    delta_aic = comparison.get(f"ΔAIC ({model_b}-{model_a})")
    delta_bic = comparison.get(f"ΔBIC ({model_b}-{model_a})")
    preferred_aic = comparison.get("preferred_model_AIC", "—").upper()
    preferred_bic = comparison.get("preferred_model_BIC", "—").upper()

    return f"""
    <table class="data-table small">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Value</th>
          <th>Preferred Model</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>ΔAIC ({_html_escape(model_b.upper())}-{_html_escape(model_a.upper())})</td>
          <td>{_fmt(delta_aic, 4)}</td>
          <td>{_html_escape(preferred_aic)}</td>
        </tr>
        <tr>
          <td>ΔBIC ({_html_escape(model_b.upper())}-{_html_escape(model_a.upper())})</td>
          <td>{_fmt(delta_bic, 4)}</td>
          <td>{_html_escape(preferred_bic)}</td>
        </tr>
      </tbody>
    </table>
    <p class="explain">
      Positive ΔAIC/ΔBIC values favour {_html_escape(model_a.upper())}; negative values favour {_html_escape(model_b.upper())}.
      Magnitudes &gt; 10 typically signal decisive evidence.
    </p>
    """


def _collect_average_params(stats: Dict[str, Any], model_name: str) -> Dict[str, float]:
    values: Dict[str, List[float]] = {}
    for run in stats.get("runs", []):
        primary_id = run.get("primary_scenario_id")
        scenario = next((s for s in run.get("scenarios", []) if s.get("id") == primary_id), None)
        if not scenario:
            continue
        params = scenario.get("models", {}).get(model_name, {}).get("best_fit", {}).get("params", {})
        for key, val in params.items():
            try:
                values.setdefault(key, []).append(float(val))
            except (TypeError, ValueError):
                continue
    return {key: sum(vals) / len(vals) for key, vals in values.items() if vals}


def _build_parameter_cards(stats: Dict[str, Any], model_order: List[str]) -> str:
    cards = []
    for model in model_order:
        param_map = _collect_average_params(stats, model)
        if not param_map:
            body = "<div class='param-row'><em>No parameters reported.</em></div>"
        else:
            rows = []
            for key in sorted(param_map.keys()):
                rows.append(
                    f"""
                    <div class="param-row">
                        <div class="param-key">{_html_escape(key)}</div>
                        <div class="param-val">{_fmt(param_map[key], 6)}</div>
                    </div>
                    """
                )
            body = "".join(rows)
        cards.append(
            f"""
            <div class="param-card">
                <div class="param-header">{_html_escape(model.upper())} Parameters (run-average)</div>
                {body}
            </div>
            """
        )
    return "<div class='param-card-container'>" + "".join(cards) + "</div>"


def _build_plot_gallery(plot_dir: Path) -> str:
    plots = [
        ("aggregated_model_totals.png", "Aggregated model totals (χ², AIC, BIC)"),
        ("aggregated_dataset_breakdown.png", "Aggregated dataset χ² contributions"),
    ]
    blocks = []
    for filename, caption in plots:
        path = plot_dir / filename
        if not path.exists():
            continue
        blocks.append(
            f"""
            <div class="plot-block">
                <div class="plot-img">
                    <img src="plots/{_html_escape(filename)}" alt="{_html_escape(caption)}" />
                </div>
                <div class="plot-caption">{_html_escape(caption)}</div>
            </div>
            """
        )
    if not blocks:
        return "<p>No aggregated plots available. Run the plot generator first.</p>"
    return "<div class='plot-gallery'>" + "".join(blocks) + "</div>"


# ----------------------------------------------------------------------
# Per-run sections
# ----------------------------------------------------------------------

def _build_run_metadata(run: Dict[str, Any]) -> str:
    # Handle case where meta might be a list or dict
    meta_list = run.get("meta", [])
    if isinstance(meta_list, dict):
        meta = meta_list
    else:
        # If it's a list, convert to a single dict with combined values
        meta = {}
        for item in meta_list:
            if isinstance(item, dict):
                meta.update(item)
    
    state = run.get("state", {})
    captured = meta.get("captured_at") if isinstance(meta, dict) else None
    budgets = meta.get("budgets", {}) if isinstance(meta, dict) else {}
    env = meta.get("env_meta", {}) if isinstance(meta, dict) else {}
    seeds = meta.get("seeds", {}) if isinstance(meta, dict) else {}
    lines = []

    path = run['path']
    if isinstance(path, list) and path:
        path = path[0]  # Take the first path if it's a non-empty list
    lines.append(f"<li><strong>Directory</strong>: {_html_escape(path)}</li>")
    if captured:
        lines.append(f"<li><strong>Captured</strong>: {_html_escape(captured)}</li>")
    git_commit = meta.get("git_commit") if isinstance(meta, dict) else None
    if git_commit:
        lines.append(f"<li><strong>Git commit</strong>: {_html_escape(git_commit)}</li>")
    if budgets:
        budget_str = ", ".join(f"{k}={v}" for k, v in budgets.items())
        lines.append(f"<li><strong>Budgets</strong>: {_html_escape(budget_str)}</li>")
    if env:
        env_str = ", ".join(f"{k}={v}" for k, v in env.items())
        lines.append(f"<li><strong>Environment</strong>: {_html_escape(env_str)}</li>")
    if seeds:
        seed_str = ", ".join(f"{k}={v}" for k, v in seeds.items())
        lines.append(f"<li><strong>Seeds</strong>: {_html_escape(seed_str)}</li>")
    if state:
        last_step = state.get("last_step")
        if last_step:
            lines.append(f"<li><strong>Last step</strong>: {_html_escape(last_step)}</li>")

    return "<ul class='meta-list'>" + "".join(lines) + "</ul>"


def _build_run_scenario_table(run: Dict[str, Any], model_order: List[str]) -> str:
    scenarios = [
        s for s in run.get("scenarios", []) if not s.get("id", "").startswith("scout:")
    ]
    if not scenarios:
        return "<p>No scenario data recorded.</p>"

    has_joint = any(s.get("joint") for s in scenarios)
    header = """
    <table class="data-table run-table">
      <thead>
        <tr>
          <th>Scenario</th>
          <th>Model</th>
          <th>χ²</th>
          <th>AIC</th>
          <th>BIC</th>
          <th>Runtime [s]</th>
          <th>Phase-6a</th>
    """
    if has_joint:
        header += "<th>ΔAIC (PBUF-LCDM)</th><th>ΔBIC (PBUF-LCDM)</th>"
    header += "</tr></thead><tbody>"

    rows = []
    for scenario in scenarios:
        scenario_id = scenario.get("id", "unknown")
        joint = scenario.get("joint", {})
        entries = [
            (model, scenario.get("models", {}).get(model))
            for model in model_order
            if scenario.get("models", {}).get(model)
        ]
        if not entries:
            continue
        rowspan = len(entries)
        delta_aic = joint.get("deltas", {}).get("delta_aic")
        delta_bic = joint.get("deltas", {}).get("delta_bic")

        for idx, (model, record) in enumerate(entries):
            fit_stats = record.get("fit_stats", {})
            runtime = record.get("runtime", {}).get("wall_seconds")
            phase6a = record.get("physics_flags", {}).get("phase6a_passed")
            phase_applied = record.get("physics_flags", {}).get("phase6a_applied")
            phase_label = "—"
            if phase6a is not None:
                phase_label = "✅ pass" if phase6a else "⚠️ fail"
                if phase_applied:
                    phase_label += " (enforced)"

            row = "<tr>"
            if idx == 0:
                row += f"<td rowspan='{rowspan}'>{_html_escape(scenario_id)}</td>"
            row += f"<td>{_html_escape(model.upper())}</td>"
            row += f"<td>{_fmt(fit_stats.get('chi2_total'))}</td>"
            row += f"<td>{_fmt(fit_stats.get('aic'))}</td>"
            row += f"<td>{_fmt(fit_stats.get('bic'))}</td>"
            row += f"<td>{_fmt(runtime)}</td>"
            row += f"<td>{_html_escape(phase_label)}</td>"
            if has_joint:
                if idx == 0:
                    row += f"<td rowspan='{rowspan}'>{_fmt(delta_aic)}</td>"
                    row += f"<td rowspan='{rowspan}'>{_fmt(delta_bic)}</td>"
            row += "</tr>"
            rows.append(row)

    return header + "".join(rows) + "</tbody></table>"


def _build_run_plot_gallery(run: Dict[str, Any], plot_dir: Path) -> str:
    run_name = run["name"]
    candidates = [
        ("scenario_chi2.png", "Scenario χ² totals"),
        ("scenario_runtime.png", "Scenario wall time"),
        ("parameter_scatter.png", "Best-fit H₀ vs Ωₘ"),
        ("joint_deltas.png", "Joint ΔAIC / ΔBIC"),
    ]
    blocks = []
    for filename, caption in candidates:
        path = plot_dir / run_name / filename
        if not path.exists():
            continue
        blocks.append(
            f"""
            <div class="plot-block">
                <div class="plot-img">
                    <img src="plots/{_html_escape(run_name)}/{_html_escape(filename)}" alt="{_html_escape(caption)}" />
                </div>
                <div class="plot-caption">{_html_escape(caption)}</div>
            </div>
            """
        )
    if not blocks:
        return "<p>No run-specific plots available.</p>"
    return "<div class='plot-gallery'>" + "".join(blocks) + "</div>"


# ----------------------------------------------------------------------
# Metadata + notes
# ----------------------------------------------------------------------

def _build_metadata_block(stats: Dict[str, Any]) -> str:
    pretty = json.dumps(stats, indent=2)
    return f"""
    <details class="raw-block">
      <summary>Raw statistics object (click to expand)</summary>
      <pre>{_html_escape(pretty)}</pre>
    </details>
    """


def _build_notes_block() -> str:
    return """
    <div class="notes-block">
      <h3>Methodology Notes</h3>
      <ul>
        <li>Aggregates combine the primary scenario of each science run (highest data volume).</li>
        <li>ΔAIC/ΔBIC are computed as (MODEL<sub>2</sub> − MODEL<sub>1</sub>); positive values favour MODEL<sub>1</sub>.</li>
        <li>Phase-6a status reflects the guard-rail enforcement during the coordinate optimisation.</li>
        <li>Per-run parameter cards display the mean of best-fit values across runs.</li>
        <li>Plots are generated via <code>reports.plotter</code>; regenerate them after new science runs.</li>
      </ul>
    </div>
    """


def _base_css() -> str:
    return """
    <style>
    body {
        font-family: system-ui, -apple-system, Roboto, "Helvetica Neue", Arial, sans-serif;
        background: #0b0d10;
        color: #f0f3f6;
        margin: 0;
        padding: 2rem;
        line-height: 1.5;
    }
    h1, h2, h3 {
        font-weight: 600;
        color: #fff;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    h1 { font-size: 1.8rem; margin-top: 0; }
    h2 { font-size: 1.25rem; border-left: 4px solid #4e8cff; padding-left: 0.6rem; }
    h3 { font-size: 1.05rem; color: #9cb3ff; margin-top: 1.5rem; }

    .section { margin-top: 2rem; }

    .data-table {
        border-collapse: collapse;
        width: 100%;
        margin: 1rem 0 2rem;
        font-size: 0.9rem;
        background: #1a1f27;
        color: #d5d9e0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
        border-radius: 8px;
        overflow: hidden;
    }
    .data-table.small { width: auto; min-width: 320px; }
    .data-table th, .data-table td {
        border: 1px solid #3a4356;
        padding: 0.5rem 0.75rem;
        text-align: right;
    }
    .data-table th:first-child,
    .data-table td:first-child {
        text-align: left;
    }

    .run-table td:nth-child(2) { text-align: left; }

    .param-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .param-card {
        background: #1a1f27;
        border: 1px solid #3a4356;
        border-radius: 8px;
        padding: 1rem 1rem 0.5rem;
        min-width: 220px;
        max-width: 260px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.6);
        flex: 0 0 auto;
    }
    .param-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #3a4356;
        padding-bottom: 0.5rem;
    }
    .param-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
        color: #d9dfe8;
    }
    .param-key { color: #9cb3ff; }

    .plot-gallery {
        display: grid;
        gap: 1.5rem;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        margin: 1rem 0 2rem;
    }
    .plot-block {
        background: #161b24;
        border: 1px solid #30394a;
        border-radius: 8px;
        padding: 0.75rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.45);
    }
    .plot-img {
        display: flex;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    .plot-img img {
        max-width: 100%;
        border-radius: 4px;
        border: 1px solid #2a3140;
    }
    .plot-caption {
        font-size: 0.8rem;
        color: #a8b5d1;
        text-align: center;
    }

    .notes-block {
        background: #141922;
        border: 1px solid #2f3849;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.45);
    }
    .notes-block ul {
        margin: 0.5rem 0 0;
        padding-left: 1.1rem;
    }
    .notes-block li { margin-bottom: 0.35rem; }

    .raw-block {
        margin: 2rem 0;
        background: #121720;
        border: 1px solid #2a3242;
        border-radius: 8px;
        padding: 1rem;
    }
    .raw-block pre {
        white-space: pre-wrap;
        font-size: 0.75rem;
        overflow-x: auto;
        color: #cdd7f2;
    }

    .meta-list {
        margin: 0.5rem 0 1.5rem;
        padding-left: 1.1rem;
        color: #d5d9e0;
        font-size: 0.9rem;
    }
    .meta-list li { margin-bottom: 0.35rem; }

    .explain {
        font-size: 0.8rem;
        color: #a7b3cf;
        margin-top: 0.75rem;
    }
    </style>
    """


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def build_html_report(stats: Dict[str, Any], plot_dir: Path, output_file: Path) -> str:
    # Debug output
    print("Debug: Starting HTML report generation")
    print(f"Debug: stats keys: {list(stats.keys())}")
    if 'runs' in stats:
        print(f"Debug: Found {len(stats['runs'])} runs in stats")
        if stats['runs'] and isinstance(stats['runs'], list):
            print(f"Debug: First run keys: {list(stats['runs'][0].keys())}")
            if 'scenarios' in stats['runs'][0]:
                print(f"Debug: First run has {len(stats['runs'][0]['scenarios'])} scenarios")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        aggregated = stats.get("aggregated", {})
        print(f"Debug: Aggregated keys: {list(aggregated.keys())}")
        
        model_order = list(aggregated.get("models", {}).keys())
        if not model_order:
            raise ValueError("No model statistics available in aggregated data.")
            
        model_a = model_order[0]
        model_b = model_order[1] if len(model_order) > 1 else model_a
        print(f"Debug: Models found - {model_order}")

        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        title_block = f"""
        <header>
            <h1>PBUF Science Run Comparison Report</h1>
            <p class="subtitle">
                Generated {timestamp} — Models: {', '.join(m.upper() for m in model_order)}
            </p>
        </header>
        """

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'/>",
            "<title>PBUF Science Run Report</title>",
            _base_css(),
            "</head>",
            "<body>",
            title_block,
            _section_header("1. Aggregated Overview"),
            "<h3>1.1 Dataset Diagnostics</h3>",
            _build_dataset_table(aggregated, model_a, model_b),
            "<h3>1.2 Global Model Totals</h3>",
            _build_global_summary(aggregated),
            "<h3>1.3 Model Preference</h3>",
            _build_model_preference(aggregated, model_a, model_b),
            "<h3>1.4 Average Best-Fit Parameters</h3>",
            _build_parameter_cards(stats, model_order),
            "<h3>1.5 Aggregated Plots</h3>",
            _build_plot_gallery(Path(plot_dir)),
        ]

        # Per-run sections
        for idx, run in enumerate(stats.get("runs", []), start=1):
            html_parts.append(_section_header(f"{idx + 1}. Science Run — {run['name']}"))
            html_parts.append("<h3>Metadata</h3>")
            html_parts.append(_build_run_metadata(run))
            html_parts.append("<h3>Scenario Summary</h3>")
            html_parts.append(_build_run_scenario_table(run, model_order))
            html_parts.append("<h3>Plots</h3>")
            html_parts.append(_build_run_plot_gallery(run, Path(plot_dir)))

        html_parts.extend([
            _section_header(f"{len(stats.get('runs', [])) + 2}. Appendices"),
            "<h3>Appendix A — Raw Statistics Object</h3>",
            _build_metadata_block(stats),
            "<h3>Appendix B — Notes</h3>",
            _build_notes_block(),
            "</body>",
            "</html>",
        ])

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("".join(html_parts))

        return str(output_path.resolve())
    except Exception as e:
        print(f"Error in build_html_report: {str(e)}")
        print(f"Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise
