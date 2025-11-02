"""
PDF Exporter — Publication-Ready Reports for PBUF Framework
==========================================================

Converts HTML reports to publication-quality PDF format using:
  - HTML to PDF conversion (via weasyprint or similar)
  - Embedded plots and figures
  - Professional formatting and styling
  - Scientific publication standards

This creates PDFs suitable for:
  - Journal submissions
  - Conference proceedings
  - Technical reports
  - Archive and sharing

Usage:
------
    from reports.pdf_exporter import export_pdf
    export_pdf(stats, plot_dir, "reports/output/report.pdf")

Note: Requires weasyprint or similar PDF generation library.
Install with: pip install weasyprint
"""

from pathlib import Path
from typing import Dict, Any, List
import math

# Optional PDF generation - fallback to HTML if not available
try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False


def export_pdf(stats: Dict[str, Any], plot_dir: Path, output_file: str = "reports/output/report.pdf"):
    """
    Export statistics and plots to PDF format.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from compute_model_stats()
    plot_dir : Path
        Directory containing plot images
    output_file : str
        Path to output PDF file

    Returns
    -------
    str
        Path to the created PDF file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not WEASYPRINT_AVAILABLE:
        print("[WARN] weasyprint not available. Creating HTML fallback instead.")
        return _create_html_fallback(stats, plot_dir, output_path)

    try:
        html_content = _generate_pdf_html(stats, Path(plot_dir))

        # Convert to PDF
        font_config = FontConfiguration()
        html_doc = HTML(string=html_content)
        css_string = _get_pdf_css()

        html_doc.write_pdf(
            output_path,
            stylesheets=[CSS(string=css_string)],
            font_config=font_config
        )

        print(f"[OK] PDF report created: {output_path.resolve()}")
        return str(output_path)

    except Exception as e:
        print(f"[ERROR] PDF generation failed: {e}")
        print("[FALLBACK] Creating HTML version instead.")
        return _create_html_fallback(stats, plot_dir, output_path)


def _fmt(value: Any, precision: int = 4) -> str:
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


def _generate_pdf_html(stats: Dict[str, Any], plot_dir: Path) -> str:
    """Generate HTML content optimized for PDF conversion."""
    from datetime import datetime

    aggregated = stats.get("aggregated", {})
    models = list(aggregated.get("models", {}).keys())
    if not models:
        raise ValueError("No model statistics available for PDF export.")
    model_a = models[0]
    model_b = models[1] if len(models) > 1 else model_a

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    dataset_rows: List[str] = []
    for dataset, entry in sorted(aggregated.get("datasets", {}).items()):
        a = entry.get(model_a, {})
        b = entry.get(model_b, {})
        dataset_rows.append(
            "<tr>"
            f"<td>{dataset.upper()}</td>"
            f"<td>{_fmt(a.get('chi2'))}</td>"
            f"<td>{_fmt(b.get('chi2'))}</td>"
            f"<td>{_fmt((b.get('chi2') or 0) - (a.get('chi2') or 0))}</td>"
            f"<td>{_fmt(a.get('AIC'))}</td>"
            f"<td>{_fmt(b.get('AIC'))}</td>"
            f"<td>{_fmt((b.get('AIC') or 0) - (a.get('AIC') or 0))}</td>"
            f"<td>{_fmt(a.get('BIC'))}</td>"
            f"<td>{_fmt(b.get('BIC'))}</td>"
            f"<td>{_fmt((b.get('BIC') or 0) - (a.get('BIC') or 0))}</td>"
            "</tr>"
        )

    model_rows: List[str] = []
    for model_name, data in aggregated.get("models", {}).items():
        model_rows.append(
            "<tr>"
            f"<td>{model_name.upper()}</td>"
            f"<td>{_fmt(data.get('chi2_total'), 6)}</td>"
            f"<td>{_fmt(data.get('AIC_total'), 6)}</td>"
            f"<td>{_fmt(data.get('BIC_total'), 6)}</td>"
            f"<td>{_fmt(data.get('chi2_reduced_total'), 6)}</td>"
            f"<td>{_fmt(data.get('n_data_total'), 0)}</td>"
            f"<td>{_fmt(data.get('n_params_total'), 0)}</td>"
            f"<td>{_fmt(data.get('run_count'), 0)}</td>"
            "</tr>"
        )

    comp = aggregated.get("global", {}).get("comparison", {})
    comparison_table = ""
    if comp:
        comparison_table = (
            "<table class='table'>"
            "<thead><tr><th>Metric</th><th>Value</th><th>Preferred Model</th></tr></thead>"
            "<tbody>"
            f"<tr><td>ΔAIC ({model_b.upper()}-{model_a.upper()})</td>"
            f"<td>{_fmt(comp.get(f'ΔAIC ({model_b}-{model_a})'))}</td>"
            f"<td>{comp.get('preferred_model_AIC', '—').upper()}</td></tr>"
            f"<tr><td>ΔBIC ({model_b.upper()}-{model_a.upper()})</td>"
            f"<td>{_fmt(comp.get(f'ΔBIC ({model_b}-{model_a})'))}</td>"
            f"<td>{comp.get('preferred_model_BIC', '—').upper()}</td></tr>"
            "</tbody></table>"
        )

    run_rows: List[str] = []
    for run in stats.get("runs", []):
        primary_id = run.get("primary_scenario_id") or "—"
        scenario = next((s for s in run.get("scenarios", []) if s.get("id") == primary_id), None)
        joint = scenario.get("joint") if scenario else None
        delta_aic = joint.get("deltas", {}).get("delta_aic") if joint else None
        delta_bic = joint.get("deltas", {}).get("delta_bic") if joint else None
        run_rows.append(
            "<tr>"
            f"<td>{run['name']}</td>"
            f"<td>{primary_id}</td>"
            f"<td>{_fmt(delta_aic)}</td>"
            f"<td>{_fmt(delta_bic)}</td>"
            "</tr>"
        )

    plot_blocks: List[str] = []
    for filename in ["aggregated_model_totals.png", "aggregated_dataset_breakdown.png"]:
        path = plot_dir / filename
        if path.exists():
            plot_blocks.append(
                f"<div class='plot'><img src='file://{path.resolve()}' alt='{filename}'/></div>"
            )

    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>PBUF Science Run Report</title>
        <style>{_get_pdf_css()}</style>
    </head>
    <body>
        <h1>PBUF Science Run Report</h1>
        <p class="meta">Generated {timestamp} &mdash; Models: {', '.join(m.upper() for m in models)}</p>

        <h2>1. Dataset Summary</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Dataset</th>
                    <th>χ² {model_a.upper()}</th>
                    <th>χ² {model_b.upper()}</th>
                    <th>Δχ²</th>
                    <th>AIC {model_a.upper()}</th>
                    <th>AIC {model_b.upper()}</th>
                    <th>ΔAIC</th>
                    <th>BIC {model_a.upper()}</th>
                    <th>BIC {model_b.upper()}</th>
                    <th>ΔBIC</th>
                </tr>
            </thead>
            <tbody>
                {''.join(dataset_rows)}
            </tbody>
        </table>

        <h2>2. Global Model Totals</h2>
        <table class="table">
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
                {''.join(model_rows)}
            </tbody>
        </table>

        <h2>3. Model Preference</h2>
        {comparison_table or '<p>No comparison metrics available.</p>'}

        <h2>4. Run Overview</h2>
        <table class="table">
            <thead><tr><th>Run</th><th>Primary Scenario</th><th>ΔAIC (PBUF-LCDM)</th><th>ΔBIC (PBUF-LCDM)</th></tr></thead>
            <tbody>
                {''.join(run_rows) or '<tr><td colspan=\"4\">No runs found.</td></tr>'}
            </tbody>
        </table>

        <h2>5. Aggregated Plots</h2>
        <div class="plots">
            {''.join(plot_blocks) or '<p>Plots not available.</p>'}
        </div>

        <h2>6. Notes</h2>
        <ul>
            <li>Aggregates sum the primary scenario for each science run.</li>
            <li>Δ metrics follow the (MODEL₂ − MODEL₁) convention; positive values favour MODEL₁.</li>
            <li>Plots are sourced from <code>reports.plotter</code> in the working directory.</li>
        </ul>
    </body>
    </html>"""

    return html


def _get_pdf_css() -> str:
    """PDF-optimized CSS for professional publication appearance."""
    return """
    body {
        font-family: 'Times New Roman', Times, serif;
        font-size: 11pt;
        line-height: 1.4;
        color: #333;
        margin: 0;
        padding: 2cm;
        background: white;
    }

    h1 {
        font-size: 18pt;
        font-weight: bold;
        text-align: center;
        margin: 0 0 1cm 0;
        color: #000;
    }

    h2 {
        font-size: 14pt;
        font-weight: bold;
        margin: 1.5cm 0 0.5cm 0;
        color: #000;
        border-bottom: 1px solid #666;
        padding-bottom: 0.2cm;
        page-break-after: avoid;
    }

    h3 {
        font-size: 12pt;
        font-weight: bold;
        margin: 1cm 0 0.3cm 0;
        color: #333;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.5cm 0;
        font-size: 9pt;
        background: white;
        page-break-inside: avoid;
    }

    th {
        background: #f0f0f0;
        font-weight: bold;
        padding: 0.3cm;
        border: 1px solid #999;
        text-align: center;
    }

    td {
        padding: 0.3cm;
        border: 1px solid #999;
        text-align: right;
    }

    td:first-child {
        text-align: left;
        font-weight: bold;
    }

    .param-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1cm;
        margin: 1cm 0;
    }

    .param-card {
        background: #f8f8f8;
        border: 1px solid #999;
        padding: 0.5cm;
        width: 6cm;
        page-break-inside: avoid;
    }

    .plot-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(8cm, 1fr));
        gap: 1cm;
        margin: 1cm 0;
    }

    .plot-block {
        background: white;
        border: 1px solid #999;
        padding: 0.5cm;
        text-align: center;
        page-break-inside: avoid;
    }

    .plot-img img {
        max-width: 100%;
        height: auto;
    }

    pre {
        background: #f5f5f5;
        padding: 0.5cm;
        border: 1px solid #999;
        font-size: 8pt;
        white-space: pre-wrap;
        page-break-inside: avoid;
    }

    @page {
        size: A4;
        margin: 2cm;
    }
    """


def _create_html_fallback(stats: Dict[str, Any], plot_dir: Path, output_path: Path) -> str:
    """Create HTML fallback when PDF generation fails."""
    html_fallback = output_path.with_suffix('.html')

    from reports.html_report import build_html_report
    result = build_html_report(stats, plot_dir, str(html_fallback))

    print(f"[INFO] HTML fallback created: {html_fallback.resolve()}")
    return result
