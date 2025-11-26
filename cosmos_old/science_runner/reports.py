"""Report generation helpers for science runs."""

from __future__ import annotations

from html import escape as _html_escape
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from .utils import ensure_dir, safe_write_json, serialize_value

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


class ReportGenerator:
    def __init__(self) -> None:
        self._plt = plt

    def generate(
        self,
        *,
        run_dir: Path,
        model_dir: Path,
        model_name: str,
        run_meta: Dict[str, Any],
        best_params: dict[str, float],
        best_chi2: float,
        chi2_breakdown: dict[str, float],
        fit_outputs: dict[str, Any],
        predictions: dict[str, Any],
        report_formats: Sequence[str],
    ) -> None:
        reports_dir = model_dir / "reports"
        ensure_dir(reports_dir)
        sanitized_formats = {fmt.strip().lower() for fmt in report_formats}
        context = {
            "run_name": run_meta.get("run_name"),
            "timestamp": run_meta.get("timestamp"),
            "model": model_name,
            "engine": run_meta.get("engine"),
            "mode": run_meta.get("mode"),
            "fits": run_meta.get("fits_used"),
            "run_meta": run_meta,
            "best_fit": {
                "parameters": best_params,
                "chi2": best_chi2,
            },
            "chi2_breakdown": chi2_breakdown,
            "predictions": predictions,
            "fits": {
                name: {"chi2": info.get("chi2"), "extras": serialize_value(info.get("extras"))}
                for name, info in fit_outputs.items()
            },
        }
        if "json" in sanitized_formats:
            safe_write_json(reports_dir / "summary.json", context)
        if "html" in sanitized_formats:
            html = self._render_html(context)
            (reports_dir / "summary.html").write_text(html, encoding="utf-8")
        if "pdf" in sanitized_formats and self._plt is not None:
            self._render_pdf(context, reports_dir / "summary.pdf")

    def _render_html(self, context: Dict[str, Any]) -> str:
        run_meta = context.get("run_meta") or {}
        engine_settings = run_meta.get("engine_settings") or {}

        def _format_value(value: Any) -> str:
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.6g}"
            if isinstance(value, (list, tuple)):
                return ", ".join(str(item) for item in value)
            if isinstance(value, dict):
                return ", ".join(f"{key}={val}" for key, val in value.items())
            return str(value)

        def _format_prediction(value: Any) -> str:
            if isinstance(value, (list, tuple)):
                snippet = ", ".join(_format_value(item) for item in value[:5])
                more = f", ... ({len(value)} total)" if len(value) > 5 else ""
                return f"[{snippet}{more}]"
            return _format_value(value)

        def _render_engine_table() -> str:
            rows = []
            for key, val in sorted(engine_settings.items()):
                rows.append(f"<tr><td>{_html_escape(str(key))}</td><td>{_html_escape(_format_value(val))}</td></tr>")
            if not rows:
                return "<p>Engine settings were not recorded.</p>"
            return (
                "<table class='thin'><thead><tr><th>Setting</th><th>Value</th></tr></thead>"
                "<tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )

        def _render_fit_rows() -> str:
            rows = []
            for name, info in sorted(context["fits"].items()):
                chi2 = info.get("chi2")
                extras = info.get("extras")
                extras_html = ""
                if extras:
                    pretty = _html_escape(json.dumps(extras, indent=2))
                    extras_html = f"<details><summary>Extras</summary><pre>{pretty}</pre></details>"
                rows.append(
                    "<tr>"
                    f"<td>{_html_escape(name)}</td>"
                    f"<td>{_html_escape(_format_value(chi2))}</td>"
                    f"<td>{extras_html or '—'}</td>"
                    "</tr>"
                )
            if not rows:
                return "<p>No fit outputs were recorded.</p>"
            return (
                "<table class='thin'>"
                "<thead><tr><th>Fit</th><th>χ²</th><th>Details</th></tr></thead>"
                "<tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )

        css = """
        body {
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
            background: #f7f7f7;
            color: #1e1e1e;
            margin: 0;
            padding: 0 1rem 2rem;
        }
        .hero {
            padding: 1.5rem;
            background: #111827;
            color: #f9fafb;
            text-align: left;
        }
        .content {
            max-width: 960px;
            margin: 0 auto;
            padding: 1rem 0;
        }
        .panel {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }
        .panel h2 {
            margin-top: 0;
            margin-bottom: 0.75rem;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        table th,
        table td {
            padding: 0.35rem 0.5rem;
            border-bottom: 1px solid #e2e8f0;
            text-align: left;
        }
        table th {
            background: #f8fafc;
        }
        table.thin td {
            border: none;
            padding: 0.2rem 0.5rem;
        }
        details {
            margin-top: 0.25rem;
            font-size: 0.85rem;
        }
        pre {
            background: #0f172a;
            color: #f8fafc;
            padding: 0.4rem;
            border-radius: 0.45rem;
            max-height: 200px;
            overflow: auto;
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
            color: #6b7280;
            display: block;
        }
        .grid-item span {
            font-weight: 600;
        }
        ul.predictions {
            padding-left: 1rem;
        }
        @media (max-width: 640px) {
            .hero {
                padding: 1rem;
            }
            .panel {
                padding: 1rem;
            }
        }
        """

        lines = [
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
            "<title>Science Run Summary</title>",
            f"<style>{css}</style></head><body>",
            "<header class='hero'>",
            f"<h1>Science Run Summary</h1>",
            f"<p>Run: {_html_escape(context['run_name'] or '-') } ({_html_escape(context['mode'] or '-')})</p>",
            f"<p>Model: {_html_escape(context['model'] or '-') } · Engine: {_html_escape(context['engine'] or '-')}</p>",
            "</header>",
            "<main class='content'>",
            "<section class='panel'>",
            "<h2>Run metadata</h2>",
            "<div class='grid'>",
            "<div class='grid-item'><label>Timestamp</label><span>"
            f"{_html_escape(str(context['timestamp'] or run_meta.get('timestamp') or '-'))}</span></div>",
            "<div class='grid-item'><label>Fits</label><span>"
            f"{_html_escape(', '.join(run_meta.get('fits_used', []) or ['-']))}</span></div>",
            "<div class='grid-item'><label>Success</label><span>"
            f"{_html_escape(str(run_meta.get('success')) if 'success' in run_meta else '-')}</span></div>",
            "<div class='grid-item'><label>Total runtime (s)</label><span>"
            f"{_html_escape(str(run_meta.get('total_runtime') or '-'))}</span></div>",
            "</div>",
            "<h3>Engine settings</h3>",
            _render_engine_table(),
            "</section>",
            "<section class='panel'>",
            "<h2>Best-fit parameters</h2>",
            "<table>",
            "<thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>",
        ]

        for key, value in sorted(context["best_fit"]["parameters"].items()):
            lines.append(
                "<tr>"
                f"<td>{_html_escape(str(key))}</td>"
                f"<td>{_html_escape(_format_value(value))}</td>"
                "</tr>"
            )

        lines.append("</tbody></table></section>")

        lines.extend(
            [
                "<section class='panel'>",
                "<h2>χ² breakdown</h2>",
                "<table>",
                "<thead><tr><th>Fit</th><th>χ²</th></tr></thead>",
                "<tbody>",
            ]
        )
        for key, value in context["chi2_breakdown"].items():
            lines.append(
                "<tr>"
                f"<td>{_html_escape(str(key))}</td>"
                f"<td>{_html_escape(_format_value(value))}</td>"
                "</tr>"
            )
        lines.extend(["</tbody></table></section>"])

        lines.extend(
            [
                "<section class='panel'>",
                "<h2>Predictions</h2>",
                "<ul class='predictions'>",
            ]
        )
        for key, value in context["predictions"].items():
            lines.append(f"<li><strong>{_html_escape(str(key))}</strong>: {_html_escape(_format_prediction(value))}</li>")
        lines.extend(["</ul></section>"])

        lines.extend(
            [
                "<section class='panel'>",
                "<h2>Fit outputs</h2>",
                _render_fit_rows(),
                "</section>",
            ]
        )

        lines.extend(["</main></body></html>"])
        return "".join(lines)

    def _render_pdf(self, context: Dict[str, Any], pdf_path: Path) -> None:
        if self._plt is None:
            return
        lines = [
            f"Run: {context['run_name']} ({context['mode']})",
            f"Model: {context['model']}",
            f"Engine: {context['engine']}",
            f"Best χ²: {context['best_fit']['chi2']:.3f}",
            "Parameters:",
        ]
        for key, value in context["best_fit"]["parameters"].items():
            lines.append(f"  {key}: {value}")
        lines.append("χ² breakdown:")
        for key, value in context["chi2_breakdown"].items():
            lines.append(f"  {key}: {value:.3f}")
        fig, ax = self._plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        for idx, line in enumerate(lines):
            ax.text(0.01, 0.95 - idx * 0.04, line, fontsize=10, va="top")
        fig.tight_layout()
        fig.savefig(pdf_path)
        self._plt.close(fig)
