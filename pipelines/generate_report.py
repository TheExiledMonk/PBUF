"""
Generate a single-fit HTML report from a JSON results file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from utils import logging as log


def _environment() -> Environment:
    templates = Path("reports/templates").resolve()
    return Environment(
        loader=FileSystemLoader(str(templates)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def render_report(result: dict, run_dir: Path) -> str:
    """Render the per-fit HTML report and return the absolute path."""

    env = _environment()
    template = env.get_template("fit_report.html")
    html = template.render(run=result)
    out_path = run_dir / f"{result['model'].lower()}_report.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single-fit HTML report.")
    parser.add_argument("--input", required=True, help="Path to fit_results.json")
    parser.add_argument("--outdir", default=None, help="Optional override for output directory")
    args = parser.parse_args()

    json_path = Path(args.input).resolve()
    result = json.loads(json_path.read_text(encoding="utf-8"))

    run_dir = json_path.parent if args.outdir is None else Path(args.outdir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = render_report(result, run_dir)
    log.info(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
