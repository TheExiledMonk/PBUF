"""
Report Pipeline — PBUF Cosmological Model Comparison
====================================================

Main orchestrator for generating analytical reports (HTML, Markdown, PDF, JSON)
comparing LCDM and PBUF fits across all datasets.

Each reporting phase (summary, plots, exports) is modular and extendable.
The LLM coder should implement missing functions in the respective modules:
  - summary_builder.py
  - plotter.py
  - html_report.py
  - markdown_writer.py
  - pdf_exporter.py
  - json_exporter.py

Example:
--------
    from reports.report_pipeline import build_full_report

    build_full_report(
        models=["lcdm", "pbuf"],
        output_dir="reports/output/",
        formats=["html", "pdf", "md", "json"]
    )
"""

import json
from pathlib import Path
from datetime import datetime

# Internal module imports (LLM will implement these)
from reports.summary_builder import collect_fit_results, compute_model_stats
from reports.plotter import generate_all_plots
from reports.markdown_writer import write_markdown_summary
from reports.html_report import build_html_report
from reports.pdf_exporter import export_pdf
from reports.json_exporter import export_json


# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------

def build_full_report(
    models=("lcdm", "pbuf"),
    output_dir="reports/output/",
    formats=("html", "md", "pdf", "json"),
    science_run_root="data/science_runs",
    verbose=True,
):
    """
    Orchestrates report generation for all cosmological models.

    Parameters
    ----------
    models : list[str]
        List of models to include in report (default: ["lcdm", "pbuf"])
    output_dir : str
        Directory to store all outputs (created if missing)
    formats : list[str]
        Output formats to generate: "html", "pdf", "md", "json"
    verbose : bool
        Print progress logs if True
    """

    # ------------------------------------------------------------------
    # 1. Prepare output directory
    # ------------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    if verbose:
        print(f"\n[REPORT] Generating PBUF report suite")
        print(f"[INFO] Models: {models}")
        print(f"[INFO] Output directory: {output_path}")
        print(f"[INFO] Timestamp: {timestamp}")

    # ------------------------------------------------------------------
    # 2. Load fit results
    # ------------------------------------------------------------------
    if verbose:
        print("\n[STEP 1] Collecting science run artifacts...")

    run_bundle = collect_fit_results(science_run_root)
    runs = run_bundle.get("runs", [])
    if not runs:
        raise RuntimeError(f"No science runs found in {science_run_root}/")

    if verbose:
        print(f"[INFO] Science run root: {run_bundle.get('root')}")
        print(f"[OK] Discovered {len(runs)} science run(s).")

    # ------------------------------------------------------------------
    # 3. Compute statistics (χ², AIC, BIC, ΔAIC, etc.)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[STEP 2] Computing model statistics...")

    stats = compute_model_stats(run_bundle, models=models)

    if verbose:
        datasets = stats.get("aggregated", {}).get("datasets", {})
        print(f"[OK] Model statistics computed successfully (datasets: {len(datasets)}).")

    # ------------------------------------------------------------------
    # 4. Generate all plots (H(z), μ(z), fσ₈, etc.)
    # ------------------------------------------------------------------
    if verbose:
        print("\n[STEP 3] Generating plots...")

    plot_dir = output_path / "plots"
    plot_dir.mkdir(exist_ok=True)
    generated_plots = generate_all_plots(stats, plot_dir)

    if verbose:
        total_plots = sum(len(paths) for paths in generated_plots.values())
        print(f"[OK] Generated {total_plots} plot(s) under {plot_dir}/")

    # ------------------------------------------------------------------
    # 5. Write text-based summaries
    # ------------------------------------------------------------------
    if "md" in formats:
        if verbose:
            print("\n[STEP 4] Writing Markdown summary...")
        write_markdown_summary(stats, output_path / "summary.md")

    if "json" in formats:
        if verbose:
            print("\n[STEP 5] Exporting JSON data...")
        export_json(stats, output_path / "results.json")

        # Per-run JSON snapshots for easy inspection
        runs_dir = output_path / "runs"
        runs_dir.mkdir(exist_ok=True)
        for run in stats.get("runs", []):
            run_file = runs_dir / f"{run['name']}.json"
            run_file.write_text(
                json.dumps(run, indent=2),
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # 6. Build interactive HTML dashboard
    # ------------------------------------------------------------------
    if "html" in formats:
        if verbose:
            print("\n[STEP 6] Building HTML report...")
        build_html_report(stats, plot_dir, output_path / "report.html")

    # ------------------------------------------------------------------
    # 7. Export PDF report
    # ------------------------------------------------------------------
    if "pdf" in formats:
        if verbose:
            print("\n[STEP 7] Exporting PDF report...")
        export_pdf(stats, plot_dir, output_path / "report.pdf")

    # ------------------------------------------------------------------
    # 8. Final confirmation
    # ------------------------------------------------------------------
    if verbose:
        print("\n✅ Report generation completed successfully!")
        print(f"📊 Outputs available at: {output_path.resolve()}")
        print(f"🕒 Generated: {timestamp}")

    return {
        "timestamp": timestamp,
        "models": models,
        "output": str(output_path),
        "formats": formats,
        "science_runs": [run["name"] for run in runs],
    }


# ----------------------------------------------------------------------
# CLI Entry
# ----------------------------------------------------------------------

if __name__ == "__main__":
    build_full_report()


# Notes for the LLM Coder
# Component                    Description
# collect_fit_results()        Traverse `data/science_runs` and capture artifacts + metadata
# compute_model_stats()        Aggregate per-model/dataset totals, deltas, and run provenance
# generate_all_plots()         Produce aggregated + per-run visualisations (bars, scatter, deltas)
# write_markdown_summary()     Emit Markdown overview covering datasets, globals, and runs
# export_json()                Dump machine-readable bundle plus per-run JSON snapshots
# build_html_report()          Construct interactive dashboard with plots and run sections
# export_pdf()                 Create printable publication-ready report (requires weasyprint)

# Future Integration
# When CLI is ready, it can call:
# python cli.py report --models lcdm,pbuf --format html,pdf
#
# Internally it just runs:
# from reports.report_pipeline import build_full_report
# build_full_report(models=["lcdm", "pbuf"], formats=["html", "pdf"])
