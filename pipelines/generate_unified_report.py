#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the unified HTML report comparing multiple fit results.
"""

from __future__ import annotations
import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from utils import logging as log
from utils.io import read_yaml, write_json_atomic


# ----------------------------------------------------------------------
# Template environment setup
# ----------------------------------------------------------------------
def _environment() -> Environment:
    templates = Path("reports/templates").resolve()
    return Environment(
        loader=FileSystemLoader(str(templates)),
        autoescape=select_autoescape(["html", "xml"]),
    )


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def _strength(delta: float) -> str:
    adelta = abs(delta)
    if adelta < 2.0:
        return "Inconclusive"
    if adelta < 4.0:
        return "Weak"
    if adelta < 10.0:
        return "Moderate"
    return "Strong"


def _summary(dataset: str, pbuf_run: dict, lcdm_run: dict, delta_aic: float, strength: str) -> str:
    if abs(delta_aic) < 2.0:
        return f"On {dataset}, PBUF and ΛCDM are statistically comparable (ΔAIC={delta_aic:.2f}, {strength.lower()})."
    better = "PBUF" if delta_aic < 0 else "ΛCDM"
    worse = "ΛCDM" if better == "PBUF" else "PBUF"
    return (
        f"On {dataset}, {better} outperforms {worse} with ΔAIC={delta_aic:.2f} "
        f"({strength.lower()} evidence)."
    )


def _detect_report_path(json_path: Path) -> Optional[str]:
    run_dir = json_path.parent
    candidates = sorted(run_dir.glob("*_report.html"))
    if candidates:
        return str(candidates[0].resolve())
    return None


# ----------------------------------------------------------------------
# Run formatter — normalize metrics and figures
# ----------------------------------------------------------------------
def _format_run(run: dict, json_path: Path) -> dict:
    """
    Normalize metric structure and flatten nested totals for display.
    Adds component-level metrics (SN, BAO, BAO_ANI, CMB).
    """
    metrics = run.get("metrics", {}) or {}

    # Flatten if metrics are nested under 'total'
    if "total" in metrics and isinstance(metrics["total"], dict):
        flat = metrics["total"].copy()
        for k, v in metrics.items():
            if k.startswith("delta_"):
                flat[k] = v
        metrics = flat

    # Include per-component breakdown
    if "total" in run.get("metrics", {}):
        details = {}
        for key in ("sn", "bao", "bao_ani", "cmb"):
            if key in run["metrics"]:
                block = run["metrics"][key]
                details[key] = {
                    "chi2": block.get("chi2"),
                    "AIC": block.get("AIC"),
                    "BIC": block.get("BIC"),
                    "p_value": block.get("p_value"),
                }
        metrics["_components"] = details

    # Display-friendly formatting
    p_value = metrics.get("p_value")
    metrics["p_value_fmt"] = f"{p_value:.3g}" if isinstance(p_value, (int, float)) else "n/a"
    metrics["chi2_fmt"] = f"{metrics.get('chi2', float('nan')):.3f}" if "chi2" in metrics else "—"
    metrics["aic_fmt"] = f"{metrics.get('AIC', float('nan')):.3f}" if "AIC" in metrics else "—"
    metrics["bic_fmt"] = f"{metrics.get('BIC', float('nan')):.3f}" if "BIC" in metrics else "—"

    run["metrics"] = metrics

    # Attach figures and provenance
    run["summary_figures"] = [
        {"label": label.replace("_", " ").title(), "path": path}
        for label, path in run.get("figures", {}).items()
    ]
    report_path = run.get("report_path") or _detect_report_path(json_path)
    if report_path:
        run["report_path"] = report_path
    if json_path:
        run["result_path"] = str(json_path.resolve())
    return run


# ----------------------------------------------------------------------
# Compute delta statistics between models
# ----------------------------------------------------------------------
def _annotate_deltas(dataset: str, runs: List[dict]) -> Optional[dict]:
    models = {run.get("model", "").upper(): run for run in runs}
    if "PBUF" not in models or "LCDM" not in models:
        return None

    pbuf_run, lcdm_run = models["PBUF"], models["LCDM"]

    def _extract_metrics(run: dict) -> dict:
        m = run.get("metrics", {})
        if not m:
            return {}
        if "total" in m and isinstance(m["total"], dict):
            return m["total"]
        return m

    pbuf_m, lcdm_m = _extract_metrics(pbuf_run), _extract_metrics(lcdm_run)

    try:
        delta_chi2 = pbuf_m["chi2"] - lcdm_m["chi2"]
        delta_aic = pbuf_m["AIC"] - lcdm_m["AIC"]
        delta_bic = pbuf_m["BIC"] - lcdm_m["BIC"]
    except KeyError:
        log.warn(f"Skipping Δ comparison for {dataset}: missing metrics.")
        return None

    strength = _strength(delta_aic)
    summary = _summary(dataset, pbuf_run, lcdm_run, delta_aic, strength)

    pbuf_run.setdefault("metrics", {}).update({
        "delta_chi2_vs_lcdm": delta_chi2,
        "delta_aic_vs_lcdm": delta_aic,
        "delta_bic_vs_lcdm": delta_bic,
    })

    return {
        "delta_chi2": delta_chi2,
        "delta_aic": delta_aic,
        "delta_bic": delta_bic,
        "strength": strength,
        "summary": summary,
    }


# ----------------------------------------------------------------------
# Unified report builder
# ----------------------------------------------------------------------
def render_unified_report(runs: List[dict], out_path: Path) -> str:
    env = _environment()
    template = env.get_template("unified_report.html")
    report_cfg = read_yaml("config/report.yml")

    dataset_groups: Dict[str, List[dict]] = defaultdict(list)
    json_lookup: Dict[str, Path] = {}

    # Normalize run IDs and map JSON paths
    for run in runs:
        if "__json_path" in run:
            run_id = run.get("run_id") or Path(run["__json_path"]).parent.name
            run["run_id"] = run_id
            json_lookup[run_id] = Path(run["__json_path"])

    # Classify datasets
    for run in runs:
        ds = run.get("dataset", {})
        json_path = str(run.get("__json_path", "")).lower()

        if "sn" in ds and "bao" in ds and "cmb" in ds:
            dataset_name = "Joint SN+BAO+CMB"
        elif "joint" in json_path:
            dataset_name = "Joint SN+BAO+CMB"
        elif "bao_ani" in ds or "anisotropic" in json_path:
            dataset_name = "BAO Anisotropic"
        elif "bao" in ds or "bao" in json_path:
            dataset_name = "BAO Mixed"
        elif "sn" in ds or "pantheon" in json_path:
            dataset_name = "Pantheon+ Supernovae"
        elif "cmb" in ds or "planck" in json_path:
            dataset_name = "Planck 2018 CMB"
        else:
            dataset_name = "Unknown Dataset"

        dataset_groups[dataset_name].append(run)

    # Sort datasets in logical order
    order_priority = [
        "Pantheon+ Supernovae",
        "Planck 2018 CMB",
        "BAO Mixed",
        "BAO Anisotropic",
        "Joint SN+BAO+CMB",
    ]
    dataset_groups = dict(sorted(
        dataset_groups.items(),
        key=lambda kv: order_priority.index(kv[0]) if kv[0] in order_priority else len(order_priority)
    ))

    dataset_blocks = []
    json_updates: Dict[Path, Dict[str, object]] = {}

    # Build per-dataset report sections
    for dataset_name, run_list in dataset_groups.items():
        formatted_runs = []
        for run in run_list:
            json_path = json_lookup.get(run["run_id"])
            if json_path is None:
                vector_path = run.get("data_vectors", {}).get("z")
                json_path = Path(vector_path).parent if vector_path else Path(".")
            formatted_runs.append(_format_run(run, json_path))

        comparison = _annotate_deltas(dataset_name, formatted_runs)

        # Extract CMB summary table if present
        cmb_rows = []
        for formatted in formatted_runs:
            predictions = formatted.get("predictions", {})
            residuals = formatted.get("residuals_sigma", {})
            if "cmb" in predictions:
                pred_block = predictions.get("cmb", {})
                resid_block = residuals.get("cmb", {}) if residuals else {}
            elif "R" in predictions:
                pred_block = predictions
                resid_block = residuals or {}
            else:
                continue
            cmb_rows.append({
                "model": formatted.get("model", ""),
                "run_id": formatted.get("run_id", ""),
                "theta100": pred_block.get("100theta_*"),
                "lA": pred_block.get("lA") or pred_block.get("l_A"),
                "R": pred_block.get("R"),
                "Omegabh2": pred_block.get("Omegabh2") or pred_block.get("Obh2"),
                "ns": pred_block.get("ns"),
                "residuals": {
                    "R": resid_block.get("R"),
                    "lA": resid_block.get("lA") or resid_block.get("l_A"),
                    "Omegabh2": resid_block.get("Omegabh2") or resid_block.get("Obh2"),
                    "ns": resid_block.get("ns"),
                },
            })

        # Gather unique figures
        figures = []
        seen_paths = set()
        for formatted in formatted_runs:
            for fig in formatted.get("summary_figures", []):
                if fig["path"] in seen_paths:
                    continue
                seen_paths.add(fig["path"])
                figures.append({
                    "label": f"{formatted['model']} — {fig['label']}",
                    "path": fig["path"]
                })

        conclusion = (
            comparison["summary"]
            if comparison
            else f"Only {', '.join(sorted({r['model'] for r in formatted_runs}))} runs available; add counterparts for model comparison."
        )

        dataset_blocks.append({
            "dataset": dataset_name,
            "release_tag": formatted_runs[0].get("dataset", {}).get("release_tag"),
            "runs": formatted_runs,
            "comparison": comparison,
            "conclusion": conclusion,
            "figures": figures,
            "cmb_rows": cmb_rows,
        })

    # Metadata header/footer
    header = {
        "title": report_cfg.get("title", "Unified Fit Comparison"),
        "subtitle": report_cfg.get("subtitle", ""),
        "mock_warning": any(run.get("mock", False) for run in runs),
    }
    provenance = [run.get("provenance", {}) for run in runs if run.get("provenance")]
    footer = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "version": provenance[0].get("code_version", "v0.1.0") if provenance else "v0.1.0",
        "commit": provenance[0].get("commit", "n/a") if provenance else "n/a",
    }

    html = template.render(header=header, dataset_blocks=dataset_blocks, footer=footer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path.resolve())


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified comparison report.")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON paths or globs")
    parser.add_argument("--out", required=True, help="Output HTML path")
    args = parser.parse_args()

    json_paths = []
    for pattern in args.inputs:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            json_paths.extend(matches)
        elif Path(pattern).exists():
            json_paths.append(pattern)
        else:
            log.warn(f"No matches for pattern '{pattern}'")

    runs = []
    for path in json_paths:
        json_path = Path(path).resolve()
        if not json_path.exists():
            log.warn(f"Skipping missing JSON: {json_path}")
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["__json_path"] = str(json_path)
        runs.append(data)

    if not runs:
        raise FileNotFoundError("No fit_results.json inputs found. Check the --inputs patterns.")

    runs.sort(key=lambda item: item["timestamp"])
    output = Path(args.out).resolve()
    result = render_unified_report(runs, output)
    log.info(f"Unified report written to {result}")


if __name__ == "__main__":
    main()
