#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the unified HTML report comparing multiple fit results.
"""

from __future__ import annotations
import argparse
import glob
import json
import math
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


def _verdict(delta_aic: float) -> tuple[str, str | None]:
    """
    Classify the ΔAIC result using project-wide thresholds.

    Returns (verdict, model) where verdict ∈ {"Favors", "Comparable", "Disfavored"}
    and model indicates which cosmology the verdict refers to (None for Comparable).
    """

    if not isinstance(delta_aic, (int, float)) or math.isnan(delta_aic):
        return "Comparable", None
    if abs(delta_aic) < 2.0:
        return "Comparable", None
    if delta_aic < 0:
        return "Favors", "PBUF"
    return "Disfavored", "PBUF"


def _detect_report_path(json_path: Path) -> Optional[str]:
    run_dir = json_path.parent
    candidates = sorted(run_dir.glob("*_report.html"))
    if candidates:
        return str(candidates[0].resolve())
    return None


def _tokenise(value: Optional[str], fallback: str) -> str:
    text = (value or "").strip()
    token = "".join(ch if ch.isalnum() else "_" for ch in text)
    token = token.strip("_") or fallback
    return token.upper()

def _coerce_numeric(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalise_metric_block(block: dict) -> dict:
    if not isinstance(block, dict):
        return {}
    metrics = dict(block)

    alias_map = {
        "aic": "AIC",
        "bic": "BIC",
        "chi2": "chi2",
        "chi2_dof": "chi2_dof",
        "chi2/dof": "chi2_dof",
        "pvalue": "p_value",
        "p-val": "p_value",
        "p_val": "p_value",
    }
    for src, dest in alias_map.items():
        if src in metrics and dest not in metrics:
            metrics[dest] = metrics[src]

    for key in ("chi2", "AIC", "BIC", "chi2_dof", "p_value"):
        if key in metrics:
            coerced = _coerce_numeric(metrics[key])
            metrics[key] = coerced if coerced is not None else metrics[key]

    for key, value in list(metrics.items()):
        if isinstance(value, dict):
            metrics[key] = _normalise_metric_block(value)

    return metrics


# ----------------------------------------------------------------------
# Run formatter — normalize metrics and figures
# ----------------------------------------------------------------------
def _format_run(run: dict, json_path: Path) -> dict:
    """
    Normalize metric structure and flatten nested totals for display.
    Adds component-level metrics (SN, BAO, BAO_ANI, CMB).
    """
    metrics = _normalise_metric_block(run.get("metrics", {}) or {})

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
        for key in ("sn", "bao", "bao_ani", "cmb", "cc", "rsd"):
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
    if "AIC" not in metrics:
        metrics["AIC"] = float("nan")
    if "BIC" not in metrics:
        metrics["BIC"] = float("nan")

    metrics["p_value_fmt"] = (
        f"{p_value:.3g}" if isinstance(p_value, (int, float)) else "n/a"
    )
    metrics["chi2_fmt"] = (
        f"{metrics.get('chi2', float('nan')):.3f}" if "chi2" in metrics else "—"
    )
    metrics["aic_fmt"] = (
        f"{metrics.get('AIC', float('nan')):.3f}" if "AIC" in metrics else "—"
    )
    metrics["bic_fmt"] = (
        f"{metrics.get('BIC', float('nan')):.3f}" if "BIC" in metrics else "—"
    )

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

    pbuf_m, lcdm_m = _normalise_metric_block(_extract_metrics(pbuf_run)), _normalise_metric_block(_extract_metrics(lcdm_run))

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
        ds = run.get("dataset", {}) or {}
        json_path = str(run.get("__json_path", "")).lower()
        ds_name = str(ds.get("name", "")).lower()
        notes = " ".join(str(v) for v in ds.values()).lower()
        dataset_alias = ds.get("dataset_name")
        run_id = str(run.get("run_id", ""))
        if isinstance(dataset_alias, str) and dataset_alias.startswith("RSD_"):
            dataset_name = dataset_alias
        elif run_id.upper().startswith("RSD_"):
            release = ds.get("tag") or ds.get("release_tag") or (run_id.split("_")[1] if "_" in run_id else "JOINT")
            release_token = _tokenise(str(release), "JOINT")
            dataset_name = f"RSD_{release_token}"
        elif "sn" in ds and "bao" in ds and "cmb" in ds:
            dataset_name = "Joint SN+BAO+CMB"
        elif "joint" in json_path:
            dataset_name = "Joint SN+BAO+CMB"
        elif "bao_ani" in ds or "anisotropic" in json_path:
            dataset_name = "BAO Anisotropic"
        elif "bao" in ds or "bao" in json_path:
            dataset_name = "BAO Mixed"
        elif "chronometer" in ds_name or "chronometer" in notes or "chronometer" in json_path:
            dataset_name = "Cosmic Chronometers H(z)"
        elif "sn" in ds or "pantheon" in json_path:
            dataset_name = "Pantheon+ Supernovae"
        elif "cmb" in ds or "planck" in json_path:
            dataset_name = "Planck 2018 CMB"
        else:
            dataset_name = ds.get("name") or "Unknown Dataset"

        dataset_groups[dataset_name].append(run)

    # Sort datasets in logical order
    order_priority = [
        "Pantheon+ Supernovae",
        "Planck 2018 CMB",
        "BAO Mixed",
        "BAO Anisotropic",
        "Cosmic Chronometers H(z)",
        "Joint SN+BAO+CMB",
        "RSD_*",
    ]
    def _dataset_order_key(name: str) -> int:
        for idx, token in enumerate(order_priority):
            if token == "RSD_*" and name.startswith("RSD_"):
                return idx
            if name == token:
                return idx
        return len(order_priority)

    dataset_groups = dict(sorted(dataset_groups.items(), key=lambda kv: _dataset_order_key(kv[0])))

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

    # Build summary table (Δ statistics per dataset)
    summary_rows: List[dict] = []
    joint_row: Optional[dict] = None
    total_accumulator = {"delta_chi2": 0.0, "delta_aic": 0.0, "delta_bic": 0.0}

    def _build_row(name: str, comp: dict) -> dict:
        delta_chi2 = comp.get("delta_chi2")
        delta_aic = comp.get("delta_aic")
        delta_bic = comp.get("delta_bic")
        verdict, verdict_model = _verdict(delta_aic)
        favoured_model: Optional[str]
        if verdict == "Comparable":
            favoured_model = None
        elif verdict == "Disfavored" and verdict_model == "PBUF":
            favoured_model = "ΛCDM"
        else:
            favoured_model = verdict_model
        if verdict == "Comparable":
            verdict_display = "Comparable"
        elif verdict == "Disfavored":
            verdict_display = f"Disfavored ({verdict_model})"
        else:
            verdict_display = f"Favors {verdict_model}"
        strength = comp.get("strength")
        display_name = name.replace("_", " ") if name.startswith("RSD_") else name
        return {
            "dataset": display_name,
            "sort_key": name,
            "delta_chi2": delta_chi2,
            "delta_aic": delta_aic,
            "delta_bic": delta_bic,
            "verdict": verdict_display,
            "favoured": favoured_model,
            "strength": strength,
            "badge_class": strength.lower() if isinstance(strength, str) else "",
        }

    for block in dataset_blocks:
        comp = block.get("comparison")
        if not comp:
            continue
        row = _build_row(block["dataset"], comp)
        if block["dataset"].lower().startswith("joint"):
            joint_row = row
            continue
        summary_rows.append(row)
        for key in ("delta_chi2", "delta_aic", "delta_bic"):
            value = comp.get(key)
            if isinstance(value, (int, float)):
                total_accumulator[key] += float(value)

    summary_rows.sort(key=lambda item: _dataset_order_key(item["sort_key"]))

    global_row: Optional[dict] = None
    if summary_rows:
        verdict, verdict_model = _verdict(total_accumulator["delta_aic"])
        if verdict == "Comparable":
            verdict_display = "Comparable"
            favoured_model = None
        elif verdict == "Disfavored" and verdict_model == "PBUF":
            verdict_display = f"Disfavored ({verdict_model})"
            favoured_model = "ΛCDM"
        else:
            verdict_display = f"Favors {verdict_model}"
            favoured_model = verdict_model
        strength = _strength(total_accumulator["delta_aic"])
        global_row = {
            "dataset": "Global Total",
            "delta_chi2": total_accumulator["delta_chi2"],
            "delta_aic": total_accumulator["delta_aic"],
            "delta_bic": total_accumulator["delta_bic"],
            "verdict": verdict_display,
            "favoured": favoured_model,
            "strength": strength,
            "badge_class": strength.lower(),
        }

    summary_table = {
        "rows": summary_rows,
        "joint": joint_row,
        "global": global_row,
    }

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

    html = template.render(
        header=header,
        dataset_blocks=dataset_blocks,
        footer=footer,
        summary_table=summary_table,
    )
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
        if "metrics" in data:
            data["metrics"] = _normalise_metric_block(data["metrics"])
        runs.append(data)

    if not runs:
        raise FileNotFoundError("No fit_results.json inputs found. Check the --inputs patterns.")

    runs.sort(key=lambda item: item["timestamp"])
    output = Path(args.out).resolve()
    result = render_unified_report(runs, output)
    log.info(f"Unified report written to {result}")


if __name__ == "__main__":
    main()
