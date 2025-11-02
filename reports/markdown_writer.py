"""
Markdown Writer — Science Run Summary
=====================================

Generates compact Markdown tables and narrative for the aggregated
statistics produced by `reports.summary_builder`. The output is suitable
for README embeds, release notes, or export to other formats.

Sections include:
  * Dataset-level χ² / AIC comparisons
  * Global totals per model
  * ΔAIC / ΔBIC preference summary
  * Per-run overview (primary scenario, Δ metrics, runtime highlights)
"""

from pathlib import Path
from typing import Any, Dict, List
import math


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


def _delta(a: Any, b: Any) -> Any:
    if a is None or b is None:
        return None
    try:
        return float(b) - float(a)
    except (TypeError, ValueError):
        return None


def write_markdown_summary(
    stats: Dict[str, Any],
    output_file: str = "reports/output/summary_table.md",
) -> str:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    aggregated = stats.get("aggregated", {})
    datasets = aggregated.get("datasets", {})
    models_map = aggregated.get("models", {})
    model_order = list(models_map.keys())
    if not model_order:
        raise ValueError("No model statistics available for Markdown summary.")

    primary = model_order[0]
    comparator = model_order[1] if len(model_order) > 1 else primary

    lines: List[str] = []
    lines.append("# PBUF Science Run Summary\n")
    lines.append(f"Models compared: {', '.join(m.upper() for m in model_order)}\n")

    # Dataset table
    lines.append("## Dataset-Level Metrics\n")
    header = (
        "| Dataset | "
        f"χ² {primary.upper()} | χ² {comparator.upper()} | Δχ² ({comparator.upper()}-{primary.upper()}) | "
        f"AIC {primary.upper()} | AIC {comparator.upper()} | ΔAIC | "
        f"BIC {primary.upper()} | BIC {comparator.upper()} | ΔBIC |\n"
    )
    separator = "|---" * 9 + "|\n"
    lines.append(header)
    lines.append(separator)

    for dataset in sorted(datasets.keys()):
        entry = datasets[dataset]
        a = entry.get(primary, {})
        b = entry.get(comparator, {})
        lines.append(
            "| {dataset} | {chi2_a} | {chi2_b} | {dchi2} | {aic_a} | {aic_b} | {daic} | {bic_a} | {bic_b} | {dbic} |".format(
                dataset=dataset.upper(),
                chi2_a=_fmt(a.get("chi2")),
                chi2_b=_fmt(b.get("chi2")),
                dchi2=_fmt(_delta(a.get("chi2"), b.get("chi2"))),
                aic_a=_fmt(a.get("AIC")),
                aic_b=_fmt(b.get("AIC")),
                daic=_fmt(_delta(a.get("AIC"), b.get("AIC"))),
                bic_a=_fmt(a.get("BIC")),
                bic_b=_fmt(b.get("BIC")),
                dbic=_fmt(_delta(a.get("BIC"), b.get("BIC"))),
            )
        )

    # Global summary
    lines.append("\n## Global Totals\n")
    for model, data in models_map.items():
        lines.append(
            f"* **{model.upper()}**: "
            f"χ²_total={_fmt(data.get('chi2_total'), 6)}, "
            f"AIC_total={_fmt(data.get('AIC_total'), 6)}, "
            f"BIC_total={_fmt(data.get('BIC_total'), 6)}, "
            f"reduced χ²={_fmt(data.get('chi2_reduced_total'), 6)}, "
            f"runs={_fmt(data.get('run_count'), 0)}"
        )

    comparison = aggregated.get("global", {}).get("comparison", {})
    if comparison:
        lines.append("\n## ΔAIC / ΔBIC Preference\n")
        lines.append("| Metric | Value | Preferred Model |\n|---|---|---|\n")
        lines.append(
            f"| ΔAIC ({comparator.upper()}-{primary.upper()}) | {_fmt(comparison.get(f'ΔAIC ({comparator}-{primary})'))} | "
            f"{comparison.get('preferred_model_AIC', '—').upper()} |"
        )
        lines.append(
            f"| ΔBIC ({comparator.upper()}-{primary.upper()}) | {_fmt(comparison.get(f'ΔBIC ({comparator}-{primary})'))} | "
            f"{comparison.get('preferred_model_BIC', '—').upper()} |"
        )

    # Per-run overview
    if stats.get("runs"):
        lines.append("\n## Run Overview\n")
        lines.append("| Run | Primary Scenario | ΔAIC (PBUF-LCDM) | ΔBIC (PBUF-LCDM) | Notes |\n")
        lines.append("|---|---|---|---|---|\n")
        for run in stats["runs"]:
            primary_id = run.get("primary_scenario_id") or "—"
            scenario = next(
                (s for s in run.get("scenarios", []) if s.get("id") == primary_id),
                None,
            )
            joint = scenario.get("joint") if scenario else None
            delta_aic = joint.get("deltas", {}).get("delta_aic") if joint else None
            delta_bic = joint.get("deltas", {}).get("delta_bic") if joint else None
            runtime_lcdm = None
            runtime_pbuf = None
            if scenario:
                model_lcdm = scenario.get("models", {}).get(primary)
                model_cmp = scenario.get("models", {}).get(comparator)
                runtime_lcdm = model_lcdm.get("runtime", {}).get("wall_seconds") if model_lcdm else None
                runtime_pbuf = model_cmp.get("runtime", {}).get("wall_seconds") if model_cmp else None
            note_bits = []
            if runtime_lcdm is not None:
                note_bits.append(f"{primary.upper()} runtime { _fmt(runtime_lcdm)} s")
            if runtime_pbuf is not None and comparator != primary:
                note_bits.append(f"{comparator.upper()} runtime { _fmt(runtime_pbuf)} s")
            notes = "; ".join(note_bits) if note_bits else "—"
            lines.append(
                f"| {run['name']} | {primary_id} | {_fmt(delta_aic)} | {_fmt(delta_bic)} | {notes} |"
            )

    lines.append("\n## Notes\n")
    lines.append("- Aggregated values sum the primary scenario from each science run.")
    lines.append("- Δ metrics follow the convention (MODEL₂ − MODEL₁); positive values favour MODEL₁.")
    lines.append("- Runtimes refer to wall-clock seconds captured in the artifacts.")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Markdown summary written to {output_path.resolve()}")
    return str(output_path.resolve())
