#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRef:
    seed: int
    run_dir: Path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_runs(root: Path) -> list[RunRef]:
    runs: list[RunRef] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config_used.json"
        jk_path = entry / "jackknife_results.json"
        if not config_path.exists() or not jk_path.exists():
            continue
        cfg = _load_json(config_path)
        seed = cfg.get("jackknife", {}).get("random_seed")
        if seed is None:
            continue
        runs.append(RunRef(seed=int(seed), run_dir=entry))
    runs.sort(key=lambda r: r.seed)
    return runs


def _sanitize_meta(meta: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
    if meta is None:
        return None
    if isinstance(meta, list):
        sanitized = []
        for item in meta:
            if not isinstance(item, dict):
                continue
            item_out = {k: v for k, v in item.items() if k not in {"components"}}
            sanitized.append(item_out)
        return sanitized
    if isinstance(meta, dict):
        # Drop file path hints if present.
        blocked = {"mean", "cov", "path", "paths", "filename", "files", "components"}
        return {k: v for k, v in meta.items() if k not in blocked}
    return None


def _extract_dataset_meta(reference_run: RunRef) -> dict[str, Any]:
    pbuf_fit_dir = reference_run.run_dir / "pbuf" / "fits"
    dataset_meta: dict[str, Any] = {}
    for key in ("cmb", "sn", "bao_iso_full", "cc", "rsd"):
        fit_path = pbuf_fit_dir / f"{key}.json"
        if not fit_path.exists():
            continue
        payload = _load_json(fit_path)
        extras = payload.get("extras") or {}
        dataset = extras.get("dataset") or {}
        meta = dataset.get("meta")
        observed = extras.get("observed")
        n_points = None
        try:
            n_points = len(observed) if observed is not None else None
        except Exception:
            n_points = None
        z_info: dict[str, Any] = {}
        if isinstance(meta, dict) and isinstance(meta.get("z_min"), (int, float)) and isinstance(meta.get("z_max"), (int, float)):
            z_info = {"z_min": float(meta["z_min"]), "z_max": float(meta["z_max"])}
        if isinstance(meta, dict) and isinstance(meta.get("z"), (int, float)):
            z_info.setdefault("z_eff", float(meta["z"]))
        # Special-case: BAO iso full metadata stores source file hints. Infer z values by matching means.
        if key == "bao_iso_full":
            z_vals = _infer_bao_iso_redshifts(meta, observed)
            if z_vals:
                z_info = {"z_min": min(z_vals), "z_max": max(z_vals), "z_values": z_vals}
        dataset_meta[key] = {
            "dataset_name": dataset.get("name"),
            "meta": _sanitize_meta(meta),
            "n_points": n_points,
            "z_info": z_info or None,
        }
    return dataset_meta


def _infer_bao_iso_redshifts(meta: Any, observed: Any) -> list[float]:
    if observed is None:
        return []
    try:
        values = [float(v) for v in observed]
    except Exception:
        return []
    if not values:
        return []
    if not isinstance(meta, list):
        return []
    # Extract any mean files listed in the meta blob.
    mean_files: list[str] = []
    for item in meta:
        if not isinstance(item, dict):
            continue
        for comp in item.get("components") or []:
            if isinstance(comp, dict) and comp.get("mean"):
                mean_files.append(str(comp["mean"]))
    if not mean_files:
        return []
    repo_root = Path(__file__).resolve().parents[2]
    z_found: list[float] = []
    remaining = list(values)
    for rel in mean_files:
        path = repo_root / rel
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            if len(parts) < 3:
                continue
            try:
                z = float(parts[0])
                val = float(parts[1])
                qty = str(parts[2])
            except Exception:
                continue
            if "DV_over" not in qty:
                continue
            # Greedy match by numeric equality within a tight tolerance.
            for idx, target in enumerate(list(remaining)):
                if abs(val - target) <= 1e-6:
                    z_found.append(z)
                    remaining.pop(idx)
                    break
        if not remaining:
            break
    return sorted(z_found)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Harvest reference tables/plots from a unified_joint run.")
    parser.add_argument(
        "--reference-run",
        help=(
            "Explicit run directory name under data/science_runs/unified_joint to treat as "
            "the reference run (e.g. 2025-12-19T101316_unified_joint-4). If omitted, "
            "seed 42 (if present) or the first sorted run is used."
        ),
    )
    parser.add_argument(
        "--runs-root",
        help="Path to the unified_joint runs root (defaults to data/science_runs/unified_joint).",
    )
    args = parser.parse_args()
    runs_root = Path(args.runs_root) if args.runs_root else repo_root / "data" / "science_runs" / "unified_joint"
    out_root = repo_root / "paper"

    runs = _find_runs(runs_root)
    if not runs:
        raise SystemExit(f"No run directories found under {runs_root}")

    reference: RunRef | None = None
    if args.reference_run:
        reference = next((run for run in runs if run.run_dir.name == args.reference_run), None)
        if reference is None:
            raise SystemExit(f"Reference run '{args.reference_run}' not found under {runs_root}")
    if reference is None:
        reference = next((run for run in runs if run.seed == 42), runs[0])

    # Best-fit payloads for tables.
    best_fit = {
        "reference_seed": reference.seed,
        "reference_run_id": reference.run_dir.name,
        "pbuf_best_fit": _load_json(reference.run_dir / "pbuf" / "best_fit.json"),
        "lcdm_best_fit": _load_json(reference.run_dir / "lcdm" / "best_fit.json"),
        "pbuf_chi2_breakdown": _load_json(reference.run_dir / "pbuf" / "chi2_breakdown.json"),
        "lcdm_chi2_breakdown": _load_json(reference.run_dir / "lcdm" / "chi2_breakdown.json"),
        "pbuf_parameters": _load_json(reference.run_dir / "pbuf" / "parameters.json"),
        "lcdm_parameters": _load_json(reference.run_dir / "lcdm" / "parameters.json"),
        "datasets_meta": _extract_dataset_meta(reference),
    }
    (out_root / "data").mkdir(parents=True, exist_ok=True)
    (out_root / "data" / "reference_best_fit.json").write_text(
        json.dumps(best_fit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_reference_tables(out_root, best_fit)

    # Aggregate jackknife draws across all seeds.
    rows: list[dict[str, Any]] = []
    for run in runs:
        jk = _load_json(run.run_dir / "jackknife_results.json")
        draws = jk.get("draws") or []
        for draw in draws:
            idx = draw.get("draw_index")
            models = draw.get("jackknife_models") or {}
            pbuf = (models.get("pbuf") or {}).get("chi_squared")
            lcdm = (models.get("lcdm") or {}).get("chi_squared")
            if pbuf is None or lcdm is None:
                continue
            delta = float(lcdm) - float(pbuf)
            rows.append(
                {
                    "seed": int(run.seed),
                    "fold": int(idx) if idx is not None else "",
                    "chi2_pbuf": float(pbuf),
                    "chi2_lcdm": float(lcdm),
                    "delta_chi2": float(delta),
                }
            )

    rows.sort(key=lambda r: (r["seed"], r["fold"]))
    _write_csv(
        out_root / "data" / "jackknife_all.csv",
        rows,
        fieldnames=["seed", "fold", "chi2_pbuf", "chi2_lcdm", "delta_chi2"],
    )

    chi2_pbuf = [r["chi2_pbuf"] for r in rows]
    chi2_lcdm = [r["chi2_lcdm"] for r in rows]
    deltas = [r["delta_chi2"] for r in rows]

    summary = {
        "n_total": len(rows),
        "seeds": [run.seed for run in runs],
        "all_folds_pbuf_better": all(d > 0.0 for d in deltas),
        "median_chi2_pbuf": _median(chi2_pbuf),
        "median_chi2_lcdm": _median(chi2_lcdm),
        "median_delta_chi2": _median(deltas),
        "min_delta_chi2": min(deltas) if deltas else None,
        "max_delta_chi2": max(deltas) if deltas else None,
    }
    (out_root / "data" / "jackknife_summary_all.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Per-seed boxplot stats for delta chi2 (LCDM - PBUF).
    delta_by_seed: dict[int, list[float]] = {}
    for row in rows:
        delta_by_seed.setdefault(int(row["seed"]), []).append(float(row["delta_chi2"]))
    box_rows: list[dict[str, Any]] = []
    for seed, values in sorted(delta_by_seed.items()):
        stats = _boxplot_stats(values)
        box_rows.append(
            {
                "seed": seed,
                "lw": stats["lw"],
                "lq": stats["lq"],
                "med": stats["med"],
                "uq": stats["uq"],
                "uw": stats["uw"],
            }
        )
    _write_csv(
        out_root / "data" / "delta_boxplot_prepared.csv",
        box_rows,
        fieldnames=["seed", "lw", "lq", "med", "uq", "uw"],
    )
    return 0


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    data = sorted(float(v) for v in values)
    n = len(data)
    mid = n // 2
    if n % 2 == 1:
        return data[mid]
    return 0.5 * (data[mid - 1] + data[mid])


def _quantile(sorted_data: list[float], q: float) -> float:
    if not sorted_data:
        raise ValueError("Cannot compute quantile of empty data.")
    if q <= 0.0:
        return float(sorted_data[0])
    if q >= 1.0:
        return float(sorted_data[-1])
    pos = (len(sorted_data) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_data) - 1)
    frac = pos - lo
    return float(sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac)


def _boxplot_stats(values: list[float]) -> dict[str, float]:
    data = sorted(float(v) for v in values)
    lq = _quantile(data, 0.25)
    med = _quantile(data, 0.5)
    uq = _quantile(data, 0.75)
    iqr = uq - lq
    lo_cut = lq - 1.5 * iqr
    hi_cut = uq + 1.5 * iqr
    lw = next((v for v in data if v >= lo_cut), data[0])
    uw = next((v for v in reversed(data) if v <= hi_cut), data[-1])
    return {"lw": float(lw), "lq": float(lq), "med": float(med), "uq": float(uq), "uw": float(uw)}


def _fmt(value: Any, *, digits: int = 6) -> str:
    try:
        num = float(value)
    except Exception:
        return str(value)
    if abs(num) >= 1e5 or (abs(num) > 0 and abs(num) < 1e-3):
        return f"{num:.{digits}e}"
    return f"{num:.{digits}f}"

def _tex_escape(text: str) -> str:
    return text.replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _write_reference_tables(out_root: Path, payload: dict[str, Any]) -> None:
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pbuf_params = (payload.get("pbuf_parameters") or {}).get("model_parameters") or {}
    lcdm_params = (payload.get("lcdm_parameters") or {}).get("model_parameters") or {}
    pbuf_best = (payload.get("pbuf_best_fit") or {}).get("parameters") or {}
    lcdm_best = (payload.get("lcdm_best_fit") or {}).get("parameters") or {}

    # Best-fit parameters table.
    param_rows = [
        ("H0", lcdm_best.get("H0"), pbuf_best.get("H0"), "km/s/Mpc"),
        ("Omega_m0", lcdm_best.get("Omega_m0"), pbuf_params.get("Omega_m0"), ""),
        ("Omega_b0", lcdm_best.get("Omega_b0"), pbuf_params.get("Omega_b0"), ""),
        ("Omega_k0", lcdm_best.get("Omega_k0"), pbuf_params.get("alpha_resolved"), ""),
        ("Omega_r0", lcdm_params.get("Omega_r0"), pbuf_params.get("Omega_r0"), ""),
        ("Rmax", "", pbuf_best.get("Rmax"), "GeV-1"),
    ]
    lines = []
    lines.append(r"\begin{tabular}{llll}")
    lines.append(r"\toprule")
    lines.append(r"Parameter & LambdaCDM & PBUF V11 & Units \\")
    lines.append(r"\midrule")
    for name, lcdm_val, pbuf_val, units in param_rows:
        name = _tex_escape(str(name))
        lcdm_txt = "fixed" if lcdm_val == "" else _fmt(lcdm_val, digits=6)
        pbuf_txt = "fixed" if pbuf_val == "" else _fmt(pbuf_val, digits=6)
        lines.append(f"{name} & {lcdm_txt} & {pbuf_txt} & {units} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (tables_dir / "best_fit_parameters.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Chi2 breakdown table.
    chi2_pbuf = payload.get("pbuf_chi2_breakdown") or {}
    chi2_lcdm = payload.get("lcdm_chi2_breakdown") or {}
    fits = sorted(set((chi2_pbuf.get("fits") or {}).keys()) | set((chi2_lcdm.get("fits") or {}).keys()))
    lines = []
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & LambdaCDM chi2 & PBUF chi2 \\")
    lines.append(r"\midrule")
    for fit in fits:
        fit_name = _tex_escape(str(fit))
        l_val = (chi2_lcdm.get("fits") or {}).get(fit)
        p_val = (chi2_pbuf.get("fits") or {}).get(fit)
        lines.append(f"{fit_name} & {_fmt(l_val, digits=3)} & {_fmt(p_val, digits=3)} \\\\")
    lines.append(r"\midrule")
    lines.append(f"Total & {_fmt(chi2_lcdm.get('total'), digits=3)} & {_fmt(chi2_pbuf.get('total'), digits=3)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    (tables_dir / "chi2_breakdown.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
