from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3,
    "savefig.dpi": 200,
})

VALID_COLOUR = "#1f77b4"
INVALID_COLOUR = "#d62728"
BEST_COLOUR = "#2ca02c"
COUPLED_COLOUR = "#9467bd"
RESEED_COLOUR = "#ff7f0e"


def _save(fig: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


def _resolve_trace_path(trace_source: Path | str) -> Path:
    path = Path(trace_source)
    if path.is_dir():
        candidate = path / "basin_trace.json"
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Basin trace file not found: {path}")
    return path


def _load_trace(trace_source: Path | str) -> Tuple[Dict[str, Any], Path]:
    trace_path = _resolve_trace_path(trace_source)
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Basin trace must be a JSON object (got {type(data).__name__})")
    return data, trace_path


def _sorted_scans(scans: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _sort_key(scan: Dict[str, Any]) -> Tuple[int, int, int, int]:
        cycle = scan.get("cycle")
        cycle_key = -1 if cycle is None else int(cycle)
        pass_id = scan.get("pass")
        pass_key = -1 if pass_id is None else int(pass_id)
        edge_iter = scan.get("edge_iteration")
        edge_key = -1 if edge_iter is None else int(edge_iter)
        scan_index = scan.get("_scan_index")
        index_key = -1 if scan_index is None else int(scan_index)
        return (cycle_key, pass_key, edge_key, index_key)

    return sorted(scans, key=_sort_key)


def _plot_parameter_scans(trace: Dict[str, Any], plot_dir: Path) -> Dict[str, Path]:
    scans = trace.get("scans")
    if not scans:
        scans = (trace.get("result") or {}).get("axis_scans", [])
    if not scans:
        return {}

    by_param: Dict[str, List[Dict[str, Any]]] = {}
    for record in scans:
        param = record.get("param")
        if not isinstance(param, str):
            continue
        by_param.setdefault(param, []).append(record)

    outputs: Dict[str, Path] = {}
    for param, records in by_param.items():
        ordered = _sorted_scans(records)
        if not ordered:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, record in enumerate(ordered):
            curve = record.get("curve") or []
            if not isinstance(curve, list):
                continue
            valid_values: List[float] = []
            valid_chi2: List[float] = []
            invalid_values: List[float] = []
            invalid_chi2: List[float] = []
            for point in curve:
                value = point.get("value")
                chi2 = point.get("chi2")
                if value is None or chi2 is None:
                    continue
                try:
                    value_f = float(value)
                    chi2_f = float(chi2)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value_f) or not np.isfinite(chi2_f):
                    continue
                if point.get("valid"):
                    valid_values.append(value_f)
                    valid_chi2.append(chi2_f)
                else:
                    invalid_values.append(value_f)
                    invalid_chi2.append(chi2_f)

            label_bits: List[str] = []
            cycle = record.get("cycle")
            if cycle is not None:
                label_bits.append(f"cycle {cycle}")
            stage_pass = record.get("pass")
            if stage_pass is not None:
                label_bits.append(f"pass {stage_pass}")
            edge_iter = record.get("edge_iteration")
            if edge_iter not in (None, 0):
                label_bits.append(f"edge {edge_iter}")
            label = ", ".join(label_bits) if label_bits else f"scan {idx + 1}"

            if valid_values:
                ax.plot(valid_values, valid_chi2, color=VALID_COLOUR, linewidth=1.6, alpha=0.75, label=label)
            if invalid_values:
                ax.scatter(invalid_values, invalid_chi2, color=INVALID_COLOUR, marker="x", alpha=0.6, label=f"{label} (invalid)")

            best = record.get("best")
            chi2_min = record.get("chi2_min")
            if best is not None and chi2_min is not None:
                try:
                    best_f = float(best)
                    chi2_min_f = float(chi2_min)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(best_f) and np.isfinite(chi2_min_f):
                    ax.scatter([best_f], [chi2_min_f], color=BEST_COLOUR, marker="*", s=80)

        ax.set_xlabel(param)
        ax.set_ylabel("χ²")
        ax.set_title(f"{param} basin scans")
        handles, labels = ax.get_legend_handles_labels()
        if handles and len(handles) <= 10:
            ax.legend(loc="best", fontsize=9)
        outputs[f"{param}_scans"] = _save(fig, plot_dir / f"{param.lower()}_scans.png")

    return outputs


def _plot_plateau_reseeds(trace: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    records = trace.get("plateau_reseeds") or []
    if not records:
        return None
    cycles: List[int] = []
    base_scores: List[float] = []
    best_scores: List[float] = []
    improvements: List[float] = []
    for record in records:
        cycle = record.get("cycle")
        base = record.get("base_score")
        best = record.get("best_score")
        if cycle is None or best is None:
            continue
        try:
            cycle_i = int(cycle)
            best_f = float(best)
        except (TypeError, ValueError):
            continue
        base_f = None
        if base is not None:
            try:
                base_f = float(base)
            except (TypeError, ValueError):
                base_f = None
        delta = None
        if base_f is not None:
            delta = base_f - best_f
        cycles.append(cycle_i)
        base_scores.append(base_f if base_f is not None else np.nan)
        best_scores.append(best_f)
        improvements.append(delta if delta is not None else np.nan)

    if not best_scores:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(cycles, best_scores, color=RESEED_COLOUR, label="Best reseed score", zorder=3)
    if any(np.isfinite(score) for score in base_scores):
        ax.scatter(cycles, base_scores, color="#666666", marker="^", label="Baseline score", zorder=2)
    if any(np.isfinite(delta) for delta in improvements):
        ax.plot(cycles, improvements, color="#228b22", linestyle="--", alpha=0.7, label="Score improvement")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Score")
    ax.set_title("Plateau reseed outcomes")
    ax.legend(loc="best")

    return _save(fig, plot_dir / "plateau_reseeds.png")


def _plot_coupled_updates(trace: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    records = trace.get("coupled_updates") or []
    useful = [record for record in records if record.get("delta_score") is not None]
    if not useful:
        return None

    deltas: List[float] = []
    labels: List[str] = []
    for record in useful:
        delta = record.get("delta_score")
        try:
            delta_f = float(delta)
        except (TypeError, ValueError):
            continue
        params = record.get("parameters") or ()
        label = ",".join(params) if params else "coupled"
        deltas.append(delta_f)
        labels.append(label)

    if not deltas:
        return None

    x = np.arange(len(deltas))
    colours = [COUPLED_COLOUR if delta > 0 else INVALID_COLOUR for delta in deltas]
    fig, ax = plt.subplots(figsize=(max(6, len(deltas) * 0.8), 4.5))
    ax.bar(x, deltas, color=colours, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_ylabel("Δ score")
    ax.set_title("Coupled update improvements (positive lowers χ²)")

    return _save(fig, plot_dir / "coupled_updates.png")


def generate_basin_plots(trace_source: Path | str, plot_dir: Path | str) -> Dict[str, Path]:
    """
    Produce diagnostic plots from a basin trace JSON file.
    Returns a mapping of plot labels to the generated image paths.
    """
    trace, trace_path = _load_trace(trace_source)
    destination = Path(plot_dir)
    destination.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, Path] = {}
    generated.update(_plot_parameter_scans(trace, destination))

    plateau_path = _plot_plateau_reseeds(trace, destination)
    if plateau_path is not None:
        generated["plateau_reseeds"] = plateau_path

    coupled_path = _plot_coupled_updates(trace, destination)
    if coupled_path is not None:
        generated["coupled_updates"] = coupled_path

    return generated
