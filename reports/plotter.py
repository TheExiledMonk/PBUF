"""
Plotter — Science Run Visualisation Suite
=========================================

Generates publication-ready diagnostic figures for PBUF science runs,
combining global aggregates (χ², AIC, BIC) with per-run scenario views.

Example
-------
    from reports.plotter import generate_all_plots
    generate_all_plots(stats, plot_dir=\"reports/output/plots\")
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------
# Matplotlib styling
# ----------------------------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3,
    "savefig.dpi": 200,
})

COLORS = {
    "lcdm": "#0055cc",
    "pbuf": "#cc3300",
    "default": "#777777",
}

MARKERS = {
    "lcdm": "o",
    "pbuf": "s",
    "default": "^",
}

METRIC_COLORS = ["#4e8cff", "#ff8c4e", "#3cc674"]


def _colour(model: str) -> str:
    return COLORS.get(model, COLORS["default"])


def _marker(model: str) -> str:
    return MARKERS.get(model, MARKERS["default"])


def _format_tick_label(text: Any) -> str:
    if text is None:
        return ""
    value = str(text).replace("_", " ")
    return value.replace("\n", " ").strip()


def _save(fig: plt.Figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination


# ----------------------------------------------------------------------
# Aggregated plots
# ----------------------------------------------------------------------

def plot_aggregated_model_totals(stats: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    """
    Grouped bar chart showing χ², AIC, and BIC totals per model across all runs.
    """
    models = stats.get("aggregated", {}).get("models", {})
    if not models:
        return None

    model_names = list(models.keys())
    metrics = ["chi2_total", "AIC_total", "BIC_total"]
    pretty_labels = ["χ² total", "AIC total", "BIC total"]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (metric, label) in enumerate(zip(metrics, pretty_labels)):
        values = [models[m].get(metric, 0.0) for m in model_names]
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            label=label,
            color=METRIC_COLORS[idx % len(METRIC_COLORS)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in model_names])
    ax.set_ylabel("Value")
    ax.set_title("Global Model Scores (Aggregated Across Science Runs)")
    ax.legend()

    return _save(fig, plot_dir / "aggregated_model_totals.png")


def plot_aggregated_dataset_breakdown(stats: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    """
    For each dataset, compare the aggregated χ² contribution by model.
    """
    datasets = stats.get("aggregated", {}).get("datasets", {})
    models = list(stats.get("aggregated", {}).get("models", {}).keys())
    if not datasets or not models:
        return None

    dataset_names = sorted(datasets.keys())
    x = np.arange(len(dataset_names))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(dataset_names) * 1.4), 5))
    for idx, model in enumerate(models):
        chi2_values = [
            datasets[dataset].get(model, {}).get("chi2", 0.0) for dataset in dataset_names
        ]
        ax.bar(
            x + (idx - (len(models) - 1) / 2) * width,
            chi2_values,
            width,
            label=model.upper(),
            color=_colour(model),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_format_tick_label(ds).upper() for ds in dataset_names], rotation=45, ha="right")
    ax.set_ylabel("Aggregated χ²")
    ax.set_title("Dataset Contribution to χ² (Primary Scenarios)")
    ax.legend()

    return _save(fig, plot_dir / "aggregated_dataset_breakdown.png")


# ----------------------------------------------------------------------
# Per-run plots
# ----------------------------------------------------------------------

def _collect_core_scenarios(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    scenarios = []
    for scenario in run.get("scenarios", []):
        if scenario.get("id", "").startswith("scout:"):
            continue
        scenarios.append(scenario)
    return scenarios


def plot_run_scenario_chi2(run: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    """
    Scenario-by-scenario χ² totals for each model within a single run.
    """
    scenarios = _collect_core_scenarios(run)
    if not scenarios:
        return None

    models = run.get("models_present", [])
    if not models:
        return None

    scenario_labels = [s.get("id", f"s{i}") for i, s in enumerate(scenarios)]
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.4), 5))
    for idx, model in enumerate(models):
        chi2_values = [
            scenario.get("models", {}).get(model, {}).get("fit_stats", {}).get("chi2_total", 0.0)
            for scenario in scenarios
        ]
        ax.bar(
            x + (idx - (len(models) - 1) / 2) * width,
            chi2_values,
            width,
            label=model.upper(),
            color=_colour(model),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_format_tick_label(label) for label in scenario_labels])
    ax.set_ylabel("χ² Total")
    ax.set_title(f"Scenario χ² Evolution — {run['name']}")
    ax.legend()

    return _save(fig, plot_dir / "scenario_chi2.png")


def plot_run_runtime(run: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    """
    Wall-clock runtime per scenario/model for a run.
    """
    scenarios = _collect_core_scenarios(run)
    if not scenarios:
        return None

    models = run.get("models_present", [])
    if not models:
        return None

    scenario_labels = [s.get("id", f"s{i}") for i, s in enumerate(scenarios)]
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.4), 5))
    for idx, model in enumerate(models):
        runtimes = [
            scenario.get("models", {}).get(model, {}).get("runtime", {}).get("wall_seconds", 0.0)
            for scenario in scenarios
        ]
        ax.bar(
            x + (idx - (len(models) - 1) / 2) * width,
            runtimes,
            width,
            label=model.upper(),
            color=_colour(model),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_format_tick_label(label) for label in scenario_labels])
    ax.set_ylabel("Wall time [s]")
    ax.set_title(f"Scenario Wall Time — {run['name']}")
    ax.legend()

    return _save(fig, plot_dir / "scenario_runtime.png")


def plot_run_parameter_scatter(run: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    """
    Scatter plot of best-fit H0 vs Ωm per scenario/model within a run.
    """
    scenarios = _collect_core_scenarios(run)
    if not scenarios:
        return None

    models = run.get("models_present", [])
    if not models:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    any_points = False

    for model in models:
        x_vals = []
        y_vals = []
        labels = []

        for scenario in scenarios:
            params = scenario.get("models", {}).get(model, {}).get("best_fit", {}).get("params", {})
            h0 = params.get("H0")
            om0 = params.get("Om0")
            if h0 is None or om0 is None:
                continue
            x_vals.append(float(h0))
            y_vals.append(float(om0))
            labels.append(scenario.get("id", ""))

        if not x_vals:
            continue

        any_points = True
        ax.scatter(x_vals, y_vals, label=model.upper(), marker=_marker(model), color=_colour(model), s=60)
        for xv, yv, label in zip(x_vals, y_vals, labels):
            ax.annotate(label, (xv, yv), textcoords="offset points", xytext=(5, 5), fontsize=8, color=_colour(model))

    if not any_points:
        plt.close(fig)
        return None

    ax.set_xlabel("H₀ [km/s/Mpc]")
    ax.set_ylabel("Ωₘ")
    ax.set_title(f"Best-Fit Parameter Scatter — {run['name']}")
    ax.legend()

    return _save(fig, plot_dir / "parameter_scatter.png")


def plot_run_joint_deltas(run: Dict[str, Any], plot_dir: Path) -> Optional[Path]:
    """
    Visualise ΔAIC and ΔBIC (PBUF - LCDM) across scenarios if joint artifacts exist.
    """
    scenarios = _collect_core_scenarios(run)
    deltas = []
    labels = []
    for scenario in scenarios:
        joint = scenario.get("joint")
        if not joint:
            continue
        delta_aic = joint.get("deltas", {}).get("delta_aic")
        delta_bic = joint.get("deltas", {}).get("delta_bic")
        if delta_aic is None and delta_bic is None:
            continue
        labels.append(scenario.get("id", ""))
        deltas.append((delta_aic, delta_bic))

    if not deltas:
        return None

    delta_aic_values = [d[0] for d in deltas]
    delta_bic_values = [d[1] for d in deltas]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - width / 2, delta_aic_values, width, label="ΔAIC (PBUF-LCDM)", color="#4e8cff")
    ax.bar(x + width / 2, delta_bic_values, width, label="ΔBIC (PBUF-LCDM)", color="#ff8c4e")

    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_tick_label(label) for label in labels])
    ax.set_ylabel("Δ Metric")
    ax.set_title(f"Joint Comparison Deltas — {run['name']}")
    ax.legend()

    return _save(fig, plot_dir / "joint_deltas.png")


# ----------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------

def generate_all_plots(stats: Dict[str, Any], plot_dir: str = "reports/output/plots") -> Dict[str, List[str]]:
    """
    Generate aggregated and per-run plots.

    Returns
    -------
    dict
        {
            "aggregated": [<paths>],
            "<run_name>": [<paths>],
            ...
        }
    """
    base_path = Path(plot_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    generated: Dict[str, List[str]] = {"aggregated": []}

    agg_paths = [
        plot_aggregated_model_totals(stats, base_path),
        plot_aggregated_dataset_breakdown(stats, base_path),
    ]
    generated["aggregated"] = [
        str(path) for path in agg_paths if path is not None
    ]

    for run in stats.get("runs", []):
        run_dir = base_path / run["name"]
        run_paths = [
            plot_run_scenario_chi2(run, run_dir),
            plot_run_runtime(run, run_dir),
            plot_run_parameter_scatter(run, run_dir),
            plot_run_joint_deltas(run, run_dir),
        ]
        generated[run["name"]] = [str(path) for path in run_paths if path is not None]

    print(f"[OK] Generated plots for {len(stats.get('runs', []))} science run(s) in {base_path}/")
    return generated
