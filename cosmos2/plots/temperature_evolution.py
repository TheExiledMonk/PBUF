"""Temperature Evolution plot implementation for PBUF thermal table with logarithmic axes and transitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.pbuf.microphysics import ensure_thermal_table

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:  # pragma: no cover - optional plotting
    plt = None
    matplotlib = None

K_TO_GEV = 8.617333262145e-14  # Boltzmann constant in GeV/K


def _get_thermal_table_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract temperature and scale factor data from the thermal table."""
    table = ensure_thermal_table()
    
    T_values = np.asarray(table.T, dtype=float)  # Temperature in GeV
    a_values = np.asarray(table.a, dtype=float)  # Scale factor
    
    # Sort by scale factor for proper plotting
    sort_idx = np.argsort(a_values)
    a_sorted = a_values[sort_idx]
    T_sorted = T_values[sort_idx]
    
    return a_sorted, T_sorted, a_values, T_values


def _identify_transitions(
    a_values: np.ndarray, 
    T_values: np.ndarray,
    table: Any
) -> list[dict[str, Any]]:
    """Identify important transitions in the thermal evolution."""
    transitions = []
    
    # Standard cosmological transitions (temperature in GeV)
    cosmological_transitions = [
        {"name": "Inflation End", "T": 1e13, "color": "purple", "linestyle": "--"},
        {"name": "Electroweak", "T": 100, "color": "red", "linestyle": "--"},
        {"name": "QCD Confinement", "T": 0.15, "color": "orange", "linestyle": "--"},
        {"name": "BBN Onset", "T": 0.01, "color": "green", "linestyle": "--"},
        {"name": "CMB Decoupling", "T": 2.725e-4, "color": "blue", "linestyle": "--"},
        {"name": "Present Day", "T": 2.725e-13, "color": "black", "linestyle": "-"},
    ]
    
    # Add transitions that fall within our temperature range
    for trans in cosmological_transitions:
        if trans["T"] >= T_values.min() and trans["T"] <= T_values.max():
            # Find closest point in our data
            idx = np.argmin(np.abs(T_values - trans["T"]))
            trans["a"] = a_values[idx]
            trans["T_actual"] = T_values[idx]
            transitions.append(trans)
    
    # Also identify significant changes in epsilon and alpha from the thermal table
    eps_values = np.asarray(table.eps, dtype=float)
    alpha_values = np.asarray(table.alpha, dtype=float)
    
    # Find regions where epsilon or alpha change significantly
    eps_changes = np.diff(eps_values)
    alpha_changes = np.diff(alpha_values)
    
    # Identify major transitions (largest changes)
    eps_threshold = np.percentile(np.abs(eps_changes), 95)  # Top 5% of changes
    alpha_threshold = np.percentile(np.abs(alpha_changes), 95)
    
    eps_transition_indices = np.where(np.abs(eps_changes) > eps_threshold)[0]
    alpha_transition_indices = np.where(np.abs(alpha_changes) > alpha_threshold)[0]
    
    # Add a few key thermal transitions from the table
    thermal_transitions = []
    
    # Sample some key epsilon transitions
    if len(eps_transition_indices) > 0:
        # Take a few representative points
        sample_indices = np.linspace(0, len(eps_transition_indices)-1, 
                                   min(5, len(eps_transition_indices)), dtype=int)
        for i in sample_indices:
            idx = eps_transition_indices[i]
            if idx < len(a_values):
                thermal_transitions.append({
                    "name": f"ε-transition {i+1}",
                    "a": a_values[idx],
                    "T": T_values[idx],
                    "color": "gray",
                    "linestyle": ":",
                    "alpha": 0.5
                })
    
    # Sample some key alpha transitions
    if len(alpha_transition_indices) > 0:
        # Take a few representative points
        sample_indices = np.linspace(0, len(alpha_transition_indices)-1, 
                                   min(5, len(alpha_transition_indices)), dtype=int)
        for i in sample_indices:
            idx = alpha_transition_indices[i]
            if idx < len(a_values):
                thermal_transitions.append({
                    "name": f"α-transition {i+1}",
                    "a": a_values[idx],
                    "T": T_values[idx],
                    "color": "brown",
                    "linestyle": ":",
                    "alpha": 0.5
                })
    
    transitions.extend(thermal_transitions)
    
    return transitions


def create_temperature_evolution(
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
    show_transitions: bool = True,
    transition_alpha: float = 0.7,
    show_thermal_transitions: bool = False,
    table_path: str | None = None,
) -> Path | None:
    """
    Create Temperature Evolution plot from PBUF thermal table with logarithmic axes and transitions.
    
    Parameters:
    -----------
    output_path : Path | str | None
        Path to save the figure. If None, returns the figure object.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    show_transitions : bool
        Whether to show cosmological transition markers.
    transition_alpha : float
        Transparency level for transition markers.
    show_thermal_transitions : bool
        Whether to show thermal field transitions (ε, α changes).
    table_path : str | None
        Optional path to thermal table file (uses default if None).
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for temperature evolution plotting")

    # Get thermal table data
    a_values, T_values, *_ = _get_thermal_table_data()
    T_values_gev = T_values * K_TO_GEV

    # Load thermal table for transition identification
    table = ensure_thermal_table()

    # Identify transitions (data already converted to GeV)
    transitions = _identify_transitions(a_values, T_values_gev, table)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot main temperature evolution curve
    ax.loglog(a_values, T_values_gev, label="PBUF Temperature Evolution", 
             color="tab:red", linewidth=2.5, zorder=4)

    # Add transition markers
    if show_transitions:
        cosmological_transitions = [t for t in transitions if not t["name"].startswith(("ε-transition", "α-transition"))]
        
        for trans in cosmological_transitions:
            ax.axvline(trans["a"], color=trans["color"], linestyle=trans["linestyle"], 
                      alpha=transition_alpha, linewidth=1.5, zorder=3)
            
            # Add transition labels
            label_y = ax.get_ylim()[1] * 0.8  # Position label at 80% of y-range
            ax.text(trans["a"], label_y, trans["name"], 
                   rotation=90, ha="right", va="bottom", fontsize=9,
                   color=trans["color"], alpha=transition_alpha,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add thermal field transitions if requested
    if show_thermal_transitions:
        thermal_transitions = [t for t in transitions if t["name"].startswith(("ε-transition", "α-transition"))]
        
        for trans in thermal_transitions:
            ax.axvline(trans["a"], color=trans["color"], linestyle=trans["linestyle"], 
                      alpha=trans.get("alpha", 0.3), linewidth=1, zorder=2)
    
    # Formatting
    ax.set_xlabel("Scale factor a")
    ax.set_ylabel("Temperature T [GeV]")
    ax.set_title("PBUF Thermal Evolution: Temperature vs Scale Factor")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    
    # Set axis limits
    ax.set_xlim(a_values.min(), a_values.max())
    ax.set_ylim(T_values_gev.min() * 0.5, T_values_gev.max() * 2)  # Some padding
    
    # Add temperature scale annotations
    temp_scales = [
        ("GeV", 1.0, 1e3),
        ("MeV", 1e-3, 1.0),
        ("keV", 1e-6, 1e-3),
        ("eV", 1e-9, 1e-6),
    ]
    
    y_pos = ax.get_ylim()[0] * 2  # Position annotations above the curve
    for scale_name, T_min, T_max in temp_scales:
        if T_min >= T_values_gev.min() and T_max <= T_values_gev.max():
            # Find position in the middle of this temperature range
            T_mid = np.sqrt(T_min * T_max)
            idx = np.argmin(np.abs(T_values_gev - T_mid))
            a_mid = a_values[idx]
            
            ax.text(a_mid, y_pos, scale_name, ha="center", va="bottom", fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.5))

    # Add thermal table information
    table_info = (f"Thermal Table:\n"
                 f"Temperature range: {T_values_gev.min():.2e} - {T_values_gev.max():.2e} GeV\n"
                 f"Scale factor range: {a_values.min():.2e} - {a_values.max():.2e}\n"
                 f"Number of points: {len(T_values)}")
    
    ax.text(0.02, 0.98, table_info, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    kelvin_axis = ax.twinx()
    kelvin_axis.set_yscale("log")
    ylim_low, ylim_high = ax.get_ylim()
    kelvin_axis.set_ylim(ylim_low / K_TO_GEV, ylim_high / K_TO_GEV)
    kelvin_axis.set_ylabel("Temperature T [K]")
    kelvin_axis.grid(False)

    present_T = T_values_gev[-1]
    log_a_values = np.log(a_values)
    log_T_values = np.log(T_values_gev)
    epoch_targets = [
        ("Present", present_T, "black", "2.35e-13 GeV"),
        ("CMB decoupling", 2.6e-10, "blue", "0.26 eV"),
        ("BBN", 1e-4, "teal", "0.1 MeV"),
        ("QCD", 0.15, "green", "0.15 GeV"),
        ("Electroweak", 100.0, "orange", "100 GeV"),
        ("GUT", 1e16, "purple", "1e16 GeV"),
        ("Inflation end", 1e13, "red", "1e13 GeV"),
    ]
    a_min, a_max = a_values.min(), a_values.max()
    for name, T_target, color, temp_label in epoch_targets:
        if T_target <= 0:
            continue
        a_epoch = present_T / T_target
        if not (a_min <= a_epoch <= a_max):
            continue
        log_a_epoch = np.log(a_epoch)
        T_epoch_log = np.interp(log_a_epoch, log_a_values, log_T_values)
        T_epoch = np.exp(T_epoch_log)
        label_text = f"{name}\na≈{a_epoch:.2e}\n{temp_label}"
        ax.text(
            a_epoch,
            T_epoch * 0.3,
            label_text,
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75),
        )

    fig.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_thermal_fields_plot(
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
) -> Path | None:
    """Show ε₀(T) and α(T) from the cached thermal table."""
    if plt is None:
        raise ImportError("matplotlib is required for thermal field plotting")

    table = ensure_thermal_table()
    a_values = np.asarray(table.a, dtype=float)
    T_values = np.asarray(table.T, dtype=float)
    eps_values = np.asarray(table.eps, dtype=float)
    alpha_values = np.asarray(table.alpha, dtype=float)

    sort_idx_T = np.argsort(T_values)
    T_sorted = T_values[sort_idx_T]
    eps_sorted = eps_values[sort_idx_T]

    sort_idx_a = np.argsort(a_values)
    a_sorted = a_values[sort_idx_a]
    alpha_sorted = alpha_values[sort_idx_a]

    fig, (ax_eps, ax_alpha) = plt.subplots(2, 1, figsize=figsize, sharex=False)

    ax_eps.loglog(T_sorted, eps_sorted, color="tab:blue", linewidth=2.0, label="ε₀(T)")
    ax_eps.set_title("Elasticity suppression")
    ax_eps.set_xlabel("Temperature T [GeV]")
    ax_eps.set_ylabel("ε₀(T)")
    ax_eps.grid(True, which="both", alpha=0.3)
    ax_eps.set_xlim(T_sorted.max(), T_sorted.min())

    ax_alpha.loglog(a_sorted, alpha_sorted, color="tab:green", linewidth=2.0, label="α(T)")
    ax_alpha.set_title("Elastic sector deformation")
    ax_alpha.set_xlabel("Scale factor a")
    ax_alpha.set_ylabel("α(T)")
    ax_alpha.grid(True, which="both", alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path
    return fig


def create_temperature_evolution_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create temperature evolution plots from a completed science run.
    
    Parameters:
    -----------
    run_dir : Path
        Directory containing science run results with model fits.
    output_dir : Path | None
        Directory to save plots. If None, uses run_dir/plots.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figures.
        
    Returns:
    --------
    dict[str, Path]
        Dictionary mapping model names to plot file paths.
    """
    run_dir = Path(run_dir)
    output_dir = output_dir or run_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}

    # Temperature evolution is independent of model parameters, but we generate it
    # for consistency with other plotting functions
    output_path = output_dir / "temperature_evolution.png"
    path = create_temperature_evolution(
        output_path=output_path,
        figsize=figsize,
        dpi=dpi,
    )
    if path:
        plot_paths["thermal"] = path
    fields_path = output_dir / "thermal_fields.png"
    fields_plot = create_thermal_fields_plot(
        output_path=fields_path,
        figsize=(10, 8),
        dpi=dpi,
    )
    if fields_plot:
        plot_paths["thermal_fields"] = fields_plot

    return plot_paths


if __name__ == "__main__":
    # Example usage when run as script
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate temperature evolution plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--no-transitions", action="store_true", help="Disable cosmological transition markers")
    parser.add_argument("--show-thermal-transitions", action="store_true", help="Show thermal field transitions")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_temperature_evolution_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} temperature evolution plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot
        path = create_temperature_evolution(
            output_path=args.output,
            show_transitions=not args.no_transitions,
            show_thermal_transitions=args.show_thermal_transitions,
        )
        if path:
            print(f"Temperature evolution plot saved to: {path}")
        else:
            print("Temperature evolution plot displayed (no output path specified)")
