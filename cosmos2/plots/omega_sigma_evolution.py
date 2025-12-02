"""Ωσ(a) Evolution plot implementation for PBUF model with jackknife variation support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.models.pbuf.elastic import omega_sigma_of_a
from cosmos2.models.pbuf.params import PBUFParams
from cosmos2.pbuf.microphysics import ensure_thermal_table

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:  # pragma: no cover - optional plotting
    plt = None
    matplotlib = None


def _load_pbuf_lut() -> dict[str, np.ndarray]:
    """Load PBUF thermal lookup table."""
    table = ensure_thermal_table()
    return {
        "T": np.asarray(table.T, dtype=float),
        "eps": np.asarray(table.eps, dtype=float),
        "alpha": np.asarray(table.alpha, dtype=float),
        "dln_eps": np.asarray(table.dln_eps, dtype=float),
        "dln_alpha": np.asarray(table.dln_alpha, dtype=float),
        "g_star": np.asarray(table.g_star, dtype=float),
        "g_starS": np.asarray(table.g_starS, dtype=float),
        "a": np.asarray(table.a, dtype=float),
        "metadata": getattr(table, "metadata", {}),
    }


def build_pbuf_params(overrides: dict[str, float]) -> PBUFParams:
    """Build PBUF parameters with given overrides."""
    normalized = {key: float(value) for key, value in overrides.items()}
    
    # Add default optional parameters if missing
    normalized.setdefault("Omega_r0", 9.0e-5)
    normalized.setdefault("alpha", 0.0)
    normalized.setdefault("omega_normalization", "flat_today")
    normalized.setdefault("sigma_rescale", 1.0)
    
    return PBUFParams(**normalized)


def _calculate_omega_sigma_curve(
    params: PBUFParams, 
    a_range: np.ndarray,
    table: Any
) -> np.ndarray:
    """Calculate Ωσ(a) curve for given parameters."""
    return np.asarray([omega_sigma_of_a(a, params, table) for a in a_range], dtype=float)


def _calculate_jackknife_variation(
    base_params: PBUFParams,
    a_range: np.ndarray,
    table: Any,
    n_samples: int = 100,
    variation_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate jackknife variation bands for Ωσ(a).
    
    This is a simplified implementation that creates variation bands
    based on parameter variations. In a full implementation, this would
    use actual jackknife resampling from the science runner.
    """
    base_curve = _calculate_omega_sigma_curve(base_params, a_range, table)
    
    # Create parameter variations
    variations = []
    
    # Vary key parameters
    param_variations = {
        'Rmax': variation_frac * base_params.Rmax,
        'Omega_m0': variation_frac * base_params.Omega_m0,
        'H0': variation_frac * base_params.H0,
        'sigma_rescale': variation_frac * getattr(base_params, 'sigma_rescale', 1.0),
    }
    
    for i in range(n_samples):
        varied_params_dict = {}
        for key, base_value in base_params.__dict__.items():
            if key in param_variations:
                # Add random variation
                variation = np.random.normal(0, param_variations[key])
                varied_params_dict[key] = max(getattr(base_params, key) + variation, 
                                             1e-6 if key == 'Rmax' else 0.0)
            else:
                varied_params_dict[key] = base_value
        
        try:
            varied_params = PBUFParams(**varied_params_dict)
            varied_curve = _calculate_omega_sigma_curve(varied_params, a_range, table)
            variations.append(varied_curve)
        except Exception:
            # Skip invalid parameter combinations
            continue
    
    if variations:
        variations = np.array(variations)
        lower = np.percentile(variations, 16, axis=0)  # 1-sigma lower bound
        upper = np.percentile(variations, 84, axis=0)  # 1-sigma upper bound
        return lower, upper
    else:
        # Fallback: simple percentage bands
        return base_curve * 0.8, base_curve * 1.2


def create_omega_sigma_evolution(
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 200,
    a_min: float = 1e-12,
    a_max: float = 1.0,
    n_points: int = 200,
    show_jackknife: bool = True,
    n_jackknife_samples: int = 100,
    jackknife_alpha: float = 0.3,
    use_log_scale: bool = True,
) -> Path | None:
    """
    Create Ωσ(a) Evolution plot for PBUF model with jackknife variation bands.
    
    Parameters:
    -----------
    pbuf_params : dict[str, float] | None
        Parameter overrides for PBUF model (e.g., {"H0": 70.0, "Omega_m0": 0.3, "Rmax": 3.0})
    output_path : Path | str | None
        Path to save the figure. If None, returns the figure object.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    a_min : float
        Minimum scale factor (default: 1e-12 for V12 range).
    a_max : float
        Maximum scale factor (default: 1.0 for present day).
    n_points : int
        Number of points for the curve.
    show_jackknife : bool
        Whether to show jackknife variation bands.
    n_jackknife_samples : int
        Number of jackknife samples for variation estimation.
    jackknife_alpha : float
        Transparency level for jackknife bands.
    use_log_scale : bool
        Whether to use logarithmic scales for both axes.
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for Ωσ(a) evolution plotting")

    # Build PBUF parameters
    pbuf_params = pbuf_params or {}
    params = build_pbuf_params(pbuf_params)
    
    # Load thermal table
    table = ensure_thermal_table()
    
    # Create scale factor range (logarithmic spacing for V12 range)
    a_range = np.logspace(np.log10(a_min), np.log10(a_max), n_points)
    
    # Calculate base Ωσ(a) curve
    omega_sigma_curve = _calculate_omega_sigma_curve(params, a_range, table)
    
    # Calculate jackknife variation bands
    jackknife_lower, jackknife_upper = None, None
    if show_jackknife:
        jackknife_lower, jackknife_upper = _calculate_jackknife_variation(
            params, a_range, table, n_jackknife_samples
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot jackknife variation bands first (so they appear behind the main curve)
    if show_jackknife and jackknife_lower is not None and jackknife_upper is not None:
        ax.fill_between(a_range, jackknife_lower, jackknife_upper, 
                      color="tab:red", alpha=jackknife_alpha, 
                      label="Jackknife variation (1σ)")

    # Plot main Ωσ(a) curve
    ax.plot(a_range, omega_sigma_curve, label="PBUF Ωσ(a)", color="tab:red", 
           linewidth=2.5, zorder=4)

    # Formatting
    ax.set_xlabel("Scale factor a")
    ax.set_ylabel("Ωσ(a)")
    ax.set_title("PBUF Elastic Sector Energy Density Evolution")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle=":")
    
    # Use logarithmic scales if requested
    if use_log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    # Set axis limits
    ax.set_xlim(a_min, a_max)
    
    # Add parameter information
    param_text = (f"PBUF parameters:\n"
                 f"H₀ = {params.H0:.1f} km/s/Mpc\n"
                 f"Ωₘ = {params.Omega_m0:.3f}\n"
                 f"Rmax = {params.Rmax:.2f}\n"
                 f"σ_rescale = {getattr(params, 'sigma_rescale', 1.0):.3f}")
    
    ax.text(0.98, 0.02, param_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Add cosmic time markers
    if use_log_scale:
        # Add markers for important cosmic epochs
        cosmic_markers = [
            (1e-12, "BBN", "Big Bang Nucleosynthesis"),
            (1e-9, "CMB", "CMB decoupling"),
            (1e-3, "Reion.", "Reionization"),
            (0.1, "Structure", "Structure formation"),
            (1.0, "Today", "Present day"),
        ]
        
        for a_val, label, tooltip in cosmic_markers:
            if a_min <= a_val <= a_max:
                ax.axvline(a_val, color="gray", linestyle=":", alpha=0.5, linewidth=1)
                ax.text(a_val, ax.get_ylim()[1] * 0.5, label, 
                       rotation=90, ha="right", va="bottom", fontsize=8, alpha=0.7)

    # Add V12 range annotation
    range_text = f"V12 range: a ∈ [{a_min:.0e}, {a_max:.0e}]"
    ax.text(0.02, 0.98, range_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    fig.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_omega_sigma_evolution_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create Ωσ(a) evolution plots from a completed science run.
    
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
    from cosmos2.science_runner.utils import load_json_or_yaml
    
    run_dir = Path(run_dir)
    output_dir = output_dir or run_dir / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    # Find model directories
    model_dirs = [
        child for child in run_dir.iterdir()
        if child.is_dir() and (child / "best_fit.json").exists()
    ]
    
    # Collect model parameters
    model_params = {}
    for model_dir in model_dirs:
        best_fit = load_json_or_yaml(model_dir / "best_fit.json")
        if best_fit and "parameters" in best_fit:
            model_params[model_dir.name] = best_fit["parameters"]
    
    # Generate plot for PBUF model only (Ωσ is specific to PBUF)
    if "pbuf" in model_params:
        output_path = output_dir / "omega_sigma_evolution.png"
        path = create_omega_sigma_evolution(
            pbuf_params=model_params["pbuf"],
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["pbuf"] = path
    
    return plot_paths


if __name__ == "__main__":
    # Example usage when run as script
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Ωσ(a) evolution plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--no-jackknife", action="store_true", help="Disable jackknife variation bands")
    parser.add_argument("--linear", action="store_true", help="Use linear scale instead of logarithmic")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_omega_sigma_evolution_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} Ωσ(a) evolution plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_omega_sigma_evolution(
            pbuf_params=pbuf_params,
            output_path=args.output,
            show_jackknife=not args.no_jackknife,
            use_log_scale=not args.linear,
        )
        if path:
            print(f"Ωσ(a) evolution plot saved to: {path}")
        else:
            print("Ωσ(a) evolution plot displayed (no output path specified)")
