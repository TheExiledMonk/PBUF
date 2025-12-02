"""Sound Horizon Evolution plot implementation for r_d comparison between ΛCDM and PBUF models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.models.model_factory import create_model as create_cosmos2_model

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:  # pragma: no cover - optional plotting
    plt = None
    matplotlib = None


def _load_pbuf_lut() -> dict[str, np.ndarray]:
    """Load PBUF thermal lookup table."""
    from cosmos2.pbuf.microphysics import ensure_thermal_table

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


def build_model(model_name: str, overrides: dict[str, float]) -> Any:
    """Build a cosmological model with given parameter overrides."""
    normalized = {key: float(value) for key, value in overrides.items()}
    
    # Add default optional parameters if missing
    if model_name == "lcdm":
        normalized.setdefault("Omega_r0", 9.0e-5)
        normalized.setdefault("Omega_k0", 0.0)
        normalized.setdefault("sigma8_0", 0.811)
    elif model_name == "pbuf":
        normalized.setdefault("Omega_r0", 9.0e-5)
        normalized.setdefault("alpha", 0.0)
        normalized.setdefault("omega_normalization", "flat_today")
        normalized.setdefault("sigma_rescale", 1.0)
    
    if model_name == "pbuf":
        lut = _load_pbuf_lut()
        return create_cosmos2_model("pbuf", lut=lut, **normalized)
    return create_cosmos2_model(model_name, **normalized)


def _calculate_rd_parameter_scaling(
    base_params_lcdm: dict[str, float],
    base_params_pbuf: dict[str, float],
    param_name: str,
    param_range: np.ndarray,
    other_params: dict[str, float] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate sound horizon scaling with a specific parameter.
    
    Parameters:
    -----------
    base_params_lcdm : dict[str, float]
        Base parameters for ΛCDM model.
    base_params_pbuf : dict[str, float]
        Base parameters for PBUF model.
    param_name : str
        Name of parameter to vary.
    param_range : np.ndarray
        Range of parameter values.
    other_params : dict[str, float] | None
        Fixed parameters for all models.
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        r_d values for ΛCDM and PBUF models.
    """
    rd_lcdm_values = []
    rd_pbuf_values = []
    
    for param_value in param_range:
        # ΛCDM parameters
        lcdm_params = base_params_lcdm.copy()
        lcdm_params[param_name] = param_value
        if other_params:
            lcdm_params.update(other_params)
        
        # PBUF parameters
        pbuf_params = base_params_pbuf.copy()
        pbuf_params[param_name] = param_value
        if other_params:
            pbuf_params.update(other_params)
        
        try:
            lcdm_model = build_model("lcdm", lcdm_params)
            pbuf_model = build_model("pbuf", pbuf_params)
            
            rd_lcdm = lcdm_model.sound_horizon()
            rd_pbuf = pbuf_model.sound_horizon()
            
            rd_lcdm_values.append(rd_lcdm)
            rd_pbuf_values.append(rd_pbuf)
        except Exception:
            # Skip invalid parameter combinations
            rd_lcdm_values.append(np.nan)
            rd_pbuf_values.append(np.nan)
    
    return np.array(rd_lcdm_values), np.array(rd_pbuf_values)


def create_sound_horizon_evolution(
    lcdm_params: dict[str, float] | None = None,
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
    plot_type: str = "parameter_scaling",
    param_to_vary: str = "Omega_m0",
    param_range: Tuple[float, float] | None = None,
    n_points: int = 50,
) -> Path | None:
    """
    Create Sound Horizon Evolution plot comparing r_d between ΛCDM and PBUF models.
    
    Parameters:
    -----------
    lcdm_params : dict[str, float] | None
        Base parameters for ΛCDM model.
    pbuf_params : dict[str, float] | None
        Base parameters for PBUF model.
    output_path : Path | str | None
        Path to save the figure. If None, returns the figure object.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    plot_type : str
        Type of plot: "parameter_scaling" or "comparison".
    param_to_vary : str
        Parameter to vary for scaling plots.
    param_range : Tuple[float, float] | None
        Range for parameter variation. If None, uses sensible defaults.
    n_points : int
        Number of points for parameter variation.
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for sound horizon evolution plotting")

    # Set default parameters
    lcdm_params = lcdm_params or {"H0": 70.0, "Omega_m0": 0.3, "Omega_b0": 0.05}
    pbuf_params = pbuf_params or {"H0": 70.0, "Omega_m0": 0.3, "Omega_b0": 0.05, "Rmax": 3.0}
    
    # Set default parameter ranges
    if param_range is None:
        if param_to_vary == "Omega_m0":
            param_range = (0.2, 0.4)
        elif param_to_vary == "H0":
            param_range = (60, 80)
        elif param_to_vary == "Omega_b0":
            param_range = (0.04, 0.06)
        elif param_to_vary == "Rmax":
            param_range = (1.0, 5.0)
        else:
            param_range = (0.8, 1.2)  # Generic 20% variation
    
    # Create parameter values
    param_values = np.linspace(param_range[0], param_range[1], n_points)
    
    if plot_type == "parameter_scaling":
        # Calculate r_d scaling with parameter
        rd_lcdm_values, rd_pbuf_values = _calculate_rd_parameter_scaling(
            lcdm_params, pbuf_params, param_to_vary, param_values
        )
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: r_d vs parameter
        ax1.plot(param_values, rd_lcdm_values, label="ΛCDM", color="tab:orange", linewidth=2.5)
        ax1.plot(param_values, rd_pbuf_values, label="PBUF", color="tab:red", linewidth=2.5, linestyle="--")
        
        # Formatting for first subplot
        param_label = param_to_vary.replace("_0", "₀").replace("Omega", "Ω").replace("H0", "H₀")
        ax1.set_xlabel(param_label)
        ax1.set_ylabel("Sound horizon r_d [Mpc]")
        ax1.set_title(f"Sound Horizon Evolution vs {param_label}")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3, linestyle=":")
        
        # Plot 2: Difference r_d(PBUF) - r_d(LCDM)
        rd_diff = rd_pbuf_values - rd_lcdm_values
        ax2.plot(param_values, rd_diff, label="Δr_d = r_d(PBUF) - r_d(LCDM)", 
                color="tab:blue", linewidth=2.5)
        ax2.axhline(0, color="black", linestyle=":", alpha=0.7)
        
        # Formatting for second subplot
        ax2.set_xlabel(param_label)
        ax2.set_ylabel("Δr_d [Mpc]")
        ax2.set_title(f"Sound Horizon Difference vs {param_label}")
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3, linestyle=":")
        
        # Add parameter information
        param_info = (f"Fixed parameters:\n"
                     f"ΛCDM: H₀={lcdm_params.get('H0', 70):.1f}, Ω_b={lcdm_params.get('Omega_b0', 0.05):.3f}\n"
                     f"PBUF: H₀={pbuf_params.get('H0', 70):.1f}, Ω_b={pbuf_params.get('Omega_b0', 0.05):.3f}, Rmax={pbuf_params.get('Rmax', 3.0):.1f}")
        
        ax1.text(0.98, 0.02, param_info, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Add statistics
        valid_diff = rd_diff[~np.isnan(rd_diff)]
        if len(valid_diff) > 0:
            stats_text = (f"Δr_d statistics:\n"
                         f"Mean: {np.mean(valid_diff):.2f} Mpc\n"
                         f"Std: {np.std(valid_diff):.2f} Mpc\n"
                         f"Range: [{np.min(valid_diff):.2f}, {np.max(valid_diff):.2f}] Mpc")
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    elif plot_type == "comparison":
        # Simple comparison plot with base parameters
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate base r_d values
        lcdm_model = build_model("lcdm", lcdm_params)
        pbuf_model = build_model("pbuf", pbuf_params)
        
        rd_lcdm = lcdm_model.sound_horizon()
        rd_pbuf = pbuf_model.sound_horizon()
        
        # Create bar plot
        models = ['ΛCDM', 'PBUF']
        rd_values = [rd_lcdm, rd_pbuf]
        colors = ['tab:orange', 'tab:red']
        
        bars = ax.bar(models, rd_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, rd_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f} Mpc', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add difference annotation
        diff = rd_pbuf - rd_lcdm
        ax.annotate(f'Δr_d = {diff:+.1f} Mpc\n({diff/rd_lcdm*100:+.1f}%)',
                   xy=(0.5, max(rd_values) * 0.8), ha='center', va='bottom',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Formatting
        ax.set_ylabel("Sound horizon r_d [Mpc]")
        ax.set_title("Sound Horizon Comparison: ΛCDM vs PBUF")
        ax.set_ylim(0, max(rd_values) * 1.2)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':', axis='y')
        
        # Add parameter information
        param_text = (f"ΛCDM: H₀={lcdm_params.get('H0', 70):.1f}, Ωₘ={lcdm_params.get('Omega_m0', 0.3):.3f}, Ω_b={lcdm_params.get('Omega_b0', 0.05):.3f}\n"
                     f"PBUF: H₀={pbuf_params.get('H0', 70):.1f}, Ωₘ={pbuf_params.get('Omega_m0', 0.3):.3f}, Ω_b={pbuf_params.get('Omega_b0', 0.05):.3f}, Rmax={pbuf_params.get('Rmax', 3.0):.1f}")
        
        ax.text(0.98, 0.02, param_text, transform=ax.transAxes, 
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'parameter_scaling' or 'comparison'.")
    
    fig.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_sound_horizon_evolution_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create sound horizon evolution plots from a completed science run.
    
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
        Dictionary mapping plot types to plot file paths.
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
    
    # Generate different types of plots
    if "lcdm" in model_params and "pbuf" in model_params:
        # Comparison plot
        output_path = output_dir / "sound_horizon_comparison.png"
        path = create_sound_horizon_evolution(
            lcdm_params=model_params["lcdm"],
            pbuf_params=model_params["pbuf"],
            output_path=output_path,
            plot_type="comparison",
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["comparison"] = path
        
        # Parameter scaling plots
        for param in ["Omega_m0", "H0"]:
            output_path = output_dir / f"sound_horizon_scaling_{param}.png"
            path = create_sound_horizon_evolution(
                lcdm_params=model_params["lcdm"],
                pbuf_params=model_params["pbuf"],
                output_path=output_path,
                plot_type="parameter_scaling",
                param_to_vary=param,
                figsize=figsize,
                dpi=dpi,
            )
            if path:
                plot_paths[f"scaling_{param}"] = path
    
    return plot_paths


if __name__ == "__main__":
    # Example usage when run as script
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sound horizon evolution plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--lcdm-params", help="ΛCDM parameters as JSON string")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--plot-type", choices=["comparison", "parameter_scaling"], 
                       default="comparison", help="Type of plot to generate")
    parser.add_argument("--param", help="Parameter to vary for scaling plots")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_sound_horizon_evolution_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} sound horizon evolution plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        lcdm_params = json.loads(args.lcdm_params) if args.lcdm_params else None
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_sound_horizon_evolution(
            lcdm_params=lcdm_params,
            pbuf_params=pbuf_params,
            output_path=args.output,
            plot_type=args.plot_type,
            param_to_vary=args.param,
        )
        if path:
            print(f"Sound horizon evolution plot saved to: {path}")
        else:
            print("Sound horizon evolution plot displayed (no output path specified)")
