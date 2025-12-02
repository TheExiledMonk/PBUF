"""Residuals Grid plot implementation showing residuals for SN, BAO, CC, RSD datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np

from cosmos2.data.registry import get_dataset
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


def _calculate_sn_residuals(model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate residuals for SN dataset (distance modulus)."""
    dataset = get_dataset("sn")
    z_sn = np.asarray(dataset["z"], dtype=float)
    mu_obs = np.asarray(dataset["obs"], dtype=float)
    
    # Calculate model predictions
    mu_model = np.asarray([model.distance_modulus(z) for z in z_sn], dtype=float)
    
    # Calculate residuals
    residuals = mu_obs - mu_model
    
    return z_sn, residuals, mu_obs


def _calculate_bao_residuals(model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate residuals for BAO dataset (distance ratios)."""
    # Combine anisotropic and isotropic BAO data
    residuals_list = []
    z_list = []
    obs_list = []
    
    # Anisotropic BAO data (DM/rd and DH/rd)
    try:
        bao_aniso = get_dataset("bao_aniso")
        z_aniso = np.asarray(bao_aniso["z"], dtype=float)
        obs_aniso = np.asarray(bao_aniso["obs"], dtype=float)
        
        rd = float(model.sound_horizon())
        
        # For anisotropic data, we need to handle the combined observable
        # This is a simplification - in practice, anisotropic BAO has multiple observables
        for i, z in enumerate(z_aniso):
            DM = model.DM(z)
            DH = 299792.458 / model.Hubble(z)  # c/H(z)
            
            # Use DM/rd as the primary observable (simplified)
            model_obs = DM / rd
            residual = obs_aniso[i] - model_obs
            
            residuals_list.append(residual)
            z_list.append(z)
            obs_list.append(obs_aniso[i])
    except Exception:
        pass
    
    # Isotropic BAO data (DV/rd)
    try:
        bao_iso = get_dataset("bao_iso")
        z_iso = np.asarray(bao_iso["z"], dtype=float)
        obs_iso = np.asarray(bao_iso["obs"], dtype=float)
        
        rd = float(model.sound_horizon())
        
        for i, z in enumerate(z_iso):
            DV = model.DV(z)
            model_obs = DV / rd
            residual = obs_iso[i] - model_obs
            
            residuals_list.append(residual)
            z_list.append(z)
            obs_list.append(obs_iso[i])
    except Exception:
        pass
    
    if residuals_list:
        return np.array(z_list), np.array(residuals_list), np.array(obs_list)
    else:
        return np.array([]), np.array([]), np.array([])


def _calculate_cc_residuals(model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate residuals for CC dataset (H(z))."""
    dataset = get_dataset("cc")
    z_cc = np.asarray(dataset["z"], dtype=float)
    h_obs = np.asarray(dataset["obs"], dtype=float)
    
    # Calculate model predictions
    h_model = np.asarray([model.Hubble(z) for z in z_cc], dtype=float)
    
    # Calculate residuals
    residuals = h_obs - h_model
    
    return z_cc, residuals, h_obs


def _calculate_rsd_residuals(model: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate residuals for RSD dataset (fσ₈)."""
    dataset = get_dataset("rsd")
    z_rsd = np.asarray(dataset["z"], dtype=float)
    fs8_obs = np.asarray(dataset["obs"], dtype=float)
    
    # Calculate model predictions (use array input for efficiency)
    fs8_model = np.asarray(model.fs8(z_rsd), dtype=float)
    
    # Calculate residuals
    residuals = fs8_obs - fs8_model
    
    return z_rsd, residuals, fs8_obs


def _calculate_residual_statistics(residuals: np.ndarray) -> Dict[str, float]:
    """Calculate statistics for residuals."""
    if len(residuals) == 0:
        return {}
    
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "rms": float(np.sqrt(np.mean(residuals**2))),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "n_points": int(len(residuals))
    }


def create_residuals_grid(
    lcdm_params: dict[str, float] | None = None,
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (15, 12),
    dpi: int = 200,
    show_statistics: bool = True,
    scatter_alpha: float = 0.6,
    line_alpha: float = 0.8,
) -> Path | None:
    """
    Create Residuals Grid plot showing residuals for SN, BAO, CC, RSD datasets.
    
    Parameters:
    -----------
    lcdm_params : dict[str, float] | None
        Parameter overrides for ΛCDM model.
    pbuf_params : dict[str, float] | None
        Parameter overrides for PBUF model.
    output_path : Path | str | None
        Path to save the figure. If None, returns the figure object.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    show_statistics : bool
        Whether to show residual statistics on each panel.
    scatter_alpha : float
        Transparency level for scatter points.
    line_alpha : float
        Transparency level for connecting lines.
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for residuals grid plotting")

    # Build models
    lcdm_params = lcdm_params or {}
    pbuf_params = pbuf_params or {}
    
    lcdm_model = None
    pbuf_model = None
    
    if lcdm_params:
        lcdm_model = build_model("lcdm", lcdm_params)
    
    if pbuf_params:
        pbuf_model = build_model("pbuf", pbuf_params)

    # Create figure with 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing
    
    # Dataset configurations
    datasets = [
        ("SN", _calculate_sn_residuals, "Distance Modulus Residuals", "[mag]"),
        ("BAO", _calculate_bao_residuals, "Distance Ratio Residuals", "[dimensionless]"),
        ("CC", _calculate_cc_residuals, "H(z) Residuals", "[km/s/Mpc]"),
        ("RSD", _calculate_rsd_residuals, "fσ₈ Residuals", "[dimensionless]"),
    ]
    
    colors = {"LCDM": "tab:orange", "PBUF": "tab:red"}
    linestyles = {"LCDM": "-", "PBUF": "--"}
    
    for idx, (name, calc_func, ylabel, units) in enumerate(datasets):
        ax = axes[idx]
        
        # Calculate residuals for each model
        if lcdm_model:
            try:
                z_data, res_lcdm, obs_data = calc_func(lcdm_model)
                stats_lcdm = _calculate_residual_statistics(res_lcdm)
            except Exception as e:
                print(f"Error calculating {name} residuals for LCDM: {e}")
                z_data, res_lcdm, obs_data = np.array([]), np.array([]), np.array([])
                stats_lcdm = {}
        else:
            z_data, res_lcdm, obs_data = np.array([]), np.array([]), np.array([])
            stats_lcdm = {}
        
        if pbuf_model:
            try:
                z_data_pbuf, res_pbuf, obs_data_pbuf = calc_func(pbuf_model)
                stats_pbuf = _calculate_residual_statistics(res_pbuf)
            except Exception as e:
                print(f"Error calculating {name} residuals for PBUF: {e}")
                z_data_pbuf, res_pbuf, obs_data_pbuf = np.array([]), np.array([]), np.array([])
                stats_pbuf = {}
        else:
            z_data_pbuf, res_pbuf, obs_data_pbuf = np.array([]), np.array([]), np.array([])
            stats_pbuf = {}
        
        # Plot residuals
        if len(res_lcdm) > 0:
            # Sort by redshift for line plotting
            sort_idx = np.argsort(z_data)
            z_sorted = z_data[sort_idx]
            res_sorted = res_lcdm[sort_idx]
            
            ax.scatter(z_sorted, res_sorted, color=colors["LCDM"], alpha=scatter_alpha, 
                      s=20, label="ΛCDM", zorder=3)
            ax.plot(z_sorted, res_sorted, color=colors["LCDM"], alpha=line_alpha, 
                   linestyle=linestyles["LCDM"], linewidth=1, zorder=2)
        
        if len(res_pbuf) > 0:
            # Sort by redshift for line plotting
            sort_idx = np.argsort(z_data_pbuf)
            z_sorted = z_data_pbuf[sort_idx]
            res_sorted = res_pbuf[sort_idx]
            
            ax.scatter(z_sorted, res_sorted, color=colors["PBUF"], alpha=scatter_alpha, 
                      s=20, label="PBUF", zorder=3)
            ax.plot(z_sorted, res_sorted, color=colors["PBUF"], alpha=line_alpha, 
                   linestyle=linestyles["PBUF"], linewidth=1, zorder=2)
        
        # Add zero line
        ax.axhline(0, color="black", linestyle=":", alpha=0.7, linewidth=1)
        
        # Formatting
        ax.set_xlabel("Redshift z")
        ax.set_ylabel(f"{ylabel} {units}")
        ax.set_title(f"{name} Dataset Residuals")
        ax.grid(True, alpha=0.3, linestyle=":")
        
        # Add legend
        if len(res_lcdm) > 0 or len(res_pbuf) > 0:
            ax.legend(loc="best", fontsize=8)
        
        # Add statistics
        if show_statistics and (stats_lcdm or stats_pbuf):
            stats_text = ""
            if stats_lcdm:
                stats_text += f"ΛCDM: μ={stats_lcdm.get('mean', 0):.3f}, σ={stats_lcdm.get('std', 0):.3f}\n"
            if stats_pbuf:
                stats_text += f"PBUF: μ={stats_pbuf.get('mean', 0):.3f}, σ={stats_pbuf.get('std', 0):.3f}"
            
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        # Add dataset info
        n_points = max(len(res_lcdm), len(res_pbuf))
        if n_points > 0:
            info_text = f"n = {n_points}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Add overall title and parameter information
    fig.suptitle("Residuals Grid: Model Fits to Observational Datasets", fontsize=14, fontweight='bold')
    
    # Add parameter information at the bottom
    param_text = ""
    if lcdm_model:
        param_text += f"ΛCDM: H₀={lcdm_params.get('H0', 70):.1f}, Ωₘ={lcdm_params.get('Omega_m0', 0.3):.3f}"
    if pbuf_model:
        if param_text:
            param_text += "\n"
        param_text += f"PBUF: H₀={pbuf_params.get('H0', 70):.1f}, Ωₘ={pbuf_params.get('Omega_m0', 0.3):.3f}, Rmax={pbuf_params.get('Rmax', 3.0):.1f}"
    
    if param_text:
        fig.text(0.5, 0.02, param_text, ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])  # Adjust for suptitle and parameter text
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_residuals_grid_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (15, 12),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create residuals grid plots from a completed science run.
    
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
    
    # Generate residuals grid plot
    output_path = output_dir / "residuals_grid.png"
    path = create_residuals_grid(
        lcdm_params=model_params.get("lcdm"),
        pbuf_params=model_params.get("pbuf"),
        output_path=output_path,
        figsize=figsize,
        dpi=dpi,
    )
    if path:
        plot_paths["residuals_grid"] = path
    
    return plot_paths


if __name__ == "__main__":
    # Example usage when run as script
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate residuals grid plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--lcdm-params", help="ΛCDM parameters as JSON string")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--no-stats", action="store_true", help="Hide statistics on panels")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_residuals_grid_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} residuals grid plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        lcdm_params = json.loads(args.lcdm_params) if args.lcdm_params else None
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_residuals_grid(
            lcdm_params=lcdm_params,
            pbuf_params=pbuf_params,
            output_path=args.output,
            show_statistics=not args.no_stats,
        )
        if path:
            print(f"Residuals grid plot saved to: {path}")
        else:
            print("Residuals grid plot displayed (no output path specified)")
