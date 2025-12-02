"""Growth Rate plot implementation for fσ₈ data with ΛCDM and PBUF predictions and uncertainty bands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

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


def _extract_rsd_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract RSD (fσ₈) data from the dataset."""
    dataset = get_dataset("rsd")
    z = np.asarray(dataset["z"], dtype=float)
    fs8_obs = np.asarray(dataset["obs"], dtype=float)
    
    # Extract errors from covariance matrix
    cov = np.asarray(dataset.get("cov"), dtype=float)
    if cov.size > 0:
        fs8_err = np.sqrt(np.diag(cov))
    else:
        # Fallback to err field if available
        err = dataset.get("err")
        if err is not None:
            fs8_err = np.asarray(err, dtype=float)
        else:
            raise ValueError("RSD dataset lacks covariance and error information")
    
    return z, fs8_obs, fs8_err


def _calculate_model_fs8(model: Any, z_range: np.ndarray) -> np.ndarray:
    """Calculate model fσ₈ predictions for a given redshift range."""
    return np.asarray(model.fs8(z_range), dtype=float)


def _calculate_uncertainty_bands(
    model: Any, 
    z_range: np.ndarray, 
    uncertainty_params: dict[str, float] | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate uncertainty bands for model predictions.
    
    This is a simplified implementation that creates small uncertainty bands
    based on parameter variations. In a full implementation, this would
    use proper error propagation or MCMC samples.
    """
    if uncertainty_params is None:
        # Default small uncertainty bands (5% variation)
        base_prediction = _calculate_model_fs8(model, z_range)
        uncertainty = 0.05 * base_prediction
        return base_prediction - uncertainty, base_prediction + uncertainty
    
    # For now, return simple bands - this could be enhanced with proper error propagation
    base_prediction = _calculate_model_fs8(model, z_range)
    return base_prediction * 0.95, base_prediction * 1.05


def create_growth_rate_plot(
    lcdm_params: dict[str, float] | None = None,
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
    z_max: float = 2.0,
    n_points: int = 100,
    show_uncertainty: bool = True,
    uncertainty_alpha: float = 0.3,
) -> Path | None:
    """
    Create Growth Rate plot with fσ₈ data, ΛCDM and PBUF predictions, and uncertainty bands.
    
    Parameters:
    -----------
    lcdm_params : dict[str, float] | None
        Parameter overrides for ΛCDM model (e.g., {"H0": 70.0, "Omega_m0": 0.3})
    pbuf_params : dict[str, float] | None
        Parameter overrides for PBUF model (e.g., {"H0": 70.0, "Omega_m0": 0.3, "Rmax": 3.0})
    output_path : Path | str | None
        Path to save the figure. If None, returns the figure object.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
    z_max : float
        Maximum redshift for model curves.
    n_points : int
        Number of points for model curves.
    show_uncertainty : bool
        Whether to show uncertainty bands around model curves.
    uncertainty_alpha : float
        Transparency level for uncertainty bands.
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for growth rate plotting")

    # Extract RSD data
    z_rsd, fs8_obs, fs8_err = _extract_rsd_data()
    
    # Build models with default or provided parameters
    lcdm_params = lcdm_params or {}
    pbuf_params = pbuf_params or {}
    
    lcdm_model = None
    pbuf_model = None
    
    if lcdm_params:
        lcdm_model = build_model("lcdm", lcdm_params)
    
    if pbuf_params:
        pbuf_model = build_model("pbuf", pbuf_params)

    # Create redshift range for model curves
    z_model = np.linspace(0.01, z_max, n_points)
    
    # Calculate model predictions
    fs8_lcdm = None
    fs8_pbuf = None
    
    if lcdm_model:
        fs8_lcdm = _calculate_model_fs8(lcdm_model, z_model)
    
    if pbuf_model:
        fs8_pbuf = _calculate_model_fs8(pbuf_model, z_model)

    # Calculate uncertainty bands
    lcdm_lower, lcdm_upper = None, None
    pbuf_lower, pbuf_upper = None, None
    
    if show_uncertainty:
        if lcdm_model:
            lcdm_lower, lcdm_upper = _calculate_uncertainty_bands(lcdm_model, z_model)
        if pbuf_model:
            pbuf_lower, pbuf_upper = _calculate_uncertainty_bands(pbuf_model, z_model)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot uncertainty bands first (so they appear behind the main curves)
    if show_uncertainty:
        if lcdm_lower is not None and lcdm_upper is not None:
            ax.fill_between(z_model, lcdm_lower, lcdm_upper, 
                          color="tab:orange", alpha=uncertainty_alpha, 
                          label="ΛCDM uncertainty")
        
        if pbuf_lower is not None and pbuf_upper is not None:
            ax.fill_between(z_model, pbuf_lower, pbuf_upper, 
                          color="tab:red", alpha=uncertainty_alpha, 
                          label="PBUF uncertainty")

    # Plot RSD data
    ax.errorbar(z_rsd, fs8_obs, yerr=fs8_err, fmt="o", markersize=6, capsize=3, 
               color="tab:blue", label="RSD fσ₈ data", alpha=0.8, zorder=5)

    # Plot model predictions
    if fs8_lcdm is not None:
        ax.plot(z_model, fs8_lcdm, label="ΛCDM prediction", color="tab:orange", 
               linewidth=2.5, zorder=4)
    
    if fs8_pbuf is not None:
        ax.plot(z_model, fs8_pbuf, label="PBUF prediction", color="tab:red", 
               linewidth=2.5, linestyle="--", zorder=4)

    # Formatting
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("fσ₈")
    ax.set_title("Growth Rate fσ₈: RSD Data with Model Predictions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")
    
    # Set reasonable axis limits
    z_min, z_max_data = z_rsd.min(), z_rsd.max()
    margin = 0.05 * (z_max_data - z_min)
    ax.set_xlim(z_min - margin, max(z_max, z_max_data) + margin)
    
    # Set y-axis limits based on data and models
    all_values = []
    all_values.extend(fs8_obs.tolist())
    if fs8_lcdm is not None:
        all_values.extend(fs8_lcdm.tolist())
    if fs8_pbuf is not None:
        all_values.extend(fs8_pbuf.tolist())
    
    y_min, y_max = min(all_values), max(all_values)
    y_margin = 0.1 * (y_max - y_min)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Add data summary information
    n_points = len(z_rsd)
    z_range_text = f"RSD data: {n_points} points, z ∈ [{z_rsd.min():.2f}, {z_rsd.max():.2f}]"
    ax.text(0.02, 0.98, z_range_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Add model parameter information
    param_text = ""
    if lcdm_model:
        param_text += f"ΛCDM: H₀={lcdm_params.get('H0', 70):.1f}, Ωₘ={lcdm_params.get('Omega_m0', 0.3):.3f}"
    if pbuf_model:
        if param_text:
            param_text += "\n"
        param_text += f"PBUF: H₀={pbuf_params.get('H0', 70):.1f}, Ωₘ={pbuf_params.get('Omega_m0', 0.3):.3f}, Rmax={pbuf_params.get('Rmax', 3.0):.1f}"
    
    if param_text:
        ax.text(0.98, 0.02, param_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_growth_rate_plot_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create growth rate plots from a completed science run.
    
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
    
    # Generate plots for different combinations
    if "lcdm" in model_params and "pbuf" in model_params:
        # Both models available
        output_path = output_dir / "growth_rate_both_models.png"
        path = create_growth_rate_plot(
            lcdm_params=model_params["lcdm"],
            pbuf_params=model_params["pbuf"],
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["both_models"] = path
    
    # Individual model plots
    if "lcdm" in model_params:
        output_path = output_dir / "growth_rate_lcdm.png"
        path = create_growth_rate_plot(
            lcdm_params=model_params["lcdm"],
            pbuf_params=None,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["lcdm"] = path
    
    if "pbuf" in model_params:
        output_path = output_dir / "growth_rate_pbuf.png"
        path = create_growth_rate_plot(
            lcdm_params=None,
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
    
    parser = argparse.ArgumentParser(description="Generate growth rate plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--lcdm-params", help="ΛCDM parameters as JSON string")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--no-uncertainty", action="store_true", help="Disable uncertainty bands")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_growth_rate_plot_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} growth rate plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        lcdm_params = json.loads(args.lcdm_params) if args.lcdm_params else None
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_growth_rate_plot(
            lcdm_params=lcdm_params,
            pbuf_params=pbuf_params,
            output_path=args.output,
            show_uncertainty=not args.no_uncertainty,
        )
        if path:
            print(f"Growth rate plot saved to: {path}")
        else:
            print("Growth rate plot displayed (no output path specified)")
