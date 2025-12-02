"""H(z) Comparison plot implementation for CC data with ΛCDM and PBUF predictions."""

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


def _extract_cc_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract CC data from the dataset."""
    dataset = get_dataset("cc")
    z = np.asarray(dataset["z"], dtype=float)
    h_obs = np.asarray(dataset["obs"], dtype=float)  # H(z) in km/s/Mpc
    
    # Extract errors from covariance matrix
    cov = np.asarray(dataset.get("cov"), dtype=float)
    if cov.size > 0:
        h_err = np.sqrt(np.diag(cov))
    else:
        # Fallback to err field if available
        err = dataset.get("err")
        if err is not None:
            h_err = np.asarray(err, dtype=float)
        else:
            raise ValueError("CC dataset lacks covariance and error information")
    
    return z, h_obs, h_err


def _calculate_model_hz(model: Any, z_range: np.ndarray) -> np.ndarray:
    """Calculate model H(z) predictions for a given redshift range."""
    return np.asarray([model.Hubble(z) for z in z_range], dtype=float)


def create_hz_comparison(
    lcdm_params: dict[str, float] | None = None,
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
    z_max: float = 2.5,
    n_points: int = 100,
) -> Path | None:
    """
    Create H(z) Comparison plot with CC data, ΛCDM and PBUF predictions, and residuals subplot.
    
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
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for H(z) comparison plotting")

    # Extract CC data
    z_cc, h_obs, h_err = _extract_cc_data()
    
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
    h_lcdm = None
    h_pbuf = None
    
    if lcdm_model:
        h_lcdm = _calculate_model_hz(lcdm_model, z_model)
    
    if pbuf_model:
        h_pbuf = _calculate_model_hz(pbuf_model, z_model)

    # Calculate residuals
    residuals_lcdm = None
    residuals_pbuf = None
    
    if lcdm_model:
        h_lcdm_at_cc = _calculate_model_hz(lcdm_model, z_cc)
        residuals_lcdm = h_obs - h_lcdm_at_cc
    
    if pbuf_model:
        h_pbuf_at_cc = _calculate_model_hz(pbuf_model, z_cc)
        residuals_pbuf = h_obs - h_pbuf_at_cc

    # Create figure with two subplots (main plot and residuals)
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, 
        sharex=True, 
        figsize=figsize,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    # Main plot: H(z) comparison
    # Plot CC data
    ax_main.errorbar(z_cc, h_obs, yerr=h_err, fmt="o", markersize=6, capsize=3, 
                    color="tab:blue", label="Cosmic Chronometers data", 
                    alpha=0.8, zorder=3)

    # Plot model predictions
    if h_lcdm is not None:
        ax_main.plot(z_model, h_lcdm, label="ΛCDM prediction", color="tab:orange", 
                    linewidth=2, zorder=4)
    
    if h_pbuf is not None:
        ax_main.plot(z_model, h_pbuf, label="PBUF prediction", color="tab:red", 
                    linewidth=2, linestyle="--", zorder=4)

    ax_main.set_ylabel("H(z) [km/s/Mpc]")
    ax_main.legend(loc="upper left")
    ax_main.grid(True, alpha=0.3, linestyle=":")
    ax_main.set_title("Hubble Parameter H(z): Cosmic Chronometers Data with Model Predictions")

    # Residuals subplot
    if residuals_lcdm is not None:
        ax_res.plot(z_cc, residuals_lcdm, label="ΛCDM residuals", 
                   color="tab:orange", linewidth=1.5, alpha=0.8)
    
    if residuals_pbuf is not None:
        ax_res.plot(z_cc, residuals_pbuf, label="PBUF residuals", 
                   color="tab:red", linewidth=1.5, linestyle="--", alpha=0.8)
    
    # Add zero line
    ax_res.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.7)
    
    ax_res.set_xlabel("Redshift z")
    ax_res.set_ylabel("Residuals\n(obs - model) [km/s/Mpc]")
    ax_res.legend(loc="upper left")
    ax_res.grid(True, alpha=0.3, linestyle=":")
    
    # Set reasonable x-axis limits based on data
    z_min, z_max_data = z_cc.min(), z_cc.max()
    margin = 0.05 * (z_max_data - z_min)
    ax_res.set_xlim(z_min - margin, max(z_max, z_max_data) + margin)

    # Add data summary information
    n_points = len(z_cc)
    z_range_text = f"CC data: {n_points} points, z ∈ [{z_cc.min():.2f}, {z_cc.max():.2f}]"
    fig.text(0.02, 0.98, z_range_text, transform=fig.transFigure, 
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


def create_hz_comparison_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create H(z) comparison plots from a completed science run.
    
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
        output_path = output_dir / "hz_comparison_both_models.png"
        path = create_hz_comparison(
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
        output_path = output_dir / "hz_comparison_lcdm.png"
        path = create_hz_comparison(
            lcdm_params=model_params["lcdm"],
            pbuf_params=None,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["lcdm"] = path
    
    if "pbuf" in model_params:
        output_path = output_dir / "hz_comparison_pbuf.png"
        path = create_hz_comparison(
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
    
    parser = argparse.ArgumentParser(description="Generate H(z) comparison plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--lcdm-params", help="ΛCDM parameters as JSON string")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_hz_comparison_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} H(z) comparison plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        lcdm_params = json.loads(args.lcdm_params) if args.lcdm_params else None
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_hz_comparison(
            lcdm_params=lcdm_params,
            pbuf_params=pbuf_params,
            output_path=args.output,
        )
        if path:
            print(f"H(z) comparison saved to: {path}")
        else:
            print("H(z) comparison displayed (no output path specified)")
