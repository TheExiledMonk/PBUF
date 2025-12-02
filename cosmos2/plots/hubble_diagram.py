"""Hubble Diagram plot implementation for SN data with ΛCDM and PBUF predictions."""

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


def create_hubble_diagram(
    lcdm_params: dict[str, float] | None = None,
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
) -> Path | None:
    """
    Create Hubble Diagram with SN data, ΛCDM and PBUF predictions, and residuals subplot.
    
    Parameters:
    -----------
    lcdm_params : dict[str, float] | None
        Parameter overrides for ΛCDM model (e.g., {"H0": 70.0, "Omega_m": 0.3})
    pbuf_params : dict[str, float] | None
        Parameter overrides for PBUF model (e.g., {"H0": 70.0, "Omega_m": 0.3, "Rmax": 3.0})
    output_path : Path | str | None
        Path to save the figure. If None, returns the figure object.
    figsize : Tuple[float, float]
        Figure size (width, height) in inches.
    dpi : int
        Resolution for saved figure.
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for Hubble diagram plotting")

    # Load SN dataset
    dataset = get_dataset("sn")
    z = np.asarray(dataset["z"], dtype=float)
    mu_obs = np.asarray(dataset["obs"], dtype=float)
    mu_err = dataset.get("err")
    
    if mu_err is not None:
        mu_err = np.asarray(mu_err, dtype=float)

    # Build models with default or provided parameters
    lcdm_params = lcdm_params or {}
    pbuf_params = pbuf_params or {}
    
    lcdm_model = None
    pbuf_model = None
    
    if lcdm_params:
        lcdm_model = build_model("lcdm", lcdm_params)
    
    if pbuf_params:
        pbuf_model = build_model("pbuf", pbuf_params)

    # Calculate model predictions
    mu_lcdm = np.asarray(lcdm_model.distance_modulus(z), dtype=float) if lcdm_model else None
    mu_pbuf = np.asarray(pbuf_model.distance_modulus(z), dtype=float) if pbuf_model else None

    # Calculate residuals
    residuals_lcdm = mu_obs - mu_lcdm if mu_lcdm is not None else None
    residuals_pbuf = mu_obs - mu_pbuf if mu_pbuf is not None else None

    # Create figure with two subplots (main plot and residuals)
    fig, (ax_main, ax_res) = plt.subplots(
        2, 1, 
        sharex=True, 
        figsize=figsize,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
    )

    # Main plot: Hubble diagram
    # Plot SN data
    ax_main.scatter(z, mu_obs, label="SN Pantheon+ data", s=8, color="tab:blue", alpha=0.6, zorder=3)
    
    # Add error bars if available
    if mu_err is not None:
        ax_main.errorbar(z, mu_obs, yerr=mu_err, fmt="none", ecolor="tab:blue", alpha=0.4, zorder=2)

    # Plot model predictions
    # Sort by redshift for smooth curves
    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]

    if mu_lcdm is not None:
        mu_lcdm_sorted = mu_lcdm[sort_idx]
        ax_main.plot(z_sorted, mu_lcdm_sorted, label="ΛCDM prediction", color="tab:orange", linewidth=2, zorder=4)
    
    if mu_pbuf is not None:
        mu_pbuf_sorted = mu_pbuf[sort_idx]
        ax_main.plot(z_sorted, mu_pbuf_sorted, label="PBUF prediction", color="tab:red", linewidth=2, linestyle="--", zorder=4)

    ax_main.set_ylabel("Distance Modulus μ (mag)")
    ax_main.legend(loc="upper left")
    ax_main.grid(True, alpha=0.3, linestyle=":")
    ax_main.set_title("Hubble Diagram: SN Pantheon+ Data with Model Predictions")

    # Residuals subplot
    if residuals_lcdm is not None:
        ax_res.plot(z_sorted, residuals_lcdm[sort_idx], label="ΛCDM residuals", color="tab:orange", linewidth=1.5, alpha=0.8)
    
    if residuals_pbuf is not None:
        ax_res.plot(z_sorted, residuals_pbuf[sort_idx], label="PBUF residuals", color="tab:red", linewidth=1.5, linestyle="--", alpha=0.8)
    
    # Add zero line
    ax_res.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.7)
    
    ax_res.set_xlabel("Redshift z")
    ax_res.set_ylabel("Residuals\n(obs - model) [mag]")
    # Only add legend if there are actual labeled artists
    if residuals_lcdm is not None or residuals_pbuf is not None:
        ax_res.legend(loc="upper left")
    ax_res.grid(True, alpha=0.3, linestyle=":")
    
    # Set reasonable x-axis limits based on data
    z_min, z_max = z.min(), z.max()
    margin = 0.05 * (z_max - z_min)
    ax_res.set_xlim(z_min - margin, z_max + margin)

    # Adjust layout and save
    # Use constrained_layout instead of tight_layout to avoid warnings
    fig.set_constrained_layout(True)
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_hubble_diagram_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create Hubble diagrams from a completed science run.
    
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
        output_path = output_dir / "hubble_diagram_both_models.png"
        path = create_hubble_diagram(
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
        output_path = output_dir / "hubble_diagram_lcdm.png"
        path = create_hubble_diagram(
            lcdm_params=model_params["lcdm"],
            pbuf_params=None,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["lcdm"] = path
    
    if "pbuf" in model_params:
        output_path = output_dir / "hubble_diagram_pbuf.png"
        path = create_hubble_diagram(
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
    
    parser = argparse.ArgumentParser(description="Generate Hubble diagram plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--lcdm-params", help="ΛCDM parameters as JSON string")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_hubble_diagram_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} Hubble diagram plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        lcdm_params = json.loads(args.lcdm_params) if args.lcdm_params else None
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_hubble_diagram(
            lcdm_params=lcdm_params,
            pbuf_params=pbuf_params,
            output_path=args.output,
        )
        if path:
            print(f"Hubble diagram saved to: {path}")
        else:
            print("Hubble diagram displayed (no output path specified)")
