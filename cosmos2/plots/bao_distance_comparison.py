"""BAO Distance Comparison plot implementation for BAO data with ΛCDM and PBUF predictions."""

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


def _extract_bao_data() -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract and organize BAO data from both anisotropic and isotropic datasets."""
    
    # Load anisotropic BAO data (D_M/rd and D_H/rd)
    bao_aniso = get_dataset("bao_aniso")
    z_aniso = np.asarray(bao_aniso["z"], dtype=float)
    obs_aniso = np.asarray(bao_aniso["obs"], dtype=float)
    labels_aniso = np.asarray(bao_aniso["labels"], dtype=object)
    cov_aniso = np.asarray(bao_aniso.get("cov"), dtype=float)
    
    # Extract D_M/rd and D_H/rd measurements
    dm_data = {"z": [], "dm_over_rd": [], "err": []}
    dh_data = {"z": [], "dh_over_rd": [], "err": []}
    
    # Calculate stride for mapping observations to redshifts
    stride = 1
    if z_aniso.size and labels_aniso.size and labels_aniso.size % z_aniso.size == 0:
        stride = max(1, labels_aniso.size // z_aniso.size)
    
    for i, label in enumerate(labels_aniso):
        z_idx = min(i // stride, z_aniso.size - 1) if z_aniso.size else 0
        z_bin = z_aniso[z_idx]
        obs_val = obs_aniso[i]
        
        # Extract error from covariance matrix (diagonal)
        err = np.sqrt(cov_aniso[i, i]) if cov_aniso.size > 0 else None
        
        label_str = str(label).lower()
        if "dm" in label_str or "d_m" in label_str:
            dm_data["z"].append(z_bin)
            dm_data["dm_over_rd"].append(obs_val)
            dm_data["err"].append(err)
        elif "dh" in label_str or "d_h" in label_str or "htimes" in label_str:
            dh_data["z"].append(z_bin)
            dh_data["dh_over_rd"].append(obs_val)
            dh_data["err"].append(err)
    
    # Convert to arrays
    for data_dict in [dm_data, dh_data]:
        for key in ["z", "dm_over_rd", "dh_over_rd", "err"]:
            if key in data_dict and data_dict[key]:
                data_dict[key] = np.asarray(data_dict[key], dtype=float)
    
    # Load isotropic BAO data (D_V/rd)
    bao_iso = get_dataset("bao_iso")
    z_iso = np.asarray(bao_iso["z"], dtype=float)
    obs_iso = np.asarray(bao_iso["obs"], dtype=float)
    cov_iso = np.asarray(bao_iso.get("cov"), dtype=float)
    
    dv_data = {"z": z_iso, "dv_over_rd": obs_iso, "err": []}
    if cov_iso.size > 0:
        dv_data["err"] = np.sqrt(np.diag(cov_iso))
    else:
        dv_data["err"] = np.full_like(obs_iso, np.nan)
    
    return dm_data, dh_data, dv_data


def _calculate_model_distances(model: Any, z_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculate model distance predictions for a given redshift range."""
    rd = float(model.sound_horizon())
    
    DM = np.asarray([model.DM(z) for z in z_range], dtype=float)
    DH = np.asarray([299_792.458 / model.Hubble(z) for z in z_range], dtype=float)  # c/H(z)
    DV = np.asarray([model.DV(z) for z in z_range], dtype=float)
    
    DM_over_rd = DM / rd
    DH_over_rd = DH / rd
    DV_over_rd = DV / rd
    
    return DM_over_rd, DH_over_rd, DV_over_rd, rd


def create_bao_distance_comparison(
    lcdm_params: dict[str, float] | None = None,
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
    z_max: float = 2.5,
    n_points: int = 100,
) -> Path | None:
    """
    Create BAO Distance Comparison plot with D_M/r_d, D_H/r_d, D_V/r_d data and model curves.
    
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
        raise ImportError("matplotlib is required for BAO distance comparison plotting")

    # Extract BAO data
    dm_data, dh_data, dv_data = _extract_bao_data()
    
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
    lcdm_predictions = None
    pbuf_predictions = None
    
    if lcdm_model:
        lcdm_predictions = _calculate_model_distances(lcdm_model, z_model)
    
    if pbuf_model:
        pbuf_predictions = _calculate_model_distances(pbuf_model, z_model)

    # Create figure with three subplots
    fig, (ax_dm, ax_dh, ax_dv) = plt.subplots(
        1, 3, 
        figsize=figsize,
        sharex=True
    )

    # Plot D_M/r_d
    if dm_data["z"].size > 0:
        ax_dm.errorbar(dm_data["z"], dm_data["dm_over_rd"], yerr=dm_data["err"], 
                      fmt="o", markersize=6, capsize=3, color="tab:blue", 
                      label="BAO data", alpha=0.8, zorder=3)
    
    if lcdm_predictions:
        ax_dm.plot(z_model, lcdm_predictions[0], label="ΛCDM", color="tab:orange", 
                  linewidth=2, zorder=4)
    
    if pbuf_predictions:
        ax_dm.plot(z_model, pbuf_predictions[0], label="PBUF", color="tab:red", 
                  linewidth=2, linestyle="--", zorder=4)
    
    ax_dm.set_ylabel(r"$D_M / r_d$")
    ax_dm.set_title("Transverse Distance")
    ax_dm.grid(True, alpha=0.3, linestyle=":")
    ax_dm.legend(loc="upper left")

    # Plot D_H/r_d
    if dh_data["z"].size > 0:
        ax_dh.errorbar(dh_data["z"], dh_data["dh_over_rd"], yerr=dh_data["err"], 
                      fmt="s", markersize=6, capsize=3, color="tab:blue", 
                      label="BAO data", alpha=0.8, zorder=3)
    
    if lcdm_predictions:
        ax_dh.plot(z_model, lcdm_predictions[1], label="ΛCDM", color="tab:orange", 
                  linewidth=2, zorder=4)
    
    if pbuf_predictions:
        ax_dh.plot(z_model, pbuf_predictions[1], label="PBUF", color="tab:red", 
                  linewidth=2, linestyle="--", zorder=4)
    
    ax_dh.set_ylabel(r"$D_H / r_d$")
    ax_dh.set_title("Hubble Distance")
    ax_dh.grid(True, alpha=0.3, linestyle=":")
    ax_dh.legend(loc="upper left")

    # Plot D_V/r_d
    if dv_data["z"].size > 0:
        ax_dv.errorbar(dv_data["z"], dv_data["dv_over_rd"], yerr=dv_data["err"], 
                      fmt="^", markersize=6, capsize=3, color="tab:blue", 
                      label="BAO data", alpha=0.8, zorder=3)
    
    if lcdm_predictions:
        ax_dv.plot(z_model, lcdm_predictions[2], label="ΛCDM", color="tab:orange", 
                  linewidth=2, zorder=4)
    
    if pbuf_predictions:
        ax_dv.plot(z_model, pbuf_predictions[2], label="PBUF", color="tab:red", 
                  linewidth=2, linestyle="--", zorder=4)
    
    ax_dv.set_ylabel(r"$D_V / r_d$")
    ax_dv.set_title("Volume-averaged Distance")
    ax_dv.grid(True, alpha=0.3, linestyle=":")
    ax_dv.legend(loc="upper left")

    # Set common x-axis label
    for ax in [ax_dm, ax_dh, ax_dv]:
        ax.set_xlabel("Redshift z")
        ax.set_xlim(0, z_max)

    # Add sound horizon information
    rd_text = ""
    if lcdm_predictions:
        rd_text += f"ΛCDM r_d = {lcdm_predictions[3]:.1f} Mpc"
    if pbuf_predictions:
        if rd_text:
            rd_text += "\n"
        rd_text += f"PBUF r_d = {pbuf_predictions[3]:.1f} Mpc"
    
    if rd_text:
        fig.text(0.02, 0.98, rd_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("BAO Distance Comparison", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_bao_distance_comparison_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create BAO distance comparison plots from a completed science run.
    
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
        output_path = output_dir / "bao_distance_comparison_both_models.png"
        path = create_bao_distance_comparison(
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
        output_path = output_dir / "bao_distance_comparison_lcdm.png"
        path = create_bao_distance_comparison(
            lcdm_params=model_params["lcdm"],
            pbuf_params=None,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["lcdm"] = path
    
    if "pbuf" in model_params:
        output_path = output_dir / "bao_distance_comparison_pbuf.png"
        path = create_bao_distance_comparison(
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
    
    parser = argparse.ArgumentParser(description="Generate BAO distance comparison plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--lcdm-params", help="ΛCDM parameters as JSON string")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_bao_distance_comparison_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} BAO distance comparison plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        lcdm_params = json.loads(args.lcdm_params) if args.lcdm_params else None
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_bao_distance_comparison(
            lcdm_params=lcdm_params,
            pbuf_params=pbuf_params,
            output_path=args.output,
        )
        if path:
            print(f"BAO distance comparison saved to: {path}")
        else:
            print("BAO distance comparison displayed (no output path specified)")
