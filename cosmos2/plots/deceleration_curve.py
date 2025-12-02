"""Deceleration Curve q(z) plot implementation for ΛCDM and PBUF models with jackknife uncertainty support."""

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


def _calculate_qz_numerical(model: Any, z_range: np.ndarray, dz: float = 1e-4) -> np.ndarray:
    """
    Calculate deceleration parameter q(z) using numerical differentiation.
    
    The deceleration parameter is defined as:
    q(z) = -(1 + z) * (dH/dz) / H(z) - 1
    
    Parameters:
    -----------
    model : Any
        Cosmological model with Hubble(z) method.
    z_range : np.ndarray
        Array of redshift values.
    dz : float
        Small step size for numerical differentiation.
        
    Returns:
    --------
    np.ndarray
        q(z) values corresponding to input redshifts.
    """
    qz_values = np.zeros_like(z_range)
    
    for i, z in enumerate(z_range):
        # Calculate H(z) and H(z + dz)
        H_z = model.Hubble(z)
        H_z_plus = model.Hubble(min(z + dz, 10.0))  # Cap at reasonable upper bound
        
        # Numerical derivative
        dH_dz = (H_z_plus - H_z) / dz
        
        # Calculate q(z)
        qz_values[i] = -(1 + z) * dH_dz / H_z - 1
    
    return qz_values


def _calculate_jackknife_variation(
    lcdm_params: dict[str, float] | None,
    pbuf_params: dict[str, float] | None,
    z_range: np.ndarray,
    n_samples: int = 100,
    variation_frac: float = 0.05
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Calculate jackknife variation bands for q(z) curves.
    
    Parameters:
    -----------
    lcdm_params : dict[str, float] | None
        ΛCDM model parameters.
    pbuf_params : dict[str, float] | None
        PBUF model parameters.
    z_range : np.ndarray
        Array of redshift values.
    n_samples : int
        Number of jackknife samples.
    variation_frac : float
        Fraction for parameter variations.
        
    Returns:
    --------
    Tuple of (lcdm_lower, lcdm_upper, pbuf_lower, pbuf_upper) arrays or None values.
    """
    lcdm_lower, lcdm_upper, pbuf_lower, pbuf_upper = None, None, None, None
    
    # ΛCDM jackknife variation
    if lcdm_params:
        base_lcdm_model = build_model("lcdm", lcdm_params)
        base_lcdm_qz = _calculate_qz_numerical(base_lcdm_model, z_range)
        
        lcdm_variations = []
        for i in range(n_samples):
            varied_params = {}
            for key, base_value in lcdm_params.items():
                # Add random variation
                variation = np.random.normal(0, variation_frac * base_value)
                if key == "Omega_m0":
                    # Ensure physical bounds
                    varied_params[key] = np.clip(base_value + variation, 0.1, 0.9)
                elif key == "H0":
                    varied_params[key] = np.clip(base_value + variation, 50, 100)
                else:
                    varied_params[key] = max(base_value + variation, 1e-6)
            
            try:
                varied_model = build_model("lcdm", varied_params)
                varied_qz = _calculate_qz_numerical(varied_model, z_range)
                lcdm_variations.append(varied_qz)
            except Exception:
                continue
        
        if lcdm_variations:
            lcdm_variations = np.array(lcdm_variations)
            lcdm_lower = np.percentile(lcdm_variations, 16, axis=0)  # 1-sigma lower bound
            lcdm_upper = np.percentile(lcdm_variations, 84, axis=0)  # 1-sigma upper bound
    
    # PBUF jackknife variation
    if pbuf_params:
        base_pbuf_model = build_model("pbuf", pbuf_params)
        base_pbuf_qz = _calculate_qz_numerical(base_pbuf_model, z_range)
        
        pbuf_variations = []
        for i in range(n_samples):
            varied_params = {}
            for key, base_value in pbuf_params.items():
                # Add random variation
                variation = np.random.normal(0, variation_frac * base_value)
                if key == "Omega_m0":
                    varied_params[key] = np.clip(base_value + variation, 0.1, 0.9)
                elif key == "H0":
                    varied_params[key] = np.clip(base_value + variation, 50, 100)
                elif key == "Rmax":
                    varied_params[key] = max(base_value + variation, 1.0)
                else:
                    varied_params[key] = max(base_value + variation, 1e-6)
            
            try:
                varied_model = build_model("pbuf", varied_params)
                varied_qz = _calculate_qz_numerical(varied_model, z_range)
                pbuf_variations.append(varied_qz)
            except Exception:
                continue
        
        if pbuf_variations:
            pbuf_variations = np.array(pbuf_variations)
            pbuf_lower = np.percentile(pbuf_variations, 16, axis=0)  # 1-sigma lower bound
            pbuf_upper = np.percentile(pbuf_variations, 84, axis=0)  # 1-sigma upper bound
    
    return lcdm_lower, lcdm_upper, pbuf_lower, pbuf_upper


def create_deceleration_curve(
    lcdm_params: dict[str, float] | None = None,
    pbuf_params: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 200,
    z_max: float = 3.0,
    n_points: int = 100,
    show_jackknife: bool = False,
    n_jackknife_samples: int = 50,
    jackknife_alpha: float = 0.3,
) -> Path | None:
    """
    Create Deceleration Curve q(z) plot for ΛCDM and PBUF models with optional jackknife uncertainty.
    
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
        Maximum redshift for the curves.
    n_points : int
        Number of points for the curves.
    show_jackknife : bool
        Whether to show jackknife uncertainty bands.
    n_jackknife_samples : int
        Number of jackknife samples for uncertainty estimation.
    jackknife_alpha : float
        Transparency level for jackknife bands.
        
    Returns:
    --------
    Path | None
        Path to saved figure if output_path provided, otherwise None.
    """
    if plt is None or matplotlib is None:
        raise ImportError("matplotlib is required for deceleration curve plotting")

    # Build models with default or provided parameters
    lcdm_params = lcdm_params or {}
    pbuf_params = pbuf_params or {}
    
    lcdm_model = None
    pbuf_model = None
    
    if lcdm_params:
        lcdm_model = build_model("lcdm", lcdm_params)
    
    if pbuf_params:
        pbuf_model = build_model("pbuf", pbuf_params)

    # Create redshift range
    z_range = np.linspace(0.0, z_max, n_points)
    
    # Calculate q(z) curves
    qz_lcdm = None
    qz_pbuf = None
    
    if lcdm_model:
        qz_lcdm = _calculate_qz_numerical(lcdm_model, z_range)
    
    if pbuf_model:
        qz_pbuf = _calculate_qz_numerical(pbuf_model, z_range)

    # Calculate jackknife variation bands
    lcdm_lower, lcdm_upper, pbuf_lower, pbuf_upper = None, None, None, None
    if show_jackknife:
        lcdm_lower, lcdm_upper, pbuf_lower, pbuf_upper = _calculate_jackknife_variation(
            lcdm_params, pbuf_params, z_range, n_jackknife_samples
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot jackknife variation bands first (so they appear behind the main curves)
    if show_jackknife:
        if lcdm_lower is not None and lcdm_upper is not None:
            ax.fill_between(z_range, lcdm_lower, lcdm_upper, 
                          color="tab:orange", alpha=jackknife_alpha, 
                          label="ΛCDM uncertainty")
        
        if pbuf_lower is not None and pbuf_upper is not None:
            ax.fill_between(z_range, pbuf_lower, pbuf_upper, 
                          color="tab:red", alpha=jackknife_alpha, 
                          label="PBUF uncertainty")

    # Plot q(z) curves
    if qz_lcdm is not None:
        ax.plot(z_range, qz_lcdm, label="ΛCDM q(z)", color="tab:orange", 
               linewidth=2.5, zorder=4)
    
    if qz_pbuf is not None:
        ax.plot(z_range, qz_pbuf, label="PBUF q(z)", color="tab:red", 
               linewidth=2.5, linestyle="--", zorder=4)

    # Add reference line at q=0 (acceleration/deceleration transition)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7, 
              label="q=0 (acceleration transition)")

    # Formatting
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Deceleration parameter q(z)")
    ax.set_title("Deceleration Parameter Evolution: ΛCDM vs PBUF")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle=":")
    
    # Set axis limits
    ax.set_xlim(0, z_max)
    
    # Set reasonable y-axis limits
    all_values = []
    if qz_lcdm is not None:
        all_values.extend(qz_lcdm.tolist())
    if qz_pbuf is not None:
        all_values.extend(qz_pbuf.tolist())
    
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_margin = 0.1 * (y_max - y_min)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Add acceleration/deceleration regions
    ax.axhspan(-10, 0, alpha=0.1, color='green', label='Accelerating (q < 0)')
    ax.axhspan(0, 10, alpha=0.1, color='red', label='Decelerating (q > 0)')

    # Add parameter information
    param_text = ""
    if lcdm_model:
        q0_lcdm = qz_lcdm[0] if qz_lcdm is not None else None
        param_text += f"ΛCDM: H₀={lcdm_params.get('H0', 70):.1f}, Ωₘ={lcdm_params.get('Omega_m0', 0.3):.3f}"
        if q0_lcdm is not None:
            param_text += f"\nq₀={q0_lcdm:.3f}"
    
    if pbuf_model:
        q0_pbuf = qz_pbuf[0] if qz_pbuf is not None else None
        if param_text:
            param_text += "\n"
        param_text += f"PBUF: H₀={pbuf_params.get('H0', 70):.1f}, Ωₘ={pbuf_params.get('Omega_m0', 0.3):.3f}, Rmax={pbuf_params.get('Rmax', 3.0):.1f}"
        if q0_pbuf is not None:
            param_text += f"\nq₀={q0_pbuf:.3f}"
    
    if param_text:
        ax.text(0.98, 0.02, param_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Add cosmic epoch annotations
    epoch_annotations = [
        (0.0, "Present", "black"),
        (0.5, "z=0.5", "gray"),
        (1.0, "z=1", "gray"),
        (2.0, "z=2", "gray"),
    ]
    
    for z_epoch, label, color in epoch_annotations:
        if z_epoch <= z_max:
            ax.axvline(z_epoch, color=color, linestyle=":", alpha=0.3, linewidth=1)
            # Add epoch label at top
            ax.text(z_epoch, ax.get_ylim()[1] * 0.9, label, 
                   ha="center", va="top", fontsize=8, color=color, alpha=0.7)

    fig.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        return fig


def create_deceleration_curve_from_run(
    run_dir: Path,
    output_dir: Path | None = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Create deceleration curve plots from a completed science run.
    
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
        output_path = output_dir / "deceleration_curve_both_models.png"
        path = create_deceleration_curve(
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
        output_path = output_dir / "deceleration_curve_lcdm.png"
        path = create_deceleration_curve(
            lcdm_params=model_params["lcdm"],
            pbuf_params=None,
            output_path=output_path,
            figsize=figsize,
            dpi=dpi,
        )
        if path:
            plot_paths["lcdm"] = path
    
    if "pbuf" in model_params:
        output_path = output_dir / "deceleration_curve_pbuf.png"
        path = create_deceleration_curve(
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
    
    parser = argparse.ArgumentParser(description="Generate deceleration curve plots")
    parser.add_argument("--run-dir", type=Path, help="Science run directory")
    parser.add_argument("--output-dir", type=Path, help="Output directory for plots")
    parser.add_argument("--lcdm-params", help="ΛCDM parameters as JSON string")
    parser.add_argument("--pbuf-params", help="PBUF parameters as JSON string")
    parser.add_argument("--output", type=Path, help="Output file path")
    parser.add_argument("--jackknife", action="store_true", help="Enable jackknife uncertainty bands")
    
    args = parser.parse_args()
    
    if args.run_dir:
        # Generate plots from science run
        plots = create_deceleration_curve_from_run(args.run_dir, args.output_dir)
        print(f"Generated {len(plots)} deceleration curve plots:")
        for name, path in plots.items():
            print(f"  {name}: {path}")
    else:
        # Generate single plot with provided parameters
        import json
        
        lcdm_params = json.loads(args.lcdm_params) if args.lcdm_params else None
        pbuf_params = json.loads(args.pbuf_params) if args.pbuf_params else None
        
        path = create_deceleration_curve(
            lcdm_params=lcdm_params,
            pbuf_params=pbuf_params,
            output_path=args.output,
            show_jackknife=args.jackknife,
        )
        if path:
            print(f"Deceleration curve plot saved to: {path}")
        else:
            print("Deceleration curve plot displayed (no output path specified)")
