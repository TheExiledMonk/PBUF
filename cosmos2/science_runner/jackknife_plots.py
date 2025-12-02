"""Jackknife plotting utilities for cosmos2 science runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_chi2_stability(
    jackknife_results: List[Dict[str, Any]], 
    output_path: Path
) -> None:
    """Plot χ² vs draw index."""
    draw_indices = []
    chi2_values = []
    
    for result in jackknife_results:
        if result.get("success", True):
            draw_idx = result.get("draw_index", 0)
            for model_summary in result.get("model_summaries", {}).values():
                draw_indices.append(draw_idx)
                chi2_values.append(model_summary.get("best_chi2", np.nan))
    
    if not chi2_values:
        print("[cosmos2] No χ² data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(draw_indices, chi2_values, alpha=0.7, s=50)
    plt.xlabel("Jackknife Draw Index")
    plt.ylabel("χ²")
    plt.title("χ² Stability Across Jackknife Draws")
    plt.grid(True, alpha=0.3)
    
    # Add reference line at mean
    mean_chi2 = np.nanmean(chi2_values)
    plt.axhline(y=mean_chi2, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_chi2:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[cosmos2] Saved χ² stability plot to {output_path}")


def plot_parameter_stability(
    jackknife_results: List[Dict[str, Any]], 
    parameter_name: str,
    output_path: Path
) -> None:
    """Plot parameter stability across jackknife draws."""
    draw_indices = []
    param_values = []
    
    for result in jackknife_results:
        if result.get("success", True):
            draw_idx = result.get("draw_index", 0)
            for model_summary in result.get("model_summaries", {}).values():
                # Check in best_params first
                if parameter_name in model_summary.get("best_params", {}):
                    param_values.append(model_summary["best_params"][parameter_name])
                # Then check in derived_quantities
                elif parameter_name in model_summary.get("derived_quantities", {}):
                    param_values.append(model_summary["derived_quantities"][parameter_name])
                else:
                    param_values.append(np.nan)
                draw_indices.append(draw_idx)
    
    if not param_values or all(np.isnan(param_values)):
        print(f"[cosmos2] No {parameter_name} data to plot")
        return
    
    # Filter out NaN values
    valid_data = [(idx, val) for idx, val in zip(draw_indices, param_values) if not np.isnan(val)]
    if not valid_data:
        print(f"[cosmos2] No valid {parameter_name} data to plot")
        return
    
    valid_indices, valid_values = zip(*valid_data)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_indices, valid_values, alpha=0.7, s=50)
    plt.xlabel("Jackknife Draw Index")
    plt.ylabel(parameter_name)
    plt.title(f"{parameter_name} Stability Across Jackknife Draws")
    plt.grid(True, alpha=0.3)
    
    # Add reference line at mean
    mean_val = np.mean(valid_values)
    std_val = np.std(valid_values)
    plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_val:.3f} ± {std_val:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[cosmos2] Saved {parameter_name} stability plot to {output_path}")


def plot_qz_evolution(
    jackknife_results: List[Dict[str, Any]], 
    output_path: Path
) -> None:
    """Plot q(z) evolution with jackknife uncertainty bands."""
    # Collect all prediction curves
    z_curves = []
    qz_curves = []
    
    for result in jackknife_results:
        if result.get("success", True):
            for model_summary in result.get("model_summaries", {}).values():
                predictions = model_summary.get("predictions", {})
                if "plot_data" in predictions:
                    plot_data = predictions["plot_data"]
                    if "z" in plot_data and "q_z" in plot_data:
                        z_curves.append(np.array(plot_data["z"]))
                        qz_curves.append(np.array(plot_data["q_z"]))
    
    if not qz_curves:
        print("[cosmos2] No q(z) data to plot")
        return
    
    # Ensure all z arrays are the same
    z_ref = z_curves[0]
    for i, z_curve in enumerate(z_curves):
        if not np.allclose(z_curve, z_ref):
            print(f"[cosmos2] Skipping curve {i} due to inconsistent z values")
            continue
    
    # Stack all q(z) curves
    qz_array = np.array([qz for qz in qz_curves if len(qz) == len(z_ref)])
    
    if qz_array.shape[0] == 0:
        print("[cosmos2] No valid q(z) curves to plot")
        return
    
    # Calculate statistics
    qz_mean = np.mean(qz_array, axis=0)
    qz_std = np.std(qz_array, axis=0)
    
    plt.figure(figsize=(10, 6))
    
    # Plot uncertainty band
    plt.fill_between(z_ref, qz_mean - qz_std, qz_mean + qz_std, 
                     alpha=0.3, color='blue', label='Jackknife ±1σ')
    
    # Plot mean curve
    plt.plot(z_ref, qz_mean, 'b-', linewidth=2, label='Mean q(z)')
    
    # Plot individual curves faintly
    for qz_curve in qz_curves[:10]:  # Limit to avoid overcrowding
        if len(qz_curve) == len(z_ref):
            plt.plot(z_ref, qz_curve, 'gray', alpha=0.1, linewidth=0.5)
    
    plt.xlabel("Redshift z")
    plt.ylabel("Deceleration Parameter q(z)")
    plt.title("q(z) Evolution with Jackknife Uncertainty")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[cosmos2] Saved q(z) evolution plot to {output_path}")


def generate_all_jackknife_figures(
    jackknife_results: List[Dict[str, Any]], 
    output_dir: Path
) -> None:
    """Generate all jackknife figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # χ² stability
    plot_chi2_stability(jackknife_results, output_dir / "jackknife_chi2_stability.png")
    
    # Parameter stability plots
    key_parameters = ["H0", "Omega_m0", "r_d", "S8"]
    for param in key_parameters:
        plot_parameter_stability(jackknife_results, param, 
                                output_dir / f"jackknife_{param.lower()}_stability.png")
    
    # q(z) evolution
    plot_qz_evolution(jackknife_results, output_dir / "jackknife_qz_evolution.png")


def generate_jackknife_figures_from_files(
    jackknife_results_path: Path,
    output_dir: Path
) -> None:
    """Generate jackknife figures from saved results file."""
    with open(jackknife_results_path, 'r') as f:
        jackknife_results = json.load(f)
    
    generate_all_jackknife_figures(jackknife_results, output_dir)
