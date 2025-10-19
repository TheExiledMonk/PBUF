"""
Plotting utilities for diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def _prepare_out_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_residuals(z: np.ndarray, residuals: np.ndarray, model_name: str, out_dir: str | Path) -> str:
    """Scatter plot of residuals versus redshift."""

    out_path = _prepare_out_dir(out_dir) / f"{model_name}_residuals_vs_z.png"
    plt.figure(figsize=(6, 4))
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.scatter(z, residuals, color="#1f77b4")
    plt.xlabel("Redshift z")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} Residuals vs Redshift")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path.resolve())


def plot_pull_distribution(pulls: np.ndarray, model_name: str, out_dir: str | Path) -> str:
    """Histogram of pulls."""

    out_path = _prepare_out_dir(out_dir) / f"{model_name}_pull_distribution.png"
    plt.figure(figsize=(6, 4))
    plt.hist(pulls, bins=10, color="#ff6f00", edgecolor="black", alpha=0.8)
    plt.xlabel("Pull")
    plt.ylabel("Count")
    plt.title(f"{model_name} Pull Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path.resolve())


def plot_chi2_surface(
    x: np.ndarray,
    y: np.ndarray,
    chi2_grid: np.ndarray,
    x_name: str,
    y_name: str,
    model_name: str,
    out_dir: str | Path,
    with_confidence: bool = True,
) -> str:
    """Contour plot of a χ² surface."""

    out_path = _prepare_out_dir(out_dir) / f"{model_name}_chi2_surface_{x_name}_{y_name}.png"
    levels = None
    if with_confidence:
        levels = np.min(chi2_grid) + np.array([2.30, 6.17, 11.8])
    plt.figure(figsize=(6, 4))
    contour = plt.contourf(x, y, chi2_grid, levels=30, cmap="viridis")
    plt.colorbar(contour, label="Δχ²")
    if levels is not None:
        plt.contour(x, y, chi2_grid, levels=levels, colors=["white", "orange", "red"], linewidths=1.2)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f"{model_name} χ² Surface")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path.resolve())


def plot_correlation_matrix(correlation: np.ndarray, model_name: str, out_dir: str | Path) -> str:
    """Heatmap of a correlation matrix."""

    out_path = _prepare_out_dir(out_dir) / f"{model_name}_correlations.png"
    plt.figure(figsize=(5, 4))
    im = plt.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm", origin="lower")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.title(f"{model_name} Correlation Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path.resolve())
