"""
Generate two-dimensional χ² grids for visual diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from utils.plotting import plot_chi2_surface


def _compute_chi2(residuals: np.ndarray, cov: np.ndarray | None, sigma: np.ndarray | None) -> float:
    if cov is not None:
        inv = np.linalg.pinv(cov)
        return float(residuals.T @ inv @ residuals)
    if sigma is not None:
        scaled = residuals / sigma
        return float(np.dot(scaled, scaled))
    return float(np.dot(residuals, residuals))


def chi2_surface_scan(
    model_func: Callable[[np.ndarray, Mapping[str, float]], np.ndarray],
    z: np.ndarray,
    y: np.ndarray,
    cov: np.ndarray | None,
    param_x: str,
    param_y: str,
    grid_x: Sequence[float],
    grid_y: Sequence[float],
    base_params: Mapping[str, float],
    out_dir: str | Path,
    model_name: str,
) -> Tuple[str, str]:
    """
    Evaluate a χ² grid over two parameters and save artefacts.

    Returns
    -------
    tuple
        (absolute path to PNG, absolute path to NPY grid)
    """

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_x = np.asarray(list(grid_x), dtype=float)
    grid_y = np.asarray(list(grid_y), dtype=float)
    chi2_grid = np.zeros((grid_y.size, grid_x.size), dtype=float)

    sigma = None
    if cov is None:
        sigma = None  # placeholder; pipelines can provide separately if needed

    for i, y_val in enumerate(grid_y):
        for j, x_val in enumerate(grid_x):
            params: Dict[str, float] = dict(base_params)
            params[param_x] = x_val
            params[param_y] = y_val
            model_y = model_func(z, params)
            residuals = y - model_y
            chi2_grid[i, j] = _compute_chi2(residuals, cov, sigma)

    png_path = plot_chi2_surface(
        grid_x,
        grid_y,
        chi2_grid,
        param_x,
        param_y,
        model_name,
        out_dir,
    )

    npy_path = out_dir / f"{model_name}_chi2_surface_{param_x}_{param_y}.npy"
    np.save(npy_path, chi2_grid)

    return png_path, str(npy_path.resolve())
