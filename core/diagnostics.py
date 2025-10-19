"""
Diagnostics helpers for cosmology fits.

These utilities centralise statistical checks so that pipelines can
reuse them when composing JSON outputs or generating reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from utils.plotting import plot_correlation_matrix

try:
    from scipy.stats import kstest, skew, kurtosis
except ImportError:  # pragma: no cover
    def kstest(values: np.ndarray, distribution: str):
        return np.nan, np.nan

    def skew(values: np.ndarray, bias: bool = False):
        return float("nan")

    def kurtosis(values: np.ndarray, fisher: bool = False, bias: bool = False):
        return float("nan")


def compute_residuals(y_obs: np.ndarray, y_model: np.ndarray) -> np.ndarray:
    """Return simple residuals y_obs - y_model."""

    return y_obs - y_model


def compute_pulls(residuals: np.ndarray, sigma: np.ndarray | None) -> np.ndarray:
    """Return pulls; zeros if uncertainties are missing."""

    if sigma is None:
        return np.zeros_like(residuals)
    return residuals / sigma


def normality_checks(pulls: np.ndarray) -> Dict[str, float]:
    """Perform K–S, skewness, and kurtosis tests on the pulls."""

    if pulls.size == 0:
        return {"ks_p": float("nan"), "skew": float("nan"), "kurtosis": float("nan")}
    ks_result = kstest(pulls, "norm")
    return {
        "ks_p": float(ks_result[1]),
        "skew": float(skew(pulls, bias=False)),
        "kurtosis": float(kurtosis(pulls, fisher=False, bias=False)),
    }


def covariance_health(cov: np.ndarray | None) -> Dict[str, object]:
    """Report simple health indicators for the covariance matrix."""

    if cov is None:
        return {"cond_num": float("nan"), "param_corrs": []}
    cond = np.linalg.cond(cov)
    return {"cond_num": float(cond), "param_corrs": []}


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Normalise a covariance matrix to its correlation counterpart."""

    diag = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_diag = np.where(diag > 0.0, 1.0 / diag, 0.0)
    corr = cov * inv_diag[:, None] * inv_diag[None, :]
    np.fill_diagonal(corr, 1.0)
    return corr


def save_correlation_heatmap(
    cov: np.ndarray,
    out_dir: str | Path,
    model_name: str,
    max_points: int = 50,
) -> str:
    """Render a heatmap of the covariance-derived correlations."""

    corr = covariance_to_correlation(cov)
    if corr.shape[0] > max_points:
        corr = corr[:max_points, :max_points]
    return plot_correlation_matrix(corr, model_name, out_dir)
