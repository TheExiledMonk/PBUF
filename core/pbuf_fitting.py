"""
Fitting utilities for PBUF and ΛCDM models.

The functions in this module keep optimisation logic decoupled from the
pipeline scripts so that physics code remains in `core/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
    from scipy.stats import chi2 as chi2_dist
    from scipy.stats import kstest, skew, kurtosis
except ImportError:  # pragma: no cover - SciPy is optional at runtime.
    minimize = None
    chi2_dist = None

    def kstest(data: np.ndarray, cdf: str):
        return np.nan, np.nan

    def skew(data: np.ndarray):
        return float("nan")

    def kurtosis(data: np.ndarray):
        return float("nan")


@dataclass
class FitSettings:
    """Container for optimisation settings."""

    tolerance: float = 1.0e-6
    max_iter: int = 5_000
    bounds: Mapping[str, Tuple[float, float]] = None  # type: ignore[assignment]


def _vector_to_params(names: Iterable[str], values: np.ndarray) -> Dict[str, float]:
    return {name: float(val) for name, val in zip(names, values)}


def _compute_chi2(residuals: np.ndarray, sigma: np.ndarray | None, cov: np.ndarray | None) -> float:
    if cov is not None:
        inv = np.linalg.pinv(cov)
        return float(residuals.T @ inv @ residuals)
    if sigma is not None:
        scaled = residuals / sigma
        return float(np.dot(scaled, scaled))
    return float(np.dot(residuals, residuals))


def _chi2_p_value(chi2_value: float, dof: int) -> float:
    if dof <= 0:
        return float("nan")
    if chi2_dist is not None:
        return float(chi2_dist.sf(chi2_value, dof))
    # Wilson–Hilferty approximation fallback.
    from math import sqrt, erfc

    mean = dof
    variance = 2.0 * dof
    z = (chi2_value - mean) / sqrt(variance)
    return 0.5 * erfc(z / np.sqrt(2.0))


def fit_model(
    model_module: object,
    data: Mapping[str, np.ndarray],
    initial_params: Mapping[str, float],
    settings: FitSettings,
) -> Dict[str, object]:
    """
    Run a χ² minimisation against the provided dataset.

    Parameters
    ----------
    model_module : module-like
        Object exposing `mu(z, params)` returning the predicted observable.
    data : mapping
        Output from `dataio.loaders`, must contain at least `z` and `y`.
    initial_params : mapping
        Starting point for the optimiser.
    settings : FitSettings
        Control tolerances and bounds.
    """

    param_names = tuple(initial_params.keys())
    x0 = np.array([initial_params[name] for name in param_names], dtype=float)
    z = np.asarray(data["z"], dtype=float)
    y_obs = np.asarray(data["y"], dtype=float)
    sigma = np.asarray(data.get("sigma"), dtype=float) if data.get("sigma") is not None else None
    cov = np.asarray(data.get("cov"), dtype=float) if data.get("cov") is not None else None

    bounds = None
    if settings.bounds:
        bounds = [settings.bounds.get(name, (None, None)) for name in param_names]

    def objective(theta: np.ndarray) -> float:
        params = _vector_to_params(param_names, theta)
        model_y = model_module.mu(z, params)
        residuals = y_obs - model_y
        return _compute_chi2(residuals, sigma, cov)

    optimisation = None
    if minimize is None:
        best_params = x0
        chi2_value = objective(best_params)
    else:
        optimisation = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": settings.tolerance, "maxiter": settings.max_iter},
        )
        best_params = optimisation.x
        chi2_value = optimisation.fun

    params_dict = _vector_to_params(param_names, best_params)
    y_model = model_module.mu(z, params_dict)
    residuals = y_obs - y_model
    pulls = residuals / sigma if sigma is not None else np.zeros_like(residuals)

    n = len(z)
    k = len(param_names)
    dof = max(n - k, 1)

    fit_cov_cond = float("nan")
    if optimisation is not None and hasattr(optimisation, "hess_inv"):
        try:
            hess_inv = optimisation.hess_inv.todense() if hasattr(optimisation.hess_inv, "todense") else optimisation.hess_inv
            hess_matrix = np.asarray(hess_inv, dtype=float)
            fit_cov_cond = float(np.linalg.cond(hess_matrix))
        except Exception:  # pragma: no cover - safeguard for unsupported hess types
            fit_cov_cond = float("nan")

    metrics = {
        "N": n,
        "N_params": k,
        "dof": dof,
        "chi2": float(chi2_value),
        "chi2_dof": float(chi2_value) / dof,
        "AIC": float(chi2_value) + 2.0 * k,
        "BIC": float(chi2_value) + k * np.log(max(n, 1)),
        "p_value": _chi2_p_value(float(chi2_value), dof),
        "normality": {
            "ks_p": float(kstest(pulls, "norm")[1]) if len(pulls) > 0 else float("nan"),
            "skew": float(skew(pulls, bias=False)) if len(pulls) > 0 else float("nan"),
            "kurtosis": float(kurtosis(pulls, fisher=False, bias=False)) if len(pulls) > 0 else float("nan"),
        },
        "cov_health": {
            "cond_num": float(np.linalg.cond(cov)) if cov is not None else float("nan"),
            "param_corrs": [],
        },
        "fit_covariance": {
            "cond_num": fit_cov_cond,
        },
    }

    return {
        "bestfit": params_dict,
        "model_y": y_model,
        "residuals": residuals,
        "pulls": pulls,
        "metrics": metrics,
    }
