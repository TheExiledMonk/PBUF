"""SN Pantheon fit for cosmos2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def run_sn_pantheon_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("sn")
    z = np.asarray(dataset["z"], dtype=float)
    mu_obs = np.asarray(dataset["obs"], dtype=float)
    cov = dataset.get("cov")
    err = dataset.get("err")
    if cov is not None:
        cov_full = np.asarray(cov, dtype=float)
    elif err is not None:
        cov_full = np.diag(np.asarray(err, dtype=float) ** 2)
    else:
        raise ValueError("SN dataset lacks covariance/errors")
    inv_cov = np.linalg.inv(cov_full)

    mu_model = np.asarray(model.distance_modulus(z), dtype=float)
    residuals = mu_model - mu_obs
    chi2 = float(residuals.T @ inv_cov @ residuals)

    extras = build_fit_extras(dataset=dataset, predictions=mu_model, observed=mu_obs, residuals=residuals)
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_sn_pantheon_fit(model, dataset)


__all__ = ["run_fit", "run_sn_pantheon_fit"]
