"""Weak lensing S8 prior fit for cosmos2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def run_wl_s8_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("wl_s8")
    S8_obs = np.asarray(dataset["S8_obs"] if "S8_obs" in dataset else dataset.get("obs"), dtype=float)
    cov = dataset.get("cov")
    err = dataset.get("S8_err") if "S8_err" in dataset else dataset.get("err")
    if cov is not None:
        cov_full = np.asarray(cov, dtype=float)
    elif err is not None:
        cov_full = np.diag(np.asarray(err, dtype=float) ** 2)
    else:
        raise ValueError("WL S8 dataset lacks covariance/errors")
    inv_cov = np.linalg.inv(cov_full)

    S8_model = np.array([model.S8()], dtype=float)
    residuals = S8_model - S8_obs
    chi2 = float(residuals.T @ inv_cov @ residuals)
    extras = build_fit_extras(dataset=dataset, predictions=S8_model, observed=S8_obs, residuals=residuals)
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_wl_s8_fit(model, dataset)


__all__ = ["run_fit", "run_wl_s8_fit"]
