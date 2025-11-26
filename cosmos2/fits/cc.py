"""Cosmic chronometers fit for cosmos2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def run_cc_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("cc")
    z = np.asarray(dataset["z"], dtype=float)
    obs = np.asarray(dataset["obs"], dtype=float)
    cov = dataset.get("cov")
    err = dataset.get("err")
    if cov is not None:
        cov_full = np.asarray(cov, dtype=float)
    elif err is not None:
        cov_full = np.diag(np.asarray(err, dtype=float) ** 2)
    else:
        raise ValueError("CC dataset lacks covariance/errors")
    inv_cov = np.linalg.inv(cov_full)

    preds = np.asarray(model.Hubble(z), dtype=float)
    residuals = preds - obs
    chi2 = float(residuals.T @ inv_cov @ residuals)
    extras = build_fit_extras(dataset=dataset, predictions=preds, observed=obs, residuals=residuals)
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_cc_fit(model, dataset)


__all__ = ["run_fit", "run_cc_fit"]
