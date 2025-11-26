"""Lensing cross-correlation fit for cosmos2 models (mirrors cosmos behavior)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def run_lensing_cross_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("lensing_cross")
    A_obs = np.asarray(dataset["A_obs"] if "A_obs" in dataset else dataset.get("obs"), dtype=float)
    A_err = np.asarray(dataset["A_err"] if "A_err" in dataset else dataset.get("err"), dtype=float)
    p_exponent = np.asarray(dataset.get("p_exponent"), dtype=float)
    q_exponent = np.asarray(dataset.get("q_exponent"), dtype=float)
    z_eff = np.asarray(dataset.get("z_eff"), dtype=float)
    S8_fid = np.asarray(dataset.get("S8_fid"), dtype=float)
    fs8_fid = np.asarray(dataset.get("fs8_fid"), dtype=float)
    gamma = np.asarray(dataset.get("gamma", 0.5), dtype=float)
    weights = np.asarray(dataset.get("weights", 1.0), dtype=float)

    n = len(A_obs)
    for arr_name, arr in (
        ("A_err", A_err),
        ("p_exponent", p_exponent),
        ("q_exponent", q_exponent),
        ("z_eff", z_eff),
        ("S8_fid", S8_fid),
        ("fs8_fid", fs8_fid),
        ("gamma", gamma),
        ("weights", weights),
    ):
        if arr is None:
            raise ValueError(f"Lensing cross dataset missing field {arr_name}")
        arr = np.asarray(arr, dtype=float)
        if arr.shape != (n,):
            raise ValueError(f"Lensing cross field {arr_name} has shape {arr.shape}, expected {(n,)}")
    if np.any(weights <= 0.0):
        raise ValueError("Lensing cross weights must be positive.")

    S8_model = np.asarray([model.S8(g) for g in gamma], dtype=float)
    fs8_model = np.asarray(model.fs8(z_eff), dtype=float).reshape(n)
    A_model = (S8_model / S8_fid) ** p_exponent * (fs8_model / fs8_fid) ** q_exponent

    diff = A_model - A_obs
    scaled_err = A_err / weights
    cov = dataset.get("cov")
    if cov is not None:
        inv_cov = dataset.get("inv_cov") or np.linalg.inv(np.asarray(cov, dtype=float))
        inv_cov = np.asarray(inv_cov, dtype=float)
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        if np.any(scaled_err <= 0.0):
            raise ValueError("Lensing cross errors must be positive when no covariance is provided.")
        chi2 = float(np.sum((diff / scaled_err) ** 2))

    additional = {
        "A_model": A_model,
        "z_eff": z_eff,
        "gamma": gamma,
        "fs8_model": fs8_model,
        "S8_model": S8_model,
        "weights": weights,
        "scaled_err": scaled_err,
    }
    if "labels" in dataset:
        additional["labels"] = dataset["labels"]

    extras = build_fit_extras(
        dataset=dataset,
        predictions=A_model,
        observed=A_obs,
        residuals=diff,
        additional=additional,
    )
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_lensing_cross_fit(model, dataset)


__all__ = ["run_fit", "run_lensing_cross_fit"]
