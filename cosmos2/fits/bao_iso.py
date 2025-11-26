"""BAO isotropic fit for cosmos2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def run_bao_iso_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("bao_iso")
    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)

    dv_model = np.asarray(model.DV(z), dtype=float)
    rd = float(model.sound_horizon())
    if rd <= 0.0:
        raise ValueError("Model returned a non-positive sound horizon")

    dv_over_rd_model = dv_model / rd
    diff = dv_over_rd_model - observed

    inv_cov = dataset.get("inv_cov")
    if inv_cov is not None:
        inv_cov = np.asarray(inv_cov, dtype=float)
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        err = dataset.get("err")
        if err is None:
            raise ValueError("BAO isotropic dataset lacks covariance and errors")
        chi2 = float(np.sum((diff / err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=dv_over_rd_model,
        observed=observed,
        residuals=diff,
        additional={"rd": rd},
    )
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_bao_iso_fit(model, dataset)


__all__ = ["run_fit", "run_bao_iso_fit"]
