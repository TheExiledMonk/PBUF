"""SH0ES H0 prior fit for cosmos2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def run_sh0es_prior(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("sh0es")
    H0_obs = float(dataset.get("H0") or dataset["obs"][0])
    sigma = float(dataset.get("sigma") or dataset.get("err", [0.0])[0])
    H0_model = float(model.Hubble(0.0))
    chi2 = ((H0_model - H0_obs) / sigma) ** 2
    extras = build_fit_extras(dataset=dataset, predictions=H0_model, observed=H0_obs, residuals=H0_model - H0_obs)
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_sh0es_prior(model, dataset)


__all__ = ["run_fit", "run_sh0es_prior"]
