"""BAO anisotropic fit for cosmos2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.data.registry import get_dataset
from cosmos2.fits.extras import build_fit_extras


def run_bao_aniso_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("bao_aniso")
    z = np.asarray(dataset["z"], dtype=float)
    obs = np.asarray(dataset["obs"], dtype=float)
    labels = np.asarray(dataset.get("labels"), dtype=object)
    cov = np.asarray(dataset.get("cov"), dtype=float)
    if cov.size == 0:
        raise ValueError("BAO anisotropic dataset lacks covariance")
    inv_cov = np.linalg.inv(cov)

    # Some standardized caches store two observables per redshift (D_M/rd, D_H/rd),
    # so the z array can be half the length of the label/obs vectors. Derive a stride
    # to map each pair back to its redshift bin.
    stride = 1
    if z.size and labels.size and labels.size % z.size == 0:
        stride = max(1, labels.size // z.size)

    residuals = np.empty_like(obs, dtype=float)
    for i in range(obs.shape[0]):
        z_idx = min(i // stride, z.size - 1) if z.size else 0
        z_bin = z[z_idx]
        raw_label = str(labels[i])
        label = "".join(ch for ch in raw_label.lower() if ch.isalnum())
        a = 1.0 / (1.0 + z_bin) if z_bin >= 0 else 1.0
        if "dm" in label:
            DM = model.DM(z_bin)
            pred = DM / model.sound_horizon()
        elif "dh" in label or "htimes" in label:
            H = model.Hubble(z_bin)
            pred = (299_792.458 / H) / model.sound_horizon()
        elif "da" in label:
            DM_here = model.DM(z_bin)
            pred = (DM_here / (1.0 + z_bin)) / model.sound_horizon()
        else:
            raise ValueError(f"Unknown BAO observable label '{raw_label}' (normalized '{label}')")
        residuals[i] = pred - obs[i]

    chi2 = float(residuals.T @ inv_cov @ residuals)
    extras = build_fit_extras(dataset=dataset, predictions=None, observed=obs, residuals=residuals)
    return chi2, extras


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    return run_bao_aniso_fit(model, dataset)


__all__ = ["run_fit", "run_bao_aniso_fit"]
