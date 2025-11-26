"""CMB distance prior χ² for cosmos2 models."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

from cosmos2.fits.extras import build_fit_extras
from cosmos2.data.registry import get_dataset


def run_fit(model: Any, dataset: Any | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("cmb")
    observed = np.asarray(dataset["obs"] if "obs" in dataset else dataset.get("observed"), dtype=float)
    inv_cov = np.asarray(dataset.get("inv_cov"), dtype=float)
    meta = dataset.get("meta")
    if isinstance(meta, np.ndarray):
        try:
            meta = meta.item()
        except Exception:
            meta = None
    z_star = dataset.get("z_star")
    if z_star is None and isinstance(meta, dict):
        z_star = meta.get("z_star") or meta.get("z")
    if z_star is None and "z" in dataset:
        try:
            z_star = float(np.asarray(dataset["z"]).flatten()[0])
        except Exception:
            pass
    z_star = z_star if z_star is not None else 1090.0
    cmb_out = model.cmb(dataset)
    predicted = np.array([cmb_out.R, cmb_out.l_A, cmb_out.theta_star], dtype=float)
    residual = predicted - observed
    chi2 = float(residual.T @ inv_cov @ residual)

    if os.getenv("COSMOS2_CMB_DEBUG"):
        print("[cosmos2][cmb] z_star", z_star)
        print("[cosmos2][cmb] observed", observed)
        print("[cosmos2][cmb] predicted", predicted)
        print("[cosmos2][cmb] residual", residual)
        print("[cosmos2][cmb] inv_cov", inv_cov)
        print(
            "[cosmos2][cmb] details",
            {
                "D_M": getattr(cmb_out, "D_M_Mpc", None),
                "D_A": getattr(cmb_out, "D_A_Mpc", None),
                "r_s": getattr(cmb_out, "r_s_Mpc", None),
            },
        )
        print("[cosmos2][cmb] chi2", chi2)
    extras = build_fit_extras(
        dataset=dataset,
        predictions=predicted,
        observed=observed,
        residuals=residual,
        additional={"cmb_output": cmb_out},
    )
    return chi2, extras


__all__ = ["run_fit"]
