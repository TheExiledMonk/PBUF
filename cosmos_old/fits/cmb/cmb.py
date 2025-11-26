"""CMB distance prior χ² wrapper for the joint fit runner."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel
from fits.cmb.data_loader import CMBDataset, load_planck_priors


def run_fit(
    model: CosmologyModel,
    dataset: CMBDataset | None = None,
) -> tuple[float, Dict[str, Any]]:
    """Compute the Planck χ² contribution for the supplied model."""

    dataset = dataset or load_planck_priors()
    output = model.cmb(dataset)
    predicted = np.array(
        [
            output.R,
            output.l_A,
            output.theta_star,
        ],
        dtype=float,
    )
    residual = predicted - dataset.observed
    chi2 = float(residual.T @ dataset.inv_covariance @ residual)

    extras = build_fit_extras(
        dataset=dataset,
        predictions=predicted,
        observed=dataset.observed,
        residuals=residual,
        additional={
            "cmb_output": output,
            "sigmas": dataset.sigmas,
        },
    )
    return chi2, extras
