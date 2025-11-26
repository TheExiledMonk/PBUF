"""Model-agnostic CMB fit orchestration."""

from __future__ import annotations

import numpy as np

from cosmos.factory import create_model
from cosmos.interfaces import CMBOutput
from fits.cmb.data_loader import CMBDataset


def run_cmb_fit(model_name: str, dataset: CMBDataset, model_params: dict):
    """
    Run a CMB-only fit for the requested model.

    Returns a dictionary containing the CMBOutput and the χ² value computed
    against the supplied dataset.
    """

    model = create_model(model_name, **model_params)
    cmb_output = model.cmb(dataset)
    predicted = _cmb_vector(cmb_output)
    residual = predicted - dataset.observed
    chi2 = float(residual.T @ dataset.inv_covariance @ residual)

    return {
        "chi2": chi2,
        "output": cmb_output,
        "residual": residual,
        "model_name": model_name,
    }


def _cmb_vector(output: CMBOutput) -> np.ndarray:
    """Vectorise the model output in the same order as the priors."""

    return np.array(
        [
            output.R,
            output.l_A,
            output.theta_star,
        ]
    )
