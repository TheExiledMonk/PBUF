"""
Cosmic chronometer parameter fitting interface.

This module provides functions to fit cosmological parameters
to cosmic chronometer H(z) data by minimizing the χ² between
observed and predicted expansion rates.
"""

import numpy as np
from scipy.optimize import minimize
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.optim.parameter_defaults import (
    LCDM_PARAMETER_DEFAULTS,
    PBUF_PARAMETER_DEFAULTS,
)
from .chi2 import chi_squared_cc

def fit_cc(model_type="lcdm", initial_params=None, bounds=None):
    """
    Fit cosmological parameters to CC H(z) data.
    Parameters
    ----------
    model_type : str
        "lcdm" or "pbuf"
    initial_params : dict
        Mapping {param_name: (low, high)} for optimizer.

    Returns
    -------
    dict
            "status": "success" or "fail",
            "chi2": float,
            "params": dict,
        }

    Notes
    -----
    LCDM parameters: H0, Om0, Ol0
    PBUF parameters: H0, Om0, alpha, Rmax, k_sat

    The fitting is performed using scipy.optimize.minimize with
    the L-BFGS-B method, which supports parameter bounds.
    """
    # Default guesses
    if initial_params is None:
        if model_type == "lcdm":
            lcdm_defaults = dict(LCDM_PARAMETER_DEFAULTS)
            inferred_ol0 = lcdm_defaults.get(
                "Ol0",
                max(
                    0.0,
                    1.0
                    - float(lcdm_defaults.get("Om0", 0.0))
                    - float(lcdm_defaults.get("Ok0", 0.0))
                    - float(lcdm_defaults.get("Or0", 0.0)),
                ),
            )
            initial_params = {
                "H0": float(lcdm_defaults.get("H0", 67.5)),
                "Om0": float(lcdm_defaults.get("Om0", 0.315)),
                "Ol0": float(inferred_ol0),
            }
        elif model_type == "pbuf":
            pbuf_defaults = dict(PBUF_PARAMETER_DEFAULTS)
            initial_params = {
                "H0": float(pbuf_defaults.get("H0", 67.5)),
                "Om0": float(pbuf_defaults.get("Om0", 0.315)),
                "alpha": float(pbuf_defaults.get("alpha", 1.0e-3)),
                "Rmax": float(pbuf_defaults.get("Rmax", 1.0e9)),
                "k_sat": float(pbuf_defaults.get("k_sat", 1.5)),
            }
        else:
            raise ValueError(f"Unknown model_type {model_type}")

    # Helper to construct the cosmology object from parameter dict
    def make_model(params):
        h = params["H0"] / 100.0

        if model_type == "lcdm":
            return LCDM(
                omega_m=params["Om0"],
                omega_lambda=params["Ol0"],
                h=h,
                omega_k=0.0,
                omega_r=None,
                omega_b=0.022 / (h**2),
            )

        elif model_type == "pbuf":
            return PBUF(
                omega_m=params["Om0"],
                h=h,
                alpha=params["alpha"],
                Rmax=params["Rmax"],
                k_sat=params["k_sat"],
                eps0=params.get("eps0", 0.7),
                n_alpha=params.get("n_alpha", 0.0),
                n_eps=params.get("n_eps", 0.0),
                n_R=params.get("n_R", 0.0),
                omega_k=0.0,
                omega_r=None,
                omega_b=0.022 / (h**2),
            )

        else:
            raise ValueError(f"Unknown model_type {model_type}")

    # Flatten params to a vector x for the optimizer
    param_names = list(initial_params.keys())          # preserve order
    x0 = np.array([initial_params[p] for p in param_names], dtype=float)

    # Default parameter bounds
    if bounds is None:
        bounds = {}

        if model_type == "lcdm":
            bounds.update({
                "H0": (50.0, 100.0),
                "Om0": (0.1, 0.5),
                "Ol0": (0.5, 0.9),
            })
        elif model_type == "pbuf":
            bounds.update({
                "H0": (50.0, 100.0),
                "Om0": (0.1, 0.5),
                "alpha": (1e-6, 1e-1),
                "Rmax": (1e6, 1e12),
                "k_sat": (0.1, 3.0),
            })

    # Bounds in scipy format, same ordering as param_names
    bnds = [bounds.get(p, (None, None)) for p in param_names]

    # Objective function: χ² from CC data
    def objective(x):
        p = dict(zip(param_names, x))
        model = make_model(p)
        return chi_squared_cc(model)

    # Minimize χ²
    res = minimize(objective, x0, bounds=bnds, method="L-BFGS-B")

    best = dict(zip(param_names, res.x))

    return {
        "status": "success" if res.success else "fail",
        "chi2": float(res.fun),
        "params": best,
    }
