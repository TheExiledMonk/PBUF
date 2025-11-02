"""
RSD parameter fitting interface.
"""

import numpy as np
from scipy.optimize import minimize
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from cosmos.optim.parameter_defaults import (
    LCDM_PARAMETER_DEFAULTS,
    PBUF_PARAMETER_DEFAULTS,
    SIGMA8_PLANCK,
)
from .chi2 import chi_squared_rsd


def fit_rsd(model_type="lcdm", initial_params=None, bounds=None, sigma8_0=SIGMA8_PLANCK):
    """
    Fit cosmological parameters to RSD fσ8 data.

    Parameters
    ----------
    model_type : str
        "lcdm" or "pbuf"
    initial_params : dict
        Starting guesses
    bounds : dict
        Parameter bounds
    sigma8_0 : float
        Normalization of σ8(z=0)

    Returns
    -------
    dict
        {
            "status": "success" or "fail",
            "chi2": float,
            "params": dict,
        }
    """
    sigma8_default = float(sigma8_0)

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
            raise ValueError(f"Unknown model_type: {model_type}")
    else:
        initial_params = dict(initial_params)

    initial_params.setdefault("sigma8_0", sigma8_default)

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
    else:
        bounds = dict(bounds)

    bounds.setdefault("sigma8_0", (0.4, 1.2))

    param_names = list(initial_params.keys())
    x0 = np.array([initial_params[p] for p in param_names], dtype=float)
    bnds = [bounds.get(p, (None, None)) for p in param_names] if bounds else None

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

    def objective(x):
        p = dict(zip(param_names, x))
        sigma8 = p.get("sigma8_0", sigma8_0)
        model_params = {k: v for k, v in p.items() if k != "sigma8_0"}
        model = make_model(model_params)
        return chi_squared_rsd(model, sigma8_0=sigma8)

    res = minimize(objective, x0, bounds=bnds, method="L-BFGS-B")
    best = dict(zip(param_names, res.x))

    return {
        "status": "success" if res.success else "fail",
        "chi2": float(res.fun),
        "params": best,
    }
