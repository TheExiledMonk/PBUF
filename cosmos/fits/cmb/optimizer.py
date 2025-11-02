"""
CMB parameter fitting interface.
"""

import numpy as np
from scipy.optimize import minimize
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from .chi2 import chi_squared_cmb
from .observables import cmb_observables

def fit_cmb(model_type="lcdm", initial_params=None, bounds=None):
    """
    Fit model parameters to minimize χ² against Planck CMB priors.

    Parameters
    ----------
    model_type : str
        "lcdm" or "pbuf"
    initial_params : dict
        Starting guess for model parameters
    bounds : dict[str, tuple[float, float]]
        Parameter bounds for optimizer

    Returns
    -------
    dict
        {
            "status": "success" or "fail",
            "chi2": best-fit χ²,
            "params": best-fit parameters,
            "observables": derived CMB quantities,
        }
    """
    if initial_params is None:
        if model_type == "lcdm":
            initial_params = {"H0": 67.5, "Om0": 0.315, "Ol0": 0.685}
        elif model_type == "pbuf":
            initial_params = {
                "H0": 67.5, "Om0": 0.315, "alpha": 0.001, "Rmax": 1e9, "k_sat": 0.8
            }

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

    # Flatten parameters for optimization
    param_names = list(initial_params.keys())
    x0 = np.array([initial_params[p] for p in param_names], dtype=float)
    bnds = [bounds.get(p, (None, None)) for p in param_names] if bounds else None

    def objective(x):
        p = dict(zip(param_names, x))
        model = make_model(p)
        return chi_squared_cmb(model)

    res = minimize(objective, x0, bounds=bnds, method="L-BFGS-B")

    best = dict(zip(param_names, res.x))
    model_best = make_model(best)
    obs_best = cmb_observables(model_best)

    return {
        "status": "success" if res.success else "fail",
        "chi2": float(res.fun),
        "params": best,
        "observables": obs_best,
    }
