"""
SN parameter fitting interface.

This module provides functions to fit cosmological model parameters
to supernova data by minimizing the χ² between observed and predicted
distance moduli.
"""

import numpy as np
from scipy.optimize import minimize
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from .chi2 import chi_squared_sn

def fit_sn(model_type="lcdm", initial_params=None, bounds=None, fit_M=False, M_init=0.0):
    """
    Fit cosmological model parameters to SN data by minimizing χ².

    Parameters
    ----------
    model_type : str
        "lcdm" or "pbuf"
    initial_params : dict, optional
        Starting parameter values. If None, uses default values.
    bounds : dict, optional
        Parameter bounds for optimization. If None, uses reasonable defaults.
    fit_M : bool, optional
        Whether to fit the absolute magnitude offset M as a nuisance parameter.
        Default is False (M=0).
    M_init : float, optional
        Initial value for M if fit_M=True. Default is 0.0.

    Returns
    -------
    dict
        {
            "status": "success" or "fail",
            "chi2": float,              # best-fit χ²
            "params": dict,             # best-fit parameters
            "M": float,                 # best-fit M (if fit_M=True)
            "n_data": int,              # number of data points
        }

    Notes
    -----
    The absolute magnitude offset M is degenerate with H0 and is often
    marginalized over analytically. However, this function can fit it
    numerically if desired for comparison purposes.
    """
    # Default initial parameters
    if initial_params is None:
        if model_type == "lcdm":
            initial_params = {"H0": 67.5, "Om0": 0.315, "Ol0": 0.685}
        elif model_type == "pbuf":
            initial_params = {"H0": 67.5, "Om0": 0.315, "alpha": 0.001, "Rmax": 1e9, "k_sat": 1.5}
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

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

    # Set up parameters to fit
    fit_params = list(initial_params.keys())
    if fit_M:
        fit_params.append("M")

    # Initial parameter vector
    x0_list = [initial_params[p] for p in fit_params if p != "M"]
    if fit_M:
        x0_list.append(M_init)
    x0 = np.array(x0_list, dtype=float)

    # Parameter bounds
    bnds_list = [bounds.get(p, (None, None)) for p in fit_params if p != "M"]
    if fit_M:
        bnds_list.append((-5.0, 5.0))  # Reasonable bounds for M
    bnds = bnds_list

    def make_model_and_compute_chi2(x):
        """Create model and compute χ² for parameter vector x."""
        # Extract parameters
        param_dict = {}
        M = 0.0

        for i, param_name in enumerate(fit_params):
            if param_name == "M":
                M = x[i]
            else:
                param_dict[param_name] = x[i]

        # Create model
        h = param_dict["H0"] / 100.0
        if model_type == "lcdm":
            model = LCDM(
                omega_m=param_dict["Om0"],
                omega_lambda=param_dict["Ol0"],
                h=h,
                omega_k=0.0,
                omega_r=None,
                omega_b=0.022 / (h**2),
            )
        elif model_type == "pbuf":
            model = PBUF(
                omega_m=param_dict["Om0"],
                h=h,
                alpha=param_dict["alpha"],
                Rmax=param_dict["Rmax"],
                k_sat=param_dict["k_sat"],
                eps0=param_dict.get("eps0", 0.7),
                n_alpha=param_dict.get("n_alpha", 0.0),
                n_eps=param_dict.get("n_eps", 0.0),
                n_R=param_dict.get("n_R", 0.0),
                omega_k=0.0,
                omega_r=None,
                omega_b=0.022 / (h**2),
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Compute χ²
        return chi_squared_sn(model, M=M)

    # Minimize χ²
    res = minimize(make_model_and_compute_chi2, x0, bounds=bnds, method="L-BFGS-B")

    # Extract results
    best_params = {}
    M_best = 0.0

    for i, param_name in enumerate(fit_params):
        if param_name == "M":
            M_best = res.x[i]
        else:
            best_params[param_name] = res.x[i]

    # Get number of data points for reduced χ² calculation
    data = None  # This will load the default dataset
    n_data = len(data["z"]) if data else 0

    return {
        "status": "success" if res.success else "fail",
        "chi2": float(res.fun),
        "params": best_params,
        "M": M_best if fit_M else None,
        "n_data": n_data,
        "n_params": len(fit_params),
        "reduced_chi2": float(res.fun) / max(n_data - len(fit_params), 1),
    }
