"""
Isotropic BAO parameter fitting interface.
"""

import numpy as np
from scipy.optimize import minimize
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from .chi2 import chi_squared_bao_iso

def fit_bao_iso(model_type="lcdm", initial_params=None, bounds=None):
    """
    Fit cosmological model to isotropic BAO data.

    Parameters
    ----------
    model_type : str
        "lcdm" or "pbuf"
    initial_params : dict
        Starting guesses
    bounds : dict
        Parameter bounds

    Returns
    -------
    dict
        {
            "status": "success" or "fail",
            "chi2": float,
            "params": dict,
        }
    """
    if initial_params is None:
        if model_type == "lcdm":
            initial_params = {"H0": 67.5, "Om0": 0.315, "Ol0": 0.685}
        elif model_type == "pbuf":
            initial_params = {
                "H0": 67.5,
                "Om0": 0.315,
                "alpha": 0.001,
                "Rmax": 1e9,
                "k_sat": 1.5,
            }

    # Default parameter bounds
    if bounds is None:
        if model_type == "lcdm":
            bounds = {
                "H0": (60.0, 80.0),
                "Om0": (0.1, 0.5),
                "Ol0": (0.5, 0.9),
            }
        elif model_type == "pbuf":
            bounds = {
                "H0": (60.0, 80.0),
                "Om0": (0.1, 0.5),  # Matter density
                "alpha": (1e-6, 1e-1),  # Elastic amplitude
                "Rmax": (1e6, 1e12),  # Saturation scale
                "k_sat": (0.1, 3.0),  # Rigidity saturation
                # Add bounds for other parameters if needed
            }

    def make_model(params):
        h = params["H0"] / 100.0
        model_params = {
            'omega_m': params["Om0"],
            'h': h,
            'omega_k': 0.0,
            'omega_r': 9.2e-5,
            'omega_b': 0.022 / (h**2)
        }
        
        if model_type == "lcdm":
            return LCDM(**model_params)
        elif model_type == "pbuf":
            # Add PBUF-specific parameters
            model_params.update({
                'alpha': params.get("alpha", 0.001),
                'Rmax': params.get("Rmax", 1e9),
                'k_sat': params.get("k_sat", 1.5)
            })
            return PBUF(**model_params)
    
    param_names = list(initial_params.keys())
    x0 = np.array([initial_params[p] for p in param_names], dtype=float)
    bnds = [bounds.get(p, (None, None)) for p in param_names]
    
    def objective(x):
        p = dict(zip(param_names, x))
        
        # Add physical constraints
        if model_type == "pbuf":
            # Ensure Om0 + Or0 + Ok0 <= 1
            total = p.get("Om0", 0) + 9.2e-5 + p.get("Ok0", 0)
            if total > 1.0:
                return 1e10
        
        try:
            model = make_model(p)
            return chi_squared_bao_iso(model)
        except (ValueError, RuntimeError) as e:
            return 1e10
            raise

    res = minimize(objective, x0, bounds=bnds, method="L-BFGS-B")
    best = dict(zip(param_names, res.x))
    return {
        "status": "success" if res.success else "fail",
        "chi2": float(res.fun),
        "params": best,
    }
