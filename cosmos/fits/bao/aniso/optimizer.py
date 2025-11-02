"""
Anisotropic BAO parameter fitting interface.
"""

import numpy as np
from scipy.optimize import minimize
from cosmos.lcdm.model import LCDM
from cosmos.pbuf.model import PBUF
from .chi2 import chi_squared_bao_aniso

def fit_bao_aniso(model_type="lcdm", initial_params=None, bounds=None):
    """
    Fit cosmological model to anisotropic BAO data.

    Parameters
    ----------
    model_type : str
        "lcdm" or "pbuf"
    initial_params : dict
        Starting guess for model parameters
    bounds : dict
        Mapping param -> (low, high)

    Returns
    -------
    dict
            "status": "success" or "fail",
            "chi2": float,
            "params": dict,
        }
    """
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
            model_params['omega_lambda'] = params.get("Ol0", 0.7)
            return LCDM(**model_params)
        elif model_type == "pbuf":
            # Add PBUF-specific parameters
            model_params.update({
                'alpha': params["alpha"],
                'Rmax': params["Rmax"],
                'k_sat': params["k_sat"],
                'eps0': params.get("eps0", 0.7),
                'n_alpha': params.get("n_alpha", 0.0),
                'n_eps': params.get("n_eps", 0.0),
                'n_R': params.get("n_R", 0.0)
            })
            return PBUF(**model_params)

        else:
            raise ValueError(f"Unknown model_type {model_type}")

    # flatten params for optimizer
    param_names = list(initial_params.keys())
    x0 = np.array([initial_params[p] for p in param_names], dtype=float)
    bnds = [bounds.get(p, (None, None)) for p in param_names] if bounds else None

    def objective(x):
        p = dict(zip(param_names, x))
        
        # Add physical constraints
        if model_type == "pbuf":
            # Ensure Om0 + Or0 + Ok0 <= 1
            total = p.get("Om0", 0) + 9.2e-5 + p.get("Ok0", 0)
            if total > 1.0:
                # Return a large chi2 for unphysical parameters
                return 1e10
                
            # Add other constraints as needed
            
        try:
            model = make_model(p)
            return chi_squared_bao_aniso(model)
        except (ValueError, RuntimeError) as e:
            # Return a large chi2 for invalid parameters
            if "closure" in str(e).lower():
                return 1e10
            raise

    res = minimize(objective, x0, bounds=bnds, method="L-BFGS-B")

    best = dict(zip(param_names, res.x))

    return {
        "status": "success" if res.success else "fail",
        "chi2": float(res.fun),
        "params": best,
    }
