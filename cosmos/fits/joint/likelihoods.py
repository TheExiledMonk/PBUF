"""
Compute total joint χ² across selected datasets.
"""

import numpy as np
from .registry import available_datasets

def compute_joint_chi2(model, datasets=None, verbose=False):
    """
    Compute total χ² for the given model across datasets.

    Parameters
    ----------
    model : LCDM or PBUF instance or dict
        Either a model instance or a dictionary of parameters
    datasets : list[str] or None
        e.g. ["cmb", "sn", "bao_iso"]
        If None, use all available datasets.
    verbose : bool
        Print per-dataset breakdown

    Returns
    -------
    tuple (chi2_total, breakdown)
        chi2_total : float
        breakdown  : dict {dataset_name: chi2_value}
    """
    from cosmos.lcdm.model import LCDM
    from cosmos.pbuf.model import PBUF
    import numpy as np
    
    # Determine if model is an instance or a parameter dictionary
    is_instance = isinstance(model, (LCDM, PBUF))
    
    # If it's a model instance, extract parameters
    if is_instance:
        if isinstance(model, LCDM):
            model_type = "lcdm"
        else:
            model_type = "pbuf"
            
        # Extract parameters from model instance
        params = {}
        params["H0"] = model.h * 100.0
        params["Om0"] = model.omega_m
        params["Ok0"] = getattr(model, 'omega_k', 0.0)
        params["Or0"] = getattr(model, 'omega_r', 9.2e-5)
        
        if model_type == "lcdm":
            params["Ol0"] = getattr(model, 'omega_lambda', 1.0 - model.omega_m - getattr(model, 'omega_k', 0.0))
            params["Obh2"] = getattr(model, 'omega_b', 0.02237) * (model.h ** 2)
        else:  # PBUF
            params["alpha"] = model.alpha
            params["Rmax"] = model.Rmax
            params["k_sat"] = getattr(model, 'k_sat', 1.0)
            params["Ol0"] = 0.0
            params["Obh2"] = getattr(model, 'omega_b', 0.02237) * (model.h ** 2)
        params["Ok0"] = getattr(model, 'omega_k', 0.0)
    else:
        # It's already a parameter dictionary
        params = dict(model)
    
    # Inject legacy aliases for compatibility with older modules
    def _alias(src_key, dst_keys):
        if src_key in params and params[src_key] is not None:
            for key in dst_keys:
                params.setdefault(key, params[src_key])

    _alias("H0", ["h0"])
    _alias("Om0", ["omega_m", "Omega_m"])
    _alias("Ol0", ["omega_lambda", "Omega_lambda"])
    _alias("Or0", ["omega_r", "Omega_r"])
    _alias("Ok0", ["omega_k", "Omega_k"])
    _alias("Obh2", ["Omega_b", "obh2"])

    # Ensure required defaults exist
    params.setdefault("Ok0", 0.0)
    params.setdefault("Or0", 9.2e-5)
    if "Ol0" not in params and "omega_lambda" in params:
        params["Ol0"] = params["omega_lambda"]
    if "Om0" not in params and "omega_m" in params:
        params["Om0"] = params["omega_m"]
    if "omega_b" not in params and "Obh2" in params:
        h_for_baryons = params.get("H0", params.get("h0", 70.0)) / 100.0
        if h_for_baryons > 0:
            params["omega_b"] = params["Obh2"] / (h_for_baryons**2)
    
    registry = available_datasets(verbose=False)

    if datasets is None:
        datasets = list(registry.keys())

    chi2_total = 0.0
    breakdown = {}
    
    # Create a model instance if needed by the registry functions
    model_instance = None
    if is_instance:
        model_instance = model
    else:
        # Create a minimal model instance if needed
        from cosmos.lcdm.model import LCDM as LCDMClass
        omega_lambda = params.get("Ol0", params.get("omega_lambda", 0.7))
        model_instance = LCDMClass(
            h=params.get("H0", params.get("h0", 70.0))/100.0,
            omega_m=params.get("Om0", params.get("omega_m", 0.3)),
            omega_lambda=omega_lambda,
            omega_k=params.get("Ok0", params.get("omega_k", 0.0)),
            omega_r=params.get("Or0", params.get("omega_r")),
        )

    for name in datasets:
        func = registry.get(name)
        if func is None:
            if verbose:
                print(f"[warn] Dataset {name} not found or missing chi² function.")
            continue

        try:
            # First try passing the model instance
            result = func(model_instance)
        except (TypeError, AttributeError):
            try:
                # If that fails, try passing the parameters
                result = func(params)
            except Exception as e:
                if verbose:
                    print(f"[warn] Error evaluating {name}: {e}, using penalty.")
                # Use a large penalty value for failed evaluations
                result = 1e20

        # Handle both old format (float) and new format (dict with chi2 key)
        if isinstance(result, dict):
            chi2_val = result.get("chi2", float("nan"))
        else:
            try:
                chi2_val = float(result)
            except (TypeError, ValueError):
                if verbose:
                    print(f"[warn] Invalid chi² value from {name}: {result}, using penalty.")
                chi2_val = 1e20

        breakdown[name] = chi2_val
        chi2_total += chi2_val

        if verbose:
            print(f"{name:10s}: χ² = {chi2_val:.3f}")

    return float(chi2_total), breakdown
