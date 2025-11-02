"""
CMB distance priors loader for Planck 2018 data.

Loads compressed CMB distance priors from Planck 2018 TT,TE,EE+lowE+lensing
analysis, providing R (shift parameter), l_a (angular scale), and theta_star.
"""

import numpy as np
from pathlib import Path


def load_cmb_priors():
    """
    Load Planck 2018 CMB distance priors from standardized data.

    Returns
    -------
    dict
        CMB distance priors following PBUF Data Object v1 schema:
        {
            "name": str,             # dataset name
            "type": str,             # "CMB"
            "z": np.ndarray | None,  # Redshifts (None for CMB)
            "obs": np.ndarray,       # CMB observables [R, l_a, theta_star]
            "err": np.ndarray | None,# 1σ uncertainties (None if covariance provided)
            "cov": np.ndarray | None,# covariance matrix
            "meta": dict             # metadata
        }
    """
    # Load from standardized data (same pattern as other datasets)
    data_path = Path(__file__).parent.parent / "data" / "standardized" / "cmb.npz"

    if not data_path.exists():
        raise FileNotFoundError(f"CMB standardized data not found at {data_path}")

    data = np.load(data_path, allow_pickle=True)

    # Extract standardized fields
    obs = data["obs"]      # Observed CMB observables
    cov = data["cov"]      # Covariance matrix
    meta = data["meta"]    # Metadata

    # For CMB, create diagonal errors from covariance if needed
    err = np.sqrt(np.diag(cov)) if cov is not None else None

    return {
        "name": "Planck2018",
        "type": "CMB",
        "z": data.get("z"),    # Redshifts (None for CMB)
        "obs": obs,
        "err": err,
        "cov": cov,
        "meta": meta.item() if hasattr(meta, 'item') else meta
    }
