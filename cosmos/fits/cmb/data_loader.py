"""
Load Planck 2018 CMB distance priors and covariance matrices.
"""

import json
import numpy as np
import os
from pathlib import Path

def load_planck_priors(path=None):
    """
    Load Planck 2018 distance priors (mean and covariance).

    Parameters
    ----------
    path : str or Path, optional
        Path to the priors JSON file. If None, search in common locations.

    Returns
    -------
    dict
        {
            "mean": {"R": ..., "la": ..., "theta_star": ...},
            "cov": np.ndarray(3x3)
        }
    """
    if path is None:
        # Search for the data file in common locations
        search_paths = [
            Path(__file__).parent.parent.parent.parent.parent / "data" / "priors" / "planck2018_distance_priors.json",
            Path.cwd() / "data" / "priors" / "planck2018_distance_priors.json",
            Path.home() / "PBUF4" / "data" / "priors" / "planck2018_distance_priors.json",
        ]

        data_path = None
        for p in search_paths:
            if p.exists():
                data_path = p
                break

        if data_path is None:
            raise FileNotFoundError(
                "Could not find planck2018_distance_priors.json. "
                f"Searched in: {[str(p) for p in search_paths]}"
            )
    else:
        data_path = Path(path)

    with open(data_path, "r") as f:
        data = json.load(f)

    mean = data["mean"]
    cov = np.array(data["covariance"], dtype=float)
    if cov.shape != (3, 3):
        raise ValueError(f"Invalid covariance shape: {cov.shape}")

    return {"mean": mean, "cov": cov}
