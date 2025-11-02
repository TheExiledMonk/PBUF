"""
Redshift Space Distortion (RSD) data loader.

Loads fσ8 measurements from galaxy redshift surveys, where f is the growth rate
and σ8 is the amplitude of matter fluctuations on 8 h⁻¹ Mpc scales.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_rsd_data():
    """
    Load RSD fσ8 data.

    Returns
    -------
    dict
        RSD data following PBUF Data Object v1 schema:
        {
            "name": str,             # dataset name
            "type": str,             # "RSD"
            "z": np.ndarray,         # Redshift
            "obs": np.ndarray,       # fσ8 measurements
            "err": np.ndarray,       # Uncertainty in fσ8
            "cov": np.ndarray | None,# covariance matrix (None for now)
            "meta": dict             # metadata
        }
    """
    # Path to the RSD data - check multiple locations
    data_paths = [
        Path(__file__).parent.parent / "data" / "raw" / "rsd" / "rsd.csv",
        Path(__file__).parent.parent / "data" / "rsd" / "rsd_data.csv",
        Path(__file__).parent.parent / "data" / "rsd" / "rsd.csv",
    ]

    for data_path in data_paths:
        if data_path.exists():
            # Load the CSV data
            df = pd.read_csv(data_path)

            z = df["z"].values
            fsigma8 = df["f_sigma8"].values
            err = df["sigma_f_sigma8"].values

            return {
                "name": "RSD_compilation",
                "type": "RSD",
                "z": z,
                "obs": fsigma8,
                "err": err,
                "cov": None,
                "meta": {
                    "survey": "Various",
                    "observable": "growth rate fσ₈",
                    "reference": "Compilation"
                }
            }

    # If no file found, return None
    return None
