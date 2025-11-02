"""
Cosmic Chronometer (CC) data loader.

Loads direct Hubble parameter measurements H(z) from cosmic chronometer
observations using the differential age method with passively evolving galaxies.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_cc_data():
    """
    Load cosmic chronometer H(z) data.

    Returns
    -------
    dict
        Cosmic chronometer data following PBUF Data Object v1 schema:
        {
            "name": str,             # dataset name
            "type": str,             # "CC"
            "z": np.ndarray,         # Redshift
            "obs": np.ndarray,       # H(z) measurements [km/s/Mpc]
            "err": np.ndarray,       # Uncertainty in H(z)
            "cov": np.ndarray | None,# covariance matrix (None for now)
            "meta": dict             # metadata
        }
    """
    # Path to the cosmic chronometer data - check multiple locations
    data_paths = [
        Path(__file__).parent.parent / "data" / "raw" / "rsd" / "cc" / "cc.csv",
        Path(__file__).parent.parent / "data" / "cc" / "cc_data.csv",
        Path(__file__).parent.parent / "data" / "cc" / "cc.csv",
    ]

    for data_path in data_paths:
        if data_path.exists():
            # Load the CSV data
            df = pd.read_csv(data_path)

            z = df["z"].values
            Hz = df["Hz"].values
            err = df["sigma_Hz"].values

            return {
                "name": "CC_compilation",
                "type": "CC",
                "z": z,
                "obs": Hz,
                "err": err,
                "cov": None,
                "meta": {
                    "survey": "Various",
                    "observable": "Hubble parameter H(z)",
                    "units": "km/s/Mpc",
                    "reference": "Compilation"
                }
            }

    # If no file found, return None
    return None
