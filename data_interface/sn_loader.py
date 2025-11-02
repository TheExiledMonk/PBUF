"""
Supernova distance modulus loader.

Loads Type Ia supernova distance modulus measurements from observational datasets
like Pantheon or Pantheon+ compilations.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_sn_data():
    """
    Load supernova distance modulus data.

    Returns
    -------
    dict
        Supernova data following PBUF Data Object v1 schema:
        {
            "name": str,             # dataset name
            "type": str,             # "SN"
            "z": np.ndarray,         # Redshift
            "obs": np.ndarray,       # Distance modulus (mu)
            "err": np.ndarray,       # Uncertainty in mu
            "cov": np.ndarray | None,# covariance matrix (None for now)
            "meta": dict             # metadata
        }
    """
    # Path to the supernova data - check multiple locations
    data_paths = [
        Path(__file__).parent.parent / "data" / "raw" / "DataRelease" / "Pantheon+_Data" / "1_DATA" / "all_redshifts_PVs.csv",
        Path(__file__).parent.parent / "data" / "supernovae" / "derived" / "supernova_index.csv",
        Path(__file__).parent.parent / "data" / "supernovae" / "derived" / "pantheon_index.csv",
    ]

    for data_path in data_paths:
        if data_path.exists():
            # Load the CSV data
            df = pd.read_csv(data_path)

            # Check if this is Pantheon+ format
            if "zHD" in df.columns and "MU_SH0ES" in df.columns:
                # Pantheon+ format
                z = df["zHD"].values
                mu = df["MU_SH0ES"].values
                mu_err = df["MU_SH0ES_ERR_DIAG"].values
            elif "redshift" in df.columns and "mu" in df.columns:
                # Generic format
                z = df["redshift"].values
                mu = df["mu"].values
                if "mu_err" in df.columns:
                    mu_err = df["mu_err"].values
                else:
                    mu_err = np.full_like(mu, 0.1)  # Default error
            else:
                continue  # Try next path

            return {
                "name": "Pantheon+",
                "type": "SN",
                "z": z,
                "obs": mu,
                "err": mu_err,
                "cov": None,
                "meta": {
                    "survey": "Pantheon+",
                    "observable": "distance modulus",
                    "units": "mag",
                    "reference": "Scolnic et al. 2022"
                }
            }

    # If no file found, return None
    return None
