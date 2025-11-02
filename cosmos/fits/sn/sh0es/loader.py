"""
Load SH0ES supernova datasets.

This module provides functions to load SH0ES supernova distance modulus data
or use SH0ES as a Gaussian prior on H0.
"""

import numpy as np
from pathlib import Path

from cosmos.fits._dataset_loader import load_sh0es_dataset

# SH0ES prior values (from Riess et al. 2021)
SH0ES_H0_OBS = 73.04  # km/s/Mpc
SH0ES_H0_ERR = 1.04   # km/s/Mpc


def load_sh0es_data(data_path=None, cov_path=None, use_prior=True):
    """
    Load SH0ES supernova dataset or use as H0 prior.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to SH0ES data file (CSV or Parquet).
        If None and use_prior=False, search in default locations.
    cov_path : str or Path, optional
        Path to covariance matrix file.
        If None and use_prior=False, search in default locations.
    use_prior : bool, optional
        If True, use SH0ES as H0 Gaussian prior. If False, load actual SN data.

    Returns
    -------
    dict
        If use_prior=True:
        {
            "H0_obs": float,      # SH0ES H0 measurement
            "H0_err": float,      # SH0ES H0 uncertainty
            "n": int              # number of constraints (1 for prior)
        }

        If use_prior=False:
        {
            "z": np.ndarray,      # redshifts
            "mu": np.ndarray,     # distance moduli
            "cov": np.ndarray | None,  # covariance matrix
            "n": int              # number of supernovae
        }

    Raises
    ------
    FileNotFoundError
        If data file cannot be found in search paths and use_prior=False.
    ValueError
        If required columns are missing or data format is invalid.
    """
    if use_prior:
        return {
            "H0_obs": SH0ES_H0_OBS,
            "H0_err": SH0ES_H0_ERR,
            "n": 1
        }

    if data_path is not None:
        custom_path = Path(data_path)
        if not custom_path.exists():
            raise FileNotFoundError(f"Custom SH0ES dataset not found at {custom_path}")
        with np.load(custom_path, allow_pickle=True) as npz:
            dataset = {key: npz[key] for key in npz.files}
    else:
        dataset = load_sh0es_dataset()

    # Normalise common fields
    dataset = dict(dataset)
    dataset.setdefault("name", "SH0ES")
    dataset.setdefault("type", "SN")

    cov = dataset.get("cov")
    if dataset.get("err") is None and cov is not None:
        diag = np.clip(np.diag(cov), 0.0, None)
        dataset["err"] = np.sqrt(diag)

    if "obs" not in dataset:
        raise ValueError("Standardized SH0ES dataset missing 'obs' field")

    z = dataset.get("z")
    if z is None:
        raise ValueError("Standardized SH0ES dataset missing redshift array")

    dataset["n"] = int(len(z))
    dataset.setdefault("meta", {}).setdefault("survey", "SH0ES")
    dataset["meta"].setdefault("observable", "distance modulus")
    dataset["meta"].setdefault("units", "mag")
    dataset["meta"].setdefault("reference", "Riess et al. 2021")

    return dataset
