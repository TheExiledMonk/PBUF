"""
Load Pantheon+SH0ES supernova datasets.

This module provides functions to load standardized Pantheon+SH0ES supernova data
with absolute magnitudes and full covariance matrices.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from cosmos.fits._dataset_loader import load_sn_dataset

def load_pantheon_data(data_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Load standardized Pantheon+SH0ES supernova dataset.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to standardized .npz file. If None, uses default location.

    Returns
    -------
    dict
        Standardized dataset with keys:
        - 'z': CMB-frame redshifts (ndarray)
        - 'obs_abs': Absolute distance moduli (SH0ES-calibrated, ndarray)
        - 'err_diag': Diagonal errors (for plotting, ndarray)
        - 'cov': Full STAT+SYS covariance matrix (ndarray, NxN)
        - 'meta': Metadata dictionary with dataset information
        - 'n_data': Number of supernovae (int)

    Raises
    ------
    FileNotFoundError
        If data file cannot be found.
    ValueError
        If the data format is invalid.
    """
    # Use default path if not specified
    if data_path is not None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Standardized Pantheon dataset not found at {data_path}")
        with np.load(data_path, allow_pickle=True) as data:
            dataset = {k: data[k] for k in data.files}
    else:
        dataset = load_sn_dataset()

    dataset = dict(dataset)
    dataset.setdefault("name", "Pantheon+")
    dataset.setdefault("type", "SN")

    # Normalise metadata if stored as numpy objects
    meta = dataset.get("meta")
    if isinstance(meta, np.ndarray):
        if meta.shape == ():
            dataset["meta"] = meta.item() if isinstance(meta.item(), dict) else {}
        elif meta.size >= 1:
            first = meta.reshape(-1)[0]
            dataset["meta"] = first if isinstance(first, dict) else {"raw_meta": first}
        else:
            dataset["meta"] = {}
    elif meta is None:
        dataset["meta"] = {}
    elif not isinstance(meta, dict):
        dataset["meta"] = {"raw_meta": meta}

    if "obs_abs" not in dataset:
        if "obs" not in dataset:
            raise ValueError("Pantheon dataset missing 'obs' field")
        dataset["obs_abs"] = np.array(dataset["obs"], dtype=float, copy=True)
    else:
        dataset["obs_abs"] = np.array(dataset["obs_abs"], dtype=float, copy=True)

    # Ensure relative-observable view remains available for chi2_sn_pantheon
    if "obs" not in dataset or dataset["obs"] is None:
        dataset["obs"] = np.array(dataset["obs_abs"], dtype=float, copy=True)
    else:
        dataset["obs"] = np.array(dataset["obs"], dtype=float, copy=True)

    # Provide diagonal uncertainties when covariance available (for diagnostics/fallbacks)
    if dataset.get("cov") is not None:
        cov = np.asarray(dataset["cov"], dtype=float)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(f"Pantheon covariance matrix has invalid shape {cov.shape}")
        dataset["cov"] = cov
        dataset["err"] = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    elif dataset.get("err") is not None:
        dataset["err"] = np.asarray(dataset["err"], dtype=float)
    else:
        dataset["err"] = None

    required = ["z", "cov", "obs_abs"]
    for field in required:
        if field not in dataset:
            raise ValueError(f"Missing required field '{field}' in dataset")

    if dataset.get("n_data") is None:
        dataset["n_data"] = len(dataset["z"])

    return dataset
