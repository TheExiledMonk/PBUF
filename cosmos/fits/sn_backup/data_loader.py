"""
Load supernova datasets (Pantheon+, SH0ES, JLA, etc.)

This module provides functions to load supernova distance modulus data
from CSV or Parquet files with optional covariance matrices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Default paths for supernova data
DEFAULT_SN_PATH = Path("data/supernovae/derived/supernova_index.csv")
DEFAULT_COV_PATH = Path("data/supernovae/derived/supernova_index.cov")

def load_sn_data(data_path=None, cov_path=None):
    """
    Load supernova dataset (z, mu, [optional covariance]).

    Parameters
    ----------
    data_path : str or Path, optional
        Path to supernova data file (CSV or Parquet).
        If None, search in default locations.
    cov_path : str or Path, optional
        Path to covariance matrix file.
        If None, search in default locations.

    Returns
    -------
    dict
        {
            "z": np.ndarray,      # redshifts
            "mu": np.ndarray,     # distance moduli
            "cov": np.ndarray | None,  # covariance matrix
            "n": int              # number of supernovae
        }

    Raises
    ------
    FileNotFoundError
        If data file cannot be found in search paths.
    ValueError
        If required columns are missing or data format is invalid.
    """
    # Search for data file if not provided
    if data_path is None:
        search_paths = [
            Path(__file__).parent.parent.parent.parent / DEFAULT_SN_PATH,
            Path.cwd() / DEFAULT_SN_PATH,
            Path.home() / "PBUF4" / DEFAULT_SN_PATH,
        ]

        data_path = None
        for p in search_paths:
            if p.exists():
                data_path = p
                break

        if data_path is None:
            raise FileNotFoundError(
                f"Could not find supernova data file. "
                f"Searched in: {[str(p) for p in search_paths]}"
            )
    else:
        data_path = Path(data_path)

    # Load the data
    try:
        if data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Failed to load {data_path}: {e}")

    # Validate required columns
    required_cols = ["redshift", "mu"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {data_path}: {missing_cols}")

    # Extract data
    z = df["redshift"].to_numpy(dtype=float)
    mu = df["mu"].to_numpy(dtype=float)
    n = len(z)

    # Basic data validation
    if n == 0:
        raise ValueError(f"No data found in {data_path}")

    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(mu)):
        raise ValueError(f"Non-finite values found in {data_path}")

    if np.any(z < 0):
        warnings.warn("Negative redshifts found in data")

    # Load covariance matrix if provided or available
    cov = None
    if cov_path is not None:
        cov_path = Path(cov_path)

    if cov_path is None or (cov_path is not None and cov_path.exists()):
        if cov_path is None:
            # Try default covariance path
            cov_search_paths = [
                data_path.parent / (data_path.stem + '.cov'),
                Path(__file__).parent.parent.parent.parent / DEFAULT_COV_PATH,
                Path.cwd() / DEFAULT_COV_PATH,
                Path.home() / "PBUF4" / DEFAULT_COV_PATH,
            ]

            for p in cov_search_paths:
                if p.exists():
                    cov_path = p
                    break

        if cov_path and cov_path.exists():
            try:
                if cov_path.suffix.lower() == '.npz':
                    cov_data = np.load(cov_path)
                    if 'cov' in cov_data:
                        cov = cov_data['cov']
                    else:
                        # Assume the whole file is the covariance matrix
                        cov = cov_data[cov_data.files[0]]
                else:
                    cov = np.loadtxt(cov_path)

                # Validate covariance matrix shape
                if cov.shape != (n, n):
                    warnings.warn(
                        f"Covariance matrix shape {cov.shape} doesn't match "
                        f"number of data points ({n}, {n}). Ignoring covariance."
                    )
                    cov = None
                else:
                    # Ensure symmetry
                    if not np.allclose(cov, cov.T, rtol=1e-10):
                        warnings.warn("Covariance matrix is not symmetric. Using as-is.")

            except Exception as e:
                warnings.warn(f"Failed to load covariance matrix {cov_path}: {e}. Continuing without covariance.")

    return {
        "z": z,
        "mu": mu,
        "cov": cov,
        "n": n
    }
