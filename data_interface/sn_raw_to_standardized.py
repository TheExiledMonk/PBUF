"""
PBUF Data Loader — Pantheon+SH0ES Supernova Distance Moduli
=========================================================

Converts raw Pantheon+SH0ES supernova distance modulus data into standardized PBUF format.

This module loads the full covariance matrix and CMB-frame redshifts from the
Pantheon+SH0ES data release.

Outputs:
--------
    dict conforming to /docs/DEVELOPER_DATA_INTERFACE.md
    (validated by ensure_standard_dataset)

Usage:
------
    from data_interface.sn_raw_to_standardized import sn_raw_to_standardized
    sn_data = sn_raw_to_standardized()
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from hashlib import md5
from data_interface.standardize import ensure_standard_dataset

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATA_VERSION = "PantheonPlusSH0ES"
DATA_URL = "https://github.com/PantheonPlusSH0ES/DataRelease/tree/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"

# Paths
RAW_DIR = Path("data/raw/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR")
DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"
CACHE_PATH = Path("data/standardized/sn_pantheon_shoes.npz")


# ----------------------------------------------------------------------
# Utility: checksum
# ----------------------------------------------------------------------

def file_md5(path: Path) -> str:
    """Return MD5 checksum for a file."""
    h = md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# Utility: auto-download placeholder
# ----------------------------------------------------------------------

def ensure_raw_exists(data_path: Path, cov_path: Path) -> tuple[Path, Path]:
    """
    Ensure raw Pantheon+SH0ES data and covariance files exist locally.
    
    Parameters
    ----------
    data_path : Path
        Path to the data file
    cov_path : Path
        Path to the covariance file
        
    Returns
    -------
    tuple[Path, Path]
        Paths to data and covariance files
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Pantheon+SH0ES data file not found at {data_path}\n"
            f"Please download from {DATA_URL} and place in {data_path.parent}"
        )
    if not cov_path.exists():
        raise FileNotFoundError(
            f"Pantheon+SH0ES covariance file not found at {cov_path}\n"
            f"Please download from {DATA_URL} and place in {cov_path.parent}"
        )
        
    print(f"[info] Found Pantheon+SH0ES data at {data_path}")
    print(f"[info] Found covariance matrix at {cov_path}")
    return data_path, cov_path


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------

def load_pantheon_shoes_data(data_path: Path, cov_path: Path) -> dict:
    """
    Load and process Pantheon+SH0ES data.
    
    Parameters
    ----------
    data_path : Path
        Path to the data file
    cov_path : Path
        Path to the covariance file
        
    Returns
    -------
    dict
        Standardized dataset dictionary
    """
    # Load data file
    # Format: space-separated with header
    data = np.genfromtxt(data_path, skip_header=1)
    
    # Extract columns (see Pantheon+ data release documentation)
    z_CMB = data[:, 4]   # CMB-frame redshift (column 5, 0-indexed as 4)
    z_err = data[:, 5]    # CMB redshift error (column 6)
    mu = data[:, 10]      # Distance modulus (MU_SH0ES, column 11)
    mu_err = data[:, 11]  # Diagonal errors (MU_SH0ES_ERR_DIAG, column 12)
    
    # Get number of supernovae from the data
    n_sne = len(z_CMB)
    
    try:
        # Read the full covariance matrix from the text file
        # The file contains N x N elements in row-major order
        cov_flat = np.loadtxt(cov_path)
        
        # The file might have an extra element due to trailing newline
        expected_size = n_sne * n_sne
        if len(cov_flat) == expected_size + 1:
            print(f"[info] Trimming extra element from covariance file")
            cov_flat = cov_flat[:-1]  # Remove the last element
        elif len(cov_flat) != expected_size:
            raise ValueError(
                f"Covariance file contains {len(cov_flat)} elements, "
                f"expected {expected_size} or {expected_size + 1} for {n_sne} SNe"
            )
        
        # Reshape to 2D matrix
        full_cov = cov_flat.reshape((n_sne, n_sne)).astype(np.float64)
        
        # Ensure the matrix is symmetric (within numerical precision)
        if not np.allclose(full_cov, full_cov.T, atol=1e-10):
            print("[warn] Covariance matrix is not symmetric. Symmetrizing...")
            full_cov = 0.5 * (full_cov + full_cov.T)
            
        # Keep the covariance matrix as-is, no scaling applied
        print("[info] Keeping covariance matrix in original units")
        
        # Verify the scale
        median_diag = np.median(np.diag(full_cov))
        median_err = np.median(mu_err) if mu_err is not None else np.sqrt(median_diag)
        print(f"[info] Final covariance check - median diagonal: {median_diag:.2e}, "
              f"median error: {median_err:.2e}")
            
    except Exception as e:
        print(f"[warn] Error loading covariance matrix: {e}")
        print("[warn] Falling back to diagonal covariance with errors from data file")
        full_cov = np.diag(mu_err ** 2)
        
    # Add a small diagonal term for numerical stability if needed
    min_eig = np.min(np.real(np.linalg.eigvals(full_cov)))
    if min_eig <= 0:
        print(f"[warn] Covariance matrix is not positive definite (min eigenvalue: {min_eig:.2e}). Adding diagonal term.")
        full_cov += np.eye(n_sne) * (1e-6 * np.max(np.diag(full_cov)) - min_eig)
    
    return {
        "z": z_CMB,
        "z_err": z_err,
        "obs": mu,
        "err": mu_err,
        "cov": full_cov,
        "n_data": n_sne
    }

def sn_raw_to_standardized(source_dir: str | Path | None = None) -> dict:
    """
    Convert raw Pantheon+SH0ES supernova data into standardized PBUF format.

    Parameters
    ----------
    source_dir : str or Path, optional
        Directory containing Pantheon+SH0ES data files. If None, uses default location.

    Returns
    -------
    dict
        Standardized dataset dictionary with full covariance matrix.
    """
    if source_dir is None:
        source_dir = RAW_DIR
    
    source_dir = Path(source_dir)
    data_path = source_dir / DATA_FILE
    cov_path = source_dir / COV_FILE
    
    # Verify files exist
    data_path, cov_path = ensure_raw_exists(data_path, cov_path)
    
    # Load and process data
    data = load_pantheon_shoes_data(data_path, cov_path)
    
    # Create standardized dataset
    standardized = {
        "name": "Pantheon+SH0ES",
        "type": "SN",
        "z": data["z"],
        "obs": data["obs"],
        "err": data["err"],
        "cov": data["cov"],
        "meta": {
            "units": {
                "z": "dimensionless",
                "obs": "mag",
                "err": "mag",
                "cov": "mag^2"
            },
            "source": "Pantheon+SH0ES supernova compilation",
            "version": DATA_VERSION,
            "reference": "Brout et al. 2022, ApJ, 938, 113",
            "redshift_frame": "CMB",
            "covariance_type": "STAT+SYS",
            "n_data": data["n_data"],
            "data_file": DATA_FILE,
            "cov_file": COV_FILE
        },
    }
    
    # Validate against schema
    standardized = ensure_standard_dataset(standardized, "SN")
    
    # Cache standardized data
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        z=standardized["z"],
        obs=standardized["obs"],
        err=standardized["err"],
        cov=standardized["cov"],
        meta=np.array([standardized["meta"]]),
        n_data=np.array([standardized["meta"]["n_data"]])
    )
    print(f"[ok] Saved standardized Pantheon+SH0ES data to {CACHE_PATH}")
    
    # Verify the saved file
    with np.load(CACHE_PATH, allow_pickle=True) as f:
        print(f"\nVerifying saved data:")
        print(f"  z: {f['z'].shape} points from {f['z'][0]:.4f} to {f['z'][-1]:.4f}")
        print(f"  obs: {f['obs'].shape}, mean = {f['obs'].mean():.2f} ± {f['obs'].std():.2f}")
        print(f"  cov: {f['cov'].shape}, diagonal: {np.diag(f['cov'])[:3]}...")
    
    return standardized


# ----------------------------------------------------------------------
# CLI test mode
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Converting Pantheon+SH0ES supernova data → standardized PBUF format...")
    try:
        sn_data = sn_raw_to_standardized()
        print("\n✅ Conversion successful!")
        print("\nDataset summary:")
        print(f"  Name: {sn_data['name']}")
        print(f"  Type: {sn_data['type']}")
        print(f"  Redshifts: {len(sn_data['z'])} points from {sn_data['z'][0]:.3f} to {sn_data['z'][-1]:.3f}")
        print(f"  Distance moduli: mean = {sn_data['obs'].mean():.2f} ± {sn_data['obs'].std():.2f} mag")
        print(f"  Errors: mean = {sn_data['err'].mean():.3f} mag, min = {sn_data['err'].min():.3f}, max = {sn_data['err'].max():.3f}")
        print(f"  Covariance: {sn_data['cov'].shape}, diagonal: {np.diag(sn_data['cov'])[:3]}...")
        
        # Check covariance matrix properties
        cov = sn_data['cov']
        diag = np.diag(cov)
        off_diag = cov - np.diag(diag)
        print("\nCovariance matrix properties:")
        print(f"  Diagonal mean: {diag.mean():.3e} ± {diag.std():.3e} mag²")
        print(f"  Off-diagonal mean: {off_diag.mean():.3e} (abs mean: {np.abs(off_diag).mean():.3e})")
        print(f"  Condition number: {np.linalg.cond(cov):.3e}")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        raise
