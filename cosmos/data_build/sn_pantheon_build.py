"""
Pantheon+SH0ES standardized data builder.

This module handles the conversion of raw Pantheon+SH0ES data files into a
standardized format with proper covariance matrix handling.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Path configuration
RAW_DIR = Path("data/raw/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR")
DATA_FILE = "Pantheon+SH0ES.dat"
COV_FILE = "Pantheon+SH0ES_STAT+SYS.cov"
CACHE_PATH = Path("data/standardized/sn_pantheon.npz")

def _read_cov_rowmajor(cov_path: Path, N: int) -> np.ndarray:
    """
    Read a covariance matrix in row-major format with NxN elements.
    
    The file format is:
    - First line: single integer N (number of SNe)
    - Followed by N*N numbers in row-major order
    """
    with cov_path.open("rt") as f:
        # Read first non-empty line to get N
        first = None
        while first is None:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF before N header")
            toks = line.strip().split()
            if toks:
                first = int(float(toks[0]))
        
        if first != N:
            raise ValueError(f"Header N={first} but data has N={N}")

        # Read N*N floats (row-major), ignoring line breaks
        need = N * N
        buf = np.empty(need, dtype=np.float64)
        filled = 0
        
        for line in f:
            if filled >= need:
                break
            toks = line.strip().split()
            if not toks:
                continue
                
            # Convert tokens to floats and add to buffer
            vals = np.fromiter((float(t) for t in toks), dtype=np.float64)
            take = min(vals.size, need - filled)
            buf[filled:filled+take] = vals[:take]
            filled += take
            
        if filled != need:
            raise ValueError(f"Covariance has {filled} entries; expected {need}")

    # Reshape and enforce symmetry
    C = buf.reshape(N, N)
    return 0.5 * (C + C.T)  # Ensure exact symmetry

def build_sn_pantheon(cache_path: Path = CACHE_PATH) -> Dict[str, Any]:
    """
    Build standardized Pantheon+SH0ES dataset from raw files.
    
    Returns:
        Dictionary containing the standardized dataset
    """
    data_path = RAW_DIR / DATA_FILE
    cov_path = RAW_DIR / COV_FILE
    
    # Load data file (space-separated with header)
    raw = np.genfromtxt(data_path, dtype=np.float64, skip_header=1)
    
    # Extract columns
    z_cmb = raw[:, 4]     # CMB-frame redshift
    mu_shoes = raw[:, 10]  # MU_SH0ES (absolute, SH0ES-calibrated)
    mu_diag = raw[:, 11]   # MU_SH0ES_ERR_DIAG (for plotting only)
    
    N = z_cmb.size
    
    # Load full STAT+SYS covariance
    C = _read_cov_rowmajor(cov_path, N)
    
    # Calculate diagnostics
    diag = np.diag(C)
    med_diag = float(np.median(diag))
    med_err = float(np.median(np.sqrt(diag)))
    
    print(f"[sn] N={N}, cov diag median={med_diag:.3e}, median σ_μ={med_err:.3f} mag")
    
    # Create output directory if it doesn't exist
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save standardized dataset
    np.savez_compressed(
        cache_path,
        z=z_cmb.astype(np.float64),
        obs_abs=mu_shoes.astype(np.float64),  # absolute μ (SH0ES-anchored)
        err_diag=mu_diag.astype(np.float64),  # for plotting only
        cov=C.astype(np.float64),
        meta=np.array([{
            "source": "Pantheon+SH0ES",
            "reference": "Brout+ 2022; Riess+ 2021",
            "redshift_frame": "CMB",
            "covariance": "STAT+SYS (full)",
            "use_mode": "ABSOLUTE_MU_NO_MARG",
            "N": int(N),
        }], dtype=object),
    )
    
    print(f"[sn] Wrote standardized dataset to {cache_path}")
    
    return {
        "z": z_cmb,
        "obs_abs": mu_shoes,
        "cov": C,
        "N": N,
        "diag_med": med_diag,
        "sigma_med": med_err,
    }

if __name__ == "__main__":
    build_sn_pantheon()
