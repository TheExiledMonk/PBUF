"""
PBUF Data Loader — Cosmic Chronometer H(z) Measurements
=======================================================

Converts raw cosmic chronometer H(z) data into standardized PBUF format.

This serves as the canonical example for all <dataset>_raw_to_standardized.py
files in /data_interface.

Outputs:
--------
    dict conforming to /docs/DEVELOPER_DATA_INTERFACE.md
    (validated by ensure_standard_dataset)

Usage:
------
    from data_interface.cc_raw_to_standardized import cc_raw_to_standardized
    cc_data = cc_raw_to_standardized()
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from hashlib import md5
from standardize import ensure_standard_dataset

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATA_VERSION = "CC_compilation"
DATA_URL = "https://arxiv.org/abs/1207.4541"  # Moresco et al. 2012 compilation

CHECKSUM_EXPECTED = "e2i65f1795e6d7a7b87a72bb7c00a43e"  # dummy example

RAW_PATH = Path("data/cc/cc_data.csv")
CACHE_PATH = Path("data/standardized/cc_compilation.npz")


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

def ensure_raw_exists(source_path: Path) -> Path:
    """
    Ensure raw cosmic chronometer file exists locally.
    Future versions may implement automatic download from literature compilation.
    """
    if not source_path.exists():
        print(f"[warn] Raw cosmic chronometer file not found at {source_path}")
        print(f"[note] Auto-download not yet implemented.")
        print(f"[note] Please place the cosmic chronometer H(z) data at:")
        print(f"       {source_path}")
    else:
        checksum = file_md5(source_path)
        print(f"[info] Found cosmic chronometer data at {source_path}")
        print(f"[info] MD5 checksum: {checksum}")
        if CHECKSUM_EXPECTED and checksum != CHECKSUM_EXPECTED:
            print(f"[warn] Checksum mismatch (expected {CHECKSUM_EXPECTED})")
    return source_path


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------

def cc_raw_to_standardized(source_path: str | None = None) -> dict:
    """
    Convert raw cosmic chronometer H(z) data into standardized PBUF format.

    Parameters
    ----------
    source_path : str or None
        Optional local file path. If None, default path is used.

    Returns
    -------
    dict
        Standardized dataset dictionary for cosmic chronometer measurements.
    """
    if source_path is None:
        source_path = RAW_PATH

    source_path = Path(source_path)
    ensure_raw_exists(source_path)

    # Load cosmic chronometer data
    df = pd.read_csv(source_path)

    z = df["z"].values
    Hz_obs = df["Hz"].values
    Hz_err = df["sigma_Hz"].values

    # No covariance matrix provided in raw data, use diagonal from errors
    cov = np.diag(Hz_err**2)

    data = {
        "name": "CC_compilation",
        "type": "CC",
        "z": z,
        "obs": Hz_obs,
        "err": Hz_err,
        "cov": cov,
        "meta": {
            "units": "km/s/Mpc",
            "source": "Cosmic chronometer compilation (Moresco et al. 2012)",
            "version": DATA_VERSION,
            "reference": "Moresco et al. 2012, JCAP, 2012, 006",
            "checksum": CHECKSUM_EXPECTED,
        },
    }

    standardized = ensure_standard_dataset(data, "CC")

    # ------------------------------------------------------------------
    # Cache standardized data
    # ------------------------------------------------------------------
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH,
        obs=standardized["obs"],
        err=standardized["err"],
        cov=standardized["cov"],
        z=standardized["z"],
        meta=np.array([standardized["meta"]], dtype=object),
    )
    print(f"[ok] Saved standardized cosmic chronometer data to {CACHE_PATH}")

    return standardized


# ----------------------------------------------------------------------
# CLI test mode
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Converting cosmic chronometer H(z) data → standardized PBUF format...")
    cc_data = cc_raw_to_standardized()
    print("\n✅ Conversion complete.")
    print(f"  Name: {cc_data['name']}")
    print(f"  Type: {cc_data['type']}")
    print(f"  Redshifts: {len(cc_data['z'])} points from {cc_data['z'][0]:.3f} to {cc_data['z'][-1]:.3f}")
    print(f"  Observables shape: {cc_data['obs'].shape}")
    print(f"  Errors shape: {cc_data['err'].shape}")
    print(f"  Covariance shape: {cc_data['cov'].shape}")
    print(f"  Metadata: {cc_data['meta']}")
