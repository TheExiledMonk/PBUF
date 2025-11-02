"""
PBUF Data Loader — Redshift Space Distortion fσ8 Measurements
=============================================================

Converts raw redshift space distortion fσ8 data into standardized PBUF format.

This serves as the canonical example for all <dataset>_raw_to_standardized.py
files in /data_interface.

Outputs:
--------
    dict conforming to /docs/DEVELOPER_DATA_INTERFACE.md
    (validated by ensure_standard_dataset)

Usage:
------
    from data_interface.rsd_raw_to_standardized import rsd_raw_to_standardized
    rsd_data = rsd_raw_to_standardized()
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

DATA_VERSION = "RSD_compilation"
DATA_URL = "https://arxiv.org/abs/1203.0302"  # Song & Percival 2012 compilation

CHECKSUM_EXPECTED = "f3j76f1795e6d7a7b87a72bb7c00a43f"  # dummy example

RAW_PATH = Path("data/rsd/rsd_data.csv")
CACHE_PATH = Path("data/standardized/rsd_compilation.npz")


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
    Ensure raw RSD file exists locally.
    Future versions may implement automatic download from literature compilation.
    """
    if not source_path.exists():
        print(f"[warn] Raw RSD file not found at {source_path}")
        print(f"[note] Auto-download not yet implemented.")
        print(f"[note] Please place the RSD fσ8 data at:")
        print(f"       {source_path}")
    else:
        checksum = file_md5(source_path)
        print(f"[info] Found RSD data at {source_path}")
        print(f"[info] MD5 checksum: {checksum}")
        if CHECKSUM_EXPECTED and checksum != CHECKSUM_EXPECTED:
            print(f"[warn] Checksum mismatch (expected {CHECKSUM_EXPECTED})")
    return source_path


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------

def rsd_raw_to_standardized(source_path: str | None = None) -> dict:
    """
    Convert raw RSD fσ8 data into standardized PBUF format.

    Parameters
    ----------
    source_path : str or None
        Optional local file path. If None, default path is used.

    Returns
    -------
    dict
        Standardized dataset dictionary for RSD measurements.
    """
    if source_path is None:
        source_path = RAW_PATH

    source_path = Path(source_path)
    ensure_raw_exists(source_path)

    # Load RSD data
    df = pd.read_csv(source_path)

    z = df["z"].values
    fsigma8_obs = df["fsigma8"].values
    fsigma8_err = df["sigma_fsigma8"].values

    # No covariance matrix provided in raw data, use diagonal from errors
    cov = np.diag(fsigma8_err**2)

    data = {
        "name": "RSD_compilation",
        "type": "RSD",
        "z": z,
        "obs": fsigma8_obs,
        "err": fsigma8_err,
        "cov": cov,
        "meta": {
            "units": "dimensionless",
            "source": "RSD compilation (Song & Percival 2012)",
            "version": DATA_VERSION,
            "reference": "Song & Percival 2012, JCAP, 2012, 010",
            "checksum": CHECKSUM_EXPECTED,
        },
    }

    standardized = ensure_standard_dataset(data, "RSD")

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
    print(f"[ok] Saved standardized RSD data to {CACHE_PATH}")

    return standardized


# ----------------------------------------------------------------------
# CLI test mode
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Converting RSD fσ8 data → standardized PBUF format...")
    rsd_data = rsd_raw_to_standardized()
    print("\n✅ Conversion complete.")
    print(f"  Name: {rsd_data['name']}")
    print(f"  Type: {rsd_data['type']}")
    print(f"  Redshifts: {len(rsd_data['z'])} points from {rsd_data['z'][0]:.3f} to {rsd_data['z'][-1]:.3f}")
    print(f"  Observables shape: {rsd_data['obs'].shape}")
    print(f"  Errors shape: {rsd_data['err'].shape}")
    print(f"  Covariance shape: {rsd_data['cov'].shape}")
    print(f"  Metadata: {rsd_data['meta']}")
