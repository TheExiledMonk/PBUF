"""
PBUF Data Loader — SDSS DR16 BAO Measurements
=============================================

Converts raw SDSS DR16 BAO data into standardized PBUF format.

This serves as the canonical example for all <dataset>_raw_to_standardized.py
files in /data_interface.

Supports both isotropic (D_V(z)/r_d) and anisotropic (D_M(z)/r_d, D_H(z)/r_d)
BAO measurements from SDSS DR16.

Outputs:
--------
    dict conforming to /docs/DEVELOPER_DATA_INTERFACE.md
    (validated by ensure_standard_dataset)

Usage:
------
    from data_interface.bao_raw_to_standardized import bao_aniso_raw_to_standardized
    bao_data = bao_aniso_raw_to_standardized()

    from data_interface.bao_raw_to_standardized import bao_iso_raw_to_standardized
    bao_data = bao_iso_raw_to_standardized()
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from hashlib import md5
from data_interface.standardize import ensure_standard_dataset
from cosmos.fits.bao.aniso.data_loader import load_bao_aniso_data

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DATA_VERSION = "DR16"
DATA_URL = "https://sdss.org/science/data/"  # placeholder (SDSS archive)

CHECKSUM_EXPECTED_ANISO = "a8f32f1795e6d7a7b87a72bb7c00a43b"  # dummy example
CHECKSUM_EXPECTED_ISO = "c9g43f1795e6d7a7b87a72bb7c00a43c"    # dummy example

RAW_PATH_ANISO = Path("data/bao/derived/bao_aniso.csv")
RAW_PATH_ISO = Path("data/bao/derived/bao_iso.csv")
CACHE_PATH_ANISO = Path("data/standardized/bao_aniso_dr16.npz")
CACHE_PATH_ISO = Path("data/standardized/bao_iso_dr16.npz")


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

def ensure_raw_exists(source_path: Path, data_type: str) -> Path:
    """
    Ensure raw BAO file exists locally.
    Future versions may implement automatic download from SDSS archive.
    """
    if not source_path.exists():
        print(f"[warn] Raw BAO {data_type} file not found at {source_path}")
        print(f"[note] Auto-download not yet implemented.")
        print(f"[note] Please place the BAO data file at:")
        print(f"       {source_path}")
    else:
        checksum = file_md5(source_path)
        print(f"[info] Found BAO {data_type} data at {source_path}")
        print(f"[info] MD5 checksum: {checksum}")
        expected = CHECKSUM_EXPECTED_ANISO if "aniso" in data_type else CHECKSUM_EXPECTED_ISO
        if expected and checksum != expected:
            print(f"[warn] Checksum mismatch (expected {expected})")
    return source_path


# ----------------------------------------------------------------------
# Conversion: Anisotropic BAO
# ----------------------------------------------------------------------

def bao_aniso_raw_to_standardized(source_path: str | None = None) -> dict:
    """
    Convert raw SDSS DR16 anisotropic BAO data into standardized PBUF format.

    Parameters
    ----------
    source_path : str or None
        Optional local file path. If None, default path is used.

    Returns
    -------
    dict
        Standardized dataset dictionary for anisotropic BAO measurements.
    """
    if source_path is None:
        source_path = RAW_PATH_ANISO

    source_path = Path(source_path)
    ensure_raw_exists(source_path, "anisotropic BAO")

    raw_data = load_bao_aniso_data(source_path)

    z = raw_data["z"]
    DM_rd = raw_data["DM_over_rd"]
    DH_rd = raw_data["DH_over_rd"]
    cov_blocks = raw_data["cov_list"]
    n_points = raw_data["n"]
    if n_points != len(z):
        raise ValueError(
            "Internal inconsistency in anisotropic BAO loader: length mismatch between z and covariance."
        )

    # Interleave measurements: [DM1, H1, DM2, H2, DM3, H3]
    obs_interleaved = np.empty((2 * len(z),), dtype=float)
    obs_interleaved[0::2] = DM_rd  # Even indices: DM measurements
    obs_interleaved[1::2] = DH_rd  # Odd indices: D_H measurements

    # Assemble per-bin covariance into a block matrix ordered like obs_interleaved.
    if cov_blocks:
        cov = np.zeros((2 * len(z), 2 * len(z)), dtype=float)
        for i, block in enumerate(cov_blocks):
            if block.shape != (2, 2):
                raise ValueError(f"Unexpected anisotropic BAO covariance block shape: {block.shape}")
            cov_idx = slice(2 * i, 2 * i + 2)
            cov[cov_idx, cov_idx] = block
    else:
        cov = None

    if cov is not None:
        diag = np.clip(np.diag(cov), 0.0, None)
        err_interleaved = np.sqrt(diag)
    else:
        # Fall back to using the diagonal terms from per-bin blocks if cov was not built.
        DM_err = np.array([np.sqrt(block[0, 0]) for block in cov_blocks], dtype=float)
        DH_err = np.array([np.sqrt(block[1, 1]) for block in cov_blocks], dtype=float)
        err_interleaved = np.empty((2 * len(z),), dtype=float)
        err_interleaved[0::2] = DM_err
        err_interleaved[1::2] = DH_err

    data = {
        "name": "BAO_DR16_aniso",
        "type": "BAO_ANISO",
        "z": z,
        "obs": obs_interleaved,
        "err": err_interleaved,
        "cov": cov,
        "meta": {
            "units": "dimensionless",
            "source": "SDSS DR16 BAO anisotropic measurements",
            "version": DATA_VERSION,
            "reference": "Alam et al. 2021, MNRAS, 504, 3309",
            "checksum": CHECKSUM_EXPECTED_ANISO,
            "observable": "D_M(z)/r_d and D_H(z)/r_d",
            "ordering": "obs interleaved as [D_M/r_d, D_H/r_d] per redshift",
        },
    }

    standardized = ensure_standard_dataset(data, "BAO_ANISO")

    # ------------------------------------------------------------------
    # Cache standardized data
    # ------------------------------------------------------------------
    CACHE_PATH_ANISO.parent.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "obs": standardized["obs"],
        "err": standardized["err"],
        "z": standardized["z"],
        "meta": np.array([standardized["meta"]], dtype=object),
    }
    if standardized["cov"] is not None:
        npz_payload["cov"] = standardized["cov"]

    np.savez(CACHE_PATH_ANISO, **npz_payload)
    print(f"[ok] Saved standardized BAO anisotropic data to {CACHE_PATH_ANISO}")

    return standardized


# ----------------------------------------------------------------------
# Conversion: Isotropic BAO
# ----------------------------------------------------------------------

def bao_iso_raw_to_standardized(source_path: str | None = None) -> dict:
    """
    Convert raw SDSS DR16 isotropic BAO data into standardized PBUF format.

    Parameters
    ----------
    source_path : str or None
        Optional local file path. If None, default path is used.

    Returns
    -------
    dict
        Standardized dataset dictionary for isotropic BAO measurements.
    """
    if source_path is None:
        source_path = RAW_PATH_ISO

    source_path = Path(source_path)
    ensure_raw_exists(source_path, "isotropic BAO")

    # Load isotropic BAO data
    df = pd.read_csv(source_path)

    z = df["z"].values
    DV_rd = df["DV_div_rd"].values
    DV_err = df["sigma_DV_div_rd"].values

    data = {
        "name": "BAO_DR16_iso",
        "type": "BAO_ISO",
        "z": z,
        "obs": DV_rd,
        "err": DV_err,
        "cov": None,  # No covariance matrix provided in raw data
        "meta": {
            "units": "dimensionless",
            "source": "SDSS DR16 BAO isotropic measurements",
            "version": DATA_VERSION,
            "reference": "Alam et al. 2021, MNRAS, 504, 3309",
            "checksum": CHECKSUM_EXPECTED_ISO,
        },
    }

    standardized = ensure_standard_dataset(data, "BAO_ISO")

    # ------------------------------------------------------------------
    # Cache standardized data
    # ------------------------------------------------------------------
    CACHE_PATH_ISO.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH_ISO,
        obs=standardized["obs"],
        err=standardized["err"],
        z=standardized["z"],
        meta=np.array([standardized["meta"]], dtype=object),
    )
    print(f"[ok] Saved standardized BAO isotropic data to {CACHE_PATH_ISO}")

    return standardized


# ----------------------------------------------------------------------
# CLI test mode
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Converting SDSS DR16 BAO data → standardized PBUF format...")

    print("\n=== Anisotropic BAO ===")
    bao_aniso_data = bao_aniso_raw_to_standardized()
    print("✅ Anisotropic conversion complete.")
    print(f"  Name: {bao_aniso_data['name']}")
    print(f"  Type: {bao_aniso_data['type']}")
    print(f"  Redshifts: {bao_aniso_data['z']}")
    print(f"  Observables shape: {bao_aniso_data['obs'].shape}")
    print(f"  Errors shape: {bao_aniso_data['err'].shape}")
    print(f"  Metadata: {bao_aniso_data['meta']}")

    print("\n=== Isotropic BAO ===")
    bao_iso_data = bao_iso_raw_to_standardized()
    print("✅ Isotropic conversion complete.")
    print(f"  Name: {bao_iso_data['name']}")
    print(f"  Type: {bao_iso_data['type']}")
    print(f"  Redshifts: {bao_iso_data['z']}")
    print(f"  Observables shape: {bao_iso_data['obs'].shape}")
    print(f"  Errors shape: {bao_iso_data['err'].shape}")
    print(f"  Metadata: {bao_iso_data['meta']}")
