"""
BAO (Baryon Acoustic Oscillations) data loader.

Loads both isotropic and anisotropic BAO measurements from galaxy surveys,
providing D_V(z)/r_d, D_M(z)/r_d, and D_H(z)/r_d measurements.
"""

import numpy as np
from pathlib import Path
from data_interface.standardize import ensure_standard_dataset


STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
STANDARDIZED_BAO_FILES = {
    "BAO_ANISO": ("bao_aniso.npz", "bao_aniso_dr16.npz"),
    "BAO_ISO": ("bao_iso.npz", "bao_iso_dr16.npz"),
}


def _extract_scalar(array):
    """Return scalar value from zero-d numpy arrays."""
    if isinstance(array, np.ndarray) and array.shape == ():
        return array.item()
    return array


def _load_standardized_bao(expected_type: str) -> dict:
    """
    Load standardized BAO dataset from .npz cache.

    Parameters
    ----------
    expected_type : str
        "BAO_ANISO" or "BAO_ISO"

    Returns
    -------
    dict
        Dataset conforming to the PBUF Data Object v1 schema.
    """
    candidates = STANDARDIZED_BAO_FILES.get(expected_type, ())
    for filename in candidates:
        dataset_path = STANDARDIZED_DIR / filename
        if dataset_path.exists():
            with np.load(dataset_path, allow_pickle=True) as npz:
                meta = {}
                if "meta" in npz:
                    meta_entry = npz["meta"]
                    if isinstance(meta_entry, np.ndarray):
                        if meta_entry.shape == ():
                            meta = dict(meta_entry.item())
                        elif meta_entry.size == 1:
                            meta = dict(meta_entry.reshape(-1)[0])
                        else:
                            meta = dict(meta_entry)
                    else:
                        meta = dict(meta_entry)

                if "labels" in npz:
                    labels = npz["labels"]
                    meta.setdefault(
                        "labels",
                        labels.tolist() if isinstance(labels, np.ndarray) else labels,
                    )

                if "n_data" in npz:
                    n_data = _extract_scalar(npz["n_data"])
                    if n_data is not None:
                        meta.setdefault("n_data", int(n_data))

                cov = np.asarray(npz["cov"], dtype=float) if "cov" in npz else None
                err = None
                if "err" in npz:
                    err = np.asarray(npz["err"], dtype=float)
                elif cov is not None:
                    diag = np.clip(np.diag(cov), 0.0, None)
                    err = np.sqrt(diag)

                dataset = {
                    "name": str(_extract_scalar(npz["name"])) if "name" in npz else "BAO",
                    "type": expected_type,
                    "z": np.asarray(npz["z"], dtype=float) if "z" in npz else None,
                    "obs": np.asarray(npz["obs"], dtype=float) if "obs" in npz else None,
                    "err": err,
                    "cov": cov,
                    "meta": meta,
                }

                dataset = ensure_standard_dataset(dataset, expected_type)
                dataset["meta"].setdefault("source_file", dataset_path.name)
                return dataset

    search_paths = ", ".join(str(STANDARDIZED_DIR / name) for name in candidates)
    raise FileNotFoundError(
        f"Standardized BAO dataset not found for {expected_type}. "
        f"Expected one of: {search_paths}. "
        "Run data_interface.bao_raw_to_standardized to generate the cache."
    )


def load_bao_data():
    """
    Load anisotropic BAO data from standardized cache.

    Returns
    -------
    dict
        BAO data following PBUF Data Object v1 schema:
        {
            "name": str,             # dataset name
            "type": str,             # "BAO_ISO" or "BAO_ANISO"
            "z": np.ndarray,         # Redshift
            "obs": np.ndarray,       # BAO observables
            "err": np.ndarray,       # Uncertainty in observables
            "cov": np.ndarray | None,# covariance matrix (None for now)
            "meta": dict             # metadata
        }
    """
    return _load_standardized_bao("BAO_ANISO")


def load_bao_iso_data():
    """
    Load isotropic BAO data (D_V(z)/r_d measurements) from standardized cache.

    Returns
    -------
    dict
        Isotropic BAO data following PBUF Data Object v1 schema:
        {
            "name": str,             # dataset name
            "type": str,             # "BAO_ISO"
            "z": np.ndarray,         # Redshift
            "obs": np.ndarray,       # D_V(z)/r_d measurements
            "err": np.ndarray,       # Uncertainty in D_V(z)/r_d
            "cov": np.ndarray | None,# covariance matrix (None for now)
            "meta": dict             # metadata
        }
    """
    return _load_standardized_bao("BAO_ISO")
