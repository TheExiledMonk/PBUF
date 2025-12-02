"""Dataset registry mirroring cosmos.datasets without importing cosmos/."""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict

Dataset = Any
DATA_ROOT = Path("data/standardized")


def _load_npz_standardized(*names: str) -> Dict[str, Any]:
    """
    Load a standardized NPZ payload, trying a list of candidate basenames.
    Mirrors cosmos.datasets get_dataset behavior but stays cosmos-free.
    """
    tried: list[Path] = []
    for name in names:
        path = DATA_ROOT / f"{name}.npz"
        tried.append(path)
        if path.exists():
            payload = np.load(path, allow_pickle=True)
            return {key: payload[key] for key in payload.files}
    tried_str = ", ".join(str(p) for p in tried)
    raise FileNotFoundError(f"Standardized dataset not found (tried {tried_str})")


def _ensure_inv_cov(dataset: Dict[str, Any]) -> Dict[str, Any]:
    cov = dataset.get("cov")
    err = dataset.get("err")
    if "inv_cov" in dataset:
        return dataset
    if cov is not None:
        dataset["inv_cov"] = np.linalg.inv(np.asarray(cov, dtype=float))
    elif err is not None:
        errs = np.asarray(err, dtype=float)
        dataset["inv_cov"] = np.diag(1.0 / (errs * errs))
    return dataset


def load_cmb() -> Dict[str, Any]:
    return _load_npz_standardized("cmb")


def load_sn() -> Dict[str, Any]:
    # Prefer the Pantheon(+SH0ES) cache, fall back through other standardized names.
    return _load_npz_standardized(
        "sn_pantheon",
        "sn_pantheon_full",
        "sn_pantheonplus",
        "sn_pantheon_shoes",
        "sn",
    )


def load_sh0es() -> Dict[str, Any]:
    return _load_npz_standardized("sh0es")


def load_bao_iso() -> Dict[str, Any]:
    return _load_npz_standardized("bao_iso")


def load_bao_aniso() -> Dict[str, Any]:
    return _load_npz_standardized("bao_aniso")


def load_cc() -> Dict[str, Any]:
    return _load_npz_standardized("cc")


def load_rsd() -> Dict[str, Any]:
    return _load_npz_standardized("rsd")


def load_weak_lensing_kids1000() -> Dict[str, Any]:
    return _load_npz_standardized("weak_lensing_kids1000_raw_v1")


def load_wl_s8() -> Dict[str, Any]:
    return _load_npz_standardized("wl_s8")


def load_lensing_cross() -> Dict[str, Any]:
    return _load_npz_standardized("lensing_cross")


def load_galaxy_pk() -> Dict[str, Any]:
    data = _load_npz_standardized("galaxy_pk", "galaxy_pks")
    # Normalize fiducial keys that sometimes arrive as separate arrays
    fiducials = data.get("fiducials")
    if isinstance(fiducials, np.ndarray):
        fiducials = fiducials.item()
    if fiducials is None:
        fiducials = {}
    if "DM_fid" in data and "DM" not in fiducials:
        fiducials = dict(fiducials)
        fiducials["DM"] = data["DM_fid"]
    if "H_fid" in data and "H" not in fiducials:
        fiducials = dict(fiducials)
        fiducials["H"] = data["H_fid"]
    if fiducials:
        data["fiducials"] = fiducials
    return data


_LOADERS: Dict[str, Callable[[], Dataset]] = {
    "cmb": load_cmb,
    "sn": load_sn,
    "sn_pantheon": load_sn,
    "sn_pantheonplus": load_sn,
    "sn_pantheon_shoes": load_sn,
    "sh0es": load_sh0es,
    "bao_iso": load_bao_iso,
    "bao_aniso": load_bao_aniso,
    "cc": load_cc,
    "rsd": load_rsd,
    "wl_s8": load_wl_s8,
    "lensing_cross": load_lensing_cross,
    "lensing_x": load_lensing_cross,
    "galaxy_pk": load_galaxy_pk,
    "weak_lensing_kids1000": load_weak_lensing_kids1000,
    "wl_kids1000": load_weak_lensing_kids1000,
}


@lru_cache(maxsize=None)
def get_dataset(name: str) -> Dataset:
    normalized = name.strip().lower()

    # Check if we have jackknife masked data first
    from cosmos2.api.engine import _jackknife_masked_datasets
    if normalized in _jackknife_masked_datasets:
        dataset = _jackknife_masked_datasets[normalized]
        size = len(dataset.get('z', dataset.get('data', [])))
        print(f"[jackknife] get_dataset: Using masked dataset for {normalized} ({size} points)")
        if isinstance(dataset, dict):
            dataset = _ensure_inv_cov(dataset)
        return dataset

    # Fall back to normal loading
    loader = _LOADERS.get(normalized)
    if loader is None:
        raise ValueError(f"Dataset '{name}' is not supported.")
    dataset = loader()
    if isinstance(dataset, dict):
        dataset = _ensure_inv_cov(dataset)
    return dataset


__all__ = ["get_dataset", "Dataset"]
