"""
Helpers for loading standardized datasets from cached ``.npz`` files.

All fitting modules must rely on these helpers (rather than raw CSV loaders)
to keep the data access layer consistent and reproducible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence, Dict, Any, Optional

import numpy as np

STANDARDIZED_DIR = Path("data/standardized")


def _extract_scalar(value: Any) -> Any:
    """Return a Python scalar from a numpy container if possible."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _extract_scalar(value.item())
        if value.size == 1:
            return _extract_scalar(value.reshape(-1)[0])
    return value


def _extract_meta(meta: Any) -> Dict[str, Any]:
    """Coerce metadata payloads into a dictionary."""
    if meta is None:
        return {}
    meta = _extract_scalar(meta)
    if isinstance(meta, Mapping):
        return dict(meta)
    return {"raw_meta": meta}


def _extract_labels(labels: Any) -> Optional[Sequence[Any]]:
    """Return labels as a simple Python sequence if they exist."""
    if labels is None:
        return None
    if isinstance(labels, np.ndarray):
        return labels.tolist()
    if isinstance(labels, (list, tuple)):
        return list(labels)
    return [labels]


def _load_npz_file(path: Path) -> Dict[str, Any]:
    """Load an npz file into a plain dictionary."""
    with np.load(path, allow_pickle=True) as npz:
        dataset = {key: npz[key] for key in npz.files}
    dataset.setdefault("name", path.stem)
    dataset["meta"] = _extract_meta(dataset.get("meta"))
    labels = _extract_labels(dataset.get("labels"))
    if labels is not None:
        dataset["labels"] = labels
    n_data = _extract_scalar(dataset.get("n_data"))
    if n_data is not None:
        dataset["n_data"] = int(n_data)
    else:
        dataset.pop("n_data", None)
    return dataset


def load_standardized_npz(
    candidate_filenames: Sequence[str],
    dataset_type: str,
) -> Dict[str, Any]:
    """
    Load the first available standardized dataset from ``candidate_filenames``.

    Parameters
    ----------
    candidate_filenames:
        Ordered list of filenames to probe within ``data/standardized``.
    dataset_type:
        Expected dataset type identifier (e.g. ``CC``, ``SN``).

    Returns
    -------
    dict
        Dataset dictionary containing numpy arrays and metadata.

    Raises
    ------
    FileNotFoundError
        If none of the candidate files exist.
    """
    for filename in candidate_filenames:
        path = STANDARDIZED_DIR / filename
        if path.exists():
            dataset = _load_npz_file(path)
            dataset["type"] = dataset_type
            dataset.setdefault("name", path.stem)
            return dataset

    search_paths = ", ".join(str(STANDARDIZED_DIR / name) for name in candidate_filenames)
    raise FileNotFoundError(
        f"Standardized dataset not found for {dataset_type}. "
        f"Expected one of: {search_paths}. "
        "Run the appropriate conversion command to generate the standardized cache."
    )


def load_cc_dataset() -> Dict[str, Any]:
    return load_standardized_npz(
        ("cc.npz", "cc_compilation.npz"),
        dataset_type="CC",
    )


def load_rsd_dataset() -> Dict[str, Any]:
    return load_standardized_npz(
        ("rsd.npz", "rsd_compilation.npz"),
        dataset_type="RSD",
    )


def load_bao_iso_dataset() -> Dict[str, Any]:
    return load_standardized_npz(
        ("bao_iso.npz", "bao_iso_dr16.npz"),
        dataset_type="BAO_ISO",
    )


def load_bao_aniso_dataset() -> Dict[str, Any]:
    return load_standardized_npz(
        ("bao_aniso.npz", "bao_aniso_dr16.npz"),
        dataset_type="BAO_ANISO",
    )


def load_sn_dataset() -> Dict[str, Any]:
    return load_standardized_npz(
        (
            "sn_pantheon.npz",
            "sn_pantheonplus.npz",
            "sn_pantheon_full.npz",
            "sn_pantheon_shoes.npz",
            "sn_sh0es.npz",
            "sh0es.npz",
            "sn.npz",
        ),
        dataset_type="SN",
    )


def load_sh0es_dataset() -> Dict[str, Any]:
    return load_standardized_npz(
        ("sn_pantheon_shoes.npz", "sn_sh0es.npz", "sh0es.npz"),
        dataset_type="SN",
    )


def load_cmb_dataset() -> Dict[str, Any]:
    return load_standardized_npz(
        ("cmb.npz", "planck2018_distance_priors.npz", "cmb_planck2018.npz"),
        dataset_type="CMB",
    )
