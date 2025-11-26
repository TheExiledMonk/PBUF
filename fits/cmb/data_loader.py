"""Loaders for CMB prior datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data/standardized")
DEFAULT_FILE = "cmb.npz"


@dataclass
class CMBDataset:
    """Planck-like CMB data with convenience accessors."""

    observed: np.ndarray
    covariance: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    _inv_cov: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def inv_covariance(self) -> np.ndarray:
        if self._inv_cov is None:
            self._inv_cov = np.linalg.inv(self.covariance)
        return self._inv_cov

    @property
    def sigmas(self) -> np.ndarray:
        """Return the 1σ uncertainties inferred from the covariance diagonals."""
        return np.sqrt(np.diag(self.covariance))


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


@lru_cache(maxsize=None)
def _load_planck_priors_cached(resolved_path: str) -> CMBDataset:
    target = Path(resolved_path)
    if not target.exists():
        raise FileNotFoundError(f"Planck CMB priors not found at {target}")

    payload = np.load(target, allow_pickle=True)
    observed = np.asarray(payload["obs"], dtype=float)
    covariance = np.asarray(payload["cov"], dtype=float)
    metadata_raw = payload.get("meta")
    metadata = metadata_raw.item() if hasattr(metadata_raw, "item") else metadata_raw

    logger.debug("Loaded CMB dataset: %s", target)
    logger.debug("Dataset metadata: %s", metadata)

    if metadata is None:
        metadata_payload: Dict[str, Any] = {}
    elif isinstance(metadata, dict):
        metadata_payload = dict(metadata)
    else:
        metadata_payload = {"meta": metadata}

    return CMBDataset(observed=observed, covariance=covariance, metadata=metadata_payload)


def load_planck_priors(path: Path | None = None) -> CMBDataset:
    """
    Load Planck CMB priors from the standardized cache under data/standardized.

    The file format matches the `.npz` payload produced by the data_interface
    converters. We read only the fields required by the fits module so we
    don't need to import heavy pandas dependencies.
    """

    resolved = _resolve_dataset_path(path)
    return _load_planck_priors_cached(resolved)
