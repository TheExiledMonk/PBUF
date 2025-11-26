"""Supernova Pantheon+ χ² evaluator that stays model neutral."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel

DATA_ROOT = Path("data/standardized")
DEFAULT_FILE = "sn_pantheon.npz"


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        value = raw.item()
        return dict(value) if isinstance(value, dict) else {"metadata": value}
    return dict(raw) if isinstance(raw, dict) else {"metadata": raw}


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


@lru_cache(maxsize=None)
def _load_sn_dataset_cached(resolved_path: str) -> Dict[str, Any]:
    target = Path(resolved_path)
    if not target.exists():
        raise FileNotFoundError(f"SN Pantheon dataset not found at {target}")

    payload = np.load(target, allow_pickle=True)
    z = np.asarray(payload["z"], dtype=float)
    obs = np.asarray(payload["obs"], dtype=float)
    cov = np.asarray(payload["cov"], dtype=float) if "cov" in payload else None
    metadata = _parse_metadata(payload.get("meta"))
    name = str(payload.get("name", "Pantheon+SH0ES").item() if hasattr(payload.get("name", None), "item") else payload.get("name", "Pantheon+SH0ES"))

    inv_cov = None
    err = None
    if cov is not None:
        inv_cov = np.linalg.inv(cov)
        err = np.sqrt(np.diag(cov))

    return {
        "name": name,
        "type": "SN",
        "z": z,
        "obs": obs,
        "err": err,
        "cov": cov,
        "inv_cov": inv_cov,
        "meta": metadata,
    }


def load_sn_pantheon_dataset(path: Path | str | None = None) -> Dict[str, Any]:
    """
    Load the standardized Pantheon+SH0ES distance moduli cache.
    """

    resolved = _resolve_dataset_path(path)
    return _load_sn_dataset_cached(resolved)


def run_sn_pantheon_fit(model: CosmologyModel, dataset: Dict[str, Any] | None = None) -> tuple[float, Dict[str, np.ndarray]]:
    """
    Evaluate the SN χ² for a cosmology model instance.
    """

    dataset = dataset or load_sn_pantheon_dataset()
    z = np.asarray(dataset["z"], dtype=float)
    observed = np.asarray(dataset["obs"], dtype=float)
    mu_model = np.asarray(model.distance_modulus(z), dtype=float)

    diff = mu_model - observed

    inv_cov = dataset.get("inv_cov")
    if inv_cov is not None:
        chi2 = float(diff.T @ inv_cov @ diff)
    else:
        err = dataset.get("err")
        if err is None:
            raise ValueError("SN dataset lacks covariance & error information")
        chi2 = float(np.sum((diff / err) ** 2))

    extras = build_fit_extras(
        dataset=dataset,
        predictions=mu_model,
        observed=observed,
        residuals=diff,
        additional={"mu_model": mu_model},
    )
    return chi2, extras
