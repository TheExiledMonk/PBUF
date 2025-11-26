"""SH0ES H0 prior expressed as a χ² evaluator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import math
import numpy as np

from cosmos.fits.extras import build_fit_extras
from cosmos.interfaces import CosmologyModel

DATA_ROOT = Path("data/standardized")
DEFAULT_FILE = "sh0es.npz"


def _unwrap_metadata(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if hasattr(raw, "item"):
        value = raw.item()
        return dict(value) if isinstance(value, dict) else {"meta": value}
    if isinstance(raw, dict):
        return dict(raw)
    return {"meta": raw}


def _resolve_dataset_path(path: Path | str | None) -> str:
    candidate = Path(path) if path is not None else DATA_ROOT / DEFAULT_FILE
    return str(candidate.expanduser().resolve())


@lru_cache(maxsize=None)
def _load_sh0es_dataset_cached(resolved_path: str) -> Dict[str, Any]:
    target = Path(resolved_path)
    if not target.exists():
        raise FileNotFoundError(f"SH0ES prior not found at {target}")

    payload = np.load(target, allow_pickle=True)
    obs = np.asarray(payload["obs"]).ravel()
    if obs.size != 1:
        raise ValueError("SH0ES dataset should contain a single measurement")
    err = payload.get("err")
    cov = payload.get("cov")

    if err is not None:
        sigma = float(np.asarray(err).ravel()[0])
    elif cov is not None:
        cov_arr = np.asarray(cov, dtype=float)
        sigma = float(math.sqrt(cov_arr[0, 0]))
    else:
        raise ValueError("SH0ES dataset lacks uncertainty information")

    metadata = _unwrap_metadata(payload.get("meta"))
    name_value = payload.get("name", "SH0ES")
    if hasattr(name_value, "item"):
        name_value = name_value.item()
    return {
        "name": str(name_value),
        "type": "SH0ES",
        "obs": float(obs[0]),
        "sigma": sigma,
        "meta": metadata,
    }


def load_sh0es_dataset(path: Path | str | None = None) -> Dict[str, Any]:
    resolved = _resolve_dataset_path(path)
    return _load_sh0es_dataset_cached(resolved)


def run_sh0es_prior(model: CosmologyModel, dataset: Dict[str, Any] | None = None) -> tuple[float, Dict[str, float]]:
    dataset = dataset or load_sh0es_dataset()
    H0_model = float(model.parameters["H0"])
    H0_target = float(dataset["obs"])
    sigma = float(dataset["sigma"])
    delta = H0_model - H0_target
    diff = delta / sigma
    chi2 = float(diff * diff)
    extras = build_fit_extras(
        dataset=dataset,
        predictions=H0_model,
        observed=H0_target,
        residuals=delta,
        additional={"sigma": sigma},
    )
    return chi2, extras
