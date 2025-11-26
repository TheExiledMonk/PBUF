"""Joint-fit orchestration utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import yaml

from cosmos.fits.registry import FIT_REGISTRY
from cosmos.interfaces import CosmologyModel

ParamDict = Dict[str, float]
ModelFactory = Callable[[ParamDict], CosmologyModel]


def load_joint_config(path: str | Path) -> Dict[str, Any]:
    payload = Path(path).read_text()
    try:
        config = json.loads(payload)
    except json.JSONDecodeError:
        config = yaml.safe_load(payload)

    if not isinstance(config, dict):
        raise ValueError(f"Joint config at {path} must be an object.")
    return config


def _sanitize_fits(raw_fits: Sequence[str]) -> list[str]:
    cleaned = []
    for entry in raw_fits:
        normalized = entry.strip().lower()
        if normalized:
            cleaned.append(normalized)
    return cleaned


def build_joint_chi2_evaluator(model_factory: ModelFactory, joint_config_path: str | Path) -> Callable[[ParamDict], float]:
    """
    Build a joint χ² target from the provided fit config.
    """

    config = load_joint_config(joint_config_path)
    raw_fits = config.get("fits") or config.get("joint_config", {}).get("fits")
    if not raw_fits:
        raise ValueError("Joint config must supply a non-empty 'fits' array.")
    if isinstance(raw_fits, (str, bytes)) or not isinstance(raw_fits, Sequence):
        raise ValueError("Joint config 'fits' entry must be a list of fit names.")

    enabled: list[tuple[str, Callable[[CosmologyModel], tuple[float, Any]], float]] = []
    fit_weights = config.get("fit_weights") or {}
    weights: Dict[str, float] = {}
    for key, value in dict(fit_weights).items():
        try:
            key_name = str(key).strip().lower()
            if not key_name:
                continue
            weights[key_name] = float(value)
        except Exception:
            continue

    for fit_name in _sanitize_fits(raw_fits):
        if fit_name not in FIT_REGISTRY:
            raise ValueError(f"Unknown fit '{fit_name}' referenced by joint config.")
        weight = weights.get(fit_name, 1.0)
        enabled.append((fit_name, FIT_REGISTRY[fit_name], weight))

    if not enabled:
        raise ValueError("Joint config did not resolve any valid fits.")

    def joint_chi2(params: ParamDict) -> float:
        sanitized = {key: float(value) for key, value in params.items()}
        model = model_factory(sanitized)
        if not model.is_valid():
            return float("inf")

        total = 0.0
        for _, fit_fn, weight in enabled:
            chi2, _ = fit_fn(model)
            if not math.isfinite(chi2):
                return float("inf")
            total += weight * chi2
        return total

    return joint_chi2
