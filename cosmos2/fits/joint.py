"""Joint-fit orchestration utilities for cosmos2."""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import yaml

from cosmos2.fits.registry import FIT_REGISTRY

ParamDict = Dict[str, float]
ModelFactory = Callable[[ParamDict], Any]
_PARALLEL_ENV = os.getenv("COSMOS2_JOINT_PARALLEL", "").strip().lower()
_PARALLEL_DEFAULT = _PARALLEL_ENV not in {"0", "false", "no", "off"}


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


def _parse_fits_and_weights(config: Dict[str, Any]) -> Tuple[list[str], Dict[str, float]]:
    return _parse_fits_and_weights_from_registry(config, FIT_REGISTRY)


def _parse_fits_and_weights_from_registry(
    config: Dict[str, Any],
    registry: Mapping[str, Callable[[Any], Any]],
) -> Tuple[list[str], Dict[str, float]]:
    raw_fits = config.get("fits") or config.get("joint_config", {}).get("fits")
    if not raw_fits:
        raise ValueError("Joint config must supply a non-empty 'fits' array.")
    if isinstance(raw_fits, (str, bytes)) or not isinstance(raw_fits, Sequence):
        raise ValueError("Joint config 'fits' entry must be a list of fit names.")

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

    fits = _sanitize_fits(raw_fits)
    return fits, weights


def resolve_joint_fits(
    joint_config_path: str | Path,
    *,
    registry: Mapping[str, Callable[[Any], Any]] = FIT_REGISTRY,
) -> Tuple[list[str], Dict[str, float]]:
    """
    Load a joint config and return the requested fits plus their weights.
    """
    config = load_joint_config(joint_config_path)
    return _parse_fits_and_weights_from_registry(config, registry)


def build_joint_chi2_evaluator(
    model_factory: ModelFactory,
    joint_config_path: str | Path,
    *,
    skip_valid: bool = False,
    parallel: bool | None = None,
    max_workers: int | None = None,
    registry: Mapping[str, Callable[[Any], Any]] = FIT_REGISTRY,
) -> Callable[[ParamDict], float]:
    """
    Build a joint χ² target from the provided fit config.

    If `parallel` is True, independent fits are evaluated concurrently using a
    thread pool; set `max_workers` to control the pool size (defaults to the
    executor default).

    Parallelism defaults to enabled unless explicitly disabled via the
    `parallel` argument or the `COSMOS2_JOINT_PARALLEL=0/false` environment
    variable.
    """

    config = load_joint_config(joint_config_path)
    fits, weights = _parse_fits_and_weights_from_registry(config, registry)

    print(f"[jackknife] Building joint evaluator with {len(fits)} fits: {fits}")
    print(f"[jackknife] Registry type: {type(registry)}")

    enabled: list[tuple[str, Callable[[Any], tuple[float, Any]], float]] = []
    for fit_name in fits:
        if fit_name not in registry:
            raise ValueError(f"Unknown fit '{fit_name}' referenced by joint config.")
        fit_fn = registry[fit_name]
        weight = weights.get(fit_name, 1.0)
        print(f"[jackknife] Fit {fit_name}: function={fit_fn}, weight={weight}")
        enabled.append((fit_name, fit_fn, weight))

    if not enabled:
        raise ValueError("Joint config did not resolve any valid fits.")

    def joint_chi2(params: ParamDict) -> float:
        sanitized = {key: float(value) for key, value in params.items()}
        model = model_factory(sanitized)
        if not skip_valid and hasattr(model, "is_valid") and not model.is_valid():
            return float("inf")

        total = 0.0
        parallel_flag = _PARALLEL_DEFAULT if parallel is None else bool(parallel)

        if parallel_flag and len(enabled) > 1:
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for fit_name, fit_fn, weight in enabled:
                    futures[pool.submit(fit_fn, model)] = (fit_name, weight)
                for future in as_completed(futures):
                    fit_name, weight = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        return float("inf")
                    chi2 = result[0] if isinstance(result, tuple) else result
                    if not math.isfinite(chi2):
                        return float("inf")
                    total += weight * float(chi2)
        else:
            for _, fit_fn, weight in enabled:
                result = fit_fn(model)
                chi2 = result[0] if isinstance(result, tuple) else result
                if not math.isfinite(chi2):
                    return float("inf")
                total += weight * float(chi2)
        return total

    return joint_chi2


__all__ = ["load_joint_config", "build_joint_chi2_evaluator", "resolve_joint_fits"]
