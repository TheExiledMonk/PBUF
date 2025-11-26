"""Shared helpers for assembling fit extras payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import numpy as np


def _normalize_array_like(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return value


def summarize_dataset(dataset: Any | None) -> Dict[str, Any]:
    if dataset is None:
        return {}
    summary: Dict[str, Any] = {}
    if isinstance(dataset, Mapping):
        for key in ("name", "type", "files"):
            value = dataset.get(key)
            if value is not None:
                summary[key] = value
        meta = dataset.get("meta")
        if meta is not None:
            summary["meta"] = meta
    metadata_attr = getattr(dataset, "metadata", None)
    if metadata_attr is not None:
        summary.setdefault("meta", metadata_attr)
    dataset_type = getattr(dataset, "type", None)
    if dataset_type is not None:
        summary.setdefault("type", dataset_type)
    if not summary:
        summary["source"] = type(dataset).__name__
    return summary


def build_fit_extras(
    *,
    dataset: Any | None = None,
    predictions: Any | None = None,
    observed: Any | None = None,
    residuals: Any | None = None,
    weights: Any | None = None,
    additional: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    dataset_summary = summarize_dataset(dataset)
    if dataset_summary:
        extras["dataset"] = dataset_summary

    normalized_predictions = _normalize_array_like(predictions)
    normalized_observed = _normalize_array_like(observed)
    normalized_residuals = _normalize_array_like(residuals)

    if normalized_predictions is not None:
        extras["predictions"] = normalized_predictions
    if normalized_observed is not None:
        extras["observed"] = normalized_observed
    if normalized_residuals is None and normalized_predictions is not None and normalized_observed is not None:
        try:
            normalized_residuals = normalized_predictions - normalized_observed
        except Exception:
            normalized_residuals = None
    if normalized_residuals is not None:
        extras["residuals"] = normalized_residuals
    if weights is not None:
        extras["weights"] = _normalize_array_like(weights)
    if additional:
        extras.update(additional)
    return extras


__all__ = ["build_fit_extras", "summarize_dataset"]
