"""Shared weak-lensing utilities (dataset helpers, theory layer, backends)."""

from importlib import import_module
from typing import Any

__all__ = ["standardize_kids1000", "WeakLensingBackend", "compute_shear_predictions"]


def __getattr__(name: str) -> Any:  # pragma: no cover - lazy import surface
    if name == "standardize_kids1000":
        return import_module(".kids", __name__).standardize_kids1000
    if name == "WeakLensingBackend":
        return import_module(".backend", __name__).WeakLensingBackend
    if name == "compute_shear_predictions":
        return import_module(".theory", __name__).compute_shear_predictions
    raise AttributeError(f"module {__name__} has no attribute {name}")
