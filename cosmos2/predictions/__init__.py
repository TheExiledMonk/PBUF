"""Prediction registry + helpers for the Cosmos CLI."""

from __future__ import annotations

import importlib
import pkgutil

from .wl_utils import wl_source_distribution
from .registry import PredictionModule, get_prediction_module, predictions_available, register_prediction
from .runner import PredictionManager, run_prediction_for_model
from .structures import PredictionPlot, PredictionResult, PredictionTable
from .model_api import PredictionModelAdapter

__all__ = [
    "PredictionModule",
    "PredictionModelAdapter",
    "PredictionManager",
    "PredictionResult",
    "PredictionPlot",
    "PredictionTable",
    "register_prediction",
    "get_prediction_module",
    "predictions_available",
    "run_prediction_for_model",
]
__all__.append("wl_source_distribution")


def _import_prediction_modules() -> None:
    """Auto-import submodules that live under :mod:`cosmos2.predictions.modules`."""
    try:
        package = importlib.import_module(__name__ + ".modules")
    except ModuleNotFoundError:  # pragma: no cover - defensive
        return

    for finder, name, is_pkg in pkgutil.iter_modules(package.__path__):
        if not name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{name}")


_import_prediction_modules()
