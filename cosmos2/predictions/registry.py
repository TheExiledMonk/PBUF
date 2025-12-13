"""Registry helpers for available prediction modules."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Type

from .model_api import PredictionModelAdapter
from .structures import PredictionResult


_MODULE_REGISTRY: Dict[str, "PredictionModule"] = {}


def register_prediction(module_class: Type["PredictionModule"]) -> "PredictionModule":
    """Decorator used by prediction modules to register themselves."""

    instance = module_class()
    normalized = instance.name.strip().lower()
    if not normalized:
        raise ValueError("Prediction modules must define a non-empty 'name'.")
    if normalized in _MODULE_REGISTRY:
        raise ValueError(f"Duplicate prediction module '{normalized}'.")
    _MODULE_REGISTRY[normalized] = instance
    return instance


def get_prediction_module(name: str) -> "PredictionModule":
    """Retrieve a prediction module by name."""
    normalized = name.strip().lower()
    try:
        return _MODULE_REGISTRY[normalized]
    except KeyError as exc:
        raise KeyError(f"No prediction module registered under '{name}'.") from exc


def predictions_available() -> tuple[str, ...]:
    """Return the list of registered prediction module names."""
    return tuple(sorted(_MODULE_REGISTRY.keys()))


class PredictionModule(ABC):
    """Base class for prediction modules."""

    name = "base"
    version = "v1"
    description = "Generic prediction module."

    def register(self, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(prediction_module=self)

    def describe(self) -> str:
        return self.description

    @abstractmethod
    def run_prediction(
        self, model: PredictionModelAdapter, config: dict[str, Any]
    ) -> PredictionResult:
        ...
