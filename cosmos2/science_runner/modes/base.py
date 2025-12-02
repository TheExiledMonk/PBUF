"""Base infrastructure for unified science runner mode plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Type

from cosmos2.science_runner.context import ModeResult, RunContext
from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.events import EventBus

_MODE_REGISTRY: Dict[str, Type["BaseModePlugin"]] = {}


def register_mode(plugin_class: Type["BaseModePlugin"]) -> Type["BaseModePlugin"]:
    _MODE_REGISTRY[plugin_class.name] = plugin_class
    return plugin_class


def get_mode(name: str) -> Type["BaseModePlugin"]:
    try:
        return _MODE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"No mode registered under '{name}'") from exc


def available_modes() -> Iterable[str]:
    return sorted(_MODE_REGISTRY.keys())


class BaseModePlugin(ABC):
    """Abstract lifecycle contract for science runner modes."""

    name = "base"

    def __init__(self, config: ScienceRunConfig, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus

    @abstractmethod
    def prepare(self, context: RunContext) -> None:
        """Prepare mode-specific state before executing."""

    @abstractmethod
    def execute(self, context: RunContext) -> ModeResult:
        """Execute the mode and return the result summary."""

    def finalize(self, context: RunContext, result: ModeResult) -> None:
        """Finalize the run after execution has completed."""
        return None
