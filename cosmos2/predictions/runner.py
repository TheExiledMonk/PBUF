"""Helpers for orchestrating prediction runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Iterable

from .model_api import PredictionModelAdapter
from .registry import get_prediction_module, predictions_available
from .structures import PredictionResult

logger = logging.getLogger(__name__)


def run_prediction_for_model(
    module_name: str, model: Any, config: dict[str, Any]
) -> PredictionResult:
    """Run a single prediction module against a model instance."""

    module = get_prediction_module(module_name)
    adapter = PredictionModelAdapter(model)
    try:
        return module.run_prediction(adapter, dict(config or {}))
    except Exception as exc:  # pragma: no cover - best-effort guard
        logger.exception("Prediction '%s' failed", module_name)
        return PredictionResult(
            name=module.name,
            version=module.version,
            metadata={"error": str(exc)},
            results={},
            tables=[],
            plots=[],
            status="error",
        )


@dataclass
class PredictionManager:
    """Run a fixed set of modules for a model."""

    modules: list[str] | None = None

    def __post_init__(self) -> None:
        if self.modules is None:
            self.modules = list(predictions_available())

    def run_for_model(
        self, model_name: str, model: Any, module_configs: dict[str, dict[str, Any]]
    ) -> list[PredictionResult]:
        """Return prediction results (one per module) for the model."""

        results: list[PredictionResult] = []
        adapter = PredictionModelAdapter(model)
        for module_name in self.modules:
            module = get_prediction_module(module_name)
            payload = module_configs.get(module_name, {})
            try:
                result = module.run_prediction(adapter, dict(payload))
            except Exception as exc:
                logger.exception("Prediction %s failed for %s", module_name, model_name)
                result = PredictionResult(
                    name=module.name,
                    version=module.version,
                    metadata={"error": str(exc)},
                    results={},
                    tables=[],
                    plots=[],
                    status="error",
                )
            results.append(result)
        return results

    def as_summary(self, model_name: str, results: Iterable[PredictionResult]) -> dict[str, Any]:
        """Create serializable payload for this model."""

        return {
            "model": model_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": [item.to_dict() for item in results],
        }
