"""SH0ES H₀ prior adapter for the joint fit runner."""

from __future__ import annotations

from typing import Any, Dict

from cosmos.interfaces import CosmologyModel
from fits.sh0es.sh0es_prior import load_sh0es_dataset, run_sh0es_prior


def run_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, float]]:
    """Delegate to the original SH0ES χ² helper."""

    return run_sh0es_prior(model, dataset)


__all__ = ["load_sh0es_dataset", "run_fit"]
