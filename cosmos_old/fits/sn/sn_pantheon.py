"""Supernova Pantheon+ wrapper for the joint fit registry."""

from __future__ import annotations

from typing import Any, Dict

from cosmos.interfaces import CosmologyModel
from fits.sn.sn_pantheon import load_sn_pantheon_dataset, run_sn_pantheon_fit


def run_fit(
    model: CosmologyModel,
    dataset: Dict[str, Any] | None = None,
) -> tuple[float, Dict[str, Any]]:
    """Delegate to the existing SN Pantheon χ² helper."""

    return run_sn_pantheon_fit(model, dataset)


__all__ = ["load_sn_pantheon_dataset", "run_fit"]
