"""BAO anisotropic fit using the `bao_aniso_full` dataset."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from cosmos2.data.registry import get_dataset
from cosmos2.fits.bao_aniso import run_bao_aniso_fit


def run_fit(model: Any, dataset: Dict[str, Any] | None = None) -> Tuple[float, Dict[str, Any]]:
    dataset = dataset or get_dataset("bao_aniso_full")
    return run_bao_aniso_fit(model, dataset)


__all__ = ["run_fit"]

