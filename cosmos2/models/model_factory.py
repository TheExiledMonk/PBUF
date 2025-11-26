"""Model factory for cosmos2 drop-in surface."""

from __future__ import annotations

from typing import Any, Mapping

from cosmos2.models.lcdm import LCDMModel
from cosmos2.models.pbuf import PBUFModel

LEGACY_ONLY_MODELS = {
    "ede_lcdm",
    "running_lambda",
    "dgp",
    "mg_lcdm",
    "desi_mod",
}


def create_model(
    name: str,
    *,
    lut: Mapping[str, Any] | None = None,
    n_grid: int | None = None,
    use_cosmos_backend: bool | None = None,
    **params: Any,
):
    """
    Create a cosmos2 model by name. Currently lcdm and pbuf are available.
    """
    normalized = name.strip().lower()

    if normalized in LEGACY_ONLY_MODELS:
        raise ValueError(
            f"Model '{name}' is only available in the legacy cosmos package. "
            "cosmos2 currently ships LCDM and PBUF only; use cosmos.models for legacy variants."
        )

    if normalized == "lcdm":
        if n_grid is not None:
            params["n_grid"] = n_grid
        return LCDMModel(use_cosmos_backend=use_cosmos_backend, **params)

    if normalized == "pbuf":
        if n_grid is not None:
            params["n_grid"] = n_grid
        return PBUFModel(**params)

    raise ValueError(f"Unknown model '{name}'. Supported: lcdm.")


__all__ = ["create_model"]
