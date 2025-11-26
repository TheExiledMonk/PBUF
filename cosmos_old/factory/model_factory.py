"""Model factory so fits stay agnostic about concrete constructors."""

from __future__ import annotations

from typing import Literal

from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.pbuf.model import PBUFModel

ModelName = Literal["pbuf", "lcdm"]


def create_model(name: ModelName, **params):
    """
    Create a cosmology model instance by name.

    Fits should use this helper exclusively to avoid importing model modules
    directly (which reduces the risk of cross-contamination).
    """

    if name == "pbuf":
        return PBUFModel(**params)

    if name == "lcdm":
        return LCDMModel(**params)

    raise ValueError(f"Unknown model: {name}")
