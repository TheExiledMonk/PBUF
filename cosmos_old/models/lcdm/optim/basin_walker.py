"""LCDM-specific wrapper around the shared basin walker."""

from __future__ import annotations

from cosmos.models.optim.basin_core import BasinConfig as _BasinConfig
from cosmos.models.optim.basin_core import BasinWalkerBase


class BasinConfig(_BasinConfig):
    """LCDM basin configuration."""


class BasinWalker(BasinWalkerBase):
    """LCDM-aware basin walker."""


__all__ = ["BasinConfig", "BasinWalker"]
