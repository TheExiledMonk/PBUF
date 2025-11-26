"""Phase-6a helpers tailored to the PBUF model."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from cosmos.models.pbuf.phase7a import make_phase7a_checker
from cosmos.models.pbuf.thermal_table import ThermalTable

ParamDict = Dict[str, float]
SanityFn = Callable[[ParamDict], Tuple[bool, str | None]]


def make_phase6a_checker(
    thermal_table: ThermalTable,
    metadata: Dict[str, Any] | None = None,
) -> SanityFn:
    return make_phase7a_checker(thermal_table, metadata)
