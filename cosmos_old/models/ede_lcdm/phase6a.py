"""Phase 6a sanity gate for the EDE-LCDM helpers."""

from __future__ import annotations

from typing import Dict, Tuple

from cosmos.models.ede_lcdm import sanity
from cosmos.optim.sanity_base import SanityResult

ParamDict = Dict[str, float]


def phase6a_ede(params: ParamDict) -> Tuple[bool, str | None]:
    sanitized = {key: float(value) for key, value in params.items()}
    result = sanity.sanity_checks(sanitized)
    if result.ok:
        return True, None
    return False, "; ".join(result.reasons)
