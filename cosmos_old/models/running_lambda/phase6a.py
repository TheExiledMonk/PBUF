"""Phase 6a sanity gate for the running-Λ helpers."""

from __future__ import annotations

from typing import Dict, Tuple

from cosmos.models.running_lambda import sanity

ParamDict = Dict[str, float]


def phase6a_running_lambda(params: ParamDict) -> Tuple[bool, str | None]:
    sanitized = {key: float(value) for key, value in params.items()}
    result = sanity.sanity_checks(sanitized)
    if result.ok:
        return True, None
    return False, "; ".join(result.reasons)
