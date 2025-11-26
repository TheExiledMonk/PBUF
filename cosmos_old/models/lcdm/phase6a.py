"""Phase-6a helpers for the LCDM model."""

from __future__ import annotations

from typing import Dict, Tuple

from cosmos.models.lcdm.model import LCDMModel
from cosmos.models.lcdm.sanity import check_closure_lcdm, check_expansion_lcdm
from cosmos.optim.sanity_base import SanityResult

ParamDict = Dict[str, float]


def phase6a_lcdm(params: ParamDict) -> Tuple[bool, str | None]:
    defaults = {"Omega_r0": 9.0e-5}
    defaults.update(params)
    sanitized = {key: float(value) for key, value in defaults.items()}
    model = LCDMModel(**sanitized)
    result = SanityResult()
    result.merge(check_closure_lcdm(sanitized, model))
    result.merge(check_expansion_lcdm(sanitized, model))

    if result.ok:
        return True, None
    return False, "; ".join(result.reasons)
