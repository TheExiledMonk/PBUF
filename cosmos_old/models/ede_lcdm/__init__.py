"""Early Dark Energy ΛCDM helpers for compatibility checks."""

from __future__ import annotations

from typing import Dict

from cosmos.models.ede_lcdm import distances, expansion, parameters, phase6a, sanity

MODEL_OBJECT: Dict[str, object] = {
    "parameters": parameters,
    "expansion": expansion,
    "distances": distances,
    "sanity": sanity,
    "phase6a": phase6a,
}
