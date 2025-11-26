"""DESI-modified ΛCDM pieces for the V11 engine.

We only expose the parameter, expansion, distance, and sanity helpers in
V11 so the full model can stay isolated until V12 handles fitting.
"""

from __future__ import annotations

from typing import Dict

from cosmos.models.desi_mod import distances, expansion, parameters, sanity

MODEL_OBJECT: Dict[str, object] = {
    "parameters": parameters,
    "expansion": expansion,
    "distances": distances,
    "sanity": sanity,
}
