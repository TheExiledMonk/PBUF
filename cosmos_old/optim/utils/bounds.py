"""Helpers to read the Basin Walker boundary declarations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict



def load_bounds(model_name: str) -> Dict[str, Any]:
    """Load the system-level bounds for the supplied model."""
    path = Path("configs/basin_walker") / f"{model_name}_bounds.json"
    with path.open("r") as handle:
        payload = json.load(handle)

    declared_model = payload.get("model")
    if declared_model and declared_model != model_name:
        raise ValueError(
            f"Bounds file for {model_name} declares an unexpected model '{declared_model}'."
        )

    return payload
