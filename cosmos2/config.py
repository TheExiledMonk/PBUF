"""Configuration helpers for cosmos2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple


def load_bounds_for_model(model_name: str, datasets: Sequence[str] | None = None) -> Dict[str, Tuple[float, float]]:
    normalized = model_name.strip().lower()
    path = Path("configs/basin_walker") / f"{normalized}_bounds.json"
    payload = json.loads(path.read_text())
    raw = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else payload
    bounds: Dict[str, Tuple[float, float]] = {}
    for name, interval in raw.items():
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            continue
        lo, hi = float(interval[0]), float(interval[1])
        if hi < lo:
            continue
        bounds[name] = (lo, hi)
    return bounds


__all__ = ["load_bounds_for_model"]
