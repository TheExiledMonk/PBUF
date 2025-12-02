"""Utility helpers for the cosmos2 science runner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_json_or_yaml(path: Path) -> dict[str, Any]:
    """Load JSON or YAML payload from the given path."""

    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected mapping in '{path}', got {type(payload)}")
        return payload


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_run_name(name: str) -> str:
    trimmed = name.strip().lower()
    if not trimmed:
        return "run"
    result = []
    for char in trimmed:
        if char.isalnum() or char in "-_":
            result.append(char)
        else:
            result.append("_")
    sanitized = "".join(result)
    return sanitized.strip("_") or "run"


def serialize_value(value: Any) -> Any:
    """Normalize arbitrary values into JSON-friendly primitives."""

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            # Return a special marker for infinity instead of string "inf"
            if math.isinf(value):
                return {"__type__": "infinity", "value": "inf" if value > 0 else "-inf"}
            else:  # NaN
                return {"__type__": "nan", "value": "nan"}
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialize_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): serialize_value(val) for key, val in value.items()}

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if dataclasses.is_dataclass(value):
        return serialize_value(dataclasses.asdict(value))

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "__dict__"):
        return serialize_value(vars(value))

    return str(value)


def hash_payload(payload: Any) -> str:
    serialized = json.dumps(serialize_value(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def safe_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialize_value(payload), handle, indent=indent)


__all__ = [
    "ensure_dir",
    "hash_payload",
    "load_json_or_yaml",
    "sanitize_run_name",
    "safe_write_json",
    "serialize_value",
]
