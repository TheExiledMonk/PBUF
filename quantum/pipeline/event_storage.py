"""Persistence helpers for event lists."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import log_pipeline_error
from .event_builder import validate_event_schema


def _validate_events(events: List[Dict[str, Any]]) -> None:
    for idx, event in enumerate(events):
        is_valid, error = validate_event_schema(event)
        if not is_valid:
            log_pipeline_error(f"Event {idx} invalid: {error}")
            raise ValueError(f"Event {idx} invalid: {error}")


def save_events_json(events: List[Dict[str, Any]], path: str) -> None:
    """Serialize events to JSON with schema validation."""
    _validate_events(events)
    output_path = Path(path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(events, handle, indent=2)


def save_events_npz(events: List[Dict[str, Any]], path: str) -> None:
    """Serialize events to compressed NPZ after validation."""
    _validate_events(events)
    output_path = Path(path)
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(events)
    np.savez_compressed(output_path, events_json=np.array(payload))


def load_events_npz(path: str) -> List[Dict[str, Any]]:
    """Load events stored via save_events_npz."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as npz:
        data = npz["events_json"]
        events_json = data.item() if hasattr(data, "item") else str(data)
    return json.loads(events_json)


__all__ = [
    "save_events_json",
    "save_events_npz",
    "load_events_npz",
]
