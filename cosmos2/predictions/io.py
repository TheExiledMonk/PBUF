"""File helpers for persisting prediction payloads."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .structures import PredictionResult, PredictionTable


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_prediction_json(result: PredictionResult, destination: Path) -> None:
    """Serialize the prediction payload to JSON."""

    path = _ensure_parent(destination)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def write_prediction_table(table: PredictionTable, destination: Path) -> None:
    """Write a prediction table as CSV."""

    path = _ensure_parent(destination)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(table.columns)
        for row in table.rows:
            writer.writerow([cell if cell is not None else "" for cell in row])
