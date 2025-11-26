"""Helpers for reading GWOSC metadata into event-friendly dicts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from . import log_pipeline_error


def extract_distance_Mpc(gw_metadata: Dict[str, Any]) -> float:
    """Extract and validate luminosity distance from GWOSC metadata."""
    if "luminosity_distance" not in gw_metadata:
        raise KeyError("luminosity_distance not found in GWOSC metadata")
    L_Mpc = float(gw_metadata["luminosity_distance"])
    if L_Mpc <= 0.0:
        raise ValueError(f"Invalid distance: {L_Mpc} Mpc")
    return L_Mpc


def _load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_gwosc_event(filepath: str | Path) -> Dict[str, Any]:
    """Load a single GWOSC event metadata JSON."""
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".json":
        data = _load_json_file(path)
    elif ext == ".csv":
        raise NotImplementedError("Use load_gwosc_csv for CSV inputs")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    try:
        gps_time = float(data["GPS"])
        event_name = data["name"]
    except KeyError as exc:
        raise KeyError(f"Missing key {exc} in {path}") from exc

    return {
        "event_name": str(event_name),
        "gps_time": gps_time,
        "L_Mpc": extract_distance_Mpc(data),
    }


def load_gwosc_csv(filepath: str | Path) -> List[Dict[str, Any]]:
    """Load GWOSC events from a CSV exported by GWOSC."""
    path = Path(filepath)
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = (row.get("name") or row.get("Name") or "").strip()
            gps_raw = row.get("gps") or row.get("GPS")
            dist_raw = row.get("luminosity_distance")
            if not name or gps_raw in (None, "") or dist_raw in (None, ""):
                log_pipeline_error(
                    f"Skipping GWOSC row in {path.name}: missing name/gps/distance"
                )
                continue
            try:
                gps_time = float(gps_raw)
                L_Mpc = float(dist_raw)
            except (TypeError, ValueError):
                log_pipeline_error(
                    f"Skipping GWOSC row {name}: invalid numeric fields in {path.name}"
                )
                continue
            if L_Mpc <= 0:
                log_pipeline_error(
                    f"Skipping GWOSC row {name}: non-positive distance {L_Mpc}"
                )
                continue
            events.append({
                "event_name": name,
                "gps_time": gps_time,
                "L_Mpc": L_Mpc,
            })
    return events


def _iter_gwosc_sources(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for pattern in ("*.json", "*.csv"):
        yield from sorted(path.glob(pattern))


def load_gwosc_directory(dirpath: str | Path) -> List[Dict[str, Any]]:
    """Load GWOSC metadata from JSON or CSV sources."""
    base_path = Path(dirpath)
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    events: List[Dict[str, Any]] = []
    for source in _iter_gwosc_sources(base_path):
        if source.suffix.lower() == ".json":
            try:
                events.append(load_gwosc_event(source))
            except Exception as exc:  # pragma: no cover - logging path
                log_pipeline_error(f"Error loading {source}: {exc}")
        elif source.suffix.lower() == ".csv":
            try:
                events.extend(load_gwosc_csv(source))
            except Exception as exc:  # pragma: no cover - logging path
                log_pipeline_error(f"Error loading {source}: {exc}")

    return events


__all__ = [
    "extract_distance_Mpc",
    "load_gwosc_event",
    "load_gwosc_directory",
    "load_gwosc_csv",
]
