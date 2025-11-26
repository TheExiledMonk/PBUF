"""Loaders for massive messenger channel metadata (e.g., neutrinos, optical)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ENERGY_UNIT_CONVERSIONS = {
    "ev": 1.0,
    "kev": 1.0e3,
    "mev": 1.0e6,
    "gev": 1.0e9,
    "tev": 1.0e12,
    "pev": 1.0e15,
}

_REQUIRED_FIELDS = ("event_id", "channel", "t_obs", "sigma_t", "mass_eV")


def load_massive_messenger_channels(filepath: str | Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load additional messenger channels (neutrinos, optical) from CSV or JSON.

    Returns a mapping of event_id -> channel_name -> channel_dict that can be
    passed directly into build_event(additional_channels=...).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = _load_massive_csv(path)
    elif suffix == ".json":
        records = _load_massive_json(path)
    else:
        raise ValueError(f"Unsupported massive messenger format: {path}")

    events: Dict[str, Dict[str, Dict[str, float]]] = {}
    for record in records:
        event_id = record["event_id"]
        channel_name = record["channel"]
        events.setdefault(event_id, {})[channel_name] = {
            "t_obs": record["t_obs"],
            "sigma_t": record["sigma_t"],
            "mass_eV": record["mass_eV"],
            "E_eV": record["E_eV"],
        }
    return events


def _load_massive_csv(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        for idx, row in enumerate(reader, start=2):
            if not any(row.values()):
                continue
            records.append(_normalize_record(row, source=f"{path}:{idx}"))
    return records


def _load_massive_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        iterable: Iterable[Mapping[str, Any]] = data.get("channels", [])
    else:
        iterable = data
    if not isinstance(iterable, Iterable):
        raise ValueError(f"Invalid JSON schema in {path}")
    return [_normalize_record(item, source=str(path)) for item in iterable]


def _normalize_record(raw: Mapping[str, Any], source: str) -> Dict[str, Any]:
    for field in _REQUIRED_FIELDS:
        if raw.get(field) in (None, ""):
            raise ValueError(f"{source}: missing required field '{field}'")

    def _to_float(value: Any, field: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: invalid numeric value for '{field}': {value}") from exc

    event_id = str(raw["event_id"]).strip()
    channel_name = str(raw["channel"]).strip()
    t_obs = _to_float(raw["t_obs"], "t_obs")
    sigma_t = _to_float(raw["sigma_t"], "sigma_t")
    mass_eV = _to_float(raw["mass_eV"], "mass_eV")

    E_eV = _extract_energy(raw, source)
    if mass_eV > 0.0 and E_eV is None:
        raise ValueError(f"{source}: channel '{channel_name}' with mass > 0 requires energy")

    return {
        "event_id": event_id,
        "channel": channel_name,
        "t_obs": t_obs,
        "sigma_t": sigma_t,
        "mass_eV": mass_eV,
        "E_eV": E_eV,
    }


def _extract_energy(raw: Mapping[str, Any], source: str) -> float | None:
    if "E_eV" in raw and raw["E_eV"] not in (None, ""):
        return _convert_to_eV(raw["E_eV"], "eV", source)

    value = raw.get("E_value")
    if value in (None, ""):
        return None
    unit = str(raw.get("energy_unit", "eV")).lower()
    return _convert_to_eV(value, unit, source)


def _convert_to_eV(value: Any, unit: str, source: str) -> float:
    try:
        magnitude = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid energy value {value}") from exc

    unit_norm = unit.strip().lower()
    if unit_norm not in ENERGY_UNIT_CONVERSIONS:
        raise ValueError(f"{source}: unsupported energy unit '{unit}'")

    energy = magnitude * ENERGY_UNIT_CONVERSIONS[unit_norm]
    if energy <= 0.0:
        raise ValueError(f"{source}: energy must be positive, got {energy}")
    return energy


__all__ = ["load_massive_messenger_channels"]
