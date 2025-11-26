"""Read the compact NPZ artifacts generated for quantum data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nth(array: np.ndarray, idx: int) -> Any:
    if idx < len(array):
        return array[idx]
    return None


def _optional_dict(**fields: Optional[str]) -> Optional[Dict[str, str]]:
    payload = {key: value for key, value in fields.items() if value}
    return payload or None


def _load_detectors(raw: Any, mask: Optional[str]) -> Dict[str, Any]:
    text = _to_text(raw)
    try:
        detectors = json.loads(text) if text else {}
    except json.JSONDecodeError:
        detectors = {}
    if mask:
        detectors["mask"] = mask
    return detectors


def _build_spectral(arrays: Dict[str, np.ndarray], idx: int) -> Dict[str, Any]:
    flux_field = _to_text(_nth(arrays["spectral_flux_field"], idx))
    source = _to_text(_nth(arrays["spectral_source"], idx))
    peak = _to_float(_nth(arrays["spectral_E_peak_keV"], idx))
    flux = _to_float(_nth(arrays["spectral_flux"], idx))

    spectral: Dict[str, Any] = {}
    if peak == peak:
        spectral["E_peak_keV"] = peak
    if flux == flux:
        spectral["flux"] = flux
    if flux_field:
        spectral["flux_field"] = flux_field
    if source:
        spectral["source"] = source
    return spectral


def _load_array(data: np.lib.npyio.NpzFile, key: str, default_dtype: Any = object) -> np.ndarray:
    if key in data:
        return data[key]
    return np.array([], dtype=default_dtype)


def _collect_arrays(data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    return {
        "trigger_name": _load_array(data, "trigger_name"),
        "trig_met": _load_array(data, "trig_met", default_dtype=float),
        "sigma_t": _load_array(data, "sigma_t", default_dtype=float),
        "E_eV": _load_array(data, "E_eV", default_dtype=float),
        "detector": _load_array(data, "detector"),
        "detectors_json": _load_array(data, "detectors_json"),
        "detector_mask": _load_array(data, "detector_mask"),
        "detectors_triggered": _load_array(data, "detectors_triggered"),
        "detectors_ranked": _load_array(data, "detectors_ranked"),
        "channel_lo": _load_array(data, "channel_lo", default_dtype=float),
        "channel_hi": _load_array(data, "channel_hi", default_dtype=float),
        "energy_lo_keV": _load_array(data, "energy_lo_keV", default_dtype=float),
        "energy_hi_keV": _load_array(data, "energy_hi_keV", default_dtype=float),
        "source_dir": _load_array(data, "source_dir"),
        "source_tcat": _load_array(data, "source_tcat"),
        "source_trigdat": _load_array(data, "source_trigdat"),
        "spectral_flux_field": _load_array(data, "spectral_flux_field"),
        "spectral_source": _load_array(data, "spectral_source"),
        "spectral_E_peak_keV": _load_array(data, "spectral_E_peak_keV", default_dtype=float),
        "spectral_flux": _load_array(data, "spectral_flux", default_dtype=float),
    }


def load_fermi_triggers_from_npz(path: Path) -> List[Dict[str, Any]]:
    """Recreate the minimal trigger representation from the compact NPZ."""
    with np.load(path, allow_pickle=True) as data:
        arrays = _collect_arrays(data)
        length = int(len(arrays["trig_met"]))
        records: List[Dict[str, Any]] = []

        for idx in range(length):
            spectral = _build_spectral(arrays, idx)
            source = _optional_dict(
                directory=_to_text(_nth(arrays["source_dir"], idx)),
                tcat=_to_text(_nth(arrays["source_tcat"], idx)),
                trigdat=_to_text(_nth(arrays["source_trigdat"], idx)),
            )
            detectors = _load_detectors(
                _nth(arrays["detectors_json"], idx),
                _to_text(_nth(arrays["detector_mask"], idx)) or None,
            )
            channel_range = (
                _to_float(_nth(arrays["channel_lo"], idx)),
                _to_float(_nth(arrays["channel_hi"], idx)),
            )
            energy_range = (
                _to_float(_nth(arrays["energy_lo_keV"], idx)),
                _to_float(_nth(arrays["energy_hi_keV"], idx)),
            )

            records.append(
                {
                    "trigger_name": _to_text(_nth(arrays["trigger_name"], idx)),
                    "trig_met": _to_float(_nth(arrays["trig_met"], idx)),
                    "sigma_t": _to_float(_nth(arrays["sigma_t"], idx)),
                    "detector": _to_text(_nth(arrays["detector"], idx)),
                    "detectors": detectors,
                    "channel_range": channel_range,
                    "channel_energy_keV": energy_range,
                    "E_eV": _to_float(_nth(arrays["E_eV"], idx)),
                    "spectral": spectral or None,
                    "source": source or None,
                }
            )
        return records


def load_real_events_from_npz(path: Path) -> List[Dict[str, Any]]:
    """Deserialize the JSON events that were compacted into an NPZ."""
    with np.load(path, allow_pickle=True) as data:
        raw = _load_array(data, "events")
        events: List[Dict[str, Any]] = []
        for entry in raw:
            text = _to_text(entry)
            if not text:
                continue
            try:
                events.append(json.loads(text))
            except json.JSONDecodeError:
                continue
        return events


__all__ = [
    "load_fermi_triggers_from_npz",
    "load_real_events_from_npz",
]
