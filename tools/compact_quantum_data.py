#!/usr/bin/env python3
"""Create compressed NPZ artifacts for quantum triggers and events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from quantum.pipeline.fermi_loader import load_fermi_directory


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_tuple(pair: Any, length: int = 2) -> tuple[float, ...]:
    if isinstance(pair, Sequence) and len(pair) >= length:
        return tuple(float(v) if v is not None else float("nan") for v in pair[:length])
    return tuple(float("nan") for _ in range(length))


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


def _collect_trigger_data(triggers: Sequence[Dict[str, Any]]) -> Dict[str, Iterable[Any]]:
    floats = {
        "trig_met": [],
        "sigma_t": [],
        "E_eV": [],
        "channel_lo": [],
        "channel_hi": [],
        "energy_lo_keV": [],
        "energy_hi_keV": [],
        "spectral_E_peak_keV": [],
        "spectral_flux": [],
    }
    strings = {
        "trigger_name": [],
        "detector": [],
        "spectral_flux_field": [],
        "spectral_source": [],
        "source_dir": [],
        "source_tcat": [],
        "source_trigdat": [],
        "detector_mask": [],
        "detectors_json": [],
        "detectors_triggered": [],
        "detectors_ranked": [],
    }
    ints = {"num_triggered": []}

    for trigger in triggers:
        floats["trig_met"].append(_as_float(trigger.get("trig_met", float("nan"))))
        floats["sigma_t"].append(_as_float(trigger.get("sigma_t", float("nan"))))
        floats["E_eV"].append(_as_float(trigger.get("E_eV", float("nan"))))

        channel_range = _as_tuple(trigger.get("channel_range"))
        floats["channel_lo"].append(channel_range[0])
        floats["channel_hi"].append(channel_range[1])

        energy_range = _as_tuple(trigger.get("channel_energy_keV"))
        floats["energy_lo_keV"].append(energy_range[0])
        floats["energy_hi_keV"].append(energy_range[1])

        spectral = trigger.get("spectral") or {}
        floats["spectral_E_peak_keV"].append(_as_float(spectral.get("E_peak_keV")))
        floats["spectral_flux"].append(_as_float(spectral.get("flux")))

        strings["trigger_name"].append(_stringify(trigger.get("trigger_name")))
        strings["detector"].append(_stringify(trigger.get("detector")))
        strings["spectral_flux_field"].append(_stringify(spectral.get("flux_field")))
        strings["spectral_source"].append(_stringify(spectral.get("source")))

        source = trigger.get("source") or {}
        if isinstance(source, dict):
            strings["source_dir"].append(_stringify(source.get("directory")))
            strings["source_tcat"].append(_stringify(source.get("tcat")))
            strings["source_trigdat"].append(_stringify(source.get("trigdat")))
        else:
            strings["source_dir"].append(_stringify(source))
            strings["source_tcat"].append("")
            strings["source_trigdat"].append("")

        detectors = trigger.get("detectors") or {}
        strings["detector_mask"].append(_stringify(detectors.get("mask")))
        strings["detectors_json"].append(_stringify(detectors))
        strings["detectors_triggered"].append(_stringify(detectors.get("triggered")))
        strings["detectors_ranked"].append(_stringify(detectors.get("ranked_rates")))
        ints["num_triggered"].append(len(detectors.get("triggered") or []))

    return {**floats, **strings, **ints}


DATA_ROOT = Path("data")
QUANTUM_DIR = DATA_ROOT / "quantum"
STANDARDIZED_DIR = DATA_ROOT / "standardized"
FERMI_DIR = QUANTUM_DIR / "fermi"
REAL_EVENTS_JSON = QUANTUM_DIR / "events" / "real_events.json"
FERMI_OUTPUT = STANDARDIZED_DIR / "quantum_fermi_trigger.npz"
EVENT_OUTPUT = STANDARDIZED_DIR / "quantum_real_events.npz"
STRATEGY = "brightest"


def _save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _serialize_real_events(json_path: Path) -> Dict[str, np.ndarray]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    events = raw if isinstance(raw, list) else [raw]
    serialized = [
        json.dumps(event, separators=(",", ":"), sort_keys=True)
        for event in events
    ]
    ids = [str(event.get("id", "")) for event in events]
    return {
        "events": np.array(serialized, dtype=object),
        "event_id": np.array(ids, dtype=object),
    }


def _to_numpy(data: Dict[str, Iterable[Any]]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for key, values in data.items():
        sample = list(values)
        if not sample:
            arrays[key] = np.array([], dtype=float if key in {"trig_met", "sigma_t", "E_eV"} else object)
            continue
        if all(isinstance(value, (int, float)) for value in sample):
            arrays[key] = np.array(sample, dtype=float)
        else:
            arrays[key] = np.array(sample, dtype=object)
    return arrays


def main() -> None:
    triggers = load_fermi_directory(FERMI_DIR, strategy=STRATEGY)
    if not triggers:
        raise RuntimeError(f"No triggers resolved from {FERMI_DIR}")

    data = _collect_trigger_data(triggers)
    arrays = _to_numpy(data)

    _save_npz(FERMI_OUTPUT, arrays)
    print(f"[OK] Wrote {len(triggers)} triggers → {FERMI_OUTPUT}")

    if REAL_EVENTS_JSON.exists():
        event_arrays = _serialize_real_events(REAL_EVENTS_JSON)
        _save_npz(EVENT_OUTPUT, event_arrays)
        print(f"[OK] Wrote {len(event_arrays['event_id'])} events → {EVENT_OUTPUT}")
    else:
        print(f"[WARN] Real events JSON not found at {REAL_EVENTS_JSON}; skipping NPZ generation.")


if __name__ == "__main__":
    main()
