"""Wrapper around the integrated E₀ scanner with configuration + metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .config import QuantumEngineConfig
from .data_access import list_event_files
from .e0 import compute_rigidity, load_event, load_events_from_directory, validate_event


@dataclass(frozen=True)
class E0Result:
    eps0: float
    eps0_error: float
    interval_68: Tuple[float, float]
    interval_95: Tuple[float, float]
    event_count: int
    raw_event_count: int
    scan_steps: int
    eps_range: Tuple[float, float]
    best_loglike: float
    warnings: Tuple[str, ...]

    def stats(self) -> Dict[str, Any]:
        return {
            "eps_range": list(self.eps_range),
            "scan_steps": self.scan_steps,
            "event_count": self.event_count,
            "raw_event_count": self.raw_event_count,
            "interval_68": list(self.interval_68),
            "interval_95": list(self.interval_95),
            "best_loglike": self.best_loglike,
        }


def _load_events(path: Path, patterns: Sequence[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    events: List[Dict[str, Any]] = []
    if path.is_file():
        try:
            payload = load_event(str(path))
            if isinstance(payload, list):
                events.extend(payload)
            else:
                events.append(payload)
        except Exception as exc:  # pragma: no cover - passthrough
            warnings.append(f"Failed to parse {path.name}: {exc}")
        event_files = [path]
    else:
        events = load_events_from_directory(str(path))
        event_files = list_event_files(path, patterns)
    if not events:
        warnings.append(f"No valid events discovered in {path}")
    elif not event_files:
        warnings.append(f"No files matched {patterns} in {path}")
    return events, warnings


def run_e0_pipeline(config: QuantumEngineConfig, event_path: Path) -> E0Result:
    raw_events, warnings = _load_events(event_path, config.data.events_patterns)
    if not raw_events:
        raise RuntimeError(f"No events available at {event_path}")

    valid_events: List[Dict[str, Any]] = []
    for event in raw_events:
        is_valid, error = validate_event(event)
        if is_valid:
            valid_events.append(event)
        else:
            warnings.append(f"Event {event.get('id', 'unknown')} failed validation: {error}")

    if not valid_events:
        raise RuntimeError("All events failed validation; cannot compute ε₀")

    results = compute_rigidity(
        valid_events,
        eps_range=config.eps_range(),
        steps=config.steps,
        k_eps=config.k_eps,
        progress=False,
        threads=config.threads,
    )
    best_eps0 = float(results["best_eps0"])
    uncertainty = results["uncertainty"]
    lower_68 = float(uncertainty["lower_68"])
    upper_68 = float(uncertainty["upper_68"])
    lower_95 = float(uncertainty["lower_95"])
    upper_95 = float(uncertainty["upper_95"])
    eps_error = max(best_eps0 - lower_68, upper_68 - best_eps0)
    loglikes = results.get("loglikes")
    best_loglike = float(np.max(loglikes)) if loglikes is not None else float("nan")

    return E0Result(
        eps0=best_eps0,
        eps0_error=eps_error,
        interval_68=(lower_68, upper_68),
        interval_95=(lower_95, upper_95),
        event_count=len(valid_events),
        raw_event_count=len(raw_events),
        scan_steps=config.steps,
        eps_range=config.eps_range(),
        best_loglike=best_loglike,
        warnings=tuple(warnings),
    )


__all__ = ["E0Result", "run_e0_pipeline"]
