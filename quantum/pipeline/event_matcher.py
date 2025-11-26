"""Matching utilities between GW and Fermi triggers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import log_pipeline_error
from .time_conversion import met_to_gps


def match_fermi_to_gw(
    gw_gps: float,
    fermi_triggers: List[Dict[str, Any]],
    time_window: float,
) -> Optional[Dict[str, Any]]:
    """Return the closest-in-time Fermi trigger within the matching window."""
    if time_window <= 0.0:
        raise ValueError("time_window must be positive")

    best_match: Optional[Dict[str, Any]] = None
    min_dt = float("inf")
    for trigger in fermi_triggers:
        try:
            gamma_gps = met_to_gps(trigger["trig_met"])
        except KeyError:
            log_pipeline_error(f"Trigger missing trig_met field: {trigger}")
            continue
        dt = abs(gamma_gps - float(gw_gps))
        if dt < time_window and dt < min_dt:
            min_dt = dt
            best_match = trigger
    return best_match


def pair_matched_events(
    gw_events: List[Dict[str, Any]],
    fermi_triggers: List[Dict[str, Any]],
    time_window: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return GW/Fermi event pairs within the time window."""
    matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for gw_event in gw_events:
        match = match_fermi_to_gw(gw_event["gps_time"], fermi_triggers, time_window)
        if match is None:
            log_pipeline_error(
                f"No Fermi match for {gw_event.get('event_name', 'UNKNOWN')} "
                f"within {time_window}s"
            )
            continue
        matches.append((gw_event, match))
    return matches


__all__ = ["match_fermi_to_gw", "pair_matched_events"]
