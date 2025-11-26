"""Time conversion utilities for GW ⇄ Fermi coordination."""

from __future__ import annotations

from typing import Final

# Difference between GPS epoch (1980-01-06) and Fermi MET epoch (2001-01-01) in seconds,
# including leap seconds (GPS runs ahead of UTC by 13 s at that epoch).
FERMI_MET_TO_GPS: Final[float] = 662_342_413.0


def met_to_gps(met: float) -> float:
    """Convert Fermi Mission Elapsed Time (MET) to GPS seconds."""
    return float(met) + FERMI_MET_TO_GPS


def compute_dt_gamma(gw_gps: float, gamma_met: float) -> float:
    """Return gamma-ray arrival time relative to the GW merger (gamma - gw)."""
    gamma_gps = met_to_gps(float(gamma_met))
    return gamma_gps - float(gw_gps)


__all__ = [
    "FERMI_MET_TO_GPS",
    "met_to_gps",
    "compute_dt_gamma",
]
