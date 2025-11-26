"""Temperature lookups sourced from the Quantum thermal table (ported from cosmos_old)."""

from __future__ import annotations

from .thermal_table import ThermalTable


def T_of_a(a: float, table: ThermalTable) -> float:
    """Return the photon temperature at the supplied scale factor."""

    try:
        return table.fast_get("T", at_scale_factor=a)
    except Exception:
        return table.get("T", at_scale_factor=a)


def T_of_z(z: float, table: ThermalTable) -> float:
    """Return the photon temperature at the supplied redshift."""

    a_val = 1.0 / (1.0 + float(z))
    try:
        return table.fast_get("T", at_scale_factor=a_val)
    except Exception:
        return table.get_by_z("T", at_redshift=z)


__all__ = ["T_of_a", "T_of_z"]
