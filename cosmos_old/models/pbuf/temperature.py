"""Temperature lookups sourced from the Quantum thermal table."""

from __future__ import annotations

from cosmos.models.pbuf.thermal_table import ThermalTable


def T_of_a(a: float, table: ThermalTable) -> float:
    """Return the photon temperature at the supplied scale factor."""

    return table.get("T", at_scale_factor=a)


def T_of_z(z: float, table: ThermalTable) -> float:
    """Return the photon temperature at the supplied redshift."""

    return table.get_by_z("T", at_redshift=z)


__all__ = ["T_of_a", "T_of_z"]
