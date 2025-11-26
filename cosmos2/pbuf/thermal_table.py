"""Thin wrapper to share the per-model ThermalTable implementation."""

from cosmos2.models.pbuf.thermal_table import InMemoryTable, ThermalTable

__all__ = ["ThermalTable", "InMemoryTable"]
