"""
Lightweight dataclasses that describe the α_QM island summary artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IslandSummary:
    regulator: str
    field_set: str
    hits: int
    mixing_min: float
    mixing_max: float
    alpha_min: float
    alpha_max: float
    alpha_mean: float


@dataclass(frozen=True)
class ScanMetadata:
    global_alpha_min: float
    global_alpha_max: float
    total_island_hits: int


__all__ = ["IslandSummary", "ScanMetadata"]
