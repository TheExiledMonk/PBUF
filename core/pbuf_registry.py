"""
Formula registry for PBUF cosmology expressions.

The registry stores provenance metadata for canonical and legacy
formulae, making it easy to track changes and ensure reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass(frozen=True)
class FormulaRecord:
    """Metadata for a registered formula."""

    name: str
    expression: str
    source: str
    status: str


_REGISTRY: Dict[str, FormulaRecord] = {}


def register(name: str, expression: str, source: str, status: str = "canonical") -> None:
    """
    Register or update a formula record.

    Parameters
    ----------
    name : str
        Unique identifier.
    expression : str
        Human-readable expression or reference.
    source : str
        Citation or provenance note.
    status : str
        One of {"canonical", "legacy", "experimental"}.
    """

    _REGISTRY[name] = FormulaRecord(name, expression, source, status)


def get(name: str) -> FormulaRecord:
    """Return the recorded formula metadata."""

    return _REGISTRY[name]


def list_formulas() -> Iterable[FormulaRecord]:
    """Yield all registered formulas."""

    return _REGISTRY.values()


# Seed registry with the baseline distance modulus relation.
register(
    name="distance_modulus",
    expression="μ(z) = 5 log10(D_L / 10 pc)",
    source="Standard SN cosmology relation",
    status="canonical",
)

register(
    name="z_recomb_hu96",
    expression="Hu & Sugiyama (1996) fitting formula for z_*",
    source="Hu & Sugiyama, ApJ 471, 542 (1996)",
    status="legacy",
)

register(
    name="r_s_integral",
    expression="r_s(z) = ∫ c_s(z) / H(z) dz",
    source="FRW background with photon-baryon sound speed",
    status="canonical",
)
