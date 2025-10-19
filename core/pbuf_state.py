"""
State containers for PBUF cosmology runs.

The goal is to keep parameter bookkeeping, metadata, and versioning
concerns out of the pipeline scripts so that cosmology logic stays
centralised inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional


@dataclass
class ModelParameters:
    """
    Container for cosmological parameters.

    Attributes
    ----------
    values : dict
        Mapping from parameter name to value.
    evolution_policy : dict
        Policy controlling redshift/time evolution (PBUF only).
    """

    values: Dict[str, float]
    evolution_policy: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        """Return a shallow copy of the parameter dictionary."""
        return dict(self.values)


@dataclass
class RunState:
    """
    Track execution metadata for a fitting run.

    Parameters
    ----------
    run_id : str
        Unique identifier for the run (e.g. `SN_MOCK_20250101-000000`).
    model : str
        Model short name (`PBUF` or `LCDM`).
    dataset : Mapping[str, Any]
        Dataset metadata returned by the loaders.
    parameters : ModelParameters
        Tunable parameters at the best fit point.
    version : str, optional
        Semantic version for the codebase snapshot.
    extras : dict
        Additional fields (e.g., git commit hash) propagated to outputs.
    """

    run_id: str
    model: str
    dataset: Mapping[str, Any]
    parameters: ModelParameters
    version: str = "v0.1.0"
    extras: MutableMapping[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Return a dictionary serialisable to JSON for provenance tracking."""
        payload = {
            "run_id": self.run_id,
            "model": self.model,
            "dataset": dict(self.dataset),
            "parameters": self.parameters.as_dict(),
            "evolution_policy": dict(self.parameters.evolution_policy),
            "code_version": self.version,
        }
        payload.update(self.extras)
        return payload


def with_defaults(params: Optional[Mapping[str, float]], defaults: Mapping[str, float]) -> Dict[str, float]:
    """
    Merge user-specified parameters with defaults.

    This helper ensures that optimisation routines always receive a full
    parameter vector.
    """

    merged = dict(defaults)
    if params:
        merged.update(params)
    return merged
