"""Helpers that power the basin optimisation targets for PBUF."""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

from cosmos.models.pbuf.microphysics import ensure_thermal_table, run_microphysics_bootstrap
from cosmos.models.pbuf.phase7a import make_phase7a_checker
from cosmos.models.pbuf.thermal_table import ThermalTable
from cosmos.optim.sanity import evaluate_candidate

ParamDict = Dict[str, float]


class PBUFBasinModel:
    """Encapsulates the quantum bootstrap plus dataset evaluation."""

    def __init__(self, *, dataset_weights: dict[str, float] | None = None) -> None:
        self._thermal_table: ThermalTable | None = None
        self._thermal_metadata: Dict[str, Any] | None = None
        self._dataset_weights = {
            key.lower(): float(value)
            for key, value in (dataset_weights or {}).items()
        }

    def ensure_quantum_and_thermal_table(self, *, datasets: Sequence[str] | None = None) -> None:
        if self._thermal_table is not None:
            return

        ordered = [name.lower() for name in (datasets or [])]
        metadata = run_microphysics_bootstrap(list(dict.fromkeys(ordered)))
        self._thermal_metadata = metadata
        self._thermal_table = ensure_thermal_table()

    def evaluate(self, params: ParamDict, dataset_names: Sequence[str]) -> float:
        if self._thermal_table is None:
            raise RuntimeError("Quantum thermal table has not been prepared.")

        normalized_datasets = [name.lower() for name in dataset_names]
        sanitized = {key: float(value) for key, value in params.items()}
        chi2, extras = evaluate_candidate("pbuf", sanitized, normalized_datasets)
        dataset_summaries = extras.get("dataset_summaries", {})

        weighted_total = 0.0
        for dataset_name in normalized_datasets:
            summary = dataset_summaries.get(dataset_name)
            if summary is None:
                continue
            dataset_chi2 = summary.get("chi2")
            if dataset_chi2 is None:
                continue
            weight = self._dataset_weights.get(dataset_name, 1.0)
            weighted_total += float(weight) * float(dataset_chi2)

        return float(weighted_total if weighted_total != 0.0 else chi2)

    def phase7a_checker(self) -> Callable[[ParamDict], Tuple[bool, str | None]]:
        if self._thermal_table is None:
            raise RuntimeError("Quantum thermal table has not been prepared.")

        return make_phase7a_checker(self._thermal_table, self._thermal_metadata)

    def phase6a_checker(self) -> Callable[[ParamDict], Tuple[bool, str | None]]:
        return self.phase7a_checker()
