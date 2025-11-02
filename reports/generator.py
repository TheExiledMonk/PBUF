"""Helpers for building science-run reporting payloads."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


def collect_dataset_partitions(results: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    partitions: Dict[str, List[Dict[str, Any]]] = {}
    for entry in results:
        breakdown = entry.get("fiducial_breakdown") or entry.get("chi2_breakdown")
        if not isinstance(breakdown, Mapping):
            continue
        scenario_id = entry.get("scenario_id")
        model = entry.get("model")
        runtime = entry.get("metadata", {}).get("runtime_seconds")
        for dataset, value in breakdown.items():
            try:
                chi2_value = float(value)
            except (TypeError, ValueError):
                continue
            payload: Dict[str, Any] = {
                "scenario_id": scenario_id,
                "model": model,
                "chi2": chi2_value,
            }
            if runtime is not None:
                payload["runtime_seconds"] = float(runtime)
            partitions.setdefault(dataset, []).append(payload)
    return partitions


def summarise_run(results: Iterable[Mapping[str, Any]], *, include_partitions: bool = False) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_results": 0,
    }
    aggregated: List[Mapping[str, Any]] = []
    for entry in results:
        aggregated.append(entry)
    summary["num_results"] = len(aggregated)
    if include_partitions:
        summary["per_dataset"] = collect_dataset_partitions(aggregated)
    return summary


__all__ = ["collect_dataset_partitions", "summarise_run"]
