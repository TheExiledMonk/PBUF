"""
Summary Builder — Science Run Aggregator for ΛCDM & PBUF
========================================================

This module ingests the structured outputs generated under
`data/science_runs/` and produces a rich statistics object that powers
the reporting stack (Markdown, HTML, PDF, JSON).

For every science run directory it captures:
  * run metadata (`meta.json`, `state.json`)
  * per-model artifacts (`artifacts/*-done.json`)
  * joint comparison artifacts (`*-joint-comparison.json`)
  * scout (single-dataset) diagnostics

The resulting data model contains both aggregated metrics (χ², AIC, BIC,
reduced χ², dataset breakdowns) and deep per-run provenance that can be
rendered in detailed reports.
"""

from collections import defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Mapping


# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------

def _normalise_model_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return str(name).strip().lower()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON file {path}: {exc}") from exc


def _extract_order(artifact_path: Path) -> int:
    """Extract the numeric prefix order from an artifact filename."""
    stem = artifact_path.stem  # e.g. "13-geom-lcdm-done"
    prefix = stem.split("-", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return 0


def _infer_scenario_from_filename(artifact_path: Path) -> Optional[str]:
    parts = artifact_path.stem.split("-")
    if len(parts) >= 2:
        return parts[1]
    return None


def _expected_raw_path(run_path: Path, artifact_path: Path) -> Optional[str]:
    """Map an artifact file back to the raw CLI output if available."""
    raw_dir = run_path / "raw"
    if not raw_dir.exists():
        return None

    candidate_name = artifact_path.name.replace("-done", "").replace("-started", "")
    candidate = raw_dir / candidate_name
    if candidate.exists():
        return str(candidate)
    return None


def _init_dataset_model_entry() -> Dict[str, Any]:
    return {
        "chi2": 0.0,
        "n_data": 0,
        "n_params": 0,
        "dof": 0,
        "AIC": 0.0,
        "BIC": 0.0,
        "runs": [],
        "scout_runs": [],
        "parameters": [],
    }


def _init_model_aggregate() -> Dict[str, Any]:
    return {
        "chi2_total": 0.0,
        "AIC_total": 0.0,
        "BIC_total": 0.0,
        "chi2_reduced_total": None,
        "n_data_total": 0,
        "n_params_total": 0,
        "dof_total": 0,
        "run_count": 0,
        "runs": [],
    }


def _select_primary_scenario(
    scenarios: List[Dict[str, Any]],
    models: List[str],
) -> Optional[Dict[str, Any]]:
    """
    Choose the scenario with the largest data volume across the requested models.
    Scout scenarios (prefixed with "scout:") are ignored.
    """
    best_scenario = None
    best_score = -1
    best_priority = -1

    for scenario in scenarios:
        scenario_id = scenario.get("id", "")
        if scenario_id.startswith("scout:"):
            continue

        score = 0
        for model in models:
            entry = scenario.get("models", {}).get(model)
            if not entry:
                continue
            fit_stats = entry.get("fit_stats", {})
            score = max(score, int(fit_stats.get("n_data", 0) or 0))

        priority = 1 if _uses_absolute_pantheon(scenario) else 0

        if score > best_score or (score == best_score and priority > best_priority):
            best_scenario = scenario
            best_score = score
            best_priority = priority

    return best_scenario


def _uses_absolute_pantheon(scenario: Dict[str, Any]) -> bool:
    """Return True if the scenario explicitly runs the absolute Pantheon path."""
    datasets = set()

    for name in scenario.get("datasets", []) or []:
        datasets.add(str(name).strip().lower())

    for model_entry in (scenario.get("models") or {}).values():
        for name in model_entry.get("datasets", []) or []:
            datasets.add(str(name).strip().lower())

    if any(name in {"sn_pantheon_abs", "pantheon_abs"} for name in datasets):
        return True

    options = scenario.get("options") or {}
    sn_mode = options.get("sn_mode")
    if sn_mode and str(sn_mode).strip().lower() == "absolute":
        return True

    return False


# ----------------------------------------------------------------------
# 1. Collect raw science run data
# ----------------------------------------------------------------------

def collect_fit_results(science_runs_root: str = "data/science_runs") -> Dict[str, Any]:
    """
    Traverse `data/science_runs/` and collect all per-run artifacts.

    Returns
    -------
    dict
        {
            "root": "<absolute path to science_runs>",
            "collected_at": "<UTC timestamp>",
            "runs": [
                {
                    "name": "<run directory>",
                    "path": "<absolute path>",
                    "meta": {...},
                    "state": {...},
                    "models_present": [...],
                    "scouts": {
                        "<dataset>": {
                            "<model>": {
                                "artifact_path": "...",
                                "raw_path": "...",
                                "fit_stats": {...},
                                ...
                            }
                        }
                    },
                    "scenarios": [
                        {
                            "id": "geom_plus_sn_abs",
                            "order": 25,
                            "models": { "<model>": {...} },
                            "joint": {...}
                        },
                        ...
                    ]
                }
            ]
        }
    """
    root_path = Path(science_runs_root)
    bundle: Dict[str, Any] = {
        "root": str(root_path.resolve()) if root_path.exists() else str(root_path),
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "runs": [],
    }

    if not root_path.exists():
        return bundle

    for run_path in sorted(p for p in root_path.iterdir() if p.is_dir()):
        run_record: Dict[str, Any] = {
            "name": run_path.name,
            "path": str(run_path.resolve()),
            "meta": _read_json(run_path / "meta.json") or {},
            "state": _read_json(run_path / "state.json") or {},
            "models_present": [],
            "scouts": {},
            "scenarios": [],
        }

        artifacts_dir = run_path / "artifacts"
        if artifacts_dir.exists():
            scenario_map: Dict[str, Dict[str, Any]] = {}

            for artifact_file in sorted(artifacts_dir.glob("*.json")):
                filename = artifact_file.name
                if filename.endswith("-started.json") or filename.endswith("-failed.json"):
                    continue

                data = _read_json(artifact_file)
                if data is None:
                    continue

                order = _extract_order(artifact_file)
                scenario_id = data.get("scenario") or _infer_scenario_from_filename(artifact_file)
                if not scenario_id:
                    continue

                # Joint comparison artifacts store both models together.
                if "models" in data and "model" not in data:
                    scenario_entry = scenario_map.setdefault(
                        scenario_id,
                        {"id": scenario_id, "order": order, "models": {}, "joint": None},
                    )
                    scenario_entry["joint"] = {
                        "artifact_path": str(artifact_file.resolve()),
                        "deltas": data.get("deltas", {}),
                        "options": data.get("options", {}),
                        "parity": data.get("parity", {}),
                        "provenance": data.get("provenance", {}),
                        "models": data.get("models", {}),
                    }
                    continue

                model_name = _normalise_model_name(data.get("model") or data.get("model_type"))
                if not model_name:
                    continue

                record = {
                    "artifact_path": str(artifact_file.resolve()),
                    "raw_path": _expected_raw_path(run_path, artifact_file),
                    "datasets": data.get("datasets", []),
                    "best_fit": data.get("best_fit", {}),
                    "fit_stats": data.get("fit_stats", {}),
                    "runtime": data.get("runtime", {}),
                    "physics_flags": data.get("physics_flags", {}),
                    "provenance": data.get("provenance", {}),
                }

                if scenario_id.startswith("scout:"):
                    dataset = scenario_id.split(":", 1)[1]
                    run_record["scouts"].setdefault(dataset, {})[model_name] = record
                else:
                    scenario_entry = scenario_map.setdefault(
                        scenario_id,
                        {"id": scenario_id, "order": order, "models": {}, "joint": None},
                    )
                    # Keep the smallest order in case of multiple artifacts per scenario.
                    scenario_entry["order"] = min(scenario_entry.get("order", order), order)
                    scenario_entry["models"][model_name] = record

            run_record["scenarios"] = sorted(
                scenario_map.values(),
                key=lambda item: (item.get("order", 0), item.get("id", "")),
            )
        if not run_record["scenarios"]:
            _populate_scenarios_from_summary(run_path, run_record)

        model_names = set()
        for scenario in run_record["scenarios"]:
            model_names.update(scenario.get("models", {}).keys())
        for dataset_models in run_record["scouts"].values():
            model_names.update(dataset_models.keys())
        run_record["models_present"] = sorted(model_names)

        bundle["runs"].append(run_record)

    return bundle


def _union_datasets(results: Sequence[Mapping[str, Any]]) -> List[str]:
    datasets: List[str] = []
    seen = set()
    for entry in results:
        for dataset in entry.get("datasets_used", []) or []:
            ds = str(dataset).strip()
            if ds and ds not in seen:
                datasets.append(ds)
                seen.add(ds)
    return datasets


def _load_dataset_catalog(names: List[str]) -> Dict[str, Dict[str, Any]]:
    if not names:
        return {}
    try:
        from scripts.run_science import collect_dataset_catalog  # type: ignore
    except Exception:
        return {name.lower(): {"n_data": None} for name in names}
    return collect_dataset_catalog(names)


def _populate_scenarios_from_summary(run_path: Path, run_record: Dict[str, Any]) -> None:
    summary_path = run_path / "summary.json"
    summary = _read_json(summary_path)
    if not summary:
        return

    results = summary.get("results")
    if not isinstance(results, list) or not results:
        return

    scenario_order = {
        scenario_id: idx for idx, scenario_id in enumerate(summary.get("scenarios") or [])
    }

    datasets = _union_datasets(results)
    dataset_catalog = _load_dataset_catalog(datasets)

    scenario_map: Dict[str, Dict[str, Any]] = {}

    for entry in results:
        scenario_id = entry.get("scenario_id")
        model_name = _normalise_model_name(entry.get("model") or entry.get("model_type"))
        if not scenario_id or not model_name:
            continue

        datasets_used = entry.get("datasets_used") or []
        params = entry.get("fiducial_params") or {}
        breakdown = entry.get("fiducial_breakdown") or {}
        chi2_total = entry.get("fiducial_chi2")

        n_params = len(params)
        n_data_per_dataset: Dict[str, Optional[int]] = {}
        total_n_data = 0
        for dataset in datasets_used:
            info = dataset_catalog.get(str(dataset).lower(), {})
            n_data = info.get("n_data")
            n_data_per_dataset[dataset] = n_data
            if isinstance(n_data, int):
                total_n_data += n_data

        dof = total_n_data - n_params if total_n_data and n_params else None
        aic = chi2_total + 2 * n_params if chi2_total is not None else None
        bic = (
            chi2_total + n_params * math.log(total_n_data)
            if chi2_total is not None and total_n_data
            else None
        )
        chi2_reduced = chi2_total / dof if dof and chi2_total is not None else None

        physics_flags: Dict[str, Any] = {
            "phase6a_applied": bool(entry.get("phase6a_enforced", model_name == "pbuf")),
        }
        phase6a_passed = entry.get("fiducial_passes_phase6a")
        if phase6a_passed is not None:
            physics_flags["phase6a_passed"] = bool(phase6a_passed)
        if "fiducial_validation" in entry:
            physics_flags["validation"] = entry["fiducial_validation"]

        runtime_seconds = entry.get("metadata", {}).get("runtime_seconds")
        runtime = {
            "wall_seconds": runtime_seconds,
        }

        record = {
            "artifact_path": None,
            "raw_path": None,
            "datasets": list(datasets_used),
            "best_fit": {
                "params": params,
                "derived": entry.get("fiducial_derived", {}),
            },
            "fit_stats": {
                "chi2_total": chi2_total,
                "chi2_per_dataset": breakdown,
                "n_params": n_params,
                "n_data": total_n_data if total_n_data else None,
                "n_data_per_dataset": n_data_per_dataset,
                "dof": dof,
                "chi2_reduced": chi2_reduced,
                "aic": aic,
                "bic": bic,
            },
            "runtime": runtime,
            "physics_flags": physics_flags,
            "provenance": {
                "source": str(summary_path),
                "timestamp": entry.get("timestamp_utc"),
            },
        }

        scenario_entry = scenario_map.setdefault(
            scenario_id,
            {
                "id": scenario_id,
                "order": scenario_order.get(scenario_id, len(scenario_map) + 1),
                "models": {},
                "joint": None,
                "options": {},
            },
        )
        scenario_entry["models"][model_name] = record

    # Build joint comparison payloads when both LCDM and PBUF are present
    for scenario_id, scenario_entry in scenario_map.items():
        models = scenario_entry.get("models", {})
        if "lcdm" in models and "pbuf" in models:
            lcdm_stats = models["lcdm"]["fit_stats"]
            pbuf_stats = models["pbuf"]["fit_stats"]
            delta_chi2 = None
            if lcdm_stats.get("chi2_total") is not None and pbuf_stats.get("chi2_total") is not None:
                delta_chi2 = pbuf_stats["chi2_total"] - lcdm_stats["chi2_total"]
            delta_aic = None
            if lcdm_stats.get("aic") is not None and pbuf_stats.get("aic") is not None:
                delta_aic = pbuf_stats["aic"] - lcdm_stats["aic"]
            delta_bic = None
            if lcdm_stats.get("bic") is not None and pbuf_stats.get("bic") is not None:
                delta_bic = pbuf_stats["bic"] - lcdm_stats["bic"]

            scenario_entry["joint"] = {
                "artifact_path": None,
                "deltas": {
                    "delta_chi2": delta_chi2,
                    "delta_aic": delta_aic,
                    "delta_bic": delta_bic,
                },
                "models": {
                    "lcdm": {
                        "chi2_total": lcdm_stats.get("chi2_total"),
                        "aic": lcdm_stats.get("aic"),
                        "bic": lcdm_stats.get("bic"),
                        "dof": lcdm_stats.get("dof"),
                        "chi2_per_dataset": lcdm_stats.get("chi2_per_dataset"),
                        "params": models["lcdm"]["best_fit"].get("params"),
                    },
                    "pbuf": {
                        "chi2_total": pbuf_stats.get("chi2_total"),
                        "aic": pbuf_stats.get("aic"),
                        "bic": pbuf_stats.get("bic"),
                        "dof": pbuf_stats.get("dof"),
                        "chi2_per_dataset": pbuf_stats.get("chi2_per_dataset"),
                        "params": models["pbuf"]["best_fit"].get("params"),
                    },
                },
                "options": {},
                "parity": {},
            }

    run_record["scenarios"] = sorted(
        scenario_map.values(),
        key=lambda item: (item.get("order", 0), item.get("id", "")),
    )

    model_names = set()
    for scenario in run_record["scenarios"]:
        model_names.update(scenario.get("models", {}).keys())
    run_record["models_present"] = sorted(model_names)


# ----------------------------------------------------------------------
# 2. Compute aggregated statistics
# ----------------------------------------------------------------------

def compute_model_stats(run_bundle: Dict[str, Any], models: List[str]) -> Dict[str, Any]:
    """
    Aggregate statistics across all collected science runs.

    Parameters
    ----------
    run_bundle : dict
        Output from `collect_fit_results`.
    models : list[str]
        Models to include (case-insensitive). Example: ["lcdm", "pbuf"].

    Returns
    -------
    dict
        {
            "runs": [...],                          # enriched run records
            "aggregated": {
                "models": {...},                    # global per-model totals
                "datasets": {...},                  # dataset-level aggregates
                "global": {...},                    # comparison metadata
            },
            "metadata": {...}                       # provenance of aggregation
        }
    """
    runs: List[Dict[str, Any]] = run_bundle.get("runs", [])
    requested_models = [_normalise_model_name(m) for m in models if _normalise_model_name(m)]

    aggregated_models: Dict[str, Dict[str, Any]] = {
        model: _init_model_aggregate() for model in requested_models
    }
    aggregated_datasets: Dict[str, Dict[str, Any]] = defaultdict(dict)
    global_info: Dict[str, Any] = {
        "run_count": len(runs),
        "primary_scenarios": {},
        "requested_models": requested_models,
    }

    # Primary scenario aggregation (multi-dataset runs)
    for run in runs:
        scenarios = run.get("scenarios", [])
        primary = _select_primary_scenario(scenarios, requested_models)
        run["primary_scenario_id"] = primary.get("id") if primary else None
        if primary:
            global_info["primary_scenarios"][run["name"]] = primary["id"]

        if not primary:
            continue

        for model in requested_models:
            model_entry = primary.get("models", {}).get(model)
            if not model_entry:
                continue

            fit_stats = model_entry.get("fit_stats", {}) or {}
            aggregate = aggregated_models.setdefault(model, _init_model_aggregate())

            chi2_total = float(fit_stats.get("chi2_total", 0.0) or 0.0)
            aggregate["chi2_total"] += chi2_total
            aggregate["AIC_total"] += float(fit_stats.get("aic", 0.0) or 0.0)
            aggregate["BIC_total"] += float(fit_stats.get("bic", 0.0) or 0.0)

            n_data = int(fit_stats.get("n_data", 0) or 0)
            n_params = int(fit_stats.get("n_params", 0) or 0)
            aggregate["n_data_total"] += n_data
            aggregate["n_params_total"] += n_params

            dof = fit_stats.get("dof")
            if dof is None:
                dof = n_data - n_params
            aggregate["dof_total"] += int(dof)

            aggregate["runs"].append(
                {
                    "run": run["name"],
                    "scenario": primary["id"],
                    "chi2_total": chi2_total,
                    "AIC": float(fit_stats.get("aic", 0.0) or 0.0),
                    "BIC": float(fit_stats.get("bic", 0.0) or 0.0),
                    "n_data": n_data,
                    "n_params": n_params,
                    "runtime": model_entry.get("runtime", {}),
                }
            )

            chi2_per_dataset = fit_stats.get("chi2_per_dataset", {}) or {}
            n_data_per_dataset = fit_stats.get("n_data_per_dataset", {}) or {}

            for dataset, chi2_value in chi2_per_dataset.items():
                dataset_key = str(dataset).strip().lower()
                if dataset_key == "sn_pantheon":
                    continue
                dataset_entry = aggregated_datasets.setdefault(dataset, {})
                dataset_model = dataset_entry.setdefault(model, _init_dataset_model_entry())
                dataset_model["chi2"] += float(chi2_value or 0.0)
                dataset_model["n_data"] += int(n_data_per_dataset.get(dataset, 0) or 0)
                dataset_model["runs"].append(
                    {
                        "run": run["name"],
                        "scenario": primary["id"],
                        "chi2": float(chi2_value or 0.0),
                        "n_data": int(n_data_per_dataset.get(dataset, 0) or 0),
                    }
                )

    # Scout aggregation (single-dataset diagnostics: AIC/BIC, params)
    for run in runs:
        scouts = run.get("scouts", {})
        for dataset, models_data in scouts.items():
            dataset_key = str(dataset).strip().lower()
            if dataset_key == "sn_pantheon":
                continue
            dataset_entry = aggregated_datasets.setdefault(dataset, {})

            for model, scout_data in models_data.items():
                model_key = _normalise_model_name(model) or model
                dataset_model = dataset_entry.setdefault(model_key, _init_dataset_model_entry())

                fit_stats = scout_data.get("fit_stats", {}) or {}
                dataset_model["AIC"] += float(fit_stats.get("aic", 0.0) or 0.0)
                dataset_model["BIC"] += float(fit_stats.get("bic", 0.0) or 0.0)
                dataset_model["n_params"] += int(fit_stats.get("n_params", 0) or 0)

                dof = fit_stats.get("dof")
                if dof is None:
                    dof = int(fit_stats.get("n_data", 0) or 0) - int(fit_stats.get("n_params", 0) or 0)
                dataset_model["dof"] += int(dof)

                dataset_model["scout_runs"].append(
                    {
                        "run": run["name"],
                        "chi2": float(fit_stats.get("chi2_total", 0.0) or 0.0),
                        "AIC": float(fit_stats.get("aic", 0.0) or 0.0),
                        "BIC": float(fit_stats.get("bic", 0.0) or 0.0),
                        "n_data": int(fit_stats.get("n_data", 0) or 0),
                        "n_params": int(fit_stats.get("n_params", 0) or 0),
                        "runtime": scout_data.get("runtime", {}),
                    }
                )

                params = scout_data.get("best_fit", {}).get("params")
                if params:
                    dataset_model["parameters"].append({"run": run["name"], "params": params})

    # Post-processing derived metrics
    for model, aggregate in aggregated_models.items():
        run_count = len(aggregate["runs"])
        aggregate["run_count"] = run_count
        dof_total = max(int(aggregate.get("dof_total", 0)), 1)
        aggregate["chi2_reduced_total"] = (
            aggregate["chi2_total"] / dof_total if dof_total else None
        )

    # Remove models with no supporting runs
    aggregated_models = {
        model: data for model, data in aggregated_models.items() if data["run_count"] > 0
    }

    for dataset_models in aggregated_datasets.values():
        for model_name, dataset_model in dataset_models.items():
            dof_total = dataset_model.get("dof", 0)
            if not dof_total:
                dof_total = dataset_model.get("n_data", 0) - dataset_model.get("n_params", 0)
            dof_total = max(int(dof_total), 1)
            dataset_model["chi2_reduced"] = dataset_model["chi2"] / dof_total

            unique_runs = {
                entry["run"] for entry in dataset_model["runs"] + dataset_model["scout_runs"]
            }
            run_count = max(len(unique_runs), 1)
            dataset_model["AIC_average"] = dataset_model["AIC"] / run_count if run_count else None
            dataset_model["BIC_average"] = dataset_model["BIC"] / run_count if run_count else None

    # Global comparison (ΔAIC, ΔBIC) for the first two requested models
    ordered_models = [m for m in requested_models if m in aggregated_models]
    if len(ordered_models) >= 2:
        m1, m2 = ordered_models[:2]
        a1, a2 = aggregated_models[m1], aggregated_models[m2]
        delta_aic = a2["AIC_total"] - a1["AIC_total"]
        delta_bic = a2["BIC_total"] - a1["BIC_total"]
        global_info["comparison"] = {
            f"ΔAIC ({m2}-{m1})": delta_aic,
            f"ΔBIC ({m2}-{m1})": delta_bic,
            "preferred_model_AIC": m1 if delta_aic > 0 else m2,
            "preferred_model_BIC": m1 if delta_bic > 0 else m2,
        }

    aggregated = {
        "models": aggregated_models,
        "datasets": dict(aggregated_datasets),
        "global": global_info,
    }

    stats = {
        "runs": runs,
        "aggregated": aggregated,
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "science_runs_root": run_bundle.get("root"),
            "collected_at": run_bundle.get("collected_at"),
            "requested_models": models,
        },
    }
    return stats


# ----------------------------------------------------------------------
# 3. Persist aggregated statistics
# ----------------------------------------------------------------------

def export_stats_to_json(
    stats: Dict[str, Any],
    output_file: str = "reports/output/stats_summary.json",
) -> str:
    """
    Write the computed statistics bundle to disk.

    Parameters
    ----------
    stats : dict
        Output from `compute_model_stats`.
    output_file : str
        Destination path for the JSON export.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return str(output_path.resolve())


# ----------------------------------------------------------------------
# Developer notes
# ----------------------------------------------------------------------
#
#  * collect_fit_results() now focuses on science run directories and captures
#    every artifact required for comprehensive reporting.
#  * compute_model_stats() builds both high-level aggregates and run-specific
#    detail, enabling HTML/PDF reports to drill down without re-reading files.
#  * export_stats_to_json() keeps the object JSON-friendly for downstream tools.
#
# Numerical safeguards:
#  * Degrees of freedom default to max(…, 1) to avoid division by zero.
#  * All numeric fields are cast to native Python floats/ints to ensure JSON
#    serialisation without NumPy dependencies.
#
# Scientific rationale:
#  * Joint comparison artifacts provide fair, like-for-like metrics across
#    models; scouts preserve dataset-level diagnostics.
#  * Aggregates over multiple runs allow longitudinal tracking of performance
#    and reproducibility for publication-ready summaries.
