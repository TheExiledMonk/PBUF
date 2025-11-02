"""
Unified science run orchestration built around a single JSON configuration.

This module exposes two public entry points:

* load_config(path) — parse a unified configuration file with sensible defaults
* ScienceRunner     — execute the configured scenarios with checkpointing support

The implementation is intentionally lightweight so it can be exercised both by
the CLI and the unit tests.  Heavy-weight optimisers are injected via the
`executor` callback to keep the science runner agnostic of solver specifics.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from cosmos.optim.chi2_targets import Chi2TargetRegistry, Chi2TargetRule
from cosmos.optim.coord_optimizer.basin_walker import CoordinateBasinWalker, DEFAULT_REFERENCES
from reports.generator import collect_dataset_partitions
from cosmos.utils.io import (
    atomic_write_json,
    merge_dict_with_defaults,
    read_json,
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "run_id": "science_run",
    "models": ["lcdm", "pbuf"],
    "scenarios": [],
    "budgets": {
        "island_samples": 200,
        "island_delta": 20.0,
        "workers": 1,
        "eval_cap_per_model": None,
    },
    "phase6a": {
        "enabled_for_pbuf": True,
        "ksat_bounds": [0.9, 1.0],
    },
    "walker": {
        "converge": False,
        "max_rebalances": None,
        "reseed_on_plateau": False,
        "plateau_delta": 1.0,
        "plateau_window": 3,
    },
    "priors": {},
    "checkpointing": {
        "enabled": False,
        "frequency": "per_scenario",
        "resume": False,
    },
    "reporting": {
        "record_dof": False,
        "store_per_dataset_partitions": False,
        "write_optimizer_trace": False,
    },
    "targets": {},
    "seeds": {
        "island_seed": None,
        "global_random_seed": None,
    },
    "output_root": "data/science_runs",
    "env_meta": {},
}


EQUIVALENT_TARGET_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("sn_pantheon", "sn_pantheon_abs", "sn_sh0es", "pantheon", "sh0es"),
)


class ScienceConfigError(ValueError):
    """Raised when the configuration payload is malformed."""


def _ensure_list(payload: Any, label: str) -> List[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return list(payload)
    raise ScienceConfigError(f"Configuration field '{label}' must be a list.")


def _default_run_directory(root: Path, run_id: str, *, timestamp: Optional[str] = None) -> Path:
    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return root / f"{stamp}_{run_id}"


def _sanitize_targets(targets: Mapping[str, Sequence[float]]) -> Dict[str, Chi2TargetRule]:
    rules: Dict[str, Chi2TargetRule] = {}
    for dataset, entry in targets.items():
        dataset_key = str(dataset).lower()
        if isinstance(entry, Mapping):
            target = entry.get("target")
            tolerance = entry.get("tolerance", entry.get("sigma"))
        else:
            try:
                target, tolerance = float(entry[0]), float(entry[1])
            except Exception:
                continue
        try:
            rule = Chi2TargetRule.from_mapping(
                {"target": float(target), "tolerance": float(tolerance)}
            )
        except Exception:
            continue
        rules[dataset_key] = rule

    for group in EQUIVALENT_TARGET_GROUPS:
        canonical_rule: Optional[Chi2TargetRule] = None
        for name in group:
            if name in rules:
                canonical_rule = rules[name]
                break
        if canonical_rule is None:
            continue
        for name in group:
            rules.setdefault(name, canonical_rule)
    return rules


def load_config(path: Path | str) -> Dict[str, Any]:
    """
    Load a unified science configuration file, applying defaults for missing sections.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Science configuration not found: {path}")
    raw = read_json(path, allow_comments=True)
    if not isinstance(raw, Mapping):
        raise ScienceConfigError("Science configuration must contain a JSON object.")

    merged = merge_dict_with_defaults(raw, DEFAULT_CONFIG)
    merged["run_id"] = str(merged.get("run_id") or "science_run")
    merged["models"] = [str(model).lower() for model in _ensure_list(merged.get("models"), "models")]

    scenarios = _ensure_list(merged.get("scenarios"), "scenarios")
    normalised_scenarios: List[Dict[str, Any]] = []
    for item in scenarios:
        if not isinstance(item, Mapping):
            raise ScienceConfigError("Each scenario entry must be an object.")
        scenario_id = str(item.get("id") or "")
        if not scenario_id:
            raise ScienceConfigError("Scenario missing 'id'.")
        datasets = _ensure_list(item.get("datasets"), f"scenario[{scenario_id}].datasets")
        normalised_scenarios.append(
            {
                "id": scenario_id,
                "datasets": [str(ds) for ds in datasets],
                "options": dict(item.get("options") or {}),
            }
        )
    if not normalised_scenarios:
        raise ScienceConfigError("Configuration must contain at least one scenario.")
    merged["scenarios"] = normalised_scenarios

    if not isinstance(merged.get("output_root"), str):
        merged["output_root"] = "data/science_runs"

    merged["_config_path"] = str(path)
    return merged


def _checkpoint_path(run_dir: Path, scenario_id: str, model: str) -> Path:
    return run_dir / "checkpoints" / f"{scenario_id}_{model}.json"


def _extensions_applied(config: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "walker": bool(config.get("walker")),
        "priors": bool(config.get("priors")),
        "checkpointing": bool(config.get("checkpointing", {}).get("enabled")),
        "reporting": bool(config.get("reporting")),
        "targets": bool(config.get("targets")),
    }


def _build_walker(
    model: str,
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
    chi2_registry: Optional[Chi2TargetRegistry],
) -> CoordinateBasinWalker:
    budgets = config.get("budgets", {})
    walker_opts = config.get("walker", {})
    priors_section = config.get("priors", {})
    model_priors = priors_section.get(model, {})

    walker = CoordinateBasinWalker(
        model_type=model,
        datasets=scenario.get("datasets", []),
        enforce_phase6a=(bool(config.get("phase6a", {}).get("enabled_for_pbuf", True)) if model == "pbuf" else False),
        delta_chi2=float(budgets.get("island_delta", 20.0) or 20.0),
        reference_params=DEFAULT_REFERENCES.get(model, {}),
        verbose=bool(config.get("walker", {}).get("verbose", False)),
        progress=False,
        max_workers=int(budgets.get("workers", 1) or 1),
        chi2_targets=chi2_registry,
        priors=model_priors,
        walker_settings=walker_opts,
    )
    return walker


def _default_executor(
    scenario: Mapping[str, Any],
    model: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run the coordinate walker end-to-end for the provided scenario/model."""
    targets = config.get("targets") or {}
    chi2_registry = None
    if targets:
        chi2_registry = Chi2TargetRegistry(_sanitize_targets(targets))

    walker = _build_walker(model, scenario, config, chi2_registry)
    start = time.time()
    result = walker.run()
    runtime = time.time() - start
    result.setdefault("metadata", {})
    result["metadata"]["runtime_seconds"] = runtime
    result["scenario_id"] = scenario["id"]
    result["model"] = model
    return result


ExecutorType = Callable[[Mapping[str, Any], str, Mapping[str, Any]], Dict[str, Any]]


@dataclass(slots=True)
class ScienceRunner:
    """
    Coordinate multiple scenarios/models using the unified science configuration.
    """

    config: Dict[str, Any]
    executor: ExecutorType = field(default=_default_executor)
    output_root: Path = field(init=False)
    run_dir: Optional[Path] = field(default=None, init=False)
    results: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _resume_completed: Dict[Tuple[str, str], Dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        output_root = Path(self.config.get("output_root", "data/science_runs"))
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root

    # ------------------------------------------------------------------
    # High level execution
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        fresh: bool = False,
        resume_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the configured science run.
        """
        start_time = time.time()
        self.run_dir = self._prepare_run_directory(fresh=fresh, resume_dir=resume_dir)
        self._load_resume_state()

        checkpoints_enabled = bool(self.config.get("checkpointing", {}).get("enabled"))

        last_completed: Optional[str] = None

        for scenario in self.config["scenarios"]:
            scenario_id = scenario["id"]
            for model in self.config["models"]:
                key = (scenario_id, model)
                if key in self._resume_completed:
                    self.results.append(self._resume_completed[key])
                    last_completed = scenario_id
                    continue

                payload = self.executor(scenario, model, self.config)
                payload.setdefault("scenario_id", scenario_id)
                payload.setdefault("model", model)
                payload.setdefault("metadata", {})
                payload["metadata"].setdefault("runtime_seconds", time.time() - payload["metadata"].get("start_ts", time.time()))
                payload["metadata"].pop("start_ts", None)
                payload["timestamp_utc"] = datetime.now(UTC).isoformat()

                self.results.append(payload)
                last_completed = scenario_id

                if checkpoints_enabled:
                    self._write_checkpoint(scenario_id, model, payload)

        summary = self.finalise_results(runtime=time.time() - start_time, last_completed=last_completed)
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_run_directory(
        self,
        *,
        fresh: bool,
        resume_dir: Optional[Path | str],
    ) -> Path:
        if resume_dir:
            run_dir = Path(resume_dir)
        else:
            if fresh or not self._checkpoint_resume_enabled():
                run_dir = _default_run_directory(self.output_root, self.config["run_id"])
            else:
                existing = self._find_existing_run()
                run_dir = existing or _default_run_directory(self.output_root, self.config["run_id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        return run_dir

    def _checkpoint_resume_enabled(self) -> bool:
        checkpointing = self.config.get("checkpointing", {})
        return bool(checkpointing.get("enabled") and checkpointing.get("resume"))

    def _find_existing_run(self) -> Optional[Path]:
        prefix = f"_{self.config['run_id']}"
        candidates = sorted(
            [path for path in self.output_root.glob(f"*{prefix}") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if (candidate / "checkpoints").exists():
                return candidate
        return candidates[0] if candidates else None

    def _load_resume_state(self) -> None:
        if not self._checkpoint_resume_enabled() or self.run_dir is None:
            return
        checkpoints_dir = self.run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return
        for path in checkpoints_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            scenario, model = path.stem.split("_", 1)
            self._resume_completed[(scenario, model)] = payload

    def _write_checkpoint(self, scenario_id: str, model: str, payload: Mapping[str, Any]) -> None:
        if self.run_dir is None:
            return
        atomic_write_json(_checkpoint_path(self.run_dir, scenario_id, model), payload)

    # ------------------------------------------------------------------
    # Finalisation & reporting
    # ------------------------------------------------------------------

    def finalise_results(
        self,
        *,
        runtime: float,
        last_completed: Optional[str],
    ) -> Dict[str, Any]:
        """
        Aggregate results, write summary, and update meta.json.
        """
        if self.run_dir is None:
            raise RuntimeError("Science runner has not been executed yet.")

        summary = {
            "run_id": self.config["run_id"],
            "models": list(self.config["models"]),
            "scenarios": [scenario["id"] for scenario in self.config["scenarios"]],
            "results": list(self.results),
            "runtime_seconds": float(runtime),
        }

        reporting = self.config.get("reporting", {})
        if reporting.get("store_per_dataset_partitions"):
            summary["per_dataset"] = collect_dataset_partitions(self.results)

        summary_path = self.run_dir / "summary.json"
        atomic_write_json(summary_path, summary)

        meta_entry = {
            "run_id": self.config["run_id"],
            "models": list(self.config["models"]),
            "extensions_applied": _extensions_applied(self.config),
            "last_completed_scenario": last_completed,
            "runtime_seconds": float(runtime),
        }
        meta_path = self.run_dir / "meta.json"
        meta_payload: List[Any] = []
        if meta_path.exists():
            try:
                meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(meta_payload, list):
                    meta_payload = []
            except Exception:
                meta_payload = []
        meta_payload.append(meta_entry)
        atomic_write_json(meta_path, meta_payload)
        return summary


__all__ = [
    "ScienceRunner",
    "load_config",
    "ScienceConfigError",
]
