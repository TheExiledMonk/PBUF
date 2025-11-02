#!/usr/bin/env python3
"""
Science run orchestrator for the PBUF v10 workflow.

This script automates the tuning → scenario → joint comparison workflow for
ΛCDM and PBUF models using the existing coordinate basin walker CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "science_run.json"

STATE_VERSION = 1
STATE_FILENAME = "state.json"
META_FILENAME = "meta.json"

SCOUT_DATASETS_DEFAULT: Tuple[str, ...] = (
    "cmb",
    "bao_iso",
    "bao_aniso",
    "cc",
    "sn_pantheon",
    "sn_pantheon_abs",
)


DATASET_FILE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "cmb": ("cmb.npz", "planck2018_distance_priors.npz", "cmb_planck2018.npz"),
    "bao_iso": ("bao_iso.npz", "bao_iso_dr16.npz"),
    "bao_aniso": ("bao_aniso.npz", "bao_aniso_dr16.npz"),
    "cc": ("cc.npz", "cc_compilation.npz"),
    "rsd": ("rsd.npz", "rsd_compilation.npz"),
    "sn_pantheon": (
        "sn_pantheon.npz",
        "sn_pantheonplus.npz",
        "sn_pantheon_full.npz",
        "sn_pantheon_shoes.npz",
        "sn_sh0es.npz",
    ),
    "sn_pantheon_abs": (
        "sn_pantheon.npz",
        "sn_pantheonplus.npz",
        "sn_pantheon_full.npz",
        "sn_pantheon_shoes.npz",
        "sn_sh0es.npz",
    ),
    "sn_sh0es": ("sn_pantheon_shoes.npz", "sn_sh0es.npz", "sh0es.npz"),
}


@dataclass
class Budget:
    workers: int
    island_samples: int
    island_delta: float
    eval_cap_per_model: Optional[int]
    island_seed: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PBUF v10 science orchestration pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to science run configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--skip-scouts",
        action="store_true",
        help="Skip the single-dataset scout runs (Stage 1).",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Explicit run directory to resume. Overrides auto-detection.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing state and start a fresh run directory.",
    )
    parser.add_argument(
        "--update-parameter-defaults",
        action="store_true",
        help="Allow CLI runs to update cosmos/optim/parameter_defaults.py (default: disabled).",
    )
    parser.add_argument(
        "--quiet-cli",
        action="store_true",
        help="Pass --quiet to the coordinate walker to reduce console noise.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable coordinate walker progress bars (--no-progress).",
    )
    return parser.parse_args()


def strip_json_comments(text: str) -> str:
    """
    Remove // and /* */ comments from JSON-like text.
    """
    result: List[str] = []
    in_string = False
    string_char = ""
    in_single_line_comment = False
    in_block_comment = False
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        next_ch = text[i + 1] if i + 1 < length else ""

        if in_single_line_comment:
            if ch == "\n":
                in_single_line_comment = False
                result.append(ch)
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and next_ch == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            result.append(ch)
            if ch == "\\":
                if i + 1 < length:
                    result.append(text[i + 1])
                    i += 2
                    continue
            elif ch == string_char:
                in_string = False
            i += 1
            continue
        if ch == "/" and next_ch == "/":
            in_single_line_comment = True
            i += 2
            continue
        if ch == "/" and next_ch == "*":
            in_block_comment = True
            i += 2
            continue
        if ch in {'"', "'"}:
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    text = path.read_text(encoding="utf-8")
    clean = strip_json_comments(text)
    cfg = json.loads(clean)
    required = ["run_id", "models", "scenarios", "budgets", "seeds", "output_root"]
    for field in required:
        if field not in cfg:
            raise ValueError(f"Configuration missing required field '{field}'")
    return cfg


def set_global_seeds(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:  # pragma: no cover - guard against numpy absence
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def find_existing_run(output_root: Path, run_id: str) -> Optional[Path]:
    if not output_root.exists():
        return None
    candidates = sorted(
        (path for path in output_root.iterdir() if path.is_dir() and path.name.endswith(f"_{run_id}")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if (candidate / STATE_FILENAME).exists():
            return candidate
    return candidates[0] if candidates else None


def ensure_directories(run_dir: Path) -> None:
    for subdir in ("logs", "raw", "artifacts", "plots"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_candidates(dataset: str) -> Sequence[Path]:
    filenames = DATASET_FILE_CANDIDATES.get(dataset, ())
    base_dir = PROJECT_ROOT / "data" / "standardized"
    return [base_dir / name for name in filenames]


def locate_dataset_file(dataset: str) -> Optional[Path]:
    for candidate in dataset_candidates(dataset):
        if candidate.exists():
            return candidate
    return None


def load_dataset_n_data(path: Path) -> Optional[int]:
    if path is None or not path.exists():
        return None
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            if "n_data" in npz:
                value = npz["n_data"]
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        return int(value.item())
                    if value.size >= 1:
                        return int(value.reshape(-1)[0])
                try:
                    return int(value)
                except Exception:
                    pass
            for key in ("z", "obs", "mu", "data", "values"):
                if key in npz:
                    arr = npz[key]
                    if hasattr(arr, "shape"):
                        return int(arr.shape[0])
    return None


def collect_dataset_catalog(datasets: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for name in sorted(set(dataset.lower() for dataset in datasets)):
        path = locate_dataset_file(name)
        info = {
            "name": name,
            "path": str(path.relative_to(PROJECT_ROOT)) if path and path.exists() else None,
            "exists": bool(path and path.exists()),
            "sha256": sha256_file(path) if path else None,
            "n_data": load_dataset_n_data(path) if path else None,
        }
        catalog[name] = info
    return catalog


def run_git_command(args: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def snapshot_environment(config: Dict[str, Any], dataset_catalog: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    python_version = sys.version.replace("\n", " ")
    packages: Dict[str, Optional[str]] = {}
    for module_name in ("numpy", "scipy", "pandas", "astropy"):
        try:
            module = __import__(module_name)
            packages[module_name] = getattr(module, "__version__", None)
        except Exception:
            packages[module_name] = None

    cpu_count = os.cpu_count()
    hostname = None
    try:
        import socket

        hostname = socket.gethostname()
    except Exception:
        hostname = None

    ram_bytes: Optional[int] = None
    try:
        import psutil  # type: ignore

        ram_bytes = int(psutil.virtual_memory().total)
    except Exception:
        ram_bytes = None

    git_commit = run_git_command(["rev-parse", "HEAD"])
    git_status = run_git_command(["status", "--short", "--branch"])

    meta = {
        "run_id": config["run_id"],
        "captured_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "git_status": git_status,
        "python_version": python_version,
        "packages": packages,
        "cpu_count": cpu_count,
        "ram_bytes": ram_bytes,
        "hostname": hostname,
        "budgets": config["budgets"],
        "seeds": config["seeds"],
        "env_meta": config.get("env_meta", {}),
        "dataset_hashes": dataset_catalog,
    }
    return meta


def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def initialize_state(state: MutableMapping[str, Any], config: Mapping[str, Any]) -> None:
    state.setdefault("version", STATE_VERSION)
    state.setdefault("run_id", config["run_id"])
    state.setdefault("timestamp", datetime.now(UTC).isoformat())
    state.setdefault("step_counter", 0)
    state.setdefault("config_snapshot", config)
    state.setdefault("environment", {"status": "pending"})
    scouts = state.setdefault("scouts", {})
    scouts.setdefault("enabled", True)
    scouts.setdefault("datasets", {})
    scenarios = state.setdefault("scenarios", [])
    if not scenarios:
        for scenario in config["scenarios"]:
            scenarios.append(
                {
                    "id": scenario["id"],
                    "datasets": scenario["datasets"],
                    "models": {},
                    "joint": {"status": "pending"},
                }
            )
    state.setdefault("last_step", None)


def advance_order(state: MutableMapping[str, Any], entry: MutableMapping[str, Any]) -> int:
    if "order" in entry:
        return int(entry["order"])
    state["step_counter"] = int(state.get("step_counter", 0)) + 1
    entry["order"] = state["step_counter"]
    return entry["order"]


def format_identifier(order: int, label: str, model: Optional[str] = None) -> str:
    if model:
        return f"{order:02d}-{label}-{model}"
    return f"{order:02d}-{label}"


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def should_attempt(entry: Mapping[str, Any]) -> bool:
    status = entry.get("status", "pending")
    if status in {"pending", "started"}:
        return True
    if status == "done":
        return False
    if status == "failed":
        retries = int(entry.get("retries", 0))
        return retries < 2
    return True


def mark_started(entry: MutableMapping[str, Any]) -> None:
    entry["status"] = "started"
    entry["started_at"] = datetime.now(UTC).isoformat()


def mark_done(entry: MutableMapping[str, Any], wall_seconds: float, workers: int) -> None:
    entry["status"] = "done"
    entry["ended_at"] = datetime.now(UTC).isoformat()
    entry["wall_seconds"] = wall_seconds
    entry["cpu_hours"] = wall_seconds * workers / 3600.0


def mark_failed(entry: MutableMapping[str, Any], message: str) -> None:
    entry["status"] = "failed"
    entry["ended_at"] = datetime.now(UTC).isoformat()
    entry["failure_reason"] = message
    entry["retries"] = int(entry.get("retries", 0)) + 1


def normalize_dataset_list(datasets: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for name in datasets:
        name_lower = name.lower()
        if name_lower not in seen:
            ordered.append(name_lower)
            seen.add(name_lower)
    return ordered


VALID_SN_MODES = {"relative", "absolute"}


def apply_scenario_options(datasets: Sequence[str], options: Optional[Mapping[str, Any]]) -> List[str]:
    """
    Apply scenario-level options (e.g. sn_mode) to the dataset list.
    """
    resolved = list(datasets)
    if not options:
        return normalize_dataset_list(resolved)

    sn_mode = options.get("sn_mode")
    if sn_mode is not None:
        sn_mode_lower = str(sn_mode).lower()
        if sn_mode_lower not in VALID_SN_MODES:
            raise ValueError(f"Invalid sn_mode '{sn_mode}'. Expected one of {sorted(VALID_SN_MODES)}.")
        if sn_mode_lower == "absolute":
            resolved = [
                "sn_pantheon_abs" if name in ("sn_pantheon", "sn_pantheon_abs") else name
                for name in resolved
            ]
            if "sn_pantheon_abs" not in resolved:
                resolved.append("sn_pantheon_abs")
        elif sn_mode_lower == "relative":
            resolved = [
                "sn_pantheon" if name in ("sn_pantheon", "sn_pantheon_abs") else name
                for name in resolved
            ]
            if "sn_pantheon" not in resolved:
                resolved.append("sn_pantheon")

    return normalize_dataset_list(resolved)


def build_cli_command(
    model: str,
    datasets: Sequence[str],
    budgets: Budget,
    output_path: Path,
    args: argparse.Namespace,
    phase6a_for_pbuf: bool,
) -> List[str]:
    cmd = [sys.executable, "cli.py", "fit", "coord", "--model", model]
    if datasets:
        cmd.extend(["--datasets", ",".join(datasets)])
    cmd.extend(
        [
            "--workers",
            str(budgets.workers),
            "--island-samples",
            str(budgets.island_samples),
            "--island-delta",
            str(budgets.island_delta),
            "--output",
            str(output_path),
            "--converge",
        ]
    )
    if budgets.island_seed is not None:
        cmd.extend(["--island-seed", str(budgets.island_seed)])
    if model == "pbuf" and phase6a_for_pbuf:
        cmd.append("--phase6a")
    if args.quiet_cli:
        cmd.append("--quiet")
    if args.no_progress:
        cmd.append("--no-progress")
    if not args.update_parameter_defaults:
        cmd.append("--dry-run")
    return cmd


def best_fit_artifact(
    run_id: str,
    scenario_label: str,
    model: str,
    datasets: Sequence[str],
    scenario_options: Mapping[str, Any],
    result: Mapping[str, Any],
    runtime: Mapping[str, Any],
    dataset_catalog: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    command: Sequence[str],
) -> Dict[str, Any]:
    params = result.get("fiducial_params") or {}
    chi2_total = result.get("fiducial_chi2")
    breakdown = result.get("fiducial_breakdown") or {}
    n_params = len(params)

    total_n_data = 0
    n_data_per_dataset: Dict[str, Optional[int]] = {}
    for dataset in datasets:
        info = dataset_catalog.get(dataset)
        n_data = info.get("n_data") if info else None
        n_data_per_dataset[dataset] = n_data
        if isinstance(n_data, int):
            total_n_data += n_data

    dof = total_n_data - n_params if total_n_data and n_params else None
    chi2_reduced = chi2_total / dof if dof and chi2_total is not None else None
    aic = chi2_total + 2 * n_params if chi2_total is not None else None
    bic = chi2_total + n_params * math.log(total_n_data) if chi2_total is not None and total_n_data else None

    phase6a_enforced = bool(result.get("phase6a_enforced", model == "pbuf"))
    phase6a_passed = result.get("fiducial_passes_phase6a")
    physics_flags: Dict[str, Any] = {
        "phase6a_applied": phase6a_enforced,
    }
    if phase6a_passed is not None:
        physics_flags["phase6a_passed"] = bool(phase6a_passed)

    if model == "pbuf":
        bounds = (config.get("phase6a") or {}).get("ksat_bounds")
        k_sat_value = params.get("k_sat")
        if bounds and k_sat_value is not None:
            lower, upper = bounds
            physics_flags["ksat_within_bounds"] = bool(lower <= k_sat_value <= upper)

    if "fiducial_validation" in result:
        physics_flags["validation"] = result["fiducial_validation"]

    best_fit = {
        "params": params,
        "derived": result.get("fiducial_derived", {}),
    }

    fit_stats = {
        "chi2_total": chi2_total,
        "chi2_per_dataset": breakdown,
        "n_params": n_params,
        "n_data": total_n_data if total_n_data else None,
        "n_data_per_dataset": n_data_per_dataset,
        "dof": dof,
        "chi2_reduced": chi2_reduced,
        "aic": aic,
        "bic": bic,
    }

    runtime_block = {
        "workers": runtime.get("workers"),
        "started_at": runtime.get("started_at"),
        "ended_at": runtime.get("ended_at"),
        "wall_seconds": runtime.get("wall_seconds"),
        "cpu_hours": runtime.get("cpu_hours"),
    }

    provenance = {
        "git_commit": runtime.get("git_commit"),
        "dataset_hashes": runtime.get("dataset_hashes"),
        "seeds": runtime.get("seeds"),
        "cli": " ".join(command),
    }

    options_payload = dict(scenario_options) if scenario_options else None

    artifact: Dict[str, Any] = {
        "run_id": run_id,
        "scenario": scenario_label,
        "model": model,
        "datasets": list(datasets),
        "best_fit": best_fit,
        "fit_stats": fit_stats,
        "physics_flags": physics_flags,
        "predictives": {},
        "growth": {},
        "elastic": {} if model == "pbuf" else {},
        "runtime": runtime_block,
        "provenance": provenance,
    }

    if "convergence" in result:
        artifact["optimizer"] = {
            "convergence": {
                "converged": result["convergence"].get("converged"),
                "cycles_completed": result["convergence"].get("cycles_completed"),
                "chi2_history": result["convergence"].get("chi2_history"),
                "param_shift_history": result["convergence"].get("param_shift_history"),
            }
        }
    if options_payload:
        artifact["scenario_options"] = options_payload

    return artifact


def write_status_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def execute_cli_step(
    step_label: str,
    scenario_label: str,
    model: str,
    datasets: Sequence[str],
    scenario_options: Mapping[str, Any],
    budgets: Budget,
    run_dir: Path,
    state: MutableMapping[str, Any],
    entry: MutableMapping[str, Any],
    args: argparse.Namespace,
    dataset_catalog: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    run_meta: Mapping[str, Any],
) -> None:
    order = advance_order(state, entry)
    identifier = format_identifier(order, step_label, model)

    raw_path = run_dir / "raw" / f"{identifier}.json"
    log_path = run_dir / "logs" / f"{identifier}.log"
    started_path = run_dir / "artifacts" / f"{identifier}-started.json"
    failed_path = run_dir / "artifacts" / f"{identifier}-failed.json"
    done_path = run_dir / "artifacts" / f"{identifier}-done.json"

    entry["raw_path"] = relative_path(raw_path, run_dir)
    entry["log_path"] = relative_path(log_path, run_dir)
    entry["artifact_path"] = relative_path(done_path, run_dir)
    entry["started_marker"] = relative_path(started_path, run_dir)
    if scenario_options:
        entry["options"] = dict(scenario_options)

    if not should_attempt(entry):
        return

    mark_started(entry)
    write_status_artifact(started_path, {"status": "started", "identifier": identifier, "model": model})

    state["last_step"] = f"{scenario_label}:{model}"
    atomic_write_json(run_dir / STATE_FILENAME, state)

    phase6a_for_pbuf = bool(config.get("phase6a", {}).get("enabled_for_pbuf", False))
    command = build_cli_command(
        model,
        datasets,
        budgets,
        raw_path,
        args,
        phase6a_for_pbuf=phase6a_for_pbuf,
    )
    started_at = datetime.now(UTC)
    wall_start = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    wall_seconds = time.monotonic() - wall_start
    ended_at = datetime.now(UTC)

    log_payload = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    atomic_write_json(log_path, log_payload)

    runtime_meta = {
        "workers": budgets.workers,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "wall_seconds": wall_seconds,
        "cpu_hours": wall_seconds * budgets.workers / 3600.0,
        "git_commit": run_meta.get("git_commit"),
        "dataset_hashes": run_meta.get("dataset_hashes"),
        "seeds": config.get("seeds"),
    }

    if completed.returncode != 0:
        message = f"CLI exited with status {completed.returncode}"
        mark_failed(entry, message)
        write_status_artifact(
            failed_path,
            {
                "status": "failed",
                "identifier": identifier,
                "model": model,
                "reason": message,
                "returncode": completed.returncode,
            },
        )
        atomic_write_json(run_dir / STATE_FILENAME, state)
        return

    if not raw_path.exists():
        message = "CLI completed but raw output missing."
        mark_failed(entry, message)
        write_status_artifact(
            failed_path,
            {
                "status": "failed",
                "identifier": identifier,
                "model": model,
                "reason": message,
            },
        )
        atomic_write_json(run_dir / STATE_FILENAME, state)
        return

    try:
        result = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        message = f"Failed to parse CLI output JSON: {exc}"
        mark_failed(entry, message)
        write_status_artifact(
            failed_path,
            {
                "status": "failed",
                "identifier": identifier,
                "model": model,
                "reason": message,
            },
        )
        atomic_write_json(run_dir / STATE_FILENAME, state)
        return

    mark_done(entry, wall_seconds, budgets.workers)

    artifact = best_fit_artifact(
        run_id=config["run_id"],
        scenario_label=scenario_label,
        model=model,
        datasets=datasets,
        scenario_options=scenario_options,
        result=result,
        runtime=runtime_meta,
        dataset_catalog=dataset_catalog,
        config=config,
        command=command,
    )
    write_status_artifact(done_path, artifact)
    atomic_write_json(run_dir / STATE_FILENAME, state)


def ensure_scout_entry(state: MutableMapping[str, Any], dataset: str, model: str) -> MutableMapping[str, Any]:
    scouts = state.setdefault("scouts", {"enabled": True, "datasets": {}})
    datasets_dict = scouts.setdefault("datasets", {})
    entry = datasets_dict.setdefault(dataset, {"dataset": dataset, "models": {}})
    model_entry = entry["models"].setdefault(model, {"status": "pending"})
    return model_entry


def get_scenario_entry(state: MutableMapping[str, Any], scenario_id: str) -> MutableMapping[str, Any]:
    for entry in state.get("scenarios", []):
        if entry.get("id") == scenario_id:
            return entry
    new_entry: Dict[str, Any] = {"id": scenario_id, "datasets": [], "models": {}, "joint": {"status": "pending"}}
    state.setdefault("scenarios", []).append(new_entry)
    return new_entry


def ensure_scenario_model_entry(
    scenario_entry: MutableMapping[str, Any],
    model: str,
) -> MutableMapping[str, Any]:
    models = scenario_entry.setdefault("models", {})
    return models.setdefault(model, {"status": "pending"})


def maybe_emit_joint_artifact(
    scenario_entry: MutableMapping[str, Any],
    scenario_label: str,
    run_dir: Path,
    state: MutableMapping[str, Any],
    dataset_catalog: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    models = scenario_entry.get("models", {})
    if not models:
        return
    if any(model_data.get("status") != "done" for model_data in models.values()):
        return
    joint_entry = scenario_entry.setdefault("joint", {"status": "pending"})
    if joint_entry.get("status") == "done":
        return
    order = advance_order(state, joint_entry)
    identifier = format_identifier(order, scenario_label, "joint-comparison")
    artifact_path = run_dir / "artifacts" / f"{identifier}.json"
    joint_entry["artifact_path"] = relative_path(artifact_path, run_dir)

    artifacts: Dict[str, Any] = {}
    for model, details in models.items():
        artifact_path_model = run_dir / details.get("artifact_path", "")
        if not artifact_path_model.exists():
            continue
        try:
            artifacts[model] = json.loads(artifact_path_model.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

    if not all(model in artifacts for model in config["models"]):
        return

    lcdm = artifacts.get("lcdm", {})
    pbuf = artifacts.get("pbuf", {})
    chi2_lcdm = (lcdm.get("fit_stats") or {}).get("chi2_total")
    chi2_pbuf = (pbuf.get("fit_stats") or {}).get("chi2_total")
    aic_lcdm = (lcdm.get("fit_stats") or {}).get("aic")
    aic_pbuf = (pbuf.get("fit_stats") or {}).get("aic")
    bic_lcdm = (lcdm.get("fit_stats") or {}).get("bic")
    bic_pbuf = (pbuf.get("fit_stats") or {}).get("bic")

    joint_payload = {
        "run_id": config["run_id"],
        "scenario": scenario_label,
        "options": scenario_entry.get("options", {}),
        "models": {
            model: {
                "chi2_total": (artifact.get("fit_stats") or {}).get("chi2_total"),
                "aic": (artifact.get("fit_stats") or {}).get("aic"),
                "bic": (artifact.get("fit_stats") or {}).get("bic"),
                "dof": (artifact.get("fit_stats") or {}).get("dof"),
                "params": (artifact.get("best_fit") or {}).get("params"),
                "chi2_per_dataset": (artifact.get("fit_stats") or {}).get("chi2_per_dataset"),
                "runtime": artifact.get("runtime"),
                "phase6a": (artifact.get("physics_flags") or {}),
            }
            for model, artifact in artifacts.items()
        },
        "parity": {
            "compute_budget_equal": True,
            "dataset_masks_equal": True,
            "priors_equal": True,
        },
        "deltas": {
            "delta_chi2": (chi2_pbuf - chi2_lcdm) if chi2_pbuf is not None and chi2_lcdm is not None else None,
            "delta_aic": (aic_pbuf - aic_lcdm) if aic_pbuf is not None and aic_lcdm is not None else None,
            "delta_bic": (bic_pbuf - bic_lcdm) if bic_pbuf is not None and bic_lcdm is not None else None,
        },
        "provenance": {
            "git_commit": run_git_command(["rev-parse", "HEAD"]),
            "dataset_hashes": dataset_catalog,
        },
    }

    atomic_write_json(artifact_path, joint_payload)
    joint_entry["status"] = "done"
    joint_entry["generated_at"] = datetime.now(UTC).isoformat()
    state["last_step"] = f"{scenario_label}:joint"
    atomic_write_json(run_dir / STATE_FILENAME, state)


def orchestrate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    set_global_seeds(config.get("seeds", {}).get("global_random_seed"))

    output_root = Path(config["output_root"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.resume_dir:
        run_dir = args.resume_dir
    elif args.fresh:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"{timestamp}_{config['run_id']}"
    else:
        existing = find_existing_run(output_root, config["run_id"])
        if existing is not None:
            run_dir = existing
        else:
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            run_dir = output_root / f"{timestamp}_{config['run_id']}"

    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_directories(run_dir)

    state_path = run_dir / STATE_FILENAME
    state = load_state(state_path)
    initialize_state(state, config)

    # Determine dataset universe
    scenario_datasets: List[str] = []
    scenario_resolved_map: Dict[str, List[str]] = {}
    for scenario in config["scenarios"]:
        raw_list = normalize_dataset_list(scenario["datasets"])
        options = scenario.get("options") or {}
        resolved = apply_scenario_options(raw_list, options)
        scenario_resolved_map[scenario["id"]] = resolved
        scenario_datasets.extend(resolved)
    scout_candidates = list(SCOUT_DATASETS_DEFAULT)
    for ds in scenario_datasets:
        if ds not in scout_candidates:
            scout_candidates.append(ds)
    dataset_catalog = collect_dataset_catalog(scout_candidates)

    # Environment snapshot
    if state.get("environment", {}).get("status") != "done":
        meta = snapshot_environment(config, dataset_catalog)
        atomic_write_json(run_dir / META_FILENAME, meta)
        state["environment"] = {
            "status": "done",
            "meta_path": META_FILENAME,
            "captured_at": meta["captured_at"],
        }
        atomic_write_json(state_path, state)
    else:
        meta = json.loads((run_dir / META_FILENAME).read_text(encoding="utf-8"))

    # Stage 1 - Scouts
    state.setdefault("scouts", {"enabled": True, "datasets": {}})
    if args.skip_scouts:
        state["scouts"]["enabled"] = False
    if state["scouts"].get("enabled", True) and not args.skip_scouts:
        total_scouts = len(scout_candidates) * len(config["models"])
        completed_scouts = 0
        for dataset in scout_candidates:
            for model in config["models"]:
                entry = ensure_scout_entry(state, dataset, model)
                if entry.get("status") == "done":
                    completed_scouts += 1
                    continue
                current_index = completed_scouts + 1
                print(f"[Stage 1/2] Scout {current_index}/{total_scouts} → dataset={dataset}, model={model}")
                execute_cli_step(
                    step_label=f"scout-{dataset}",
                    scenario_label=f"scout:{dataset}",
                    model=model,
                    datasets=[dataset],
                    scenario_options={},
                    budgets=Budget(
                        workers=int(config["budgets"]["workers"]),
                        island_samples=int(config["budgets"]["island_samples"]),
                        island_delta=float(config["budgets"]["island_delta"]),
                        eval_cap_per_model=config["budgets"].get("eval_cap_per_model"),
                        island_seed=config["seeds"].get("island_seed"),
                    ),
                    run_dir=run_dir,
                    state=state,
                    entry=entry,
                    args=args,
                    dataset_catalog=dataset_catalog,
                    config=config,
                    run_meta=meta,
                )
                completed_scouts += 1
                print(f"[Stage 1/2] ✔ Scout {completed_scouts}/{total_scouts} finished ({dataset}, {model})")
        print(f"[Stage 1/2] Scouts complete ({completed_scouts}/{total_scouts} tasks).")

    # Stage 2 - Scenarios
    scenario_total = len(config["scenarios"]) * len(config["models"])
    scenario_count = len(config["scenarios"])
    scenario_index = 0
    scenario_completed = 0
    for scenario in config["scenarios"]:
        scenario_id = scenario["id"]
        scenario_options = scenario.get("options") or {}
        dataset_list = scenario_resolved_map.get(scenario_id)
        if dataset_list is None:
            dataset_list = apply_scenario_options(normalize_dataset_list(scenario["datasets"]), scenario_options)
        scenario_entry = get_scenario_entry(state, scenario_id)
        scenario_entry["datasets"] = dataset_list
        if scenario_options:
            scenario_entry["options"] = scenario_options
        for model in config["models"]:
            model_entry = ensure_scenario_model_entry(scenario_entry, model)
            if model_entry.get("status") == "done":
                scenario_completed += 1
                continue
            current_task = scenario_completed + 1
            print(f"[Stage 2/2] Task {current_task}/{scenario_total} → scenario={scenario_id}, model={model}")
            execute_cli_step(
                step_label=scenario_id,
                scenario_label=scenario_id,
                model=model,
                datasets=dataset_list,
                scenario_options=scenario_options,
                budgets=Budget(
                    workers=int(config["budgets"]["workers"]),
                    island_samples=int(config["budgets"]["island_samples"]),
                    island_delta=float(config["budgets"]["island_delta"]),
                    eval_cap_per_model=config["budgets"].get("eval_cap_per_model"),
                    island_seed=config["seeds"].get("island_seed"),
                ),
                run_dir=run_dir,
                state=state,
                entry=model_entry,
                args=args,
                dataset_catalog=dataset_catalog,
                config=config,
                run_meta=meta,
            )
            maybe_emit_joint_artifact(
                scenario_entry,
                scenario_id,
                run_dir,
                state,
                dataset_catalog,
                config,
            )
            scenario_completed += 1
            print(f"[Stage 2/2] ✔ Task {scenario_completed}/{scenario_total} finished ({scenario_id}, {model})")
        scenario_index += 1
        print(f"[Stage 2/2] Scenario {scenario_index}/{scenario_count} complete ({scenario_id}).")
    print(f"[Stage 2/2] Scenarios complete ({scenario_completed}/{scenario_total} tasks).")

    atomic_write_json(state_path, state)
    print(f"✅ Orchestration complete. Run directory: {run_dir}")


def main() -> None:
    args = parse_args()
    orchestrate(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.", file=sys.stderr)
        sys.exit(1)
