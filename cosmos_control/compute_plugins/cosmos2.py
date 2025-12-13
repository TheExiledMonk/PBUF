"""Compute plugin backed by the Cosmos2ScienceRunner."""

from __future__ import annotations

import copy
import json
import tempfile
import time
import shutil
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from cosmos2.api.engine import clear_jackknife_masked_datasets, set_jackknife_masked_datasets
from cosmos2.data.registry import get_dataset, _LOADERS
from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.jackknife import (
    JackknifeConfig,
    JackknifeResampler,
    apply_mask_to_dataset,
)
from cosmos2.science_runner.runner import Cosmos2ScienceRunner

from ..models import SliceDescriptor

logger = logging.getLogger("cosmos_control.compute.cosmos2")


def _dataset_aliases(dataset_name: str) -> set[str]:
    normalized = dataset_name.strip().lower()
    loader = _LOADERS.get(normalized)
    if loader is None:
        return {normalized}
    return {name for name, fn in _LOADERS.items() if fn is loader}


def _dataset_length(dataset: Dict[str, Any]) -> int:
    for key in ("data", "z", "mu", "distances"):
        value = dataset.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            return len(value)
    max_length = 0
    for value in dataset.values():
        if isinstance(value, (list, tuple, np.ndarray)):
            max_length = max(max_length, len(value))
    return max_length


def _hydrate_jackknife_draw(metadata: dict[str, Any], config: dict[str, Any]) -> bool:
    draw_info = metadata.get("jackknife_draw") or {}
    draw_index = draw_info.get("index")
    if draw_index is None:
        return False
    dataset_names = list(draw_info.get("datasets") or metadata.get("jackknife_datasets") or [])
    if not dataset_names:
        return False
    jackknife_payload = metadata.get("jackknife_config") or config.get("jackknife")
    if not isinstance(jackknife_payload, dict):
        return False
    jackknife_config = JackknifeConfig.from_dict(jackknife_payload)
    resampler = JackknifeResampler(jackknife_config, dataset_names)
    dataset_map: dict[str, dict[str, Any]] = {}
    for dataset_name in dataset_names:
        try:
            payload = get_dataset(dataset_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping jackknife dataset %s: %s", dataset_name, exc)
            continue
        dataset_map[dataset_name] = payload
        resampler.set_dataset_size(dataset_name, _dataset_length(payload))
    if not resampler.dataset_sizes:
        logger.warning("No jackknife dataset sizes resolved for draw %s", draw_index)
        return False
    resampler.generate_masks()
    if draw_index < 0 or draw_index >= len(resampler.masks):
        logger.warning("Requested jackknife draw %s is out of range", draw_index)
        return False
    mask = resampler.masks[draw_index]
    masked: dict[str, dict[str, Any]] = {}
    for name, mask_array in mask.dataset_masks.items():
        source = dataset_map.get(name)
        if source is None:
            continue
        masked_dataset = apply_mask_to_dataset(source, mask_array)
        for alias in _dataset_aliases(name):
            masked[alias] = masked_dataset
    if not masked:
        logger.warning("Jackknife draw %s produced no masked datasets", draw_index)
        return False
    set_jackknife_masked_datasets(masked)
    logger.info("Applied jackknife draw %s mask", draw_index)
    return True


def _hydrate_prediction_metadata(metadata: dict[str, Any], config: dict[str, Any]) -> None:
    modules = metadata.get("prediction_modules")
    if not modules:
        return
    predictions = config.setdefault("predictions", {})
    predictions["enabled"] = True
    predictions["modules"] = list(modules)
    module_configs = metadata.get("prediction_module_configs") or {}
    for key, value in module_configs.items():
        predictions[key] = value


def _apply_package_metadata(config: dict[str, Any]) -> dict[str, bool]:
    metadata = config.get("metadata") or {}
    state: dict[str, bool] = {}
    if metadata.get("jackknife_draw"):
        state["jackknife_applied"] = _hydrate_jackknife_draw(metadata, config)
    if metadata.get("prediction_modules"):
        _hydrate_prediction_metadata(metadata, config)
    return state


def compute_slice(
    config: Dict[str, Any],
    slice_descriptor: SliceDescriptor,
    datasets: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Run a slice using the Cosmos2 science runner."""
    working_config = copy.deepcopy(config)
    run_name = working_config.get("run_name") or "cosmos2-run"
    working_config["run_name"] = f"{run_name}-{slice_descriptor.slice_id}"
    working_config["slice_descriptor"] = asdict(slice_descriptor)

    worker_dir = Path(".worker_runs").resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    output_section = dict(working_config.get("output") or {})
    output_section["base_dir"] = str(worker_dir)
    working_config["output"] = output_section

    if datasets:
        working_config.setdefault("metadata", {})["dataset_hashes"] = dict(datasets)
    metadata_state = _apply_package_metadata(working_config)
    metadata = working_config.get("metadata")
    if metadata:
        logger.info("Compute slice %s metadata=%s", slice_descriptor.slice_id, json.dumps(metadata))

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(working_config, tmp)
        tmp.flush()
        temp_path = Path(tmp.name)

    start = time.monotonic()
    run_path: Path | None = None
    try:
        science_config = ScienceRunConfig.from_path(temp_path)
        runner = Cosmos2ScienceRunner(science_config)
        run_path = runner.execute()
        duration = time.monotonic() - start
        success_info = _capture_run_summary(run_path)
        _cleanup_run_dir(run_path)
        return {
            "success": True,
            "metrics": {
                "duration_seconds": duration,
                "artifact_count": success_info.get("artifact_count", 0),
            },
            "data": {
                "run_dir": str(run_path),
                "slice_id": slice_descriptor.slice_id,
                "artifact_preview": success_info.get("artifact_preview", []),
                "summaries": success_info.get("summaries", {}),
            },
        }
    except Exception as exc:  # noqa: BLE001
        duration = time.monotonic() - start
        failure_info = {}
        if run_path and run_path.exists():
            failure_info = _capture_run_summary(run_path)
            _cleanup_run_dir(run_path)
        return {
            "success": False,
            "logs": str(exc),
            "metrics": {
                "duration_seconds": duration,
                "artifact_count": failure_info.get("artifact_count", 0),
            },
            "data": {
                "slice_id": slice_descriptor.slice_id,
                "artifact_preview": failure_info.get("artifact_preview", []),
                "summaries": failure_info.get("summaries", {}),
            },
        }
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass
        if metadata_state.get("jackknife_applied"):
            clear_jackknife_masked_datasets()


def _capture_run_summary(run_dir: Path, *, max_preview: int = 64) -> Dict[str, Any]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(run_dir)))
    previews = files[:max_preview]
    summaries: Dict[str, Any] = {}
    for candidate in ("history_entry.json", "run_meta.json", "config_used.json"):
        value = _safe_load_json(run_dir / candidate)
        if value is not None:
            summaries[candidate] = value
    return {
        "artifact_count": len(files),
        "artifact_preview": previews,
        "summaries": summaries,
    }


def _safe_load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cleanup_run_dir(run_dir: Path) -> None:
    try:
        shutil.rmtree(run_dir)
    except Exception:
        pass
