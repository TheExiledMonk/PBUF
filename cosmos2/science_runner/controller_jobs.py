"""Controller job packaging helpers for the Cosmos2 science runner."""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from cosmos2.science_runner.config import ScienceRunConfig


class ControllerAPIError(Exception):
    """Base class for controller API failures."""


class ControllerJobFailedError(ControllerAPIError):
    """Raised when a controller job terminates in a terminal failure state."""


@dataclass
class JobPackage:
    """Descriptor for a controller-bound basin walker job."""

    package_id: str
    package_type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    slice_count: int | None = None


class ControllerAPI:
    """Thin HTTP client for the controller REST API."""

    def __init__(self, endpoint: str, *, timeout: float = 10.0) -> None:
        if not endpoint:
            raise ValueError("Controller endpoint must be provided.")
        self.base_url = endpoint.rstrip("/")
        self.timeout = float(timeout)

    def submit_job(
        self,
        payload: dict[str, Any],
        *,
        slice_count: int | None = None,
        dataset_id: str | None = None,
    ) -> dict[str, Any]:
        request_payload: dict[str, Any] = {"config": payload}
        if slice_count is not None:
            request_payload["slice_count"] = slice_count
        if dataset_id is not None:
            request_payload["dataset_id"] = dataset_id
        return self._request("POST", "/controller/jobs", request_payload)

    def get_job(self, execution_id: str) -> dict[str, Any]:
        return self._request("GET", f"/controller/jobs/{execution_id}")

    def wait_for_completion(
        self,
        execution_id: str,
        *,
        timeout: float = 3600.0,
        poll_interval: float = 5.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        while True:
            job = self.get_job(execution_id)
            status = (job.get("status") or "").lower()
            if status == "completed":
                return job
            if status in {"failed", "canceled"}:
                raise ControllerJobFailedError(
                    f"Controller job {execution_id} ended with status '{status}'"
                )
            if time.monotonic() >= deadline:
                raise ControllerAPIError(f"Timeout waiting for job {execution_id}")
            time.sleep(float(poll_interval))

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = (
            json.dumps(payload, default=str).encode("utf-8") if payload is not None else None
        )
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise ControllerAPIError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ControllerAPIError(str(exc)) from exc


def _merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        target = merged.get(key)
        if isinstance(target, dict) and isinstance(value, dict):
            merged[key] = {**target, **copy.deepcopy(value)}
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build_metadata(
    config: ScienceRunConfig,
    *,
    package_type: str,
    package_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "origin": "basin_walker",
        "run_name": config.run_name,
        "models": list(config.models),
        "engine": config.engine,
        "package_type": package_type,
    }
    if package_id:
        metadata["package_id"] = package_id
    if extra:
        metadata.update(extra)
    return metadata


def _prepare_package_payload(
    config: ScienceRunConfig,
    *,
    overrides: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = _merge_overrides(config.to_dict(), overrides)
    payload.setdefault("engine_settings", {})
    payload_metadata = payload.setdefault("metadata", {})
    payload_metadata.update(metadata)
    return payload


def generate_seed_packages(
    config: ScienceRunConfig,
    *,
    package_size: int | None = None,
    slice_count: int | None = None,
) -> list[JobPackage]:
    engine_settings = dict(config.engine_settings or {})
    total_batches = int(engine_settings.get("n_batches") or engine_settings.get("n_seeds") or 1)
    if total_batches <= 0:
        total_batches = 1
    resolved_package_size = int(package_size or engine_settings.get("package_size") or total_batches)
    resolved_package_size = max(1, resolved_package_size)
    if slice_count is None:
        slice_count = int(engine_settings.get("workers") or 1)
    packages: list[JobPackage] = []
    index = 0
    for start in range(0, total_batches, resolved_package_size):
        index += 1
        chunk = min(resolved_package_size, total_batches - start)
        overrides = {
            "engine_settings": {"n_batches": chunk},
            "jackknife": {"enabled": False},
            "predictions": {"enabled": False, "modules": []},
        }
        metadata = _build_metadata(
            config,
            package_type="seed_batch",
            package_id=f"seed-{index:02d}",
            extra={
                "package_index": index,
                "seed_start": start,
                "seed_end": start + chunk,
                "total_batches": total_batches,
                "workers": int(engine_settings.get("workers") or 1),
            },
        )
        payload = _prepare_package_payload(config, overrides=overrides, metadata=metadata)
        packages.append(
            JobPackage(
                package_id=f"seed-{index:02d}",
                package_type="seed_batch",
                payload=payload,
                metadata=metadata,
                slice_count=slice_count,
            )
        )
    return packages


def generate_jackknife_packages(
    config: ScienceRunConfig,
    *,
    slice_count: int | None = None,
) -> list[JobPackage]:
    jackknife = config.jackknife
    if not (jackknife and jackknife.enabled):
        return []
    draws = max(0, int(jackknife.n_draws))
    if draws <= 0:
        return []
    if slice_count is None:
        slice_count = int(config.engine_settings.get("workers") or 1)

    jackknife_payload = jackknife.to_dict()
    datasets = list(jackknife.datasets_to_test)
    base_extra = {
        "jackknife_config": jackknife_payload,
        "jackknife_datasets": datasets,
    }

    packages: list[JobPackage] = []
    for index in range(draws):
        draw_seed = (
            (jackknife.random_seed + index) if jackknife.random_seed is not None else index
        )
        overrides = {
            "jackknife": {**jackknife_payload, "enabled": False},
            "predictions": {"enabled": False, "modules": []},
        }
        metadata = _build_metadata(
            config,
            package_type="jackknife_draw",
            package_id=f"jackknife-draw-{index+1:02d}",
            extra={
                **base_extra,
                "package_index": index + 1,
                "total_draws": draws,
                "jackknife_draw": {
                    "index": index,
                    "seed": draw_seed,
                    "datasets": datasets,
                    "fraction_removed": float(jackknife.fraction_removed),
                },
            },
        )
        payload = _prepare_package_payload(config, overrides=overrides, metadata=metadata)
        packages.append(
            JobPackage(
                package_id=f"jackknife-draw-{index+1:02d}",
                package_type="jackknife_draw",
                payload=payload,
                metadata=metadata,
                slice_count=slice_count,
            )
        )
    return packages


def generate_prediction_packages(
    config: ScienceRunConfig,
    *,
    slice_count: int | None = None,
) -> list[JobPackage]:
    predictions_cfg = config.predictions
    if not (predictions_cfg.enabled and predictions_cfg.modules):
        return []
    if slice_count is None:
        slice_count = int(config.engine_settings.get("workers") or 1)

    packages: list[JobPackage] = []
    for index, module_name in enumerate(predictions_cfg.modules, start=1):
        normalized = module_name.strip().lower()
        module_config = predictions_cfg.get_module_config(normalized)
        overrides = {
            "jackknife": {"enabled": False},
            "predictions": {
                "enabled": True,
                "modules": [module_name],
                **({normalized: dict(module_config)} if module_config else {}),
            },
        }
        metadata = _build_metadata(
            config,
            package_type="prediction_batch",
            package_id=f"prediction-{index:02d}",
            extra={
                "package_index": index,
                "prediction_modules": [module_name],
                "prediction_module_configs": {normalized: dict(module_config)}
                if module_config
                else {},
                "prediction_phase": "post_fit",
            },
        )
        payload = _prepare_package_payload(config, overrides=overrides, metadata=metadata)
        packages.append(
            JobPackage(
                package_id=f"prediction-{index:02d}",
                package_type="prediction_batch",
                payload=payload,
                metadata=metadata,
                slice_count=slice_count,
            )
        )
    return packages


def build_job_packages(
    config: ScienceRunConfig,
    *,
    package_size: int | None = None,
) -> list[JobPackage]:
    """Return job packages derived from the science configuration."""
    packages: list[JobPackage] = []
    packages.extend(generate_seed_packages(config, package_size=package_size))
    packages.extend(generate_jackknife_packages(config))
    packages.extend(generate_prediction_packages(config))
    return packages
