"""High-level engine entrypoints for cosmos2."""

from __future__ import annotations

import queue
import json
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from data_interface.standardize import ensure_standard_dataset

from cosmos2.fits import FIT_REGISTRY
from cosmos2.fits.joint import build_joint_chi2_evaluator, resolve_joint_fits
from cosmos2.models.model_factory import create_model
from cosmos2.models.pbuf import PBUF_FIT_REGISTRY, build_pbuf_joint_chi2, resolve_pbuf_joint_fits
from cosmos2.threads.collector_thread import run_collector_thread
from cosmos2.threads.model_thread import run_model_thread
from cosmos2.threads.monitor_thread import run_monitor_thread


def _load_standardized_npz(path: Path) -> Dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    data = {key: payload[key] for key in payload.files}
    return ensure_standard_dataset(data, data.get("type", ""))


def _load_dataset(name: str) -> Dict[str, Any]:
    path = Path("data/standardized") / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Standardized dataset not found: {path}")
    return _load_standardized_npz(path)


def _make_joint_evaluator(model_name: str, joint_config_path: Path, *, lut: Dict[str, np.ndarray] | None = None) -> Any:
    normalized = model_name.strip().lower()
    config_path = Path(joint_config_path)
    skip_valid = False

    if normalized == "pbuf":
        joint_fn = build_pbuf_joint_chi2(
            lambda params: _create_model(model_name, params, lut=lut),
            config_path,
            skip_valid=skip_valid,
        )
    else:
        joint_fn = build_joint_chi2_evaluator(
            lambda params: _create_model(model_name, params, lut=lut),
            config_path,
            skip_valid=skip_valid,
        )

    def evaluator(params: Dict[str, float]) -> float:
        return float(joint_fn(params))

    return evaluator


def _create_model(
    model_name: str,
    params: Dict[str, float],
    *,
    lut: Dict[str, np.ndarray] | None = None,
):
    normalized = model_name.strip().lower()
    return create_model(model_name, lut=lut, **params)


def _evaluate_fit_breakdown(
    model_name: str,
    params: Dict[str, float],
    *,
    fits: Sequence[str],
    fit_weights: Dict[str, float],
    model_kwargs: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], float]:
    """Run registered fits for the supplied best-fit parameters and collect extras."""
    model_kwargs = dict(model_kwargs or {})
    model = create_model(model_name, **model_kwargs, **params)
    fit_results: Dict[str, Any] = {}
    weighted_total = 0.0
    registry = PBUF_FIT_REGISTRY if model_name.strip().lower() == "pbuf" else FIT_REGISTRY

    for fit_name in fits:
        fit_fn = registry.get(fit_name)
        if fit_fn is None:
            continue
        try:
            result = fit_fn(model)
        except Exception as exc:  # noqa: BLE001
            fit_results[fit_name] = {"error": str(exc)}
            continue

        if isinstance(result, tuple):
            chi2_val = float(result[0])
            extras = result[1] if len(result) > 1 else {}
        else:
            chi2_val = float(result)
            extras = {}
        weight = float(fit_weights.get(fit_name, 1.0))
        weighted_chi2 = weight * chi2_val if np.isfinite(chi2_val) else float("inf")
        fit_results[fit_name] = {
            "chi2": chi2_val,
            "weight": weight,
            "weighted_chi2": weighted_chi2,
            "extras": extras,
        }
        if np.isfinite(chi2_val):
            weighted_total += weighted_chi2

    return fit_results, weighted_total


def build_pbuf_model_config(
    bounds: Dict[str, Sequence[float]],
    lut: Dict[str, np.ndarray] | None,
    joint_config: str | Path,
    *,
    grid_points: int | None = None,
) -> Dict[str, Any]:
    """Helper to create a model_config dict for PBUF using the per-model fit registry."""
    joint_path = Path(joint_config)
    fits, fit_weights = resolve_pbuf_joint_fits(joint_path)
    config = {
        "name": "pbuf",
        "bounds": bounds,
        "evaluator": _make_joint_evaluator("pbuf", Path(joint_config)),
        "joint_config_path": joint_path,
        "fits": fits,
        "fit_weights": fit_weights,
        "model_kwargs": {"lut": lut} if lut is not None else {},
    }
    if grid_points is not None:
        config["grid_points"] = int(grid_points)
    return config


def build_lcdm_model_config(
    bounds: Dict[str, Sequence[float]],
    joint_config: str | Path,
    *,
    grid_points: int | None = None,
) -> Dict[str, Any]:
    """Helper to create a model_config dict for LCDM."""
    joint_path = Path(joint_config)
    fits, fit_weights = resolve_joint_fits(joint_path)
    config = {
        "name": "lcdm",
        "bounds": bounds,
        "evaluator": _make_joint_evaluator("lcdm", Path(joint_config)),
        "joint_config_path": joint_path,
        "fits": fits,
        "fit_weights": fit_weights,
        "model_kwargs": {},
    }
    if grid_points is not None:
        config["grid_points"] = int(grid_points)
    return config


def run_optimisation(
    model_configs: Iterable[Dict],
    *,
    monitor: bool = False,
    grid_points: int | None = None,
    workers: int | None = None,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
    checkpoint_path: str | Path | None = None,
) -> Dict:
    """
    Run optimisation for the provided model configurations.

    Each model_config should include:
      - "name": identifier
      - "bounds": parameter bounds
      - "evaluator": callable(params)->chi2
      - optional "n_batches"/"batch_size"/"grid_points"

    Returns a summary dict containing per-model results and the global best.
    """

    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
    event_history: list[Dict[str, Any]] = []
    collector_state: Dict[str, Any] = {}

    def _write_checkpoint(extra: Dict[str, Any] | None = None, *, complete: bool = False) -> None:
        if checkpoint_file is None:
            return
        payload: Dict[str, Any] = {
            "events": event_history,
            "complete": complete,
        }
        if extra:
            payload.update(extra)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            checkpoint_file.write_text(json.dumps(payload, default=float), encoding="utf-8")
        except Exception:
            pass

    def _emit(event: Dict[str, Any]) -> None:
        if progress_callback is not None:
            try:
                progress_callback(event)
            except Exception:
                # Progress hooks should never break optimisation flow.
                pass
        event_history.append(event)
        if monitor and state_lock is not None:
            try:
                with state_lock:
                    event_type = event.get("type")
                    if event_type == "collector_update":
                        monitor_state["best_overall"] = event.get("best_overall")
                        monitor_state["chi2_history"] = event_history[-10:]
                    elif event_type == "model_complete":
                        model_name = event.get("model") or "model"
                        models_state = monitor_state.setdefault("models", {})
                        models_state[model_name] = {
                            "model": model_name,
                            "best_chi2": event.get("best_chi2"),
                            "best_so_far": event.get("best_chi2"),
                            "last_chi2": event.get("best_chi2"),
                            "chi2_breakdown": event.get("chi2_breakdown"),
                        }
                        monitor_state["best_overall"] = monitor_state.get("best_overall") or event
                        monitor_state["chi2_history"] = event_history[-10:]
            except Exception:
                pass
        _write_checkpoint({"best_overall": collector_state.get("best_overall") if collector_state else None})

    model_summaries = []
    monitor_state: Dict[str, Any] = {"models": {}}
    state_lock: Lock | None = None
    monitor_stop: Event | None = None
    monitor_thread: Thread | None = None
    try:
        state_lock = Lock()
    except Exception:
        state_lock = None

    try:
        if monitor and state_lock is not None:
            monitor_state["meta"] = {"started_at": time.time()}
            monitor_stop = Event()
            monitor_thread = Thread(
                target=run_monitor_thread,
                args=(monitor_state,),
                kwargs={"refresh_hz": 0.2, "iterations": None, "stop_event": monitor_stop, "lock": state_lock},
                daemon=True,
            )
            monitor_thread.start()
        for config in model_configs:
            config_with_defaults = dict(config)
            if grid_points is not None and "grid_points" not in config_with_defaults:
                config_with_defaults["grid_points"] = int(grid_points)
            if "workers" not in config_with_defaults:
                worker_override = workers if workers is not None else None
                if worker_override is None:
                    worker_override = config.get("workers") or config.get("threads")
                if worker_override:
                    config_with_defaults["workers"] = int(worker_override)
            elif workers is not None:
                config_with_defaults["workers"] = int(workers)
            summary = run_model_thread(
                config_with_defaults,
                queue=None,
                shared_state=monitor_state if monitor and state_lock is not None else None,
                lock=state_lock,
            )
            summary["name"] = config.get("name")
            model_summaries.append(summary)
    finally:
        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2)

    # Evaluate per-fit breakdowns + extras for monitoring/reporting.
    enriched_summaries = []
    for config, summary in zip(model_configs, model_summaries):
        best_params = summary.get("best_params") or {}
        fits = config.get("fits") or []
        fit_weights = config.get("fit_weights") or {}
        model_kwargs = config.get("model_kwargs") or {}
        summary["name"] = summary.get("name") or config.get("name")
        if best_params and fits:
            fit_results, weighted_total = _evaluate_fit_breakdown(
                config.get("name", "model"),
                best_params,
                fits=fits,
                fit_weights=fit_weights,
                model_kwargs=model_kwargs,
            )
            summary["fit_results"] = fit_results
            summary["chi2_breakdown"] = {
                k: {
                    "chi2": v.get("chi2"),
                    "weight": v.get("weight"),
                    "weighted_chi2": v.get("weighted_chi2"),
                    "extras": v.get("extras"),
                }
                for k, v in fit_results.items()
                if isinstance(v, dict)
            }
            summary["weighted_chi2"] = weighted_total
        summary["fits"] = fits
        summary["fit_weights"] = fit_weights
        summary["joint_config_path"] = config.get("joint_config_path")
        enriched_summaries.append(summary)
        _emit(
            {
                "type": "model_complete",
                "model": summary.get("name"),
                "best_chi2": summary.get("best_chi2"),
                "weighted_chi2": summary.get("weighted_chi2"),
                "chi2_breakdown": summary.get("chi2_breakdown"),
                "fits": fits,
            }
        )

    q: queue.Queue = queue.Queue()
    for summary in enriched_summaries:
        q.put(summary)
    q.put(None)
    run_collector_thread(q, collector_state)
    if collector_state.get("best_overall") is not None:
        _emit({"type": "collector_update", "best_overall": collector_state.get("best_overall")})

    if monitor:
        if state_lock is not None:
            with state_lock:
                monitor_state["best_overall"] = collector_state.get("best_overall")
                monitor_state["chi2_history"] = collector_state.get("chi2_history")
        if monitor_stop is not None:
            monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2)
        else:
            # If we could not spawn a background thread, emit a single snapshot.
            shared = dict(collector_state)
            if monitor_state:
                shared["latest_batch"] = monitor_state.get("last_batch")
            run_monitor_thread(shared, iterations=1)

    _write_checkpoint(
        {
            "best_overall": collector_state.get("best_overall"),
            "models": enriched_summaries,
        },
        complete=True,
    )

    return {
        "models": enriched_summaries,
        "best_overall": collector_state.get("best_overall"),
        "chi2_history": collector_state.get("chi2_history"),
    }
