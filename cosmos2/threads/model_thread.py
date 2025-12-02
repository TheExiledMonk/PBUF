"""Model thread orchestration for cosmos2."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, Optional
from threading import Lock

from cosmos2.walkers.basin_walker import BasinWalker
from cosmos2.threads.executor import make_process_pool

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore


def _capture_process_cpu_time(process: "psutil.Process" | None) -> float | None:
    if process is None:
        return None
    try:
        times = process.cpu_times()
        return float(times.user + times.system)
    except Exception:
        return None


def _capture_process_memory_mb(process: "psutil.Process" | None) -> float | None:
    if process is None:
        return None
    try:
        rss = process.memory_info().rss
        return float(rss) / (1024 ** 2)
    except Exception:
        return None


def _sample_gpu_stats() -> Dict[str, float] | None:
    """Try to query NVIDIA GPU utilization via nvidia-smi."""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    line = output.strip().splitlines()[0] if output.strip() else ""
    if not line:
        return None
    parts = [segment.strip() for segment in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        gpu_util = float(parts[0])
        mem_used = float(parts[1])
        return {"utilization_pct": gpu_util, "memory_used_mb": mem_used}
    except ValueError:
        return None


def run_model_thread(model_config: Dict, queue=None, shared_state: Optional[Dict] = None, lock: Optional[Lock] = None) -> Dict:
    """
    Run a single-model optimisation loop.

    model_config expects:
      - "bounds": parameter bounds dict
      - "evaluator": callable(params)->chi2
      - optional "n_batches": iterations to run
      - optional "batch_size": proposals per batch
      - optional sampler tuning: n_scatter, scatter_scale, island_fraction, grid_points, rng_seed
    """
    bounds = model_config["bounds"]
    evaluator = model_config["evaluator"]
    gate = model_config.get("param_gate") or model_config.get("gate")
    if gate is not None:
        def _guarded_evaluator(params):
            verdict = gate(params)
            ok = verdict
            reason = None
            if isinstance(verdict, (list, tuple)) and len(verdict) >= 1:
                ok = verdict[0]
                reason = verdict[1] if len(verdict) > 1 else None
            if not ok:
                if reason:
                    # Optional hook for later debugging/monitoring
                    _ = reason  # noqa: F841
                return float("inf")
            return evaluator(params)

        evaluator_fn = _guarded_evaluator
    else:
        evaluator_fn = evaluator
    n_batches = int(model_config.get("n_batches", 1))
    batch_size = int(model_config.get("batch_size", 32))
    rng_seed = model_config.get("rng_seed")
    model_name = model_config.get("name", "model")
    rng = None
    if rng_seed is not None:
        import numpy as np

        rng = np.random.default_rng(int(rng_seed))

    walker = BasinWalker(
        bounds=bounds,
        evaluator=evaluator_fn,
        batch_size=batch_size,
        rng=rng,
        n_scatter=int(model_config.get("n_scatter", 0)),
        scatter_scale=float(model_config.get("scatter_scale", 0.05)),
        island_fraction=float(model_config.get("island_fraction", 0.5)),
        grid_points=model_config.get("grid_points"),
    )
    worker_count = int(model_config.get("workers", 1))
    map_fn = None
    shutdown = None
    pool_fallbacks = 0
    if worker_count > 1:
        map_fn, shutdown = make_process_pool(evaluator_fn, worker_count)
    process = psutil.Process() if psutil else None
    start_cpu = _capture_process_cpu_time(process)
    start_mem = _capture_process_memory_mb(process)
    start_ts = time.time()
    all_results = []
    chi2_history = []
    running_best = float("inf")
    eval_counter = 0
    for batch_idx in range(n_batches):
        batch_results = walker.run_batch(map_fn=map_fn)
        all_results.extend(batch_results)
        for params, chi2 in batch_results:
            eval_counter += 1
            running_best = chi2 if chi2 < running_best else running_best
            chi2_history.append(
                {
                    "batch": batch_idx,
                    "params": params,
                    "chi2": chi2,
                    "best_so_far": running_best,
                    "model": model_name,
                }
            )
        if shared_state is not None and lock is not None:
            with lock:
                models_state = shared_state.setdefault("models", {})
                started_at = models_state.get(model_name, {}).get("started_at") or time.time()
                models_state[model_name] = {
                    "model": model_name,
                    "batch": batch_idx + 1,
                    "total_batches": n_batches,
                    "best_chi2": walker.best_chi2,
                    "last_chi2": batch_results[-1][1] if batch_results else float("inf"),
                    "best_so_far": running_best,
                    "recent_history": chi2_history[-5:],
                    "evals": eval_counter,
                    "workers": worker_count,
                    "started_at": started_at,
                }
                if map_fn is not None and hasattr(map_fn, "fallback_hits"):
                    pool_fallbacks = getattr(map_fn, "fallback_hits", 0)
                    if pool_fallbacks:
                        models_state[model_name]["pool_fallbacks"] = pool_fallbacks
                shared_state["latest_batch"] = models_state[model_name]

    summary = {
        "best_params": walker.best_params,
        "best_chi2": walker.best_chi2,
        "results": all_results,
        "chi2_history": chi2_history,
    }

    if shared_state is not None and lock is not None:
        with lock:
            shared_state["last_batch"] = {
                "model": model_name,
                "chi2_history": chi2_history[-5:],
                "best_chi2": walker.best_chi2,
            }
    if shutdown is not None:
        shutdown()
    end_ts = time.time()
    end_cpu = _capture_process_cpu_time(process)
    end_mem = _capture_process_memory_mb(process)
    gpu_snapshot = _sample_gpu_stats()
    tolerance_value = (
        model_config.get("tolerance")
        or model_config.get("tol")
        or model_config.get("stop_tolerance")
        or model_config.get("convergence_tol")
    )
    performance: Dict[str, float | str | Dict[str, float] | None] = {
        "start_time": datetime.fromtimestamp(start_ts, timezone.utc).isoformat(),
        "end_time": datetime.fromtimestamp(end_ts, timezone.utc).isoformat(),
        "duration_seconds": float(end_ts - start_ts),
        "cpu_seconds": float(end_cpu - start_cpu) if start_cpu is not None and end_cpu is not None else None,
        "memory_rss_mb": float(end_mem) if end_mem is not None else None,
        "memory_rss_start_mb": float(start_mem) if start_mem is not None else None,
        "worker_count": worker_count,
        "batch_iterations": n_batches,
        "batch_size": batch_size,
        "evaluations": eval_counter,
        "stop_reason": f"Completed {n_batches} batches",
        "stop_tolerance": tolerance_value,
        "gpu": gpu_snapshot,
    }
    summary["performance"] = performance
    if queue is not None:
        queue.put(summary)
    return summary
