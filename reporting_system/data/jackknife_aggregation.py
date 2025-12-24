"""Jackknife aggregation helpers for the reporting system.

This module is intentionally reporting-only: it reads existing jackknife outputs and
aggregates fold-level quantities across multiple runs without recomputing fits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_name: str
    timestamp: str | None
    jackknife_seed: int | None
    models: list[str]


@dataclass(frozen=True)
class AggregatedJackknife:
    """Fold-level pooled jackknife values across runs."""

    runs: list[RunInfo]
    models: list[str]
    pooled_chi2: dict[str, np.ndarray]
    pooled_delta_chi2: dict[tuple[str, str], np.ndarray]
    delta_by_seed: dict[str, np.ndarray]


@dataclass(frozen=True)
class PooledDrawSeries:
    """Ordered pooled series across all jackknife draws in the provided runs."""

    runs: list[RunInfo]
    models: list[str]
    chi2_by_model: dict[str, np.ndarray]
    h0_by_model: dict[str, np.ndarray]
    seed_by_draw: list[str]


_RUN_DIR_STAMP_RE = re.compile(r"^(\\d{4}-\\d{2}-\\d{2}T\\d{6})_")


def discover_candidate_runs(root: Path) -> list[Path]:
    """Return run directories under root (depth=1) that look like science runs."""
    root = Path(root)
    if not root.exists():
        return []
    if (root / "config_used.json").exists():
        return [root]
    runs: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "config_used.json").exists() or (entry / "run_meta.json").exists():
            runs.append(entry)
    return runs


def select_latest_per_run_name(run_dirs: Iterable[Path]) -> list[Path]:
    """Group by run_name and return the latest directory per group."""

    best: dict[str, tuple[str, Path]] = {}
    for run_dir in run_dirs:
        info = _read_run_info(run_dir)
        if not info:
            continue
        stamp = info.timestamp or _timestamp_from_dirname(run_dir.name) or ""
        current = best.get(info.run_name)
        if current is None or stamp > current[0]:
            best[info.run_name] = (stamp, run_dir)
    return [value[1] for value in sorted(best.values(), key=lambda item: item[0])]


def _timestamp_from_dirname(name: str) -> str | None:
    match = _RUN_DIR_STAMP_RE.match(name)
    if not match:
        return None
    return match.group(1)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _read_run_info(run_dir: Path) -> RunInfo | None:
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config_used.json") or {}
    run_meta = _load_json(run_dir / "run_meta.json") or {}

    run_name = config.get("run_name") or run_meta.get("run_name") or run_dir.name
    run_name = str(run_name)
    timestamp = run_meta.get("timestamp") or _timestamp_from_dirname(run_dir.name)
    timestamp = str(timestamp) if timestamp else None

    jk_seed = None
    jackknife = config.get("jackknife") or {}
    if isinstance(jackknife, dict):
        raw = jackknife.get("random_seed")
        if raw is not None:
            try:
                jk_seed = int(raw)
            except Exception:
                jk_seed = None

    models = config.get("models")
    if not isinstance(models, list) or not models:
        # fallback to top-level directories
        models = sorted([p.name for p in run_dir.iterdir() if p.is_dir() and (p / "best_fit.json").exists()])
    models = [str(m).strip().lower() for m in models if str(m).strip()]
    return RunInfo(
        run_dir=run_dir,
        run_name=run_name,
        timestamp=timestamp,
        jackknife_seed=jk_seed,
        models=models,
    )


def _load_draws_payload(run_dir: Path) -> dict[str, Any] | None:
    """Load the jackknife draws container from any supported filename."""

    for name in ("jackknife_results.json", "jackknife_summary.json"):
        payload = _load_json(Path(run_dir) / name)
        if payload and isinstance(payload.get("draws"), list):
            return payload
    return None


def _extract_draw_chi2(draw: dict[str, Any]) -> dict[str, float]:
    jackknife_models = draw.get("jackknife_models") or {}
    original_models = draw.get("original_models") or {}
    out: dict[str, float] = {}
    if not isinstance(jackknife_models, dict):
        jackknife_models = {}
    if not isinstance(original_models, dict):
        original_models = {}
    for model_name, payload in jackknife_models.items():
        if not isinstance(payload, dict):
            continue
        chi2 = payload.get("chi_squared", payload.get("chi2"))
        if chi2 is None:
            chi2 = (original_models.get(model_name) or {}).get("chi_squared")
        if chi2 is None:
            continue
        try:
            out[str(model_name).strip().lower()] = float(chi2)
        except Exception:
            continue
    return out


def aggregate_jackknife_runs(run_dirs: Iterable[Path]) -> AggregatedJackknife:
    run_infos: list[RunInfo] = []
    chi2_values: dict[str, list[float]] = {}

    inferred_models: set[str] = set()
    run_dirs_list = [Path(p) for p in run_dirs]
    for run_dir in run_dirs_list:
        info = _read_run_info(run_dir)
        if not info:
            continue
        draws_payload = _load_draws_payload(run_dir)
        if not draws_payload:
            continue
        draws = draws_payload.get("draws") or []
        if not isinstance(draws, list) or not draws:
            continue
        run_infos.append(info)
        inferred_models.update(info.models)

        for draw in draws:
            if not isinstance(draw, dict):
                continue
            per_model = _extract_draw_chi2(draw)
            for model, chi2 in per_model.items():
                chi2_values.setdefault(model, []).append(float(chi2))

    models = sorted(inferred_models) if inferred_models else sorted(chi2_values.keys())

    pooled_chi2: dict[str, np.ndarray] = {}
    for model in models:
        pooled_chi2[model] = np.asarray(chi2_values.get(model, []), dtype=float)

    pooled_delta: dict[tuple[str, str], np.ndarray] = {}
    delta_by_seed_values: dict[str, list[float]] = {}
    if len(models) >= 2:
        # Prefer LCDM baseline if present, otherwise take first two sorted models.
        if "lcdm" in models:
            baseline = "lcdm"
            other = next((m for m in models if m != baseline), None)
            if other is not None:
                a, b = other, baseline
            else:
                a, b = models[0], models[0]
        else:
            a, b = models[1], models[0]

        deltas: list[float] = []
        for info in run_infos:
            draws_payload = _load_draws_payload(info.run_dir)
            draws = (draws_payload or {}).get("draws") or []
            seed_key = str(info.jackknife_seed) if info.jackknife_seed is not None else info.run_name
            for draw in draws:
                if not isinstance(draw, dict):
                    continue
                per_model = _extract_draw_chi2(draw)
                if a not in per_model or b not in per_model:
                    continue
                delta = float(per_model[a]) - float(per_model[b])
                deltas.append(delta)
                delta_by_seed_values.setdefault(seed_key, []).append(delta)

        if deltas:
            pooled_delta[(a, b)] = np.asarray(deltas, dtype=float)

    delta_by_seed = {k: np.asarray(v, dtype=float) for k, v in delta_by_seed_values.items()}
    return AggregatedJackknife(
        runs=run_infos,
        models=models,
        pooled_chi2=pooled_chi2,
        pooled_delta_chi2=pooled_delta,
        delta_by_seed=delta_by_seed,
    )


def pool_draw_series(run_dirs: Iterable[Path], *, models: list[str] | None = None) -> PooledDrawSeries:
    """Pool jackknife draw series across runs, preserving draw ordering.

    This is used to render the same draw-index plots as the single-run report, but
    across the concatenated ensemble of runs.
    """
    run_infos: list[RunInfo] = []
    collected_models: set[str] = set(models or [])
    chi2_values: dict[str, list[float]] = {m: [] for m in (models or [])}
    h0_values: dict[str, list[float]] = {m: [] for m in (models or [])}
    seed_by_draw: list[str] = []

    for run_dir in [Path(p) for p in run_dirs]:
        info = _read_run_info(run_dir)
        if not info:
            continue
        draws_payload = _load_draws_payload(run_dir)
        if not draws_payload:
            continue
        draws = draws_payload.get("draws") or []
        if not isinstance(draws, list) or not draws:
            continue
        run_infos.append(info)
        collected_models.update(info.models)
        seed_key = str(info.jackknife_seed) if info.jackknife_seed is not None else info.run_name

        for draw in draws:
            if not isinstance(draw, dict):
                continue
            seed_by_draw.append(seed_key)
            per_model = _extract_draw_chi2(draw)
            jackknife_models = draw.get("jackknife_models") or {}
            if not isinstance(jackknife_models, dict):
                jackknife_models = {}
            for model in collected_models:
                chi2 = per_model.get(model)
                chi2_values.setdefault(model, []).append(float(chi2) if chi2 is not None else float("nan"))
                h0 = None
                payload = jackknife_models.get(model)
                if isinstance(payload, dict):
                    params = payload.get("parameters") or {}
                    if isinstance(params, dict):
                        h0 = params.get("H0")
                try:
                    h0_values.setdefault(model, []).append(float(h0) if h0 is not None else float("nan"))
                except Exception:
                    h0_values.setdefault(model, []).append(float("nan"))

    final_models = sorted(collected_models)
    return PooledDrawSeries(
        runs=run_infos,
        models=final_models,
        chi2_by_model={m: np.asarray(chi2_values.get(m, []), dtype=float) for m in final_models},
        h0_by_model={m: np.asarray(h0_values.get(m, []), dtype=float) for m in final_models},
        seed_by_draw=seed_by_draw,
    )


def distribution_stats(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    q16, q50, q84 = np.quantile(arr, [0.16, 0.5, 0.84])
    q25, q75 = np.quantile(arr, [0.25, 0.75])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(q50),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "q16": float(q16),
        "q84": float(q84),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
