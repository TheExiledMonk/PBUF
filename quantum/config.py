"""Configuration management for the Quantum + E₀ subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple
import json

import yaml

from .paths import resolve_path, quantum_root, repo_root


CONFIG_ROOT = repo_root() / "configs" / "quantum"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "config" / "defaults.yaml"


def _read_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    return data or {}


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    result: MutableMapping[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class DataConfig:
    root: Path
    events_dir: Optional[Path]
    events_search_roots: Tuple[Path, ...]
    events_patterns: Tuple[str, ...]
    reports_dir: Path
    downloader_log: Optional[Path]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "root": str(self.root),
            "events_dir": str(self.events_dir) if self.events_dir else None,
            "events_search_roots": [str(path) for path in self.events_search_roots],
            "events_patterns": list(self.events_patterns),
            "reports_dir": str(self.reports_dir),
            "downloader_log": str(self.downloader_log) if self.downloader_log else None,
        }


@dataclass(frozen=True)
class AlphaConfig:
    alpha_band: Tuple[float, float]
    mixing_range: Tuple[float, float]
    mixing_samples: int
    regulators: Dict[str, float]
    field_sets: Dict[str, float]
    enforce_reproducibility: bool
    reference_field: Optional[str]
    target_regulator: str
    target_field_set: str
    warnings_threshold: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alpha_band": list(self.alpha_band),
            "mixing_range": list(self.mixing_range),
            "mixing_samples": self.mixing_samples,
            "regulators": self.regulators,
            "field_sets": self.field_sets,
            "enforce_reproducibility": self.enforce_reproducibility,
            "reference_field": self.reference_field,
            "target_regulator": self.target_regulator,
            "target_field_set": self.target_field_set,
            "warnings_threshold": self.warnings_threshold,
        }


@dataclass(frozen=True)
class QuantumEngineConfig:
    source: str
    data: DataConfig
    time_window: float
    eps_min: float
    eps_max: float
    steps: int
    k_eps: float
    threads: int
    alpha: AlphaConfig

    def eps_range(self) -> Tuple[float, float]:
        return (self.eps_min, self.eps_max)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "time_window": self.time_window,
            "eps_min": self.eps_min,
            "eps_max": self.eps_max,
            "steps": self.steps,
            "k_eps": self.k_eps,
            "threads": self.threads,
            "data": self.data.as_dict(),
            "alpha": self.alpha.as_dict(),
        }


def _build_data_config(raw: Mapping[str, Any]) -> DataConfig:
    root = resolve_path(raw.get("root"), default=quantum_root().parent / "data" / "quantum")
    events_dir_raw = raw.get("events_dir")
    events_dir = resolve_path(events_dir_raw) if events_dir_raw else None
    search_roots = tuple(resolve_path(path) for path in raw.get("events_search_roots", []))
    patterns = tuple(str(pattern) for pattern in raw.get("events_patterns", ("*.json", "*.csv")))
    reports_dir = resolve_path(raw.get("reports_dir"), default=root / "reports")
    downloader_log_raw = raw.get("downloader_log")
    downloader_log = resolve_path(downloader_log_raw) if downloader_log_raw else None
    return DataConfig(
        root=root,
        events_dir=events_dir,
        events_search_roots=search_roots,
        events_patterns=patterns,
        reports_dir=reports_dir,
        downloader_log=downloader_log,
    )


def _build_alpha_config(raw: Mapping[str, Any]) -> AlphaConfig:
    def _tuple_floats(values: Any, length: int) -> Tuple[float, float]:
        if not isinstance(values, (list, tuple)) or len(values) != length:
            raise ValueError(f"Expected a sequence of length {length}")
        return tuple(float(v) for v in values)  # type: ignore[return-value]

    alpha_band = _tuple_floats(raw.get("alpha_band"), 2)
    mixing_range = _tuple_floats(raw.get("mixing_range"), 2)
    regulators_raw = raw.get("regulators") or {}
    field_sets_raw = raw.get("field_sets") or {}
    if not regulators_raw or not field_sets_raw:
        raise ValueError("Alpha configuration requires both regulators and field_sets")
    regulators = {str(k): float(v) for k, v in regulators_raw.items()}
    field_sets = {str(k): float(v) for k, v in field_sets_raw.items()}
    target_reg = str(raw.get("target_regulator") or next(iter(regulators)))
    target_field = str(raw.get("target_field_set") or next(iter(field_sets)))
    return AlphaConfig(
        alpha_band=alpha_band,
        mixing_range=mixing_range,
        mixing_samples=int(raw.get("mixing_samples", 1)),
        regulators=regulators,
        field_sets=field_sets,
        enforce_reproducibility=bool(raw.get("enforce_reproducibility", True)),
        reference_field=raw.get("reference_field"),
        target_regulator=target_reg,
        target_field_set=target_field,
        warnings_threshold=float(raw.get("warnings_threshold", 0.05)),
    )


def load_config(user_config: str | Path | None = None) -> QuantumEngineConfig:
    base = _read_config_file(DEFAULT_CONFIG_PATH)
    merged = base
    if user_config:
        user_path = resolve_path(user_config)
        overrides = _read_config_file(user_path)
        merged = _deep_merge(base, overrides)
    data_cfg = _build_data_config(merged.get("data", {}))
    alpha_cfg = _build_alpha_config(merged.get("alpha", {}))
    e0_cfg = merged.get("e0", {})
    return QuantumEngineConfig(
        source=str(merged.get("source", "quantum_engine_v11")),
        data=data_cfg,
        time_window=float(e0_cfg.get("time_window", 1000.0)),
        eps_min=float(e0_cfg.get("eps_min", 0.8)),
        eps_max=float(e0_cfg.get("eps_max", 1.2)),
        steps=int(e0_cfg.get("steps", 500000)),
        k_eps=float(e0_cfg.get("k_eps", 1.0)),
        threads=int(e0_cfg.get("threads", 1)),
        alpha=alpha_cfg,
    )


__all__ = ["QuantumEngineConfig", "AlphaConfig", "DataConfig", "load_config", "DEFAULT_CONFIG_PATH"]
