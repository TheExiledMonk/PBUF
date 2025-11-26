"""Microphysics bootstrap helpers for the cosmos2 PBUF model."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from quantum.api import run_quantum_engine
from quantum.thermal import ThermalModelConfig, ThermalTableSpec, generate_thermal_table, save_table

# Use the per-model ThermalTable (interpolation helpers, clamp, metadata summary).
_BASE_DIR = Path("configs/quantum")
_BASE_DIR.mkdir(parents=True, exist_ok=True)
MICRO_CACHE_PATH = _BASE_DIR / "micro_cache.json"
THERMAL_CACHE_PATH = _BASE_DIR / "thermal_table_cache.json"

_LAST_BOOTSTRAP_METADATA: Dict[str, Any] | None = None
_THERMAL_TABLE_CACHE: object | None = None


def run_microphysics_bootstrap(datasets: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute the Quantum engine once and persist the derived thermal table."""

    micro = run_quantum_engine()
    metadata = _install_microphysics(micro, datasets or [])
    return metadata


def ensure_thermal_table() -> ThermalTable:
    """Return a cached ThermalTable instance, loading it (or regenerating it) as needed."""

    from cosmos2.models.pbuf.thermal_table import ThermalTable

    global _THERMAL_TABLE_CACHE
    if _THERMAL_TABLE_CACHE is not None:
        return _THERMAL_TABLE_CACHE

    table_path = _ensure_thermal_table_path()
    _THERMAL_TABLE_CACHE = ThermalTable(table_path)
    _maybe_inject_quantum_metadata(_THERMAL_TABLE_CACHE)
    return _THERMAL_TABLE_CACHE


def _ensure_thermal_table_path() -> Path:
    if THERMAL_CACHE_PATH.exists():
        return THERMAL_CACHE_PATH

    cached = _load_micro_cache()
    if cached is None:
        metadata = run_microphysics_bootstrap()
        return Path(metadata["thermal_table_path"])

    _generate_table_file(cached, _hash_microphysics(cached))
    return THERMAL_CACHE_PATH


def _reset_cached_table() -> None:
    global _THERMAL_TABLE_CACHE
    _THERMAL_TABLE_CACHE = None


def get_last_bootstrap_metadata() -> Dict[str, Any] | None:
    """Return metadata recorded during the most recent bootstrap run."""

    return _LAST_BOOTSTRAP_METADATA


def get_cached_microphysics() -> Dict[str, Any] | None:
    """Return the cached microphysics payload if available on disk."""

    return _load_micro_cache()


def _install_microphysics(micro: Dict[str, Any], datasets: List[str]) -> Dict[str, Any]:
    _cache_microphysics(micro)
    micro_hash = _hash_microphysics(micro)
    _ensure_internal_table(micro, micro_hash)
    metadata = {
        "micro_hash": micro_hash,
        "thermal_table_path": str(THERMAL_CACHE_PATH),
        "alpha_qm": float(micro.get("alpha_qm", micro.get("alpha_QM", 0.0))),
        "eps0_base": float(micro.get("eps0_base", micro.get("eps0", 0.0))),
        "beta": float(micro.get("beta", 0.0)),
        "datasets": list(datasets),
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_source": micro.get("engine_source") or micro.get("engine_override_path"),
    }
    global _LAST_BOOTSTRAP_METADATA
    _LAST_BOOTSTRAP_METADATA = metadata
    return metadata


def _cache_microphysics(payload: Dict[str, Any]) -> None:
    MICRO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MICRO_CACHE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_micro_cache() -> Optional[Dict[str, Any]]:
    if MICRO_CACHE_PATH.exists():
        return json.loads(MICRO_CACHE_PATH.read_text())
    return None


def _hash_microphysics(micro: Dict[str, Any]) -> str:
    serialized = json.dumps(micro, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _ensure_internal_table(micro: Dict[str, Any], micro_hash: str) -> Path:
    if THERMAL_CACHE_PATH.exists():
        try:
            payload = json.loads(THERMAL_CACHE_PATH.read_text())
            metadata = payload.get("metadata", {})
        except Exception:
            metadata = {}
        if metadata.get("micro_hash") == micro_hash:
            return THERMAL_CACHE_PATH

    _generate_table_file(micro, micro_hash)
    return THERMAL_CACHE_PATH


def _generate_table_file(micro: Dict[str, Any], micro_hash: str) -> None:
    config = ThermalModelConfig(
        mode=str(micro.get("thermal_mode", "linear")),
        beta=float(micro.get("beta", 0.0)),
        t_star=float(micro.get("T_star", 1.0e6)),
        power=float(micro.get("power_index", 1.0)),
        alpha_qm=float(micro.get("alpha_qm", micro.get("alpha_QM", 0.02))),
        eps_min=float(micro.get("eps_min", 1.0e-4)),
    )

    num_points = int(micro.get("temperature_points") or micro.get("iterations") or 256)
    dense_points = int(micro.get("dense_points", 24))
    t_min = float(micro.get("t_min", 2.7255))
    t_max = float(micro.get("t_max", 1.0e12))

    spec = ThermalTableSpec(
        model=config,
        t_min=t_min,
        t_max=t_max,
        num_points=max(num_points, 32),
        dense_points=max(dense_points, 0),
        table_version=int(micro.get("table_version", 1)),
        method_version=int(micro.get("method_version", 1)),
        regulator=str(micro.get("regulator", "thermal_default")),
        field_content=str(micro.get("field_content", "SM_full")),
        f_cut_T=float(micro.get("f_cut", 1.0)),
        f_coup_T=float(micro.get("f_coup", 1.0)),
        notes="auto-generated via cosmos2 PBUF bootstrap",
    )

    table = generate_thermal_table(spec)
    table.metadata["micro_hash"] = micro_hash
    table.metadata["micro_source"] = micro.get("engine_source")
    table.metadata["quantum_run_metadata"] = micro.get("run_metadata")
    # Persist key quantum parameters so downstream LUT consumers can derive elasticity.
    table.metadata.setdefault("alpha_qm", micro.get("alpha_qm", micro.get("alpha_QM")))
    table.metadata.setdefault("eps0_base", micro.get("eps0_base", micro.get("eps0")))
    table.metadata.setdefault("beta", micro.get("beta"))
    THERMAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_table(table, THERMAL_CACHE_PATH)
    _reset_cached_table()


def _maybe_inject_quantum_metadata(table: ThermalTable) -> None:
    """
    Ensure required quantum metadata is present on the loaded table by seeding from cache.
    """
    metadata = getattr(table, "metadata", None)
    if metadata is None:
        return
    # Flatten nested metadata blob if present.
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        for key, value in nested.items():
            metadata.setdefault(key, value)
    if metadata.get("alpha_qm") is not None and metadata.get("eps0_base") is not None:
        return
    cached = _load_micro_cache()
    if not cached:
        return
    if metadata.get("alpha_qm") is None and cached.get("alpha_qm") is not None:
        metadata["alpha_qm"] = cached.get("alpha_qm", cached.get("alpha_QM"))
    if metadata.get("eps0_base") is None and cached.get("eps0_base") is not None:
        metadata["eps0_base"] = cached.get("eps0_base", cached.get("eps0"))
    if metadata.get("beta") is None and cached.get("beta") is not None:
        metadata["beta"] = cached.get("beta")


__all__ = [
    "MICRO_CACHE_PATH",
    "THERMAL_CACHE_PATH",
    "ensure_thermal_table",
    "get_cached_microphysics",
    "get_last_bootstrap_metadata",
    "run_microphysics_bootstrap",
]
