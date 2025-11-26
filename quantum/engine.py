"""Public entry point for the Quantum + E₀ subsystem."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .alpha_runner import run_alpha_pipeline
from .config import QuantumEngineConfig, load_config
from .data_access import discover_event_source, list_event_files
from .e0_runner import run_e0_pipeline


def _serialize_paths(paths: Dict[str, Optional[Path]]) -> Dict[str, Optional[str]]:
    return {key: (str(value) if value else None) for key, value in paths.items()}


def run_quantum_engine(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Execute the integrated pipeline and return a structured quantum state record.
    """
    config = load_config(config_path)
    start_time = datetime.now(timezone.utc)

    event_source = discover_event_source(config)
    if event_source.is_file():
        event_files = [event_source]
    else:
        event_files = list_event_files(event_source, config.data.events_patterns)

    e0_result = run_e0_pipeline(config, event_source)
    alpha_result = run_alpha_pipeline(e0_result.eps0, config.alpha)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    warnings = list(e0_result.warnings) + list(alpha_result.warnings)

    metadata_paths = _serialize_paths(
        {
            "data_root": config.data.root,
            "events": event_source,
            "reports_dir": config.data.reports_dir,
            "downloader_log": config.data.downloader_log,
        }
    )
    metadata_paths["event_files"] = [str(path) for path in event_files]

    run_metadata = {
        "config_used": config.as_dict(),
        "stats": {
            "runtime_seconds": duration,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "time_window": config.time_window,
            "k_eps": config.k_eps,
            "threads": config.threads,
            "event_source_type": "file" if event_source.is_file() else "directory",
            "events": e0_result.stats(),
            "alpha": alpha_result.metadata,
        },
        "paths": metadata_paths,
        "warnings": warnings,
    }

    quantum_state = {
        "eps0": e0_result.eps0,
        "eps0_error": e0_result.eps0_error,
        "alpha_QM": alpha_result.alpha_value,
        "alpha_error": alpha_result.alpha_error,
        "derived_parameters": alpha_result.derived_parameters,
        "run_metadata": run_metadata,
        "source": config.source,
    }
    return quantum_state


__all__ = ["run_quantum_engine"]
