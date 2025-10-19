"""
I/O helpers for JSON and YAML with atomic writes.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as tmp_file:
        tmp_file.write(data)
        tmp_name = tmp_file.name
    os.replace(tmp_name, path)


def write_json_atomic(path: str | Path, payload: Any, indent: int = 2) -> None:
    data = json.dumps(payload, indent=indent, sort_keys=False)
    _atomic_write(Path(path), data)


def write_yaml_atomic(path: str | Path, payload: Any) -> None:
    data = yaml.safe_dump(payload, sort_keys=False)
    _atomic_write(Path(path), data)

# ---------------------------------------------------------------------
#  Result file helpers
# ---------------------------------------------------------------------
import glob, json
from typing import Optional

def read_latest_result(model: str, kind: str, results_dir: str = "proofs/results") -> dict:
    """
    Load the most recent calibration JSON result for a given model and type.

    Parameters
    ----------
    model : str
        Model name (e.g. 'PBUF', 'LCDM')
    kind : str
        Result kind (e.g. 'CMB', 'BAO', 'SN')
    results_dir : str
        Root directory to search in (default: 'proofs/results').

    Returns
    -------
    dict
        Parsed JSON content, or {} if not found.
    """
    pattern = f"{results_dir}/{kind}_{model}_*/fit_results.json"
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not matches:
        return {}
    latest_path = matches[0]
    try:
        with open(latest_path, "r") as f:
            data = json.load(f)
        data["_source_path"] = latest_path
        return data
    except Exception as e:
        print(f"[WARN] Failed to load latest result from {latest_path}: {e}")
        return {}
