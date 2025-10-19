"""
Dataset registry handling provenance metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    description: str
    data_path: str | None = None
    covariance_path: str | None = None
    metadata_path: str | None = None
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetManifest:
    path: Path
    payload: Mapping[str, object]

    @property
    def transform_version(self) -> str:
        return str(self.payload.get("transform_version", ""))

    @property
    def prepared_at(self) -> str:
        return str(self.payload.get("prepared_at", ""))

    def get(self, key: str, default=None):
        return self.payload.get(key, default)


def load_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetManifest(path=manifest_path, payload=payload)


def _flatten_datasets(config: Mapping[str, Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    flattened: Dict[str, Mapping[str, object]] = {}
    for key, entry in config["datasets"].items():
        if isinstance(entry, Mapping) and "prepared" in entry:
            flattened[key] = entry
            continue
        if isinstance(entry, Mapping):
            # nested structure (e.g., datasets -> supernovae -> pantheon_plus)
            for inner_key, inner_value in entry.items():
                if isinstance(inner_value, Mapping):
                    flattened[inner_key] = inner_value
        else:
            flattened[key] = entry
    return flattened


def build_registry(config: Mapping[str, Mapping[str, object]]) -> Dict[str, DatasetRecord]:
    """Construct dataset metadata records from the YAML configuration."""

    records: Dict[str, DatasetRecord] = {}
    flattened = _flatten_datasets(config)
    for key, entry in flattened.items():
        records[key] = DatasetRecord(
            name=entry.get("name", key),
            description=entry.get("description", ""),
            data_path=entry.get("data_path"),
            covariance_path=entry.get("covariance_path") or entry.get("cov"),
            metadata_path=entry.get("metadata_path") or entry.get("meta"),
            tags=tuple(entry.get("tags", [])),
        )
    return records
