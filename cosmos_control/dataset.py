"""Simple dataset registry and hash-based sync helper."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple


class DatasetManager:
    """Manage canonical dataset hashes and payloads."""

    def __init__(self, search_paths: Sequence[Path] | None = None):
        self.search_paths = list(search_paths or [Path("data_interface/datasets"), Path("data")])
        self._hash_cache: Dict[str, Tuple[str, float]] = {}

    def list_datasets(self) -> list[str]:
        ids: list[str] = []
        for path in self.search_paths:
            if not path.exists():
                continue
            for entry in path.iterdir():
                if entry.is_dir():
                    ids.append(entry.name)
                else:
                    ids.append(entry.stem)
        return sorted(set(ids))

    def get_hash(self, dataset_id: str) -> Optional[str]:
        path = self._find_dataset_path(dataset_id)
        if not path or not path.exists():
            return None

        stat = path.stat()
        cached = self._hash_cache.get(dataset_id)
        if cached and cached[1] == stat.st_mtime:
            return cached[0]

        digest = self._hash_file(path)
        self._hash_cache[dataset_id] = (digest, stat.st_mtime)
        return digest

    def get_payload(self, dataset_id: str) -> Optional[bytes]:
        path = self._find_dataset_path(dataset_id)
        if not path or not path.exists():
            return None
        return path.read_bytes()

    def needs_update(self, dataset_id: str, worker_hash: str | None) -> bool:
        canonical = self.get_hash(dataset_id)
        return canonical is not None and canonical != worker_hash

    def encode_payload(self, dataset_id: str) -> Optional[str]:
        payload = self.get_payload(dataset_id)
        if payload is None:
            return None
        return base64.b64encode(payload).decode("ascii")

    def _find_dataset_path(self, dataset_id: str) -> Optional[Path]:
        for base in self.search_paths:
            candidate = base / dataset_id
            if candidate.exists():
                return candidate
            alt = base / f"{dataset_id}.json"
            if alt.exists():
                return alt
        return None

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
