"""Data discovery utilities for the integrated quantum engine."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .config import QuantumEngineConfig


def _contains_event_files(directory: Path, patterns: Sequence[str]) -> bool:
    if not directory.is_dir():
        return False
    for pattern in patterns:
        if any(directory.glob(pattern)):
            return True
    return False


def _looks_like_event_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return True
    name = path.stem.lower()
    return any(token in name for token in ("event", "gw", "rigidity", "quantum"))


def _candidate_files(directory: Path, patterns: Sequence[str]) -> List[Path]:
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(sorted(directory.glob(pattern)))
    matches = [path for path in matches if path.is_file() and _looks_like_event_file(path)]
    return matches


def discover_event_source(config: QuantumEngineConfig) -> Path:
    """Locate the event directory or file specified by the configuration."""
    patterns = config.data.events_patterns
    explicit = config.data.events_dir
    if explicit:
        if explicit.is_file() and _looks_like_event_file(explicit):
            return explicit
        if explicit.is_dir() and _contains_event_files(explicit, patterns):
            return explicit
    for root in config.data.events_search_roots:
        if root.is_file() and _looks_like_event_file(root):
            return root
        if root.is_dir():
            potential_subdir = root / "events"
            if _contains_event_files(potential_subdir, patterns):
                return potential_subdir
            targeted = _candidate_files(root, patterns)
            if targeted:
                return targeted[0]
            if _contains_event_files(root, patterns):
                return root
    checked = []
    if explicit:
        checked.append(str(explicit))
    checked.extend(str(path) for path in config.data.events_search_roots)
    raise FileNotFoundError("Unable to locate event files. Checked: " + ", ".join(checked))


def list_event_files(path: Path, patterns: Sequence[str]) -> List[Path]:
    """Return concrete event files from a directory or a single file path."""
    if path.is_file():
        return [path]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(path.glob(pattern)))
    return files


__all__ = ["discover_event_source", "list_event_files"]
