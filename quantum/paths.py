"""Path helpers for the Quantum + E₀ subsystem."""

from __future__ import annotations

from pathlib import Path


def quantum_root() -> Path:
    """Return the root directory of the quantum engine package."""
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Return the repository root (one level above the quantum package)."""
    return quantum_root().parent


def resolve_path(value: str | Path | None, *, default: Path | None = None) -> Path:
    """Resolve user/config supplied paths relative to the repo root."""
    base = repo_root()
    if value is None:
        if default is None:
            raise ValueError("No path provided and no default available")
        value = default
    path = Path(value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["quantum_root", "repo_root", "resolve_path", "ensure_dir"]
