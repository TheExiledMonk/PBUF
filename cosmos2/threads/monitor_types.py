"""Registry for available monitor modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class MonitorMode:
    name: str
    category: str
    description: str | None = None


_MODE_REGISTRY: dict[str, MonitorMode] = {}
_ALIAS_MAP: dict[str, str] = {}


def register_monitor_mode(
    name: str,
    category: str,
    *,
    aliases: Iterable[str] | None = None,
    description: str | None = None,
) -> None:
    """Register a monitor mode with optional aliases."""
    canonical = name.strip().lower()
    if not canonical:
        raise ValueError("Monitor mode name must not be empty.")
    if canonical in _MODE_REGISTRY:
        raise ValueError(f"Monitor mode already registered: {canonical}")
    mode = MonitorMode(canonical, category, description=description)
    _MODE_REGISTRY[canonical] = mode
    _ALIAS_MAP[canonical] = canonical
    for alias in aliases or []:
        key = alias.strip().lower()
        if not key:
            continue
        _ALIAS_MAP[key] = canonical


def normalize_monitor_mode(value: str | bool | None) -> Optional[str]:
    """Normalize a monitor value to its canonical name."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "ansi" if value else None
    key = str(value).strip().lower()
    if not key:
        return None
    canonical = _ALIAS_MAP.get(key)
    if canonical is None:
        raise ValueError(f"Unknown monitor mode: {value!r}")
    return canonical


def get_monitor_mode(name: str | None) -> Optional[MonitorMode]:
    """Get the MonitorMode for the canonical name."""
    if name is None:
        return None
    return _MODE_REGISTRY.get(name)


def available_monitor_modes() -> List[str]:
    """Return the sorted list of registered monitor names."""
    return sorted(_MODE_REGISTRY.keys())


# Register default modes
register_monitor_mode("ansi", "ansi", aliases=["simple", "legacy", "basic"], description="Legacy ANSI monitor")
register_monitor_mode(
    "plugin",
    "plugin",
    aliases=["dashboard", "enhanced", "fancy"],
    description="Plugin-based console dashboard",
)
register_monitor_mode(
    "textual",
    "textual",
    aliases=["rich"],
    description="Textual/LIVE monitor with plugin panels",
)
