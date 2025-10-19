"""
Lightweight logging utilities for CLI scripts.
"""

from __future__ import annotations

import datetime as _dt


def _prefix(level: str) -> str:
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    return f"[{level}] {timestamp}"


def info(message: str, *args) -> None:
    if args:
        try:
            message = message % args
        except TypeError:
            message = message.format(*args)
    print(f"{_prefix('INFO')} {message}")


def warn(message: str, *args) -> None:
    if args:
        try:
            message = message % args
        except TypeError:
            message = message.format(*args)
    print(f"{_prefix('WARN')} {message}")


def fit(model: str, chi2: float, dof: int) -> None:
    print(f"{_prefix('FIT ')} model={model} chi2={chi2:.3f} dof={dof}")
