"""Expose the compact_quantum_data CLI under a toolbox namespace."""

from __future__ import annotations

import sys
from typing import Sequence

from tools import compact_quantum_data


def run_compact(*, args: Sequence[str] | None = None) -> None:
    """
    Run the compacting script inside the toolbox context.

    Parameters
    ----------
    args : optional sequence of str
        Additional command-line flags forwarded to the script.
    """
    original_argv = sys.argv
    try:
        sys.argv = ["compact_quantum_data.py", *(args or [])]
        compact_quantum_data.main()
    finally:
        sys.argv = original_argv
