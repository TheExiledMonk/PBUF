"""Thin wrapper that exposes the multimessenger ingestion CLI as a callable target."""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

from quantum.pipeline import multimessenger_ingest


def run_ingest(*, args: Sequence[str] | None = None) -> None:
    """
    Execute the ingestion pipeline from within Python.

    Parameters
    ----------
    args : optional sequence of str
        Additional CLI-style flags passed to the importer (e.g., ["--max-gcn", "100"]).
    """
    original_argv = sys.argv
    try:
        sys.argv = ["multimessenger_ingest.py", *(args or [])]
        multimessenger_ingest.main()
    finally:
        sys.argv = original_argv
