"""Run the Quantum-specific download_data.py helper from inside the toolbox."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "data" / "download_data.py"


def run_quantum_downloader(*, args: Sequence[str] | None = None) -> None:
    """Execute the download_data.py script with optional arguments."""
    original_argv = sys.argv
    try:
        sys.argv = [str(SCRIPT_PATH), *(args or [])]
        runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        sys.argv = original_argv
