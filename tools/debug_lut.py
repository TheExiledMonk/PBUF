#!/usr/bin/env python
"""
Debug a thermal LUT JSON file for PBUF/COSMOS2.

Features:
- Loads the LUT JSON directly.
- Prints full arrays for a, T, epsilon0_T, alpha_T, g_star, g_starS.
- Reports min/max, NaN/inf counts, and monotonicity.
- Checks that array lengths match and that a-grid spacing is roughly uniform in log-space.
- Optional plotting if matplotlib is available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


def _load_lut(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    # LUT may be stored directly or under a subkey like "lut".
    if all(k in data for k in ("a", "T")):
        return data
    if "rows" in data and isinstance(data["rows"], list):
        rows = data["rows"]
        # Accept either Kelvin T under T_K or generic T
        lut = {
            "a": [row.get("a") for row in rows],
            "T": [row.get("T_K", row.get("T")) for row in rows],
            "epsilon0_T": [row.get("epsilon0_T") for row in rows],
            "alpha_T": [row.get("alpha_T") for row in rows],
            "g_star": [row.get("g_star") for row in rows],
            "g_starS": [row.get("g_starS") for row in rows],
        }
        return lut
    for key in ("lut", "table", "payload"):
        if key in data and isinstance(data[key], dict) and "a" in data[key]:
            return data[key]
    raise ValueError("Could not find LUT fields (a, T, etc.) in file")


def _as_array(d: Dict[str, Any], key: str) -> np.ndarray:
    if key not in d:
        return np.array([], dtype=float)
    return np.asarray(d[key], dtype=float)


def _monotonic(arr: np.ndarray) -> Tuple[bool, str]:
    if arr.size < 2:
        return True, "size<2"
    diff = np.diff(arr)
    if np.all(diff > 0):
        return True, "strictly increasing"
    if np.all(diff < 0):
        return True, "strictly decreasing"
    return False, "non-monotonic"


def _stats(name: str, arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    n_nan = np.count_nonzero(~finite)
    txt = (
        f"{name:12s} | n={arr.size:5d} "
        f"min={np.nanmin(arr): .6e} max={np.nanmax(arr): .6e} "
        f"nan/inf={n_nan:3d}"
    )
    mono_ok, mono_desc = _monotonic(arr[finite])
    txt += f" | monotonic: {mono_desc}"
    return txt


def _check_alignment(a: np.ndarray) -> str:
    if a.size < 3:
        return "a-grid too small to assess spacing"
    loga = np.log10(a)
    dlog = np.diff(loga)
    spread = np.max(dlog) - np.min(dlog)
    return f"a-grid log-spacing spread={spread:.3e} (ideal ~0)"


def _print_array(name: str, arr: np.ndarray) -> None:
    print(f"\n{name}:")
    print(arr)


def _maybe_plot(args, lut: Dict[str, np.ndarray]) -> None:
    if not args.plot:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not available; skipping plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    a = lut["a"]
    ax[0].plot(a, lut["T"], label="T(a)")
    ax[0].set_ylabel("T")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend()

    if lut["alpha_T"].size:
        ax[1].plot(lut["T"], lut["alpha_T"], label="alpha(T)")
    if lut["epsilon0_T"].size:
        ax[1].plot(lut["T"], lut["epsilon0_T"], label="epsilon0(T)")
    ax[1].set_xlabel("T")
    ax[1].set_ylabel("elastic params")
    ax[1].set_xscale("log")
    ax[1].legend()

    if lut["g_star"].size:
        ax[2].plot(lut["T"], lut["g_star"], label="g_star(T)")
    if lut["g_starS"].size:
        ax[2].plot(lut["T"], lut["g_starS"], label="g_starS(T)")
    ax[2].set_xlabel("T")
    ax[2].set_ylabel("g_star")
    ax[2].set_xscale("log")
    ax[2].legend()

    fig.tight_layout()
    plt.show()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a thermal LUT JSON.")
    parser.add_argument("path", type=Path, help="Path to thermal_table_cache.json (or similar)")
    parser.add_argument("--plot", action="store_true", help="Show quick diagnostic plots (requires matplotlib)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    lut_raw = _load_lut(args.path)
    fields = {k: _as_array(lut_raw, k) for k in ("a", "T", "epsilon0_T", "alpha_T", "g_star", "g_starS")}

    # Basic checks
    print("=== LUT stats ===")
    for k, arr in fields.items():
        if arr.size == 0:
            print(f"{k:12s} | MISSING")
        else:
            print(_stats(k, arr))
    lengths = {k: arr.size for k, arr in fields.items()}
    print("lengths:", lengths)
    print(_check_alignment(fields["a"]))

    # Print full arrays
    _print_array("a", fields["a"])
    _print_array("T", fields["T"])
    _print_array("alpha_T", fields["alpha_T"])
    _print_array("epsilon0_T", fields["epsilon0_T"])
    _print_array("g_star", fields["g_star"])
    _print_array("g_starS", fields["g_starS"])

    _maybe_plot(args, fields)
    return 0


if __name__ == "__main__":
    sys.exit(main())
