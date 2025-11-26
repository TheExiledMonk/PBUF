#!/usr/bin/env python
"""
Quick thermal table plotter for cosmos2 PBUF LUTs.

Usage:
    python tools/plot_thermal_table.py configs/quantum/thermal_table_cache.json --out plot.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("rows") or payload.get("data") or payload.get("table")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No row data found in {path}")
    return rows


def _extract(rows: list[dict]) -> dict[str, np.ndarray]:
    def arr(key: str) -> np.ndarray:
        return np.array([float(row[key]) for row in rows], dtype=float)

    return {
        "T": arr("T_K"),
        "a": arr("a"),
        "eps": arr("epsilon0_T"),
        "alpha": arr("alpha_T"),
        "g_star": arr("g_star"),
        "g_starS": arr("g_starS"),
    }


def _make_plot(data: dict[str, np.ndarray], out: Path) -> None:
    T = data["T"]
    a = data["a"]
    eps = data["eps"]
    alpha = data["alpha"]
    g_star = data["g_star"]
    g_starS = data["g_starS"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")

    ax = axes[0, 0]
    ax.loglog(T, eps, label="epsilon0_T")
    ax.set_ylabel("epsilon0_T")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.loglog(T, alpha, label="alpha_T", color="tab:orange")
    ax.set_xlabel("T [K]")
    ax.set_ylabel("alpha_T")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.semilogx(T, g_star, label="g_star")
    ax.semilogx(T, g_starS, label="g_starS", ls="--")
    ax.set_ylabel("g_star / g_starS")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.loglog(a, eps, label="epsilon0_T(a)")
    ax.loglog(a, alpha, label="alpha_T(a)")
    ax.set_xlabel("scale factor a")
    ax.set_ylabel("elasticity vs a")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot thermal table curves.")
    parser.add_argument("json_path", type=Path, help="Path to thermal_table_cache.json")
    parser.add_argument("--out", type=Path, default=Path("thermal_table_plot.png"), help="Output image path")
    args = parser.parse_args()

    rows = _load_rows(args.json_path)
    data = _extract(rows)
    _make_plot(data, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
