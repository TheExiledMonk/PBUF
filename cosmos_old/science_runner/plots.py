"""Simple plotting helpers for science run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - fallback tested indirectly
    plt = None


class SciencePlotter:
    def __init__(self) -> None:
        self._plt = plt

    def generate(self, *, predictions: dict[str, Any], model_dir: Path) -> None:
        if self._plt is None:
            return
        plot_data = predictions.get("plot_data") or {}
        z = np.asarray(plot_data.get("z", []), dtype=float)
        h = np.asarray(plot_data.get("H_z", []), dtype=float)
        dm = np.asarray(plot_data.get("DM_z", []), dtype=float)
        fs8 = np.asarray(plot_data.get("fs8_z", []), dtype=float)
        plots_dir = model_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        if z.size and h.size:
            self._line_plot(z, h, "H(z)", plots_dir / "H_z.png", "redshift", "H(z)")
        if z.size and dm.size:
            self._line_plot(z, dm, "D_M(z)", plots_dir / "DM_z.png", "redshift", "D_M(z)")
        if z.size and fs8.size:
            self._line_plot(z, fs8, "fσ₈(z)", plots_dir / "fs8_z.png", "redshift", "fσ₈(z)")
        self._scatter_point(
            predictions.get("Omega_m0"),
            predictions.get("S8"),
            plots_dir / "S8_vs_Om0.png",
        )

    def _line_plot(
        self,
        x: Sequence[float],
        y: Sequence[float],
        title: str,
        path: Path,
        xlabel: str,
        ylabel: str,
    ) -> None:
        fig, ax = self._plt.subplots()
        ax.plot(x, y, marker="", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(path)
        self._plt.close(fig)

    def _scatter_point(self, omega_m0: Any, s8: Any, path: Path) -> None:
        try:
            omega = float(omega_m0)
            s = float(s8)
        except (TypeError, ValueError):
            return
        fig, ax = self._plt.subplots()
        ax.scatter([omega], [s], color="tab:blue")
        ax.set_title("S₈ vs Ωₘ,₀")
        ax.set_xlabel("Ωₘ,₀")
        ax.set_ylabel("S₈")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(path)
        self._plt.close(fig)
