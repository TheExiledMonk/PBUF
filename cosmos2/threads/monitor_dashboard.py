"""Plugin-driven console monitor for Cosmos2."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Sequence

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cosmos2.threads.monitor_state import MonitorState


class MonitorPlugin(ABC):
    """Abstract base for monitor plugins."""

    def __init__(
        self,
        *,
        name: str | None = None,
        priority: int | None = None,
        preferred_refresh_rate: float = 1.0,
    ) -> None:
        self.name = name or self.__class__.__name__
        if priority is not None:
            self.priority = priority
        else:
            self.priority = getattr(self, "priority", 100)
        self.preferred_refresh_rate = max(0.1, preferred_refresh_rate)

    @abstractmethod
    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        """Render text lines for the plugin."""

    def render_rich(self, snapshot: Dict[str, Any]) -> Sequence[Panel]:
        """Render rich panels for a Textual-style monitor (default wraps lines)."""
        lines = list(self.render_lines(snapshot))
        if not lines:
            return []
        content = Text("\n".join(lines))
        return [Panel(content, title=self.name)]


def _progress_bar(fraction: float, width: int = 32) -> str:
    """Draw a simple ASCII progress bar."""
    clamped = max(0.0, min(1.0, fraction))
    filled = int(clamped * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _sparkline(values: Sequence[float], width: int = 24) -> str:
    """Build a compact sparkline for the supplied values."""
    chars = "▁▂▃▄▅▆▇█"
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    if not filtered:
        return "-" * width
    mini, maxi = min(filtered), max(filtered)
    span = maxi - mini if maxi != mini else 1.0
    spark_chars = []
    for val in filtered[-width:]:
        index = int((val - mini) / span * (len(chars) - 1))
        clamped = max(0, min(len(chars) - 1, index))
        spark_chars.append(chars[clamped])
    return "".join(spark_chars).ljust(width, chars[0])


def _format_eta(seconds: float) -> str:
    """Format seconds into h/m/s text."""
    seconds = max(0, int(seconds))
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs:
        return f"{hrs:d}h {mins:02d}m"
    if mins:
        return f"{mins:d}m {secs:02d}s"
    return f"{secs:d}s"


class SystemSummaryPlugin(MonitorPlugin):
    """Render header including runtime and top-line system metrics."""

    priority = 10

    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        model = snapshot.get("current_model") or "None"
        dataset = snapshot.get("current_dataset") or "None"
        run_time = snapshot.get("run_time", 0.0)
        cpu = snapshot.get("cpu", 0.0)
        gpu = snapshot.get("gpu", 0.0)
        ram = snapshot.get("ram", 0.0)
        return [
            "Cosmos2 Monitoring",
            f"  Model: {model}   Dataset: {dataset}   Runtime: {run_time:.1f}s",
            f"  CPU: {cpu:.1f}%   GPU: {gpu:.1f}%   RAM: {ram:.1f} GB",
        ]


class ProgressPlugin(MonitorPlugin):
    """Show overall optimisation progress and fitter state."""

    priority = 20

    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        progress = float(snapshot.get("progress", 0.0))
        cand_idx = snapshot.get("candidate_index", 0)
        total_cands = snapshot.get("total_candidates") or 0
        current_chi2 = snapshot.get("current_chi2") or float("nan")
        best_chi2 = snapshot.get("best_chi2") or float("nan")
        progress_line = f"Progress: {progress * 100:5.1f}% {_progress_bar(progress)}"
        stats_line = (
            f"  Candidates: {cand_idx}/{total_cands}   χ²: {current_chi2:.3g}   Best: {best_chi2:.3g}"
        )
        return [progress_line, stats_line]


class EtaPlugin(MonitorPlugin):
    """Estimate remaining runtime based on progress."""

    priority = 25
    preferred_refresh_rate = 2.0

    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        progress = snapshot.get("progress", 0.0)
        run_time = snapshot.get("run_time", 0.0)
        history = snapshot.get("history", {})
        chi2_history = history.get("chi2", [])
        lines: List[str] = []
        if progress and run_time:
            eta = run_time / max(progress, 1e-6) * max(0.0, 1.0 - progress)
            lines.append(f"ETA ≈ {_format_eta(eta)} ({progress * 100:5.1f}% complete)")
        if len(chi2_history) > 1:
            delta = chi2_history[-2] - chi2_history[-1]
            lines.append(f"χ² change: {delta:+.3f} (latest slope)")
        return lines

    def render_rich(self, snapshot: Dict[str, Any]) -> Sequence[Panel]:
        lines = list(self.render_lines(snapshot))
        if not lines:
            return []
        content = Text("\n".join(lines), justify="left")
        return [Panel(content, title="ETA & χ² slope", border_style="cyan")]


class HistoryPlugin(MonitorPlugin):
    """Render lightweight sparkline summaries for tracked history."""

    priority = 30

    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        history = snapshot.get("history", {})
        chi2 = history.get("chi2", [])
        gpu = history.get("gpu", [])
        cpu = history.get("cpu", [])
        lines: List[str] = []
        if chi2:
            lines.append(f"  χ² trend: {_sparkline(chi2)}")
        if gpu:
            lines.append(f"  GPU%   trend: {_sparkline(gpu)}")
        if cpu:
            lines.append(f"  CPU%   trend: {_sparkline(cpu)}")
        return ["History"] + lines if lines else []


class ProcessPlugin(MonitorPlugin):
    """Render a compact view of tracked process timings."""

    priority = 40

    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        table = snapshot.get("process_table", {}) or {}
        if not table:
            return ["Processes: no tracked entries yet."]
        lines = ["Processes:"]
        for name, info in sorted(table.items())[:6]:
            status = "RUN" if info.get("running") else "IDLE"
            last_time = info.get("last_time_ms", 0.0)
            calls = info.get("call_count", 0)
            lines.append(
                f"  {name[:18]:18} {last_time:7.2f}ms  {status:4}  calls={calls}"
            )
        return lines


class JackknifePlugin(MonitorPlugin):
    """Render jackknife draw chi² snapshots when available."""

    priority = 45
    preferred_refresh_rate = 5.0

    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        draws = snapshot.get("jackknife_history", [])
        if not draws:
            return []
        lines = ["Jackknife draws:"]
        tail = draws[-5:]
        for draw in tail:
            label = draw.get("label", "draw")
            chi2 = draw.get("chi2", float("nan"))
            lines.append(f"  {label:14} χ²={chi2:.2f}")
        return lines

    def render_rich(self, snapshot: Dict[str, Any]) -> Sequence[Panel]:
        draws = snapshot.get("jackknife_history", [])
        if not draws:
            return []
        table = Table(title="Jackknife χ² trace", expand=True)
        table.add_column("Draw")
        table.add_column("χ²", justify="right")
        for draw in draws[-8:]:
            label = draw.get("label", "draw")
            chi2 = draw.get("chi2", float("nan"))
            table.add_row(label, f"{chi2:.2f}")
        return [Panel(table, border_style="magenta")]


class LogPlugin(MonitorPlugin):
    """Show the most recent log entries from the monitor state."""

    priority = 50

    def render_lines(self, snapshot: Dict[str, Any]) -> Sequence[str]:
        logs = snapshot.get("logs", [])[-5:]
        if not logs:
            return ["Logs: waiting for entries..."]
        lines = ["Logs:"]
        lines.extend(f"  {line}" for line in logs)
        return lines

    def render_rich(self, snapshot: Dict[str, Any]) -> Sequence[Panel]:
        lines = list(self.render_lines(snapshot))
        if not lines:
            return []
        return [Panel(Text("\n".join(lines)), title=self.name, border_style="green")]


def create_default_monitor_plugins() -> List[MonitorPlugin]:
    """Return the default ordered monitor plugins."""
    return [
        SystemSummaryPlugin(),
        ProgressPlugin(),
        EtaPlugin(),
        HistoryPlugin(),
        ProcessPlugin(),
        JackknifePlugin(),
        LogPlugin(),
    ]


class PluginBasedMonitor:
    """Threaded monitor that renders plugin output to the console."""

    def __init__(
        self,
        state: MonitorState,
        plugins: Iterable[MonitorPlugin],
        *,
        refresh_rate: float = 1.0,
    ) -> None:
        self.state = state
        self.plugins = sorted(list(plugins), key=lambda plugin: plugin.priority)
        self.refresh_rate = max(0.1, refresh_rate)
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_rendered: Dict[str, float] = {}
        self._cached_output: Dict[str, Sequence[str]] = {}

    @staticmethod
    def _clear_screen() -> None:
        print("\033[2J\033[H", end="", flush=True)

    def _render_snapshot(self) -> str:
        snapshot = self.state.get_snapshot()
        now = time.time()
        lines: List[str] = []
        for plugin in self.plugins:
            name = plugin.name
            last = self._last_rendered.get(name, 0.0)
            if now - last >= plugin.preferred_refresh_rate or name not in self._cached_output:
                try:
                    rendered = list(plugin.render_lines(snapshot))
                except Exception as exc:  # pragma: no cover - defensive
                    rendered = [f"[{plugin.name}] render failed: {exc}"]
                self._cached_output[name] = rendered
                self._last_rendered[name] = now
            else:
                rendered = list(self._cached_output.get(name, []))
            if not rendered:
                continue
            lines.extend(rendered)
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) if lines else ""

    def _run(self) -> None:
        self._running.set()
        while self._running.is_set():
            content = self._render_snapshot()
            self._clear_screen()
            if content:
                print(content, flush=True)
            time.sleep(self.refresh_rate)

    def start(self) -> None:
        """Start the monitor thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitor thread."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def running(self) -> bool:
        """Return whether the monitor is currently running."""
        return self._running.is_set()
