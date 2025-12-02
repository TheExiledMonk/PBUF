"""Textual-style monitor that renders plugin panels via Rich Live."""

from __future__ import annotations

import threading
import time
from typing import Iterable

from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from cosmos2.threads.monitor_dashboard import MonitorPlugin
from cosmos2.threads.monitor_state import MonitorState


class TextualMonitor:
    """Threaded monitor that renders plugin panels inside a Rich Live layout."""

    def __init__(
        self,
        state: MonitorState,
        plugins: Iterable[MonitorPlugin],
        *,
        refresh_rate: float = 0.2,
    ) -> None:
        self.state = state
        self.plugins = sorted(list(plugins), key=lambda plugin: plugin.priority)
        self.refresh_rate = max(0.1, refresh_rate)
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    def _build_layout(self) -> Layout:
        snapshot = self.state.get_snapshot()
        panels = []
        for plugin in self.plugins:
            try:
                panels.extend(plugin.render_rich(snapshot))
            except Exception as exc:
                panels.append(Panel(Text(f"{plugin.name} failed: {exc}"), title=plugin.name, border_style="red"))

        if not panels:
            panels = [Panel(Text("Waiting for data...", justify="center"), title="Monitor")]

        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        layout["header"].update(Panel(Text("Cosmos2 Textual Monitor", justify="center", style="bold white"), border_style="bright_blue"))
        layout["body"].update(Columns(panels, expand=True))
        return layout

    def _run(self) -> None:
        console = Console()
        refresh_per_second = max(0.1, 1.0 / self.refresh_rate)
        self._running.set()
        try:
            with Live(console=console, refresh_per_second=refresh_per_second, auto_refresh=True) as live:
                while self._running.is_set():
                    live.update(self._build_layout())
                    time.sleep(self.refresh_rate)
        finally:
            self._running.clear()

    def start(self) -> None:
        """Start the textual monitor thread."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the textual monitor."""
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def running(self) -> bool:
        """Return whether the monitor is running."""
        return self._running.is_set()
