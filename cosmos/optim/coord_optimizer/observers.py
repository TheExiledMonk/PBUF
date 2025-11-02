from __future__ import annotations

import csv
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, Iterable, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .basin_walker import CoordinateBasinWalker


class BasinWalkObserver:
    """
    Observer hook for the coordinate basin walker lifecycle.
    """

    def on_attach(self, walker: "CoordinateBasinWalker") -> None:  # pragma: no cover - optional hook
        """
        Called when the observer is attached to a walker instance.
        """

    def on_detach(self, walker: "CoordinateBasinWalker") -> None:  # pragma: no cover - optional hook
        """
        Called when the observer is detached from a walker instance.
        """

    def on_run_started(self, walker: "CoordinateBasinWalker", context: Dict[str, Any]) -> None:
        """
        Invoked when a new run begins.
        """

    def on_scan_completed(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        """
        Invoked each time a scan (including edge rescans) completes.
        """

    def on_coupled_update(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        """
        Invoked when a coupled multi-parameter update concludes.
        """

    def on_plateau_reseed(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        """
        Invoked after a plateau reseed step completes.
        """

    def on_island_center(self, walker: "CoordinateBasinWalker", payload: Dict[str, Any]) -> None:
        """
        Invoked when the island center finder returns a payload.
        """

    def on_run_completed(self, walker: "CoordinateBasinWalker", result: Dict[str, Any]) -> None:
        """
        Invoked after the run payload has been fully assembled.
        """


class CompositeObserver(BasinWalkObserver):
    """
    Dispatches events to a collection of observers.
    """

    def __init__(self, observers: Iterable[BasinWalkObserver]):
        self._observers: List[BasinWalkObserver] = list(observers)

    def on_attach(self, walker: "CoordinateBasinWalker") -> None:
        for observer in self._observers:
            observer.on_attach(walker)

    def on_detach(self, walker: "CoordinateBasinWalker") -> None:
        for observer in self._observers:
            observer.on_detach(walker)

    def on_run_started(self, walker: "CoordinateBasinWalker", context: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_run_started(walker, context)

    def on_scan_completed(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_scan_completed(walker, summary)

    def on_coupled_update(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_coupled_update(walker, summary)

    def on_plateau_reseed(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_plateau_reseed(walker, summary)

    def on_island_center(self, walker: "CoordinateBasinWalker", payload: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_island_center(walker, payload)

    def on_run_completed(self, walker: "CoordinateBasinWalker", result: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_run_completed(walker, result)


@dataclass
class RecordingObserver(BasinWalkObserver):
    """
    Persist scan traces and derived artifacts for post-run analysis and plotting.
    """

    output_dir: Path | str
    include_curves: bool = True
    flatten_scans: bool = True
    write_csv: bool = True
    auto_run_subdir: bool = True
    filename: str = "basin_trace.json"

    _active_run_dir: Optional[Path] = field(init=False, default=None)
    last_run_dir: Optional[Path] = field(init=False, default=None)
    last_trace_path: Optional[Path] = field(init=False, default=None)
    _payload: Dict[str, Any] = field(init=False, default_factory=dict)
    _scan_rows: DefaultDict[str, List[Dict[str, Any]]] = field(init=False, default_factory=lambda: defaultdict(list))
    _scan_counter: int = field(init=False, default=0)

    def on_attach(self, walker: "CoordinateBasinWalker") -> None:
        base = Path(self.output_dir)
        if not base.exists():
            base.mkdir(parents=True, exist_ok=True)

    def _initialise_run(self, context: Dict[str, Any]) -> None:
        base = Path(self.output_dir)
        run_id = context.get("run_id") or uuid.uuid4().hex
        if self.auto_run_subdir:
            run_dir = base / run_id
        else:
            run_dir = base
        run_dir.mkdir(parents=True, exist_ok=True)
        self._active_run_dir = run_dir
        self.last_run_dir = run_dir
        self._payload = {
            "metadata": dict(context),
            "scans": [],
            "coupled_updates": [],
            "plateau_reseeds": [],
            "island_search": [],
        }
        self._scan_rows = defaultdict(list)
        self._scan_counter = 0

    def on_run_started(self, walker: "CoordinateBasinWalker", context: Dict[str, Any]) -> None:
        self._initialise_run(context)

    def on_scan_completed(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        if self._active_run_dir is None:
            return
        record = json.loads(json.dumps(summary))
        record["_scan_index"] = self._scan_counter
        self._scan_counter += 1
        if not self.include_curves:
            record.pop("curve", None)
        self._payload["scans"].append(record)
        if self.flatten_scans:
            param = summary.get("param", "unknown")
            pass_id = summary.get("pass")
            cycle = summary.get("cycle")
            edge_iteration = summary.get("edge_iteration")
            for point in summary.get("curve", []):
                row = {
                    "scan_index": record["_scan_index"],
                    "param": param,
                    "pass": pass_id,
                    "cycle": cycle,
                    "edge_iteration": edge_iteration,
                    "value": point.get("value"),
                    "chi2": point.get("chi2"),
                    "valid": point.get("valid"),
                    "passes_phase6a": point.get("passes_phase6a"),
                    "score": point.get("score"),
                }
                self._scan_rows[param].append(row)

    def on_coupled_update(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        if self._active_run_dir is None:
            return
        self._payload["coupled_updates"].append(json.loads(json.dumps(summary)))

    def on_plateau_reseed(self, walker: "CoordinateBasinWalker", summary: Dict[str, Any]) -> None:
        if self._active_run_dir is None:
            return
        self._payload["plateau_reseeds"].append(json.loads(json.dumps(summary)))

    def on_island_center(self, walker: "CoordinateBasinWalker", payload: Dict[str, Any]) -> None:
        if self._active_run_dir is None:
            return
        self._payload["island_search"].append(json.loads(json.dumps(payload)))

    def on_run_completed(self, walker: "CoordinateBasinWalker", result: Dict[str, Any]) -> None:
        if self._active_run_dir is None:
            return
        self._payload["result"] = json.loads(json.dumps(result))
        trace_path = self._active_run_dir / self.filename
        trace_path.write_text(json.dumps(self._payload, indent=2))
        self.last_trace_path = trace_path

        if self.write_csv and self._scan_rows:
            for param, rows in self._scan_rows.items():
                path = self._active_run_dir / f"{param.lower()}_scan_points.csv"
                fieldnames = [
                    "scan_index",
                    "param",
                    "pass",
                    "cycle",
                    "edge_iteration",
                    "value",
                    "chi2",
                    "valid",
                    "passes_phase6a",
                    "score",
                ]
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

        self._payload = {}
        self._scan_rows = defaultdict(list)
        self._active_run_dir = None
        self._scan_counter = 0
