"""Thread-safe shared state for Cosmos2 monitoring."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, List, Optional


@dataclass
class MonitorState:
    """Thread-safe shared state for Cosmos2 monitoring dashboard."""
    
    # System metrics
    cpu: float = 0.0
    gpu: float = 0.0
    ram: float = 0.0
    
    # Runtime state
    run_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    # Progress tracking
    progress: float = 0.0
    current_model: str = ""
    current_dataset: str = ""
    
    # Fitter state
    candidate_index: int = 0
    total_candidates: int = 0
    current_chi2: float = float('inf')
    best_chi2: float = float('inf')
    
    # Model evaluation
    phase7a_pass: bool = False
    
    # Logging
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=50))
    
    # Jackknife tracking
    jackknife_history: List[Dict[str, float]] = field(default_factory=list)
    
    # History for graphs (capped at 2000 samples)
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "time": [],
        "chi2": [],
        "gpu": [],
        "cpu": [],
        "kernel_ms": [],
    })
    
    # Process tracking
    process_table: Dict[str, Dict[str, any]] = field(default_factory=dict)
    
    # Thread safety
    _lock: Lock = field(default_factory=Lock)
    
    def update_system_metrics(self, cpu: float, gpu: float, ram: float) -> None:
        """Update system metrics."""
        with self._lock:
            self.cpu = cpu
            self.gpu = gpu
            self.ram = ram
            self.run_time = time.time() - self.start_time
    
    def update_progress(self, progress: float, model: str, dataset: str) -> None:
        """Update progress tracking."""
        with self._lock:
            self.progress = progress
            self.current_model = model
            self.current_dataset = dataset
    
    def update_fitter_state(self, candidate_idx: int, total_candidates: int, 
                          chi2: float, best_chi2: float) -> None:
        """Update fitter state."""
        with self._lock:
            self.candidate_index = candidate_idx
            self.total_candidates = total_candidates
            self.current_chi2 = chi2
            self.best_chi2 = best_chi2
    
    def update_phase7a(self, passed: bool) -> None:
        """Update Phase-7a status."""
        with self._lock:
            self.phase7a_pass = passed
    
    def add_log(self, message: str) -> None:
        """Add a log message."""
        with self._lock:
            timestamp = time.strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {message}")
    
    def update_history(self, current_time: Optional[float] = None) -> None:
        """Update history arrays with current state."""
        with self._lock:
            if current_time is None:
                current_time = time.time() - self.start_time
            
            # Cap history at 2000 samples
            max_samples = 2000
            
            for key in ["time", "chi2", "gpu", "cpu", "kernel_ms"]:
                if len(self.history[key]) >= max_samples:
                    self.history[key] = self.history[key][-max_samples+1:]
            
            self.history["time"].append(current_time)
            self.history["chi2"].append(self.current_chi2)
            self.history["gpu"].append(self.gpu)
            self.history["cpu"].append(self.cpu)
            # kernel_ms updated separately by process tracking
    
    def update_process_timing(self, process_name: str, execution_time_ms: float, 
                            is_running: bool = False, call_count: int = 0) -> None:
        """Update process table entry."""
        with self._lock:
            self.process_table[process_name] = {
                "last_time_ms": execution_time_ms,
                "running": is_running,
                "call_count": call_count,
                "last_updated": time.time()
            }
    
    def update_jackknife_trace(self, label: str, chi2: float) -> None:
        """Record a jackknife draw chi² point."""
        with self._lock:
            self.jackknife_history.append({"label": label, "chi2": float(chi2)})
            if len(self.jackknife_history) > 200:
                self.jackknife_history = self.jackknife_history[-200:]

    def clear_jackknife_history(self) -> None:
        """Remove jackknife trace history."""
        with self._lock:
            self.jackknife_history.clear()

    def get_snapshot(self) -> Dict[str, any]:
        """Get a thread-safe snapshot of current state."""
        with self._lock:
            return {
                "cpu": self.cpu,
                "gpu": self.gpu,
                "ram": self.ram,
                "run_time": self.run_time,
                "progress": self.progress,
                "current_model": self.current_model,
                "current_dataset": self.current_dataset,
                "candidate_index": self.candidate_index,
                "total_candidates": self.total_candidates,
                "current_chi2": self.current_chi2,
                "best_chi2": self.best_chi2,
                "phase7a_pass": self.phase7a_pass,
                "logs": list(self.logs),
                "jackknife_history": list(self.jackknife_history),
                "history": {k: list(v) for k, v in self.history.items()},
                "process_table": dict(self.process_table)
            }
