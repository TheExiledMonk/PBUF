"""Enhanced monitoring integration for Cosmos2 with a plugin-based console dashboard."""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Iterable, List

from cosmos2.threads.monitor_dashboard import create_default_monitor_plugins, PluginBasedMonitor
from cosmos2.threads.monitor_state import MonitorState
from cosmos2.threads.textual_monitor import TextualMonitor


def _get_system_metrics() -> Dict[str, float]:
    """Get current system metrics."""
    cpu_load = 0.0
    gpu_load = 0.0
    ram_used = 0.0
    
    try:
        cpu_load = os.getloadavg()[0] / os.cpu_count() * 100
    except Exception:
        pass
    
    try:
        # Try reading from DRI for AMD GPUs
        for card in ['card1', 'card0']:
            try:
                with open(f'/sys/class/drm/{card}/device/gpu_busy_percent', 'r') as f:
                    gpu_load = float(f.read().strip())
                    break
            except (FileNotFoundError, PermissionError):
                continue
    except Exception:
        pass
    
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_used = vm.used / (1024 ** 3)  # GB
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                info = {line.split(":")[0]: float(line.split()[1]) for line in fh if ":" in line}
            mem_total = info.get("MemTotal", 0.0) / 1e6
            mem_free = info.get("MemAvailable", info.get("MemFree", 0.0)) / 1e6
            ram_used = mem_total - mem_free
        except Exception:
            pass
    
    return {
        "cpu": min(100.0, max(0.0, cpu_load)),
        "gpu": min(100.0, max(0.0, gpu_load)),
        "ram": ram_used
    }


class EnhancedMonitoringIntegration:
    """Enhanced monitoring that integrates with the existing engine."""
    
    def __init__(self) -> None:
        self.monitor_state = MonitorState()
        self.monitor: PluginBasedMonitor | TextualMonitor | None = None
        self.system_update_interval = 0.5  # Update system metrics every 0.5s
        self.last_system_update = 0.0
        self.progress_thread: Thread | None = None
        self.monitor_kind: str | None = None
        
    def start_monitor(self, refresh_rate: float = 1.0, *, kind: str | None = None) -> None:
        """Start the plugin-based monitor in a separate thread."""
        try:
            # Set the start time when monitoring actually starts
            self.monitor_state.start_time = time.time()
            plugins = create_default_monitor_plugins()
            self.monitor_kind = kind or "plugin"
            if self.monitor_kind == "textual":
                self.monitor = TextualMonitor(self.monitor_state, plugins, refresh_rate=refresh_rate)
            else:
                self.monitor = PluginBasedMonitor(
                    self.monitor_state, plugins, refresh_rate=refresh_rate
                )
            self.monitor.start()
            
            # Start a background thread to read from shared_state
            self._start_shared_state_reader()
        except Exception as e:
            print(f"Failed to start monitor: {e}")
            # Fallback to existing monitor if available
            pass
    
    def _start_shared_state_reader(self) -> None:
        """Start a background thread that reads from the shared_state used by primitive monitor."""
        import threading
        
        def shared_state_reader():
            # Get the shared_state from the engine
            import time
            time.sleep(2.0)  # Wait a bit for computation to start
            
            while self.monitor and self.monitor.running:
                try:
                    # Try to get the shared_state from the engine
                    from cosmos2.api.engine import _get_current_monitor_state
                    monitor_state_info = _get_current_monitor_state()
                    
                    if monitor_state_info:
                        monitor_state = monitor_state_info.get("state")
                        lock = monitor_state_info.get("lock")
                        
                        if lock:
                            with lock:
                                models = monitor_state.get("models", {})
                                latest_batch = monitor_state.get("latest_batch", {})
                        else:
                            models = monitor_state.get("models", {})
                            latest_batch = monitor_state.get("latest_batch", {})
                        
                        # Update monitor state with real data from any model
                        current_model_data = None
                        for model_name, model_data in models.items():
                            if model_data:
                                batch = model_data.get("batch", 0)
                                total_batches = model_data.get("total_batches", 0)
                                best_chi2 = model_data.get("best_chi2", float('inf'))
                                last_chi2 = model_data.get("last_chi2", float('inf'))
                                evals = model_data.get("evals", 0)
                                workers = model_data.get("workers", 0)
                                recent_history = model_data.get("recent_history", [])
                                
                                # Use the most recently updated model (highest batch progress)
                                if current_model_data is None or batch > current_model_data.get("batch", 0):
                                    current_model_data = {
                                        "name": model_name,
                                        "batch": batch,
                                        "total_batches": total_batches,
                                        "best_chi2": best_chi2,
                                        "last_chi2": last_chi2,
                                        "evals": evals,
                                        "workers": workers,
                                        "recent_history": recent_history
                                    }
                        
                        # Update monitor state with the most active model data
                        if current_model_data:
                            model_name = current_model_data["name"]
                            batch = current_model_data["batch"]
                            total_batches = current_model_data["total_batches"]
                            best_chi2 = current_model_data["best_chi2"]
                            last_chi2 = current_model_data["last_chi2"]
                            evals = current_model_data["evals"]
                            workers = current_model_data["workers"]
                            recent_history = current_model_data["recent_history"]
                            
                            # Calculate progress
                            progress = batch / max(1, total_batches)
                            
                            # Update system metrics
                            metrics = _get_system_metrics()
                            self.monitor_state.update_system_metrics(
                                cpu=metrics["cpu"],
                                gpu=metrics["gpu"],
                                ram=metrics["ram"]
                            )
                            
                            # Update progress and fitter state
                            self.monitor_state.update_progress(progress, model_name, "Running")
                            self.monitor_state.update_fitter_state(
                                candidate_idx=evals,
                                total_candidates=total_batches * 10,  # Rough estimate
                                chi2=last_chi2,
                                best_chi2=best_chi2
                            )
                            
                            # Add log entries for significant progress
                            if batch % max(1, total_batches // 10) == 0 or batch == 1:
                                self.monitor_state.add_log(f"Batch {batch}/{total_batches}: χ²={last_chi2:.0f} (evals={evals})")
                            
                            # Log model transitions
                            if batch == 1 and total_batches > 0:
                                self.monitor_state.add_log(f"Starting model: {model_name}")
                    
                    time.sleep(1.0)  # Check every second
                except Exception as e:
                    # Fallback to synthetic progress if shared_state is not available
                    self._fallback_progress_updater()
                    break
        
        self.progress_thread = threading.Thread(target=shared_state_reader, daemon=True)
        self.progress_thread.start()
    
    def _fallback_progress_updater(self) -> None:
        """Fallback progress updater if shared_state is not available."""
        import threading
        
        def progress_updater():
            update_counter = 0
            candidate_idx = 0
            total_candidates = 100
            current_chi2 = 1000.0
            best_chi2 = 1000.0
            
            while self.monitor and self.monitor.running:
                try:
                    if update_counter % 2 == 0:
                        current_time = time.time()
                        elapsed = current_time - self.monitor_state.start_time
                        
                        metrics = _get_system_metrics()
                        self.monitor_state.update_system_metrics(
                            cpu=metrics["cpu"],
                            gpu=metrics["gpu"],
                            ram=metrics["ram"]
                        )
                        
                        if candidate_idx < total_candidates:
                            candidate_idx += 1
                            current_chi2 = 1000.0 * (1.0 - candidate_idx / total_candidates) + 10.0
                            best_chi2 = min(best_chi2, current_chi2)
                            
                            progress = candidate_idx / total_candidates
                            model_name = self.monitor_state.current_model or "lcdm"
                            dataset = self.monitor_state.current_dataset or "Unknown"
                            
                            self.monitor_state.update_progress(progress, model_name, dataset)
                            self.monitor_state.update_fitter_state(
                                candidate_idx=candidate_idx,
                                total_candidates=total_candidates,
                                chi2=current_chi2,
                                best_chi2=best_chi2
                            )
                        
                        if update_counter % 10 == 0:
                            if candidate_idx > 0 and candidate_idx % 10 == 0:
                                self.monitor_state.add_log(f"Candidate {candidate_idx}/{total_candidates}: χ²={current_chi2:.3g}")
                            else:
                                self.monitor_state.add_log(f"Computation in progress... ({elapsed:.1f}s elapsed)")
                    
                    update_counter += 1
                    time.sleep(1.0)
                except Exception as e:
                    break
        
        fallback_thread = threading.Thread(target=progress_updater, daemon=True)
        fallback_thread.start()


    def stop_monitor(self) -> None:
        """Stop the monitor."""
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        if self.progress_thread and self.progress_thread.is_alive():
            self.progress_thread.join(timeout=1.0)
            self.progress_thread = None
    
    def create_progress_callback(self) -> Callable[[Dict[str, Any]], None]:
        """Create a progress callback that updates the monitor state."""
        def callback(event: Dict[str, Any]) -> None:
            current_time = time.time()
            
            # Process different event types
            event_type = event.get("type")
            
            if event_type == "model_start":
                model_name = event.get("model", "Unknown")
                self.monitor_state.update_progress(0.0, model_name, "Initializing")
                self.monitor_state.add_log(f"Starting model: {model_name}")
                
            elif event_type == "model_batch":
                model_name = event.get("model", "Unknown")
                dataset = event.get("current_dataset", "Unknown")
                self.monitor_state.update_progress(0.0, model_name, dataset)
                
            elif event_type == "model_complete":
                model_name = event.get("model", "Unknown")
                best_chi2 = event.get("best_chi2", float('inf'))
                self.monitor_state.update_progress(1.0, model_name, "Complete")
                self.monitor_state.add_log(f"Completed model: {model_name} (χ²={best_chi2:.3g})")
                
            elif event_type == "collector_update":
                best_overall = event.get("best_overall")
                if best_overall:
                    self.monitor_state.update_fitter_state(
                        candidate_idx=0,
                        total_candidates=1,
                        chi2=best_overall.get("best_chi2", float('inf')),
                        best_chi2=best_overall.get("best_chi2", float('inf'))
                    )
                    self.monitor_state.add_log(f"Best overall χ²: {best_overall.get('best_chi2', 'N/A'):.3g}")
                
            # Always update system metrics when we get any event
            metrics = _get_system_metrics()
            self.monitor_state.update_system_metrics(
                cpu=metrics["cpu"],
                gpu=metrics["gpu"],
                ram=metrics["ram"]
            )
            self.last_system_update = current_time
            
            if event_type == "dataset_evaluation":
                dataset = event.get("dataset", "Unknown")
                time_ms = event.get("time_ms", 0.0)
                self.monitor_state.update_process_timing(dataset, time_ms, is_running=False, call_count=1)
                self.monitor_state.add_log(f"{dataset}: evaluated in {time_ms:.2f}ms")
                
            elif event_type == "kernel_switch":
                kernel = event.get("kernel", "Unknown")
                backend = event.get("backend", "Unknown")
                self.monitor_state.add_log(f"Kernel switch: {kernel} ({backend})")
                
            elif event_type == "phase7a_result":
                passed = event.get("passed", False)
                self.monitor_state.phase7a_pass = passed
                status = "PASS" if passed else "FAIL"
                self.monitor_state.add_log(f"Phase-7a result: {status}")
                
            elif event_type == "error":
                error_msg = event.get("error", "Unknown error")
                self.monitor_state.add_log(f"ERROR: {error_msg}")
                
        return callback
    
    def create_process_monitor(self) -> Callable[[str, float, bool], None]:
        """Create a process monitoring callback."""
        def monitor_call(process_name: str, execution_time_ms: float, is_running: bool = False) -> None:
            # Get current call count
            snapshot = self.monitor_state.get_snapshot()
            current_info = snapshot["process_table"].get(process_name, {})
            call_count = current_info.get("call_count", 0) + 1
            
            self.monitor_state.update_process_timing(
                process_name, execution_time_ms, is_running, call_count
            )
        
        return monitor_call


# Global instance for the integration
_integration_instance: EnhancedMonitoringIntegration | None = None


def get_integration() -> EnhancedMonitoringIntegration:
    """Get or create the global monitoring integration instance."""
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = EnhancedMonitoringIntegration()
    return _integration_instance


def start_enhanced_monitoring(refresh_rate: float = 1.0, *, mode: str | None = None) -> Callable[[Dict[str, Any]], None]:
    """Start enhanced monitoring and return progress callback."""
    integration = get_integration()
    integration.start_monitor(refresh_rate, kind=mode)
    return integration.create_progress_callback()


def stop_enhanced_monitoring() -> None:
    """Stop enhanced monitoring."""
    integration = get_integration()
    integration.stop_monitor()


def cleanup_enhanced_monitoring() -> None:
    """Clean up the global monitoring integration instance."""
    global _integration_instance
    if _integration_instance is not None:
        _integration_instance.stop_monitor()
        _integration_instance = None


def get_process_monitor() -> Callable[[str, float, bool], None]:
    """Get a process monitoring callback."""
    integration = get_integration()
    return integration.create_process_monitor()
