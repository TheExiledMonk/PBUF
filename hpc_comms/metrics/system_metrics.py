"""System metrics collector for HPC communication."""

import os
import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .collector import MetricsCollector, Gauge, Counter, Histogram, Timer


class SystemMetricsCollector:
    """Collects system-level metrics."""
    
    def __init__(self, collector: MetricsCollector, interval: float = 5.0):
        self.collector = collector
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Create metrics
        self.cpu_percent = collector.gauge(
            "system_cpu_percent",
            "CPU usage percentage",
            labels={'component': 'system'}
        )
        
        self.memory_percent = collector.gauge(
            "system_memory_percent", 
            "Memory usage percentage",
            labels={'component': 'system'}
        )
        
        self.memory_bytes = collector.gauge(
            "system_memory_bytes",
            "Memory usage in bytes",
            labels={'component': 'system'}
        )
        
        self.disk_usage = collector.gauge(
            "system_disk_usage_percent",
            "Disk usage percentage",
            labels={'component': 'system'}
        )
        
        self.disk_bytes = collector.gauge(
            "system_disk_bytes",
            "Disk usage in bytes",
            labels={'component': 'system'}
        )
        
        self.network_bytes = collector.counter(
            "system_network_bytes_total",
            "Network bytes transferred",
            labels={'component': 'system'}
        )
        
        self.process_count = collector.gauge(
            "system_process_count",
            "Number of running processes",
            labels={'component': 'system'}
        )
        
        self.load_average = collector.gauge(
            "system_load_average",
            "System load average",
            labels={'component': 'system'}
        )
        
        self.uptime = collector.gauge(
            "system_uptime_seconds",
            "System uptime in seconds",
            labels={'component': 'system'}
        )
        
        # Process-specific metrics
        self.process_cpu = collector.gauge(
            "process_cpu_percent",
            "Process CPU usage percentage",
            labels={'component': 'process'}
        )
        
        self.process_memory = collector.gauge(
            "process_memory_bytes",
            "Process memory usage in bytes",
            labels={'component': 'process'}
        )
        
        self.process_threads = collector.gauge(
            "process_thread_count",
            "Number of threads in process",
            labels={'component': 'process'}
        )
        
        self.process_fds = collector.gauge(
            "process_fd_count",
            "Number of file descriptors in process",
            labels={'component': 'process'}
        )
        
        # Store initial network stats
        self._last_network_stats = psutil.net_io_counters()
        self._last_network_time = time.time()
    
    def start(self) -> None:
        """Start collecting system metrics."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop collecting system metrics."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
    
    def _collect_loop(self) -> None:
        """Main collection loop."""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.interval)
            except Exception as e:
                print(f"System metrics collection error: {e}")
                time.sleep(self.interval)
    
    def _collect_metrics(self) -> None:
        """Collect all system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_percent.set(cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.memory_percent.set(memory.percent)
        self.memory_bytes.set(memory.used)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.disk_usage.set(disk.percent)
        self.disk_bytes.set(disk.used)
        
        # Network metrics
        current_network = psutil.net_io_counters()
        current_time = time.time()
        
        if self._last_network_stats:
            bytes_sent = current_network.bytes_sent - self._last_network_stats.bytes_sent
            bytes_recv = current_network.bytes_recv - self._last_network_stats.bytes_recv
            
            # Calculate rate
            time_diff = current_time - self._last_network_time
            if time_diff > 0:
                # Convert to per-second rate
                sent_rate = bytes_sent / time_diff
                recv_rate = bytes_recv / time_diff
                
                # Update counters with rate
                self.network_bytes.inc(sent_rate + recv_rate)
        
        self._last_network_stats = current_network
        self._last_network_time = current_time
        
        # Process count
        self.process_count.set(len(psutil.pids()))
        
        # Load average (Unix only)
        try:
            load_avg = os.getloadavg()
            self.load_average.set(load_avg[0])  # 1-minute average
        except (AttributeError, OSError):
            # Not available on Windows
            pass
        
        # Uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time
        self.uptime.set(uptime)
        
        # Process-specific metrics
        process = psutil.Process()
        
        # Process CPU
        try:
            self.process_cpu.set(process.cpu_percent())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Process memory
        try:
            memory_info = process.memory_info()
            self.process_memory.set(memory_info.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Process threads
        try:
            self.process_threads.set(process.num_threads())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        
        # Process file descriptors
        try:
            self.process_fds.set(process.num_fds())
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            # num_fds not available on Windows
            pass
    
    def collect_once(self) -> None:
        """Collect metrics once (for testing)."""
        self._collect_metrics()


class HPCMetricsCollector:
    """Collects HPC-specific metrics."""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        
        # Communication metrics
        self.messages_sent = collector.counter(
            "hpc_messages_sent_total",
            "Total number of messages sent",
            labels={'component': 'communication'}
        )
        
        self.messages_received = collector.counter(
            "hpc_messages_received_total", 
            "Total number of messages received",
            labels={'component': 'communication'}
        )
        
        self.message_size = collector.histogram(
            "hpc_message_size_bytes",
            "Message size distribution",
            buckets=[100, 1000, 10000, 100000, 1000000],
            labels={'component': 'communication'}
        )
        
        self.message_latency = collector.histogram(
            "hpc_message_latency_seconds",
            "Message latency distribution",
            buckets=[0.001, 0.01, 0.1, 1.0, 10.0],
            labels={'component': 'communication'}
        )
        
        self.connection_count = collector.gauge(
            "hpc_connection_count",
            "Number of active connections",
            labels={'component': 'communication'}
        )
        
        self.connection_errors = collector.counter(
            "hpc_connection_errors_total",
            "Total number of connection errors",
            labels={'component': 'communication'}
        )
        
        # Node registry metrics
        self.node_count = collector.gauge(
            "hpc_node_count",
            "Number of registered nodes",
            labels={'component': 'registry'}
        )
        
        self.online_nodes = collector.gauge(
            "hpc_online_nodes",
            "Number of online nodes",
            labels={'component': 'registry'}
        )
        
        self.node_heartbeats = collector.counter(
            "hpc_node_heartbeats_total",
            "Total number of node heartbeats",
            labels={'component': 'registry'}
        )
        
        # Work distribution metrics
        self.work_items_submitted = collector.counter(
            "hpc_work_items_submitted_total",
            "Total number of work items submitted",
            labels={'component': 'workload'}
        )
        
        self.work_items_completed = collector.counter(
            "hpc_work_items_completed_total",
            "Total number of work items completed",
            labels={'component': 'workload'}
        )
        
        self.work_items_failed = collector.counter(
            "hpc_work_items_failed_total",
            "Total number of work items failed",
            labels={'component': 'workload'}
        )
        
        self.work_queue_size = collector.gauge(
            "hpc_work_queue_size",
            "Current work queue size",
            labels={'component': 'workload'}
        )
        
        self.work_processing_time = collector.histogram(
            "hpc_work_processing_time_seconds",
            "Work item processing time distribution",
            buckets=[1.0, 10.0, 60.0, 300.0, 1800.0],
            labels={'component': 'workload'}
        )
        
        # Authentication metrics
        self.auth_attempts = collector.counter(
            "hpc_auth_attempts_total",
            "Total number of authentication attempts",
            labels={'component': 'auth'}
        )
        
        self.auth_successes = collector.counter(
            "hpc_auth_successes_total",
            "Total number of successful authentications",
            labels={'component': 'auth'}
        )
        
        self.auth_failures = collector.counter(
            "hpc_auth_failures_total",
            "Total number of failed authentications",
            labels={'component': 'auth'}
        )
        
        self.token_count = collector.gauge(
            "hpc_active_tokens",
            "Number of active authentication tokens",
            labels={'component': 'auth'}
        )
        
        # GPU metrics (if available)
        self.gpu_count = collector.gauge(
            "hpc_gpu_count",
            "Number of available GPUs",
            labels={'component': 'hardware'}
        )
        
        self.gpu_memory_used = collector.gauge(
            "hpc_gpu_memory_used_bytes",
            "GPU memory usage in bytes",
            labels={'component': 'hardware'}
        )
        
        self.gpu_utilization = collector.gauge(
            "hpc_gpu_utilization_percent",
            "GPU utilization percentage",
            labels={'component': 'hardware'}
        )
        
        # Initialize GPU metrics
        self._init_gpu_metrics()
    
    def _init_gpu_metrics(self) -> None:
        """Initialize GPU metrics if available."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_count.set(pynvml.nvmlDeviceGetCount())
            self._has_gpu = True
        except (ImportError, Exception):
            self._has_gpu = False
            self.gpu_count.set(0)
    
    def record_message_sent(self, size: int) -> None:
        """Record a sent message."""
        self.messages_sent.inc()
        self.message_size.observe(size)
    
    def record_message_received(self, size: int) -> None:
        """Record a received message."""
        self.messages_received.inc()
        self.message_size.observe(size)
    
    def record_message_latency(self, latency: float) -> None:
        """Record message latency."""
        self.message_latency.observe(latency)
    
    def update_connection_count(self, count: int) -> None:
        """Update connection count."""
        self.connection_count.set(count)
    
    def record_connection_error(self) -> None:
        """Record a connection error."""
        self.connection_errors.inc()
    
    def update_node_metrics(self, total_nodes: int, online_nodes: int) -> None:
        """Update node registry metrics."""
        self.node_count.set(total_nodes)
        self.online_nodes.set(online_nodes)
    
    def record_heartbeat(self) -> None:
        """Record a node heartbeat."""
        self.node_heartbeats.inc()
    
    def record_work_submitted(self) -> None:
        """Record work item submission."""
        self.work_items_submitted.inc()
    
    def record_work_completed(self, processing_time: float) -> None:
        """Record work item completion."""
        self.work_items_completed.inc()
        self.work_processing_time.observe(processing_time)
    
    def record_work_failed(self) -> None:
        """Record work item failure."""
        self.work_items_failed.inc()
    
    def update_work_queue_size(self, size: int) -> None:
        """Update work queue size."""
        self.work_queue_size.set(size)
    
    def record_auth_attempt(self, success: bool) -> None:
        """Record authentication attempt."""
        self.auth_attempts.inc()
        if success:
            self.auth_successes.inc()
        else:
            self.auth_failures.inc()
    
    def update_token_count(self, count: int) -> None:
        """Update active token count."""
        self.token_count.set(count)
    
    def collect_gpu_metrics(self) -> None:
        """Collect GPU metrics if available."""
        if not self._has_gpu:
            return
        
        try:
            import pynvml
            
            device_count = pynvml.nvmlDeviceGetCount()
            total_memory = 0
            used_memory = 0
            total_utilization = 0
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory += mem_info.total
                used_memory += mem_info.used
                
                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                total_utilization += util.gpu
            
            # Update metrics
            self.gpu_memory_used.set(used_memory)
            if device_count > 0:
                self.gpu_utilization.set(total_utilization / device_count)
            
        except Exception as e:
            print(f"GPU metrics collection error: {e}")


def create_system_collector(collector: MetricsCollector, interval: float = 5.0) -> SystemMetricsCollector:
    """Create a system metrics collector."""
    return SystemMetricsCollector(collector, interval)


def create_hpc_collector(collector: MetricsCollector) -> HPCMetricsCollector:
    """Create an HPC metrics collector."""
    return HPCMetricsCollector(collector)
