"""Example usage of the metrics collection system."""

import asyncio
import time
import random
from datetime import timedelta

from hpc_comms.metrics import (
    MetricsCollector, get_registry, create_collector,
    counter, gauge, histogram, timer
)
from hpc_comms.metrics.exporter import create_exporter
from hpc_comms.metrics.system_metrics import create_system_collector, create_hpc_collector


class ExampleService:
    """Example service that generates metrics."""
    
    def __init__(self):
        # Create a collector for this service
        self.collector = create_collector("example_service", labels={"version": "1.0"})
        
        # Create metrics
        self.requests_total = self.collector.counter(
            "http_requests_total",
            "Total number of HTTP requests",
            labels={"service": "example"}
        )
        
        self.request_duration = self.collector.histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            labels={"service": "example"}
        )
        
        self.active_connections = self.collector.gauge(
            "active_connections",
            "Number of active connections",
            labels={"service": "example"}
        )
        
        self.error_rate = self.collector.gauge(
            "error_rate_percent",
            "Error rate percentage",
            labels={"service": "example"}
        )
        
        # Create system and HPC collectors
        self.system_collector = create_system_collector(self.collector, interval=2.0)
        self.hpc_collector = create_hpc_collector(self.collector)
        
        self.running = False
    
    async def start(self):
        """Start the service."""
        self.running = True
        self.system_collector.start()
        
        # Simulate some initial connections
        self.active_connections.set(10)
    
    async def stop(self):
        """Stop the service."""
        self.running = False
        self.system_collector.stop()
    
    async def handle_request(self, request_type: str = "normal"):
        """Simulate handling a request."""
        start_time = time.time()
        
        # Increment request counter
        self.requests_total.inc()
        
        # Simulate request processing
        if request_type == "slow":
            await asyncio.sleep(random.uniform(0.1, 0.5))
        elif request_type == "fast":
            await asyncio.sleep(random.uniform(0.01, 0.05))
        else:
            await asyncio.sleep(random.uniform(0.02, 0.2))
        
        # Record duration
        duration = time.time() - start_time
        self.request_duration.observe(duration)
        
        # Simulate occasional errors
        if random.random() < 0.05:  # 5% error rate
            self.hpc_collector.record_auth_attempt(False)
            raise Exception("Simulated error")
        
        self.hpc_collector.record_auth_attempt(True)
        return {"status": "ok", "duration": duration}
    
    async def simulate_workload(self):
        """Simulate a realistic workload."""
        while self.running:
            try:
                # Random request types
                request_types = ["normal", "normal", "normal", "fast", "slow"]
                request_type = random.choice(request_types)
                
                await self.handle_request(request_type)
                
                # Occasionally change connection count
                if random.random() < 0.1:
                    change = random.randint(-5, 5)
                    new_count = max(0, self.active_connections.get() + change)
                    self.active_connections.set(new_count)
                
                # Update error rate
                if self.requests_total.get() > 0:
                    errors = self.hpc_collector.auth_failures.get()
                    total = self.hpc_collector.auth_attempts.get()
                    error_rate = (errors / total) * 100 if total > 0 else 0
                    self.error_rate.set(error_rate)
                
                # Simulate HPC operations
                await self.simulate_hpc_operations()
                
                # Small delay between requests
                await asyncio.sleep(random.uniform(0.05, 0.2))
                
            except Exception as e:
                # Continue despite errors
                await asyncio.sleep(0.1)
    
    async def simulate_hpc_operations(self):
        """Simulate HPC-specific operations."""
        # Simulate message processing
        message_size = random.randint(100, 10000)
        self.hpc_collector.record_message_sent(message_size)
        
        # Simulate work items
        if random.random() < 0.3:  # 30% chance of work item
            self.hpc_collector.record_work_submitted()
            
            # Simulate processing time
            processing_time = random.uniform(0.5, 5.0)
            await asyncio.sleep(0.01)  # Minimal simulation
            
            # Complete or fail the work
            if random.random() < 0.9:  # 90% success rate
                self.hpc_collector.record_work_completed(processing_time)
            else:
                self.hpc_collector.record_work_failed()
        
        # Update node metrics
        total_nodes = random.randint(8, 12)
        online_nodes = random.randint(6, total_nodes)
        self.hpc_collector.update_node_metrics(total_nodes, online_nodes)
        
        # Record heartbeats
        if random.random() < 0.5:
            self.hpc_collector.record_heartbeat()
        
        # Update work queue size
        queue_size = random.randint(0, 20)
        self.hpc_collector.update_work_queue_size(queue_size)


async def basic_metrics_example():
    """Basic metrics collection example."""
    print("=== Basic Metrics Example ===")
    
    # Create collector
    collector = MetricsCollector()
    
    # Create different types of metrics
    counter = collector.counter("example_counter", "An example counter")
    gauge = collector.gauge("example_gauge", "An example gauge")
    histogram = collector.histogram("example_histogram", "An example histogram")
    
    # Use the metrics
    counter.inc()
    counter.inc(5)
    
    gauge.set(42.5)
    gauge.inc(2.5)
    
    histogram.observe(0.1)
    histogram.observe(0.5)
    histogram.observe(1.0)
    histogram.observe(2.5)
    
    # Collect and display
    all_metrics = collector.collect_all()
    
    print("Collected metrics:")
    for metric_name, values in all_metrics.items():
        print(f"  {metric_name}:")
        for value in values:
            print(f"    {value.value} (labels: {value.labels})")
    
    print()


async def timer_example():
    """Timer usage example."""
    print("=== Timer Example ===")
    
    collector = MetricsCollector()
    
    # Create a timer
    timer = collector.timer("operation_duration", "Operation duration")
    
    # Use timer as context manager
    with timer:
        time.sleep(0.1)  # Simulate work
    
    # Use timer manually
    manual_timer = collector.timer("manual_operation", "Manual operation")
    manual_timer.start()
    time.sleep(0.05)
    manual_timer.stop()
    
    # Display results
    all_metrics = collector.collect_all()
    
    print("Timer metrics:")
    for metric_name, values in all_metrics.items():
        if "duration" in metric_name:
            print(f"  {metric_name}:")
            for value in values:
                if value.labels.get('quantile') == 'count':
                    print(f"    Count: {value.value}")
                elif value.labels.get('quantile') == 'sum':
                    print(f"    Sum: {value.value}")
    
    print()


async def exporter_example():
    """Metrics exporter example."""
    print("=== Exporter Example ===")
    
    # Create collector with some metrics
    collector = MetricsCollector()
    
    counter = collector.counter("requests_total", "Total requests")
    gauge = collector.gauge("temperature", "Current temperature")
    
    counter.inc(100)
    gauge.set(23.5)
    
    all_metrics = collector.collect_all()
    
    # Test different exporters
    exporters = [
        ("text", create_exporter("text")),
        ("json", create_exporter("json")),
        ("influx", create_exporter("influx")),
        ("statsd", create_exporter("statsd")),
    ]
    
    for format_name, exporter in exporters:
        print(f"\n--- {format_name.upper()} Export ---")
        output = exporter.export(all_metrics)
        print(output[:200] + "..." if len(output) > 200 else output)
    
    print()


async def system_metrics_example():
    """System metrics collection example."""
    print("=== System Metrics Example ===")
    
    collector = MetricsCollector()
    system_collector = create_system_collector(collector, interval=1.0)
    
    system_collector.start()
    
    # Collect for a few seconds
    print("Collecting system metrics for 3 seconds...")
    await asyncio.sleep(3.0)
    
    # Display system metrics
    all_metrics = collector.collect_all()
    
    print("System metrics collected:")
    for metric_name, values in all_metrics.items():
        if "system_" in metric_name or "process_" in metric_name:
            for value in values:
                print(f"  {metric_name}: {value.value}")
    
    system_collector.stop()
    print()


async def service_simulation_example():
    """Full service simulation with metrics."""
    print("=== Service Simulation Example ===")
    
    service = ExampleService()
    await service.start()
    
    # Run workload for 10 seconds
    print("Running service simulation for 10 seconds...")
    
    # Start workload in background
    workload_task = asyncio.create_task(service.simulate_workload())
    
    # Periodically print metrics
    for i in range(5):
        await asyncio.sleep(2.0)
        
        print(f"\n--- Metrics at {i*2}s ---")
        
        # Service metrics
        print(f"Requests: {service.requests_total.get()}")
        print(f"Active connections: {service.active_connections.get()}")
        print(f"Error rate: {service.error_rate.get():.2f}%")
        
        # HPC metrics
        print(f"Messages sent: {service.hpc_collector.messages_sent.get()}")
        print(f"Work items completed: {service.hpc_collector.work_items_completed.get()}")
        print(f"Online nodes: {service.hpc_collector.online_nodes.get()}")
        
        # System metrics
        cpu_percent = service.collector.collect_all().get("system_cpu_percent")
        if cpu_percent:
            print(f"CPU: {cpu_percent[0].value:.1f}%")
        
        memory_percent = service.collector.collect_all().get("system_memory_percent")
        if memory_percent:
            print(f"Memory: {memory_percent[0].value:.1f}%")
    
    # Stop service
    service.stop()
    await workload_task
    
    print("\nFinal metrics summary:")
    
    # Export final metrics in different formats
    registry = get_registry()
    all_metrics = registry.collect_all()
    
    print("\n--- Prometheus Format ---")
    prometheus_exporter = create_exporter("text")
    print(prometheus_exporter.export(all_metrics["example_service"]))
    
    print("\n--- JSON Format ---")
    json_exporter = create_exporter("json", pretty=True)
    print(json_exporter.export(all_metrics["example_service"]))


async def main():
    """Run all examples."""
    print("Starting metrics examples...\n")
    
    await basic_metrics_example()
    await timer_example()
    await exporter_example()
    await system_metrics_example()
    await service_simulation_example()
    
    print("\nMetrics examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
