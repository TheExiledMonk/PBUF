"""Tests for metrics collection system."""

import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from hpc_comms.metrics.collector import (
    Counter, Gauge, Histogram, Summary, Timer, MetricsCollector,
    MetricType, MetricValue
)
from hpc_comms.metrics.registry import MetricsRegistry, get_registry
from hpc_comms.metrics.exporter import (
    TextExporter, JsonExporter, InfluxExporter, StatsdExporter,
    GraphiteExporter, OpenTelemetryExporter, create_exporter
)
from hpc_comms.metrics.system_metrics import (
    SystemMetricsCollector, HPCMetricsCollector, create_system_collector,
    create_hpc_collector
)


class TestCounter:
    """Test counter metric."""
    
    def test_counter_creation(self):
        """Test counter creation."""
        counter = Counter("test_counter", "A test counter")
        
        assert counter.name == "test_counter"
        assert counter.description == "A test counter"
        assert counter.metric_type == MetricType.COUNTER
        assert counter.get() == 0
    
    def test_counter_increment(self):
        """Test counter increment."""
        counter = Counter("test_counter", "A test counter")
        
        counter.inc()
        assert counter.get() == 1
        
        counter.inc(5)
        assert counter.get() == 6
    
    def test_counter_reset(self):
        """Test counter reset."""
        counter = Counter("test_counter", "A test counter")
        counter.inc(10)
        
        counter.reset()
        assert counter.get() == 0
    
    def test_counter_collect(self):
        """Test counter collection."""
        counter = Counter("test_counter", "A test counter", labels={"env": "test"})
        counter.inc(3)
        
        values = counter.collect()
        assert len(values) == 1
        
        value = values[0]
        assert value.value == 3
        assert value.labels == {"env": "test"}
        assert isinstance(value.timestamp, datetime)


class TestGauge:
    """Test gauge metric."""
    
    def test_gauge_creation(self):
        """Test gauge creation."""
        gauge = Gauge("test_gauge", "A test gauge")
        
        assert gauge.name == "test_gauge"
        assert gauge.description == "A test gauge"
        assert gauge.metric_type == MetricType.GAUGE
        assert gauge.get() == 0.0
    
    def test_gauge_set(self):
        """Test gauge set."""
        gauge = Gauge("test_gauge", "A test gauge")
        
        gauge.set(42.5)
        assert gauge.get() == 42.5
    
    def test_gauge_increment(self):
        """Test gauge increment."""
        gauge = Gauge("test_gauge", "A test gauge")
        
        gauge.inc()
        assert gauge.get() == 1.0
        
        gauge.inc(2.5)
        assert gauge.get() == 3.5
    
    def test_gauge_decrement(self):
        """Test gauge decrement."""
        gauge = Gauge("test_gauge", "A test gauge")
        gauge.set(10.0)
        
        gauge.dec()
        assert gauge.get() == 9.0
        
        gauge.dec(2.5)
        assert gauge.get() == 6.5
    
    def test_gauge_collect(self):
        """Test gauge collection."""
        gauge = Gauge("test_gauge", "A test gauge", labels={"service": "test"})
        gauge.set(7.5)
        
        values = gauge.collect()
        assert len(values) == 1
        
        value = values[0]
        assert value.value == 7.5
        assert value.labels == {"service": "test"}


class TestHistogram:
    """Test histogram metric."""
    
    def test_histogram_creation(self):
        """Test histogram creation."""
        histogram = Histogram("test_histogram", "A test histogram")
        
        assert histogram.name == "test_histogram"
        assert histogram.description == "A test histogram"
        assert histogram.metric_type == MetricType.HISTOGRAM
        assert histogram.get_count() == 0
        assert histogram.get_sum() == 0.0
    
    def test_histogram_observe(self):
        """Test histogram observation."""
        histogram = Histogram("test_histogram", "A test histogram")
        
        histogram.observe(0.1)
        histogram.observe(1.0)
        histogram.observe(5.0)
        
        assert histogram.get_count() == 3
        assert histogram.get_sum() == 6.1
    
    def test_histogram_buckets(self):
        """Test histogram buckets."""
        buckets = [0.1, 0.5, 1.0, 2.0]
        histogram = Histogram("test_histogram", "A test histogram", buckets=buckets)
        
        histogram.observe(0.05)  # Should go in 0.1 bucket
        histogram.observe(0.3)   # Should go in 0.5 bucket
        histogram.observe(0.7)   # Should go in 1.0 bucket
        histogram.observe(1.5)   # Should go in 2.0 bucket
        histogram.observe(3.0)   # Should go in inf bucket
        
        values = histogram.collect()
        
        # Check bucket counts
        bucket_values = {v.labels['le']: v.value for v in values if 'le' in v.labels}
        
        assert bucket_values['0.1'] == 1
        assert bucket_values['0.5'] == 2
        assert bucket_values['1.0'] == 3
        assert bucket_values['2.0'] == 4
        assert bucket_values['inf'] == 5
    
    def test_histogram_reset(self):
        """Test histogram reset."""
        histogram = Histogram("test_histogram", "A test histogram")
        histogram.observe(1.0)
        histogram.observe(2.0)
        
        histogram.reset()
        
        assert histogram.get_count() == 0
        assert histogram.get_sum() == 0.0


class TestSummary:
    """Test summary metric."""
    
    def test_summary_creation(self):
        """Test summary creation."""
        summary = Summary("test_summary", "A test summary")
        
        assert summary.name == "test_summary"
        assert summary.description == "A test summary"
        assert summary.metric_type == MetricType.SUMMARY
        assert summary.get_count() == 0
        assert summary.get_sum() == 0.0
    
    def test_summary_observe(self):
        """Test summary observation."""
        summary = Summary("test_summary", "A test summary")
        
        summary.observe(1.0)
        summary.observe(2.0)
        summary.observe(3.0)
        
        assert summary.get_count() == 3
        assert summary.get_sum() == 6.0
    
    def test_summary_quantiles(self):
        """Test summary quantile calculation."""
        summary = Summary("test_summary", "A test summary", quantiles=[0.5, 0.9, 0.99])
        
        # Add values 1-100
        for i in range(1, 101):
            summary.observe(float(i))
        
        values = summary.collect()
        
        # Check quantiles
        quantile_values = {v.labels['quantile']: v.value for v in values if 'quantile' in v.labels and v.labels['quantile'] not in ['sum', 'count']}
        
        assert abs(quantile_values['0.5'] - 50.5) < 1.0  # Median
        assert abs(quantile_values['0.9'] - 90.1) < 1.0   # 90th percentile
        assert abs(quantile_values['0.99'] - 99.01) < 1.0 # 99th percentile


class TestTimer:
    """Test timer metric."""
    
    def test_timer_manual(self):
        """Test manual timer usage."""
        histogram = Histogram("test_timer", "A test timer")
        timer = Timer(histogram)
        
        timer.start()
        time.sleep(0.01)  # Small delay
        duration = timer.stop()
        
        assert duration > 0.01
        assert histogram.get_count() == 1
        assert histogram.get_sum() > 0.01
    
    def test_timer_context_manager(self):
        """Test timer context manager."""
        histogram = Histogram("test_timer", "A test timer")
        
        with Timer(histogram):
            time.sleep(0.01)
        
        assert histogram.get_count() == 1
        assert histogram.get_sum() > 0.01
    
    def test_timer_not_started(self):
        """Test timer stop without start."""
        histogram = Histogram("test_timer", "A test timer")
        timer = Timer(histogram)
        
        with pytest.raises(RuntimeError):
            timer.stop()


class TestMetricsCollector:
    """Test metrics collector."""
    
    def test_collector_creation(self):
        """Test collector creation."""
        collector = MetricsCollector()
        
        assert len(collector.get_metric_names()) == 0
    
    def test_collector_with_default_labels(self):
        """Test collector with default labels."""
        collector = MetricsCollector(default_labels={"service": "test"})
        
        counter = collector.counter("test_counter", "A test counter")
        counter.inc()
        
        values = counter.collect()
        assert values[0].labels == {"service": "test"}
    
    def test_collector_counter(self):
        """Test collector counter creation."""
        collector = MetricsCollector()
        
        counter1 = collector.counter("test_counter", "A test counter")
        counter2 = collector.counter("test_counter", "A test counter")
        
        # Should return the same instance
        assert counter1 is counter2
        
        counter1.inc()
        assert counter2.get() == 1
    
    def test_collector_gauge(self):
        """Test collector gauge creation."""
        collector = MetricsCollector()
        
        gauge = collector.gauge("test_gauge", "A test gauge")
        gauge.set(42.0)
        
        assert gauge.get() == 42.0
    
    def test_collector_histogram(self):
        """Test collector histogram creation."""
        collector = MetricsCollector()
        
        histogram = collector.histogram("test_histogram", "A test histogram")
        histogram.observe(1.0)
        
        assert histogram.get_count() == 1
    
    def test_collector_summary(self):
        """Test collector summary creation."""
        collector = MetricsCollector()
        
        summary = collector.summary("test_summary", "A test summary")
        summary.observe(1.0)
        
        assert summary.get_count() == 1
    
    def test_collector_timer(self):
        """Test collector timer creation."""
        collector = MetricsCollector()
        
        timer = collector.timer("test_timer", "A test timer")
        
        with timer:
            time.sleep(0.01)
        
        # Should have recorded the timing
        collected = collector.collect_all()
        assert "test_timer" in collected
    
    def test_collect_all(self):
        """Test collecting all metrics."""
        collector = MetricsCollector()
        
        counter = collector.counter("test_counter", "A test counter")
        gauge = collector.gauge("test_gauge", "A test gauge")
        
        counter.inc()
        gauge.set(42.0)
        
        all_metrics = collector.collect_all()
        
        assert len(all_metrics) == 2
        assert "test_counter" in all_metrics
        assert "test_gauge" in all_metrics
    
    def test_remove_metric(self):
        """Test removing metrics."""
        collector = MetricsCollector()
        
        counter = collector.counter("test_counter", "A test counter")
        assert len(collector.get_metric_names()) == 1
        
        removed = collector.remove_metric("test_counter")
        assert removed is True
        assert len(collector.get_metric_names()) == 0
        
        removed = collector.remove_metric("nonexistent")
        assert removed is False
    
    def test_clear_all(self):
        """Test clearing all metrics."""
        collector = MetricsCollector()
        
        collector.counter("test_counter", "A test counter")
        collector.gauge("test_gauge", "A test gauge")
        
        assert len(collector.get_metric_names()) == 2
        
        collector.clear_all()
        
        assert len(collector.get_metric_names()) == 0


class TestMetricsRegistry:
    """Test metrics registry."""
    
    def test_registry_creation(self):
        """Test registry creation."""
        registry = MetricsRegistry()
        
        assert len(registry.list_collectors()) == 1  # Default collector
        assert "default" in registry.list_collectors()
    
    def test_create_collector(self):
        """Test creating collectors."""
        registry = MetricsRegistry()
        
        collector1 = registry.create_collector("test1")
        collector2 = registry.create_collector("test2")
        
        assert collector1 is not collector2
        assert len(registry.list_collectors()) == 3  # default + test1 + test2
    
    def test_duplicate_collector(self):
        """Test creating duplicate collector."""
        registry = MetricsRegistry()
        
        registry.create_collector("test")
        
        with pytest.raises(ValueError):
            registry.create_collector("test")
    
    def test_get_collector(self):
        """Test getting collectors."""
        registry = MetricsRegistry()
        
        collector = registry.create_collector("test")
        
        retrieved = registry.get_collector("test")
        assert retrieved is collector
        
        nonexistent = registry.get_collector("nonexistent")
        assert nonexistent is None
    
    def test_remove_collector(self):
        """Test removing collectors."""
        registry = MetricsRegistry()
        
        registry.create_collector("test")
        assert len(registry.list_collectors()) == 2
        
        removed = registry.remove_collector("test")
        assert removed is True
        assert len(registry.list_collectors()) == 1
        
        removed = registry.remove_collector("nonexistent")
        assert removed is False
    
    def test_global_labels(self):
        """Test global labels."""
        registry = MetricsRegistry()
        
        registry.set_global_label("env", "test")
        
        collector = registry.create_collector("test")
        counter = collector.counter("test_counter", "A test counter")
        
        values = counter.collect()
        assert values[0].labels == {"env": "test"}
        
        registry.remove_global_label("env")
        
        counter2 = collector.counter("test_counter2", "A test counter")
        values2 = counter2.collect()
        assert "env" not in values2[0].labels
    
    def test_collect_all(self):
        """Test collecting from all collectors."""
        registry = MetricsRegistry()
        
        collector1 = registry.create_collector("test1")
        collector2 = registry.create_collector("test2")
        
        collector1.counter("counter1", "Counter 1").inc()
        collector2.counter("counter2", "Counter 2").inc()
        
        all_metrics = registry.collect_all()
        
        assert len(all_metrics) == 2
        assert "test1" in all_metrics
        assert "test2" in all_metrics
    
    def test_global_registry(self):
        """Test global registry functions."""
        from hpc_comms.metrics.registry import get_registry, create_collector
        
        registry = get_registry()
        assert isinstance(registry, MetricsRegistry)
        
        collector = create_collector("global_test")
        assert isinstance(collector, MetricsCollector)
        
        # Should be in the global registry
        retrieved = get_registry().get_collector("global_test")
        assert retrieved is collector


class TestExporters:
    """Test metrics exporters."""
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        collector = MetricsCollector()
        
        counter = collector.counter("test_counter", "A test counter")
        counter.inc(5)
        
        gauge = collector.gauge("test_gauge", "A test gauge")
        gauge.set(42.5)
        
        histogram = collector.histogram("test_histogram", "A test histogram")
        histogram.observe(0.1)
        histogram.observe(1.0)
        histogram.observe(5.0)
        
        return collector.collect_all()
    
    def test_text_exporter(self, sample_metrics):
        """Test text exporter."""
        exporter = TextExporter()
        
        output = exporter.export(sample_metrics)
        
        assert "# HELP test_counter A test counter" in output
        assert "# TYPE test_counter counter" in output
        assert "test_counter 5" in output
        assert "# HELP test_gauge A test gauge" in output
        assert "# TYPE test_gauge gauge" in output
        assert "test_gauge 42.5" in output
        
        assert exporter.content_type() == "text/plain; version=0.0.4; charset=utf-8"
    
    def test_json_exporter(self, sample_metrics):
        """Test JSON exporter."""
        exporter = JsonExporter()
        
        output = exporter.export(sample_metrics)
        
        # Parse as JSON to verify structure
        data = json.loads(output)
        
        assert "test_counter" in data
        assert "test_gauge" in data
        assert "test_histogram" in data
        
        assert len(data["test_counter"]) == 1
        assert data["test_counter"][0]["value"] == 5
        
        assert exporter.content_type() == "application/json"
    
    def test_influx_exporter(self, sample_metrics):
        """Test InfluxDB exporter."""
        exporter = InfluxExporter()
        
        output = exporter.export(sample_metrics)
        
        lines = output.strip().split('\n')
        
        # Should have lines for each metric
        assert len(lines) > 0
        
        # Check format of a line: measurement,fields timestamp
        for line in lines:
            parts = line.split(' ')
            assert len(parts) == 3
            assert parts[2].isdigit()  # Timestamp
    
    def test_statsd_exporter(self, sample_metrics):
        """Test StatsD exporter."""
        exporter = StatsdExporter()
        
        output = exporter.export(sample_metrics)
        
        lines = output.strip().split('\n')
        
        # Check format: metric:value|type
        for line in lines:
            assert '|' in line
            parts = line.split('|')
            assert len(parts) == 2
            assert ':' in parts[0]
    
    def test_graphite_exporter(self, sample_metrics):
        """Test Graphite exporter."""
        exporter = GraphiteExporter()
        
        output = exporter.export(sample_metrics)
        
        lines = output.strip().split('\n')
        
        # Check format: path value timestamp
        for line in lines:
            parts = line.split(' ')
            assert len(parts) == 3
            assert parts[1].replace('.', '').isdigit()  # Value
            assert parts[2].isdigit()  # Timestamp
    
    def test_opentelemetry_exporter(self, sample_metrics):
        """Test OpenTelemetry exporter."""
        exporter = OpenTelemetryExporter()
        
        output = exporter.export(sample_metrics)
        
        # Parse as JSON to verify structure
        data = json.loads(output)
        
        assert "resource" in data
        assert "scopeMetrics" in data
        assert len(data["scopeMetrics"]) > 0
        
        # Check scope metrics structure
        scope_metric = data["scopeMetrics"][0]
        assert "scope" in scope_metric
        assert "metrics" in scope_metric
    
    def test_exporter_factory(self):
        """Test exporter factory."""
        # Test valid exporters
        assert isinstance(create_exporter('text'), TextExporter)
        assert isinstance(create_exporter('prometheus'), TextExporter)
        assert isinstance(create_exporter('json'), JsonExporter)
        assert isinstance(create_exporter('influx'), InfluxExporter)
        assert isinstance(create_exporter('statsd'), StatsdExporter)
        assert isinstance(create_exporter('graphite'), GraphiteExporter)
        assert isinstance(create_exporter('opentelemetry'), OpenTelemetryExporter)
        
        # Test invalid exporter
        with pytest.raises(ValueError):
            create_exporter('invalid')


class TestSystemMetricsCollector:
    """Test system metrics collector."""
    
    @pytest.fixture
    def collector(self):
        """Create a metrics collector."""
        return MetricsCollector()
    
    @pytest.fixture
    def system_collector(self, collector):
        """Create a system metrics collector."""
        return SystemMetricsCollector(collector, interval=0.1)
    
    def test_system_collector_creation(self, system_collector):
        """Test system collector creation."""
        assert system_collector.collector is not None
        assert system_collector.interval == 0.1
        assert not system_collector.running
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_collect_metrics(self, mock_memory, mock_cpu, system_collector):
        """Test metrics collection."""
        # Mock psutil responses
        mock_cpu.return_value = 75.0
        mock_memory.return_value = MagicMock(percent=60.0, used=1024*1024*1024)
        
        # Collect once
        system_collector.collect_once()
        
        # Check metrics were updated
        assert system_collector.cpu_percent.get() == 75.0
        assert system_collector.memory_percent.get() == 60.0
        assert system_collector.memory_bytes.get() == 1024*1024*1024
    
    def test_start_stop(self, system_collector):
        """Test starting and stopping collector."""
        assert not system_collector.running
        
        system_collector.start()
        assert system_collector.running
        assert system_collector.thread is not None
        
        system_collector.stop()
        assert not system_collector.running


class TestHPCMetricsCollector:
    """Test HPC metrics collector."""
    
    @pytest.fixture
    def collector(self):
        """Create a metrics collector."""
        return MetricsCollector()
    
    @pytest.fixture
    def hpc_collector(self, collector):
        """Create an HPC metrics collector."""
        return HPCMetricsCollector(collector)
    
    def test_hpc_collector_creation(self, hpc_collector):
        """Test HPC collector creation."""
        assert hpc_collector.collector is not None
        
        # Should have created various metrics
        metric_names = hpc_collector.collector.get_metric_names()
        assert "hpc_messages_sent_total" in metric_names
        assert "hpc_work_items_completed_total" in metric_names
        assert "hpc_auth_attempts_total" in metric_names
    
    def test_message_metrics(self, hpc_collector):
        """Test message-related metrics."""
        hpc_collector.record_message_sent(1024)
        hpc_collector.record_message_received(512)
        hpc_collector.record_message_latency(0.1)
        
        assert hpc_collector.messages_sent.get() == 1
        assert hpc_collector.messages_received.get() == 1
        assert hpc_collector.message_latency.get_count() == 1
    
    def test_node_metrics(self, hpc_collector):
        """Test node-related metrics."""
        hpc_collector.update_node_metrics(10, 8)
        hpc_collector.record_heartbeat()
        
        assert hpc_collector.node_count.get() == 10
        assert hpc_collector.online_nodes.get() == 8
        assert hpc_collector.node_heartbeats.get() == 1
    
    def test_work_metrics(self, hpc_collector):
        """Test work-related metrics."""
        hpc_collector.record_work_submitted()
        hpc_collector.record_work_completed(5.0)
        hpc_collector.record_work_failed()
        hpc_collector.update_work_queue_size(3)
        
        assert hpc_collector.work_items_submitted.get() == 1
        assert hpc_collector.work_items_completed.get() == 1
        assert hpc_collector.work_items_failed.get() == 1
        assert hpc_collector.work_queue_size.get() == 3
        assert hpc_collector.work_processing_time.get_sum() == 5.0
    
    def test_auth_metrics(self, hpc_collector):
        """Test authentication metrics."""
        hpc_collector.record_auth_attempt(True)
        hpc_collector.record_auth_attempt(False)
        hpc_collector.record_auth_attempt(True)
        hpc_collector.update_token_count(5)
        
        assert hpc_collector.auth_attempts.get() == 3
        assert hpc_collector.auth_successes.get() == 2
        assert hpc_collector.auth_failures.get() == 1
        assert hpc_collector.token_count.get() == 5
    
    @patch('pynvml.nvmlInit')
    @patch('pynvml.nvmlDeviceGetCount')
    def test_gpu_metrics_available(self, mock_device_count, mock_init, hpc_collector):
        """Test GPU metrics when available."""
        mock_init.return_value = None
        mock_device_count.return_value = 2
        
        # Re-initialize GPU metrics
        hpc_collector._init_gpu_metrics()
        
        assert hpc_collector._has_gpu
        assert hpc_collector.gpu_count.get() == 2
    
    def test_gpu_metrics_unavailable(self, hpc_collector):
        """Test GPU metrics when unavailable."""
        # Should default to no GPUs
        assert not hpc_collector._has_gpu
        assert hpc_collector.gpu_count.get() == 0


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_system_collector(self):
        """Test system collector factory."""
        collector = MetricsCollector()
        system_collector = create_system_collector(collector)
        
        assert isinstance(system_collector, SystemMetricsCollector)
        assert system_collector.collector is collector
    
    def test_create_hpc_collector(self):
        """Test HPC collector factory."""
        collector = MetricsCollector()
        hpc_collector = create_hpc_collector(collector)
        
        assert isinstance(hpc_collector, HPCMetricsCollector)
        assert hpc_collector.collector is collector


if __name__ == "__main__":
    pytest.main([__file__])
