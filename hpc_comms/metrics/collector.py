"""Metrics collection and aggregation system."""

import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


@dataclass
class MetricValue:
    """A metric value with timestamp and labels."""
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels
        }


@dataclass
class HistogramBucket:
    """Histogram bucket with count and upper bound."""
    upper_bound: float
    count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'upper_bound': self.upper_bound,
            'count': self.count
        }


class Metric:
    """Base metric class."""
    
    def __init__(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.description = description
        self.metric_type = metric_type
        self.labels = labels or {}
        self.created_at = datetime.utcnow()
        self._lock = threading.Lock()
    
    def collect(self) -> List[MetricValue]:
        """Collect current metric values."""
        raise NotImplementedError
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'type': self.metric_type.value,
            'labels': self.labels,
            'created_at': self.created_at.isoformat()
        }


class Counter(Metric):
    """Counter metric that only goes up."""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
        initial_value: int = 0
    ):
        super().__init__(name, description, MetricType.COUNTER, labels)
        self._value = initial_value
    
    def inc(self, amount: int = 1) -> None:
        """Increment counter by amount."""
        with self._lock:
            self._value += amount
    
    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value
    
    def collect(self) -> List[MetricValue]:
        """Collect current metric value."""
        with self._lock:
            return [MetricValue(value=self._value, labels=self.labels)]
    
    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Gauge(Metric):
    """Gauge metric that can go up or down."""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
        initial_value: float = 0.0
    ):
        super().__init__(name, description, MetricType.GAUGE, labels)
        self._value = initial_value
    
    def set(self, value: float) -> None:
        """Set gauge value."""
        with self._lock:
            self._value = value
    
    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge by amount."""
        with self._lock:
            self._value += amount
    
    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge by amount."""
        with self._lock:
            self._value -= amount
    
    def get(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value
    
    def collect(self) -> List[MetricValue]:
        """Collect current metric value."""
        with self._lock:
            return [MetricValue(value=self._value, labels=self.labels)]


class Histogram(Metric):
    """Histogram metric with configurable buckets."""
    
    def __init__(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ):
        super().__init__(name, description, MetricType.HISTOGRAM, labels)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self._bucket_counts: Dict[float, int] = {bound: 0 for bound in self.buckets}
        self._bucket_counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0
    
    def observe(self, value: float) -> None:
        """Observe a value."""
        with self._lock:
            self._count += 1
            self._sum += value
            
            for bound in self.buckets + [float('inf')]:
                if value <= bound:
                    self._bucket_counts[bound] += 1
    
    def collect(self) -> List[MetricValue]:
        """Collect histogram metrics."""
        with self._lock:
            values = []
            
            # Bucket counts
            for bound, count in self._bucket_counts.items():
                bucket_labels = {**self.labels, 'le': str(bound)}
                values.append(MetricValue(value=count, labels=bucket_labels))
            
            # Sum and count
            sum_labels = {**self.labels, 'quantile': 'sum'}
            values.append(MetricValue(value=self._sum, labels=sum_labels))
            
            count_labels = {**self.labels, 'quantile': 'count'}
            values.append(MetricValue(value=self._count, labels=count_labels))
            
            return values
    
    def get_sum(self) -> float:
        """Get sum of observed values."""
        with self._lock:
            return self._sum
    
    def get_count(self) -> int:
        """Get count of observed values."""
        with self._lock:
            return self._count
    
    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            for bound in self._bucket_counts:
                self._bucket_counts[bound] = 0
            self._sum = 0.0
            self._count = 0


class Summary(Metric):
    """Summary metric with quantiles."""
    
    def __init__(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        max_age: timedelta = timedelta(minutes=10),
        age_buckets: int = 5,
        labels: Optional[Dict[str, str]] = None
    ):
        super().__init__(name, description, MetricType.SUMMARY, labels)
        self.quantiles = quantiles or [0.01, 0.05, 0.5, 0.9, 0.95, 0.99]
        self.max_age = max_age
        self.age_buckets = age_buckets
        
        # Time-based buckets for sliding window
        self._time_buckets: deque = deque(maxlen=age_buckets)
        self._current_bucket: List[float] = []
        self._last_rotation = time.time()
        
        self._sum = 0.0
        self._count = 0
    
    def observe(self, value: float) -> None:
        """Observe a value."""
        with self._lock:
            current_time = time.time()
            
            # Rotate buckets if needed
            if current_time - self._last_rotation >= self.max_age.total_seconds() / self.age_buckets:
                self._rotate_buckets()
                self._last_rotation = current_time
            
            self._current_bucket.append(value)
            self._sum += value
            self._count += 1
    
    def _rotate_buckets(self) -> None:
        """Rotate time buckets."""
        if self._current_bucket:
            self._time_buckets.append(self._current_bucket)
        self._current_bucket = []
    
    def _get_all_values(self) -> List[float]:
        """Get all values from all buckets."""
        values = list(self._current_bucket)
        for bucket in self._time_buckets:
            values.extend(bucket)
        return values
    
    def _calculate_quantile(self, values: List[float], quantile: float) -> float:
        """Calculate quantile from values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = quantile * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def collect(self) -> List[MetricValue]:
        """Collect summary metrics."""
        with self._lock:
            values = []
            all_values = self._get_all_values()
            
            # Quantiles
            for quantile in self.quantiles:
                quantile_value = self._calculate_quantile(all_values, quantile)
                quantile_labels = {**self.labels, 'quantile': str(quantile)}
                values.append(MetricValue(value=quantile_value, labels=quantile_labels))
            
            # Sum and count
            sum_labels = {**self.labels, 'quantile': 'sum'}
            values.append(MetricValue(value=self._sum, labels=sum_labels))
            
            count_labels = {**self.labels, 'quantile': 'count'}
            values.append(MetricValue(value=self._count, labels=count_labels))
            
            return values
    
    def get_sum(self) -> float:
        """Get sum of observed values."""
        with self._lock:
            return self._sum
    
    def get_count(self) -> int:
        """Get count of observed values."""
        with self._lock:
            return self._count


class Timer:
    """Timer for measuring duration."""
    
    def __init__(self, histogram: Histogram):
        self.histogram = histogram
        self.start_time: Optional[float] = None
    
    def start(self) -> 'Timer':
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self) -> float:
        """Stop the timer and record the duration."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        duration = time.time() - self.start_time
        self.histogram.observe(duration)
        self.start_time = None
        return duration
    
    def __enter__(self) -> 'Timer':
        """Context manager entry."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self.start_time is not None:
            self.stop()


class MetricsCollector:
    """Main metrics collector."""
    
    def __init__(self, default_labels: Optional[Dict[str, str]] = None):
        self.default_labels = default_labels or {}
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()
    
    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Counter:
        """Create or get a counter metric."""
        return self._get_or_create_metric(
            name, description, MetricType.COUNTER, labels
        )
    
    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Gauge:
        """Create or get a gauge metric."""
        return self._get_or_create_metric(
            name, description, MetricType.GAUGE, labels
        )
    
    def histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Histogram:
        """Create or get a histogram metric."""
        metric = self._get_or_create_metric(
            name, description, MetricType.HISTOGRAM, labels
        )
        if isinstance(metric, Histogram) and buckets:
            metric.buckets = buckets
        return metric
    
    def summary(
        self,
        name: str,
        description: str,
        quantiles: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Summary:
        """Create or get a summary metric."""
        metric = self._get_or_create_metric(
            name, description, MetricType.SUMMARY, labels
        )
        if isinstance(metric, Summary) and quantiles:
            metric.quantiles = quantiles
        return metric
    
    def timer(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> Timer:
        """Create a timer metric."""
        histogram = self.histogram(name, description, buckets, labels)
        return Timer(histogram)
    
    def _get_or_create_metric(
        self,
        name: str,
        description: str,
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ) -> Metric:
        """Get existing metric or create new one."""
        full_labels = {**self.default_labels, **(labels or {})}
        key = f"{name}:{json.dumps(full_labels, sort_keys=True)}"
        
        with self._lock:
            if key not in self._metrics:
                if metric_type == MetricType.COUNTER:
                    metric = Counter(name, description, full_labels)
                elif metric_type == MetricType.GAUGE:
                    metric = Gauge(name, description, full_labels)
                elif metric_type == MetricType.HISTOGRAM:
                    metric = Histogram(name, description, labels=full_labels)
                elif metric_type == MetricType.SUMMARY:
                    metric = Summary(name, description, labels=full_labels)
                else:
                    raise ValueError(f"Unsupported metric type: {metric_type}")
                
                self._metrics[key] = metric
            
            return self._metrics[key]
    
    def collect_all(self) -> Dict[str, List[MetricValue]]:
        """Collect all metric values."""
        with self._lock:
            result = {}
            for key, metric in self._metrics.items():
                values = metric.collect()
                if values:
                    result[key] = values
            return result
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names."""
        with self._lock:
            return list(set(metric.name for metric in self._metrics.values()))
    
    def remove_metric(self, name: str, labels: Optional[Dict[str, str]] = None) -> bool:
        """Remove a metric."""
        full_labels = {**self.default_labels, **(labels or {})}
        key = f"{name}:{json.dumps(full_labels, sort_keys=True)}"
        
        with self._lock:
            if key in self._metrics:
                del self._metrics[key]
                return True
            return False
    
    def clear_all(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary."""
        result = {}
        collected = self.collect_all()
        
        for key, values in collected.items():
            metric_name = key.split(':')[0]
            if metric_name not in result:
                result[metric_name] = []
            
            for value in values:
                metric_data = {
                    'value': value.value,
                    'timestamp': value.timestamp.isoformat(),
                    'labels': value.labels
                }
                result[metric_name].append(metric_data)
        
        return result
