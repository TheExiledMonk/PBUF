"""Metrics registry for managing multiple collectors."""

import threading
from typing import Any, Dict, List, Optional, Set
from .collector import MetricsCollector, MetricType


class MetricsRegistry:
    """Registry for managing multiple metrics collectors."""
    
    def __init__(self):
        self._collectors: Dict[str, MetricsCollector] = {}
        self._global_labels: Dict[str, str] = {}
        self._lock = threading.Lock()
        
        # Default collector
        self.default = self.create_collector("default")
    
    def create_collector(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> MetricsCollector:
        """Create a new metrics collector."""
        with self._lock:
            if name in self._collectors:
                raise ValueError(f"Collector '{name}' already exists")
            
            # Merge with global labels
            full_labels = {**self._global_labels, **(labels or {})}
            collector = MetricsCollector(full_labels)
            self._collectors[name] = collector
            
            return collector
    
    def get_collector(self, name: str) -> Optional[MetricsCollector]:
        """Get a collector by name."""
        with self._lock:
            return self._collectors.get(name)
    
    def remove_collector(self, name: str) -> bool:
        """Remove a collector."""
        with self._lock:
            if name in self._collectors:
                del self._collectors[name]
                return True
            return False
    
    def list_collectors(self) -> List[str]:
        """List all collector names."""
        with self._lock:
            return list(self._collectors.keys())
    
    def set_global_label(self, key: str, value: str) -> None:
        """Set a global label that applies to all collectors."""
        with self._lock:
            self._global_labels[key] = value
            # Update existing collectors
            for collector in self._collectors.values():
                collector.default_labels[key] = value
    
    def remove_global_label(self, key: str) -> None:
        """Remove a global label."""
        with self._lock:
            self._global_labels.pop(key, None)
            # Update existing collectors
            for collector in self._collectors.values():
                collector.default_labels.pop(key, None)
    
    def get_global_labels(self) -> Dict[str, str]:
        """Get all global labels."""
        with self._lock:
            return self._global_labels.copy()
    
    def collect_all(self) -> Dict[str, Dict[str, List[Any]]]:
        """Collect metrics from all collectors."""
        with self._lock:
            result = {}
            for name, collector in self._collectors.items():
                result[name] = collector.collect_all()
            return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        with self._lock:
            result = {
                'global_labels': self._global_labels,
                'collectors': {}
            }
            
            for name, collector in self._collectors.items():
                result['collectors'][name] = collector.to_dict()
            
            return result
    
    def clear_all(self) -> None:
        """Clear all metrics from all collectors."""
        with self._lock:
            for collector in self._collectors.values():
                collector.clear_all()
    
    def get_metric_names(self) -> Set[str]:
        """Get all unique metric names across all collectors."""
        with self._lock:
            names = set()
            for collector in self._collectors.values():
                names.update(collector.get_metric_names())
            return names
    
    def get_metric_by_name(self, metric_name: str) -> Dict[str, List[Any]]:
        """Get all metrics with a specific name across all collectors."""
        with self._lock:
            result = {}
            for collector_name, collector in self._collectors.items():
                collected = collector.collect_all()
                for key, values in collected.items():
                    if key.startswith(f"{metric_name}:"):
                        if collector_name not in result:
                            result[collector_name] = []
                        result[collector_name].extend(values)
            return result


# Global registry instance
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get the global metrics registry."""
    return _registry


def create_collector(
    name: str,
    labels: Optional[Dict[str, str]] = None
) -> MetricsCollector:
    """Create a collector in the global registry."""
    return _registry.create_collector(name, labels)


def get_collector(name: str) -> Optional[MetricsCollector]:
    """Get a collector from the global registry."""
    return _registry.get_collector(name)


# Convenience functions that use the default collector
def counter(name: str, description: str, labels: Optional[Dict[str, str]] = None):
    """Create a counter in the default collector."""
    return _registry.default.counter(name, description, labels)


def gauge(name: str, description: str, labels: Optional[Dict[str, str]] = None):
    """Create a gauge in the default collector."""
    return _registry.default.gauge(name, description, labels)


def histogram(name: str, description: str, buckets: Optional[List[float]] = None, labels: Optional[Dict[str, str]] = None):
    """Create a histogram in the default collector."""
    return _registry.default.histogram(name, description, buckets, labels)


def summary(name: str, description: str, quantiles: Optional[List[float]] = None, labels: Optional[Dict[str, str]] = None):
    """Create a summary in the default collector."""
    return _registry.default.summary(name, description, quantiles, labels)


def timer(name: str, description: str, buckets: Optional[List[float]] = None, labels: Optional[Dict[str, str]] = None):
    """Create a timer in the default collector."""
    return _registry.default.timer(name, description, buckets, labels)


def collect_all() -> Dict[str, Dict[str, List[Any]]]:
    """Collect all metrics from the global registry."""
    return _registry.collect_all()


def to_dict() -> Dict[str, Any]:
    """Convert global registry to dictionary."""
    return _registry.to_dict()
