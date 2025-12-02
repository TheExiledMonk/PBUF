"""Metrics exporters for different formats."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from .collector import MetricValue, MetricType


class MetricsExporter(ABC):
    """Base class for metrics exporters."""
    
    @abstractmethod
    def export(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Export metrics to string format."""
        pass
    
    @abstractmethod
    def content_type(self) -> str:
        """Get the content type for this export format."""
        pass


class TextExporter(MetricsExporter):
    """Prometheus-style text exporter."""
    
    def __init__(self, include_timestamp: bool = False):
        self.include_timestamp = include_timestamp
    
    def export(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        
        for metric_key, values in metrics.items():
            if not values:
                continue
            
            # Extract metric name from key (remove labels part)
            metric_name = metric_key.split(':')[0]
            
            # Add metadata comment
            lines.append(f"# HELP {metric_name} {values[0].labels.get('_description', '')}")
            lines.append(f"# TYPE {metric_name} {values[0].labels.get('_type', 'unknown')}")
            
            # Add metric values
            for value in values:
                # Build label string
                label_parts = []
                for label_key, label_value in value.labels.items():
                    if not label_key.startswith('_'):  # Skip internal labels
                        label_parts.append(f'{label_key}="{label_value}"')
                
                label_str = "{" + ",".join(label_parts) + "}" if label_parts else ""
                
                # Build metric line
                line = f"{metric_name}{label_str} {value.value}"
                
                if self.include_timestamp:
                    line += f" {int(value.timestamp.timestamp())}"
                
                lines.append(line)
            
            lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)
    
    def content_type(self) -> str:
        """Get content type."""
        return "text/plain; version=0.0.4; charset=utf-8"


class JsonExporter(MetricsExporter):
    """JSON exporter."""
    
    def __init__(self, pretty: bool = True):
        self.pretty = pretty
    
    def export(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Export metrics in JSON format."""
        result = {}
        
        for metric_key, values in metrics.items():
            if not values:
                continue
            
            metric_name = metric_key.split(':')[0]
            
            if metric_name not in result:
                result[metric_name] = []
            
            for value in values:
                metric_data = {
                    'value': value.value,
                    'timestamp': value.timestamp.isoformat(),
                    'labels': value.labels
                }
                result[metric_name].append(metric_data)
        
        if self.pretty:
            return json.dumps(result, indent=2)
        else:
            return json.dumps(result)
    
    def content_type(self) -> str:
        """Get content type."""
        return "application/json"


class InfluxExporter(MetricsExporter):
    """InfluxDB line protocol exporter."""
    
    def export(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Export metrics in InfluxDB line protocol format."""
        lines = []
        
        for metric_key, values in metrics.items():
            if not values:
                continue
            
            metric_name = metric_key.split(':')[0]
            
            for value in values:
                # Build measurement name
                measurement = metric_name.replace('.', '_').replace('-', '_')
                
                # Build tags (labels)
                tags = []
                for label_key, label_value in value.labels.items():
                    if not label_key.startswith('_'):  # Skip internal labels
                        tag_key = label_key.replace('.', '_').replace('-', '_')
                        tag_value = str(label_value).replace(' ', '\\ ').replace(',', '\\,')
                        tags.append(f"{tag_key}={tag_value}")
                
                tag_str = ",".join(tags)
                
                # Build fields
                fields = f"value={value.value}"
                
                # Build line
                if tag_str:
                    line = f"{measurement},{tag_str} {fields} {int(value.timestamp.timestamp())}000000000"
                else:
                    line = f"{measurement} {fields} {int(value.timestamp.timestamp())}000000000"
                
                lines.append(line)
        
        return "\n".join(lines)
    
    def content_type(self) -> str:
        """Get content type."""
        return "text/plain; charset=utf-8"


class StatsdExporter(MetricsExporter):
    """StatsD exporter."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
    
    def export(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Export metrics in StatsD format."""
        lines = []
        
        for metric_key, values in metrics.items():
            if not values:
                continue
            
            metric_name = metric_key.split(':')[0]
            
            for value in values:
                # Build metric name with prefix
                full_name = f"{self.prefix}.{metric_name}" if self.prefix else metric_name
                
                # Add tags as part of metric name (for dogstatsd)
                tags = []
                for label_key, label_value in value.labels.items():
                    if not label_key.startswith('_'):  # Skip internal labels
                        tags.append(f"{label_key}:{label_value}")
                
                if tags:
                    tag_str = ",".join(f"# {tag}" for tag in tags)
                    full_name_with_tags = f"{full_name},{tag_str}"
                else:
                    full_name_with_tags = full_name
                
                # Determine metric type
                metric_type = value.labels.get('_type', 'gauge')
                statsd_type = self._map_metric_type(metric_type)
                
                # Build line
                line = f"{full_name_with_tags}:{value.value}|{statsd_type}"
                lines.append(line)
        
        return "\n".join(lines)
    
    def _map_metric_type(self, metric_type: str) -> str:
        """Map metric type to StatsD type."""
        mapping = {
            'counter': 'c',
            'gauge': 'g',
            'histogram': 'ms',  # Treat as timing
            'summary': 'ms',
            'timer': 'ms'
        }
        return mapping.get(metric_type, 'g')
    
    def content_type(self) -> str:
        """Get content type."""
        return "text/plain; charset=utf-8"


class GraphiteExporter(MetricsExporter):
    """Graphite plaintext exporter."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
    
    def export(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Export metrics in Graphite plaintext format."""
        lines = []
        
        for metric_key, values in metrics.items():
            if not values:
                continue
            
            metric_name = metric_key.split(':')[0]
            
            for value in values:
                # Build metric path
                path_parts = []
                
                if self.prefix:
                    path_parts.append(self.prefix)
                
                path_parts.append(metric_name.replace('.', '_').replace('-', '_'))
                
                # Add labels as path components
                for label_key, label_value in value.labels.items():
                    if not label_key.startswith('_'):  # Skip internal labels
                        path_key = label_key.replace('.', '_').replace('-', '_')
                        path_value = str(label_value).replace('.', '_').replace('-', '_')
                        path_parts.append(f"{path_key}.{path_value}")
                
                metric_path = ".".join(path_parts)
                
                # Build line
                line = f"{metric_path} {value.value} {int(value.timestamp.timestamp())}"
                lines.append(line)
        
        return "\n".join(lines)
    
    def content_type(self) -> str:
        """Get content type."""
        return "text/plain; charset=utf-8"


class OpenTelemetryExporter(MetricsExporter):
    """OpenTelemetry JSON exporter."""
    
    def export(self, metrics: Dict[str, List[MetricValue]]) -> str:
        """Export metrics in OpenTelemetry format."""
        resource_metrics = {
            "resource": {
                "attributes": {
                    "service.name": "hpc-comms"
                }
            },
            "scopeMetrics": []
        }
        
        # Group metrics by name
        metrics_by_name = {}
        for metric_key, values in metrics.items():
            if not values:
                continue
            
            metric_name = metric_key.split(':')[0]
            if metric_name not in metrics_by_name:
                metrics_by_name[metric_name] = []
            metrics_by_name[metric_name].extend(values)
        
        for metric_name, values in metrics_by_name.items():
            # Determine metric type
            metric_type = values[0].labels.get('_type', 'gauge')
            
            # Build metric data
            metric_data = {
                "name": metric_name,
                "description": values[0].labels.get('_description', ''),
                "unit": "",
                "data": self._build_metric_data(metric_type, values)
            }
            
            scope_metric = {
                "scope": {
                    "name": "hpc-comms",
                    "version": "1.0.0"
                },
                "metrics": [metric_data]
            }
            
            resource_metrics["scopeMetrics"].append(scope_metric)
        
        return json.dumps(resource_metrics, indent=2)
    
    def _build_metric_data(self, metric_type: str, values: List[MetricValue]) -> Dict[str, Any]:
        """Build metric data based on type."""
        if metric_type == 'gauge' or metric_type == 'counter':
            return {
                "gauge": {
                    "dataPoints": [
                        {
                            "attributes": value.labels,
                            "timeUnixNano": int(value.timestamp.timestamp() * 1e9),
                            "value": value.value
                        }
                        for value in values
                    ]
                }
            }
        elif metric_type == 'histogram':
            # For histograms, we need to aggregate buckets
            return {
                "histogram": {
                    "dataPoints": [
                        {
                            "attributes": value.labels,
                            "timeUnixNano": int(value.timestamp.timestamp() * 1e9),
                            "count": value.value,
                            "sum": 0.0,  # Would need to be calculated separately
                            "bucketCounts": [],
                            "explicitBounds": []
                        }
                        for value in values if value.labels.get('quantile') == 'count'
                    ]
                }
            }
        else:
            # Default to gauge
            return {
                "gauge": {
                    "dataPoints": [
                        {
                            "attributes": value.labels,
                            "timeUnixNano": int(value.timestamp.timestamp() * 1e9),
                            "value": value.value
                        }
                        for value in values
                    ]
                }
            }
    
    def content_type(self) -> str:
        """Get content type."""
        return "application/json"


# Exporter factory
def create_exporter(format_name: str, **kwargs) -> MetricsExporter:
    """Create an exporter by format name."""
    exporters = {
        'text': TextExporter,
        'prometheus': TextExporter,
        'json': JsonExporter,
        'influx': InfluxExporter,
        'influxdb': InfluxExporter,
        'statsd': StatsdExporter,
        'graphite': GraphiteExporter,
        'opentelemetry': OpenTelemetryExporter
    }
    
    exporter_class = exporters.get(format_name.lower())
    if not exporter_class:
        raise ValueError(f"Unknown exporter format: {format_name}")
    
    return exporter_class(**kwargs)
