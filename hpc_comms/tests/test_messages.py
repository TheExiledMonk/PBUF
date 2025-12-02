"""Tests for message protocol and data structures."""

import pytest
import json
from datetime import datetime, timedelta

from hpc_comms.core.messages import (
    Message, WorkRequest, WorkResponse, NodeInfo, NodeCapabilities,
    PerformanceMetrics, WorkStatus, NodeStatus, ResourceRequirements,
    serialize_message, deserialize_message, create_work_request, create_work_response
)
from hpc_comms.core.errors import ValidationError


class TestMessage:
    """Test base Message class."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST",
            payload={"key": "value"}
        )
        
        assert msg.source_node == "node1"
        assert msg.target_node == "node2"
        assert msg.message_type == "TEST"
        assert msg.payload == {"key": "value"}
        assert msg.message_id is not None
        assert msg.timestamp is not None
    
    def test_message_expiration(self):
        """Test message expiration logic."""
        # Non-expiring message
        msg1 = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST"
        )
        assert not msg1.is_expired()
        
        # Expiring message
        msg2 = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST",
            ttl=timedelta(seconds=-1)  # Already expired
        )
        assert msg2.is_expired()
    
    def test_message_serialization(self):
        """Test message serialization and deserialization."""
        original = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST",
            payload={"key": "value"},
            correlation_id="corr123"
        )
        
        serialized = serialize_message(original)
        assert isinstance(serialized, str)
        
        deserialized = deserialize_message(serialized)
        assert isinstance(deserialized, Message)
        assert deserialized.source_node == original.source_node
        assert deserialized.target_node == original.target_node
        assert deserialized.message_type == original.message_type
        assert deserialized.payload == original.payload
        assert deserialized.correlation_id == original.correlation_id
        assert deserialized.message_id == original.message_id


class TestWorkRequest:
    """Test WorkRequest message."""
    
    def test_work_request_creation(self):
        """Test work request creation."""
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=2,
            max_execution_time=timedelta(minutes=10)
        )
        
        work_req = WorkRequest(
            source_node="controller",
            target_node="node1",
            model_configuration={"type": "lcdm"},
            parameters=[{"H0": 70, "Omega_m": 0.3}],
            requirements=requirements
        )
        
        assert work_req.message_type == "WORK_REQUEST"
        assert work_req.work_id is not None
        assert work_req.model_configuration == {"type": "lcdm"}
        assert len(work_req.parameters) == 1
        assert work_req.requirements.min_memory_mb == 1024
    
    def test_create_work_request_function(self):
        """Test work request creation helper."""
        requirements = ResourceRequirements(
            min_memory_mb=512,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=5)
        )
        
        work_req = create_work_request(
            source_node="controller",
            target_node="node1",
            model_config={"type": "pbuf"},
            parameters=[{"H0": 67}],
            requirements=requirements
        )
        
        assert isinstance(work_req, WorkRequest)
        assert work_req.source_node == "controller"
        assert work_req.target_node == "node1"
        assert work_req.model_configuration == {"type": "pbuf"}
        assert work_req.timeout == timedelta(minutes=30)  # Default timeout


class TestWorkResponse:
    """Test WorkResponse message."""
    
    def test_work_response_creation(self):
        """Test work response creation."""
        metrics = PerformanceMetrics(
            execution_time_ms=1000.0,
            backend_used="rocm",
            memory_peak_mb=512.0,
            cpu_utilization=0.8
        )
        
        work_resp = WorkResponse(
            source_node="node1",
            target_node="controller",
            work_id="work123",
            results=[{"params": {"H0": 70}, "chi2": 1.23}],
            performance_metrics=metrics,
            execution_time=timedelta(seconds=1),
            status=WorkStatus.COMPLETED
        )
        
        assert work_resp.message_type == "WORK_RESPONSE"
        assert work_resp.work_id == "work123"
        assert work_resp.status == WorkStatus.COMPLETED
        assert len(work_resp.results) == 1
        assert work_resp.performance_metrics.backend_used == "rocm"
    
    def test_work_response_with_error(self):
        """Test work response with error."""
        metrics = PerformanceMetrics(
            execution_time_ms=500.0,
            backend_used="cpu",
            memory_peak_mb=256.0,
            cpu_utilization=0.5
        )
        
        work_resp = WorkResponse(
            source_node="node1",
            target_node="controller",
            work_id="work123",
            results=[],
            performance_metrics=metrics,
            execution_time=timedelta(seconds=0.5),
            status=WorkStatus.FAILED,
            error_message="GPU out of memory"
        )
        
        assert work_resp.status == WorkStatus.FAILED
        assert work_resp.error_message == "GPU out of memory"
        assert len(work_resp.results) == 0


class TestNodeInfo:
    """Test NodeInfo and related classes."""
    
    def test_node_capabilities_validation(self):
        """Test node capabilities validation."""
        # Valid capabilities
        caps = NodeCapabilities(
            backend_type="rocm",
            device_count=2,
            memory_gb=64.0,
            supported_operations=["matrix_multiply", "chi2_calculation"],
            max_concurrent_tasks=4
        )
        assert caps.backend_type == "rocm"
        assert caps.device_count == 2
        
        # Invalid backend type
        with pytest.raises(ValueError):
            NodeCapabilities(
                backend_type="invalid",
                device_count=1,
                memory_gb=16.0
            )
    
    def test_node_info_health_check(self):
        """Test node health checking."""
        caps = NodeCapabilities(
            backend_type="cpu",
            device_count=1,
            memory_gb=16.0
        )
        
        node = NodeInfo(
            node_id="node1",
            endpoint="http://node1:8080",
            capabilities=caps,
            status=NodeStatus.ONLINE
        )
        
        # Healthy node
        assert node.is_healthy()
        
        # Offline node
        node.status = NodeStatus.OFFLINE
        assert not node.is_healthy()
        
        # Node with old heartbeat
        node.status = NodeStatus.ONLINE
        node.last_heartbeat = datetime.utcnow() - timedelta(minutes=10)
        assert not node.is_healthy(timeout=timedelta(minutes=2))
    
    def test_node_work_requirements_check(self):
        """Test node work requirements checking."""
        caps = NodeCapabilities(
            backend_type="rocm",
            device_count=2,
            memory_gb=64.0,
            max_concurrent_tasks=4
        )
        
        node = NodeInfo(
            node_id="node1",
            endpoint="http://node1:8080",
            capabilities=caps,
            status=NodeStatus.ONLINE
        )
        
        # Requirements that can be handled
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=2,
            max_execution_time=timedelta(minutes=10)
        )
        assert node.can_handle_work(requirements)
        
        # Requirements that exceed memory
        requirements_memory = ResourceRequirements(
            min_memory_mb=70000,  # 70GB > 64GB available
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=5)
        )
        assert not node.can_handle_work(requirements_memory)
        
        # Requirements that need GPU but node has no devices
        caps.device_count = 0
        requirements_gpu = ResourceRequirements(
            min_memory_mb=512,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=5),
            requires_gpu=True
        )
        assert not node.can_handle_work(requirements_gpu)
        
        # Offline node can't handle work
        caps.device_count = 2
        node.status = NodeStatus.OFFLINE
        assert not node.can_handle_work(requirements)


class TestResourceRequirements:
    """Test ResourceRequirements class."""
    
    def test_resource_requirements_creation(self):
        """Test resource requirements creation."""
        req = ResourceRequirements(
            min_memory_mb=2048,
            min_cpu_cores=4,
            preferred_backend="rocm",
            max_execution_time=timedelta(minutes=15),
            requires_gpu=True
        )
        
        assert req.min_memory_mb == 2048
        assert req.min_cpu_cores == 4
        assert req.preferred_backend == "rocm"
        assert req.max_execution_time == timedelta(minutes=15)
        assert req.requires_gpu is True


class TestPerformanceMetrics:
    """Test PerformanceMetrics class."""
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            execution_time_ms=1500.0,
            backend_used="numba",
            memory_peak_mb=1024.0,
            cpu_utilization=0.75,
            gpu_utilization=0.9,
            operations_per_second=1000.0,
            cache_hit_rate=0.85
        )
        
        assert metrics.execution_time_ms == 1500.0
        assert metrics.backend_used == "numba"
        assert metrics.memory_peak_mb == 1024.0
        assert metrics.cpu_utilization == 0.75
        assert metrics.gpu_utilization == 0.9
        assert metrics.operations_per_second == 1000.0
        assert metrics.cache_hit_rate == 0.85


class TestMessageFactory:
    """Test message factory functions."""
    
    def test_create_work_response_function(self):
        """Test work response creation helper."""
        metrics = PerformanceMetrics(
            execution_time_ms=800.0,
            backend_used="cpu",
            memory_peak_mb=512.0,
            cpu_utilization=0.6
        )
        
        work_resp = create_work_response(
            source_node="node1",
            target_node="controller",
            work_id="work123",
            results=[{"params": {"H0": 67}, "chi2": 0.98}],
            performance_metrics=metrics,
            execution_time=timedelta(seconds=0.8),
            status=WorkStatus.COMPLETED,
            correlation_id="corr123"
        )
        
        assert isinstance(work_resp, WorkResponse)
        assert work_resp.source_node == "node1"
        assert work_resp.target_node == "controller"
        assert work_resp.work_id == "work123"
        assert work_resp.correlation_id == "corr123"
        assert work_resp.status == WorkStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__])
