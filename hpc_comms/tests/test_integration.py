"""Integration tests for the complete HPC communication module."""

import pytest
import asyncio
from datetime import datetime, timedelta

from hpc_comms.core.protocol import HPCProtocol, RetryPolicy, WorkQueue
from hpc_comms.core.messages import (
    Message, WorkRequest, WorkResponse, NodeInfo, NodeCapabilities,
    PerformanceMetrics, WorkStatus, NodeStatus, ResourceRequirements
)
from hpc_comms.core.registry import NodeRegistry, LoadBalancer, WorkTracker
from hpc_comms.transports.http_transport import HTTPTransport, create_http_transport
from hpc_comms.auth.providers import TokenAuthProvider, create_token_auth_provider


class TestHPCProtocolIntegration:
    """Test HPC protocol integration."""
    
    @pytest.fixture
    def protocol(self):
        """Create a test protocol with mock transport."""
        from hpc_comms.tests.test_transport import MockTransport
        transport = MockTransport()
        return HPCProtocol(transport)
    
    @pytest.fixture
    def sample_node_info(self):
        """Create sample node info."""
        caps = NodeCapabilities(
            backend_type="rocm",
            device_count=2,
            memory_gb=64.0,
            max_concurrent_tasks=4,
            performance_profile={"score": 100}
        )
        return NodeInfo(
            node_id="node1",
            endpoint="http://node1:8080",
            capabilities=caps,
            status=NodeStatus.ONLINE
        )
    
    @pytest.mark.asyncio
    async def test_message_routing(self, protocol):
        """Test message routing through protocol."""
        received_messages = []
        
        async def test_handler(connection, message):
            received_messages.append(message)
            return Message(
                source_node="controller",
                target_node=message.source_node,
                message_type="RESPONSE",
                payload={"status": "received"}
            )
        
        protocol.register_handler("TEST", test_handler)
        
        # Create mock connection and message
        from hpc_comms.tests.test_transport import MockConnection
        conn = MockConnection("test://endpoint")
        message = Message(
            source_node="node1",
            target_node="controller",
            message_type="TEST",
            payload={"data": "test"}
        )
        
        # Route message
        response = await protocol.router.route_message(conn, message)
        
        assert len(received_messages) == 1
        assert received_messages[0].message_type == "TEST"
        assert response is not None
        assert response.message_type == "RESPONSE"
    
    @pytest.mark.asyncio
    async def test_node_registration_flow(self, protocol, sample_node_info):
        """Test complete node registration flow."""
        # Register node
        await protocol.register_node("http://controller:8080", sample_node_info)
        
        # Send heartbeat
        await protocol.send_heartbeat("http://controller:8080", sample_node_info)
        
        # Should not raise any errors
        assert True
    
    @pytest.mark.asyncio
    async def test_work_request_response_flow(self, protocol, sample_node_info):
        """Test complete work request/response flow."""
        # Create work request
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=2,
            max_execution_time=timedelta(minutes=10)
        )
        
        work_request = WorkRequest(
            source_node="controller",
            target_node="node1",
            model_configuration={"type": "lcdm"},
            parameters=[{"H0": 70, "Omega_m": 0.3}],
            requirements=requirements
        )
        
        # Submit work and get response
        work_response = WorkResponse(
            source_node="node1",
            target_node="controller",
            work_id=work_request.work_id,
            results=[{"params": {"H0": 70}, "chi2": 1.23}],
            performance_metrics=PerformanceMetrics(
                execution_time_ms=1500.0,
                backend_used="rocm",
                memory_peak_mb=512.0,
                cpu_utilization=0.8
            ),
            execution_time=timedelta(seconds=1.5),
            status=WorkStatus.COMPLETED
        )
        
        # Should not raise any errors
        await protocol.submit_work_response("http://controller:8080", work_response)
        assert True


class TestNodeRegistryIntegration:
    """Test node registry integration with load balancing."""
    
    @pytest.fixture
    def registry(self):
        """Create test registry."""
        return NodeRegistry(heartbeat_timeout=timedelta(seconds=1))
    
    @pytest.fixture
    def load_balancer(self):
        """Create test load balancer."""
        return LoadBalancer(strategy="performance_based")
    
    @pytest.fixture
    def work_tracker(self):
        """Create test work tracker."""
        return WorkTracker()
    
    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes for testing."""
        nodes = []
        
        # High-performance node
        caps1 = NodeCapabilities(
            backend_type="rocm",
            device_count=4,
            memory_gb=128.0,
            max_concurrent_tasks=8,
            performance_profile={"score": 200}
        )
        nodes.append(NodeInfo(
            node_id="gpu_node",
            endpoint="http://gpu_node:8080",
            capabilities=caps1,
            status=NodeStatus.ONLINE
        ))
        
        # Medium-performance node
        caps2 = NodeCapabilities(
            backend_type="numba",
            device_count=1,
            memory_gb=64.0,
            max_concurrent_tasks=4,
            performance_profile={"score": 100}
        )
        nodes.append(NodeInfo(
            node_id="numba_node",
            endpoint="http://numba_node:8080",
            capabilities=caps2,
            status=NodeStatus.ONLINE
        ))
        
        # CPU-only node
        caps3 = NodeCapabilities(
            backend_type="cpu",
            device_count=1,
            memory_gb=32.0,
            max_concurrent_tasks=2,
            performance_profile={"score": 50}
        )
        nodes.append(NodeInfo(
            node_id="cpu_node",
            endpoint="http://cpu_node:8080",
            capabilities=caps3,
            status=NodeStatus.ONLINE
        ))
        
        return nodes
    
    @pytest.mark.asyncio
    async def test_complete_work_distribution(self, registry, load_balancer, work_tracker, sample_nodes):
        """Test complete work distribution workflow."""
        # Register all nodes
        for node in sample_nodes:
            await registry.register_node(node)
        
        # Create work requirements
        requirements = ResourceRequirements(
            min_memory_mb=2048,
            min_cpu_cores=2,
            max_execution_time=timedelta(minutes=15),
            requires_gpu=True
        )
        
        # Find suitable nodes (only GPU node can handle this)
        suitable_nodes = await registry.find_nodes_for_work(requirements)
        assert len(suitable_nodes) == 1
        assert suitable_nodes[0].node_id == "gpu_node"
        
        # Select best node
        selected_node = await load_balancer.select_node(suitable_nodes, requirements)
        assert selected_node.node_id == "gpu_node"
        
        # Assign work
        work_id = "work_123"
        await work_tracker.assign_work(work_id, selected_node.node_id)
        
        # Verify assignment
        assigned_node = await work_tracker.get_node_for_work(work_id)
        assert assigned_node == "gpu_node"
        
        # Check node work count
        work_count = await work_tracker.get_node_work_count("gpu_node")
        assert work_count == 1
        
        # Complete work
        completed_node = await work_tracker.complete_work(work_id)
        assert completed_node == "gpu_node"
        
        # Verify completion
        work_count = await work_tracker.get_node_work_count("gpu_node")
        assert work_count == 0
    
    @pytest.mark.asyncio
    async def test_load_balancing_across_nodes(self, registry, load_balancer, work_tracker, sample_nodes):
        """Test load balancing across multiple nodes."""
        # Register all nodes
        for node in sample_nodes:
            await registry.register_node(node)
        
        # Create work requirements that all nodes can handle
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=5)
        )
        
        # Find suitable nodes
        suitable_nodes = await registry.find_nodes_for_work(requirements)
        assert len(suitable_nodes) == 3
        
        # Use least_loaded strategy to ensure distribution
        load_balancer.strategy = "least_loaded"
        
        # Distribute work
        work_ids = ["work_1", "work_2", "work_3", "work_4", "work_5"]
        
        for work_id in work_ids:
            selected_node = await load_balancer.select_node(suitable_nodes, requirements)
            await work_tracker.assign_work(work_id, selected_node.node_id)
            load_balancer.increment_load(selected_node.node_id)
        
        # Check load distribution
        loads = load_balancer.get_loads()
        assert len(loads) == 3  # All three nodes should have some work
        assert loads["gpu_node"] > 0
        assert loads["numba_node"] > 0
        assert loads["cpu_node"] > 0
        
        # GPU node should have more work due to higher performance
        assert loads["gpu_node"] >= loads["numba_node"]
        assert loads["numba_node"] >= loads["cpu_node"]


class TestWorkQueueIntegration:
    """Test work queue integration."""
    
    @pytest.mark.asyncio
    async def test_work_queue_operations(self):
        """Test work queue operations."""
        queue = WorkQueue(max_size=5)
        
        # Create work requests
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=5)
        )
        
        work_requests = []
        for i in range(3):
            work_req = WorkRequest(
                source_node="controller",
                target_node="node1",
                model_configuration={"type": "lcdm"},
                parameters=[{"H0": 70 + i}],
                requirements=requirements
            )
            work_requests.append(work_req)
        
        # Add work to queue
        for work_req in work_requests:
            await queue.put(work_req)
        
        # Check queue size
        assert await queue.size() == 3
        assert not await queue.is_empty()
        
        # Get work from queue
        retrieved_work = []
        for _ in range(3):
            work = await queue.get()
            if work:
                retrieved_work.append(work)
        
        assert len(retrieved_work) == 3
        assert await queue.is_empty()
        
        # Verify work order (FIFO)
        for i, work in enumerate(retrieved_work):
            assert work.parameters[0]["H0"] == 70 + i


class TestAuthenticationIntegration:
    """Test authentication integration."""
    
    @pytest.mark.asyncio
    async def test_token_auth_flow(self):
        """Test complete token authentication flow."""
        # Create auth provider
        auth_provider = create_token_auth_provider(
            secret_key="test-secret-key",
            use_jwt=False,
            token_expiry=timedelta(hours=1)
        )
        
        # Authenticate node
        credentials = {
            "node_id": "node1",
            "capabilities": {"backend_type": "rocm", "device_count": 2}
        }
        
        token = await auth_provider.authenticate(credentials)
        assert token is not None
        
        # Validate token
        payload = await auth_provider.validate_token(token)
        assert payload is not None
        assert payload["node_id"] == "node1"
        assert payload["type"] == "compute_node"
        
        # Create new token for controller
        controller_credentials = {
            "node_id": "controller"
        }
        
        controller_token = await auth_provider.authenticate(controller_credentials)
        controller_payload = await auth_provider.validate_token(controller_token)
        
        assert controller_payload["node_id"] == "controller"
        assert controller_payload["type"] == "controller"


class TestErrorHandlingIntegration:
    """Test error handling integration."""
    
    @pytest.mark.asyncio
    async def test_retry_policy_integration(self):
        """Test retry policy integration."""
        from hpc_comms.tests.test_transport import MockTransport, MockConnection
        
        # Create transport that fails initially
        transport = MockTransport()
        call_count = 0
        
        original_connect = transport.connect
        async def failing_connect(endpoint, timeout=None):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return await original_connect(endpoint, timeout)
        
        transport.connect = failing_connect
        
        # Create protocol with retry policy
        retry_policy = RetryPolicy(
            max_attempts=5,
            base_delay=timedelta(milliseconds=10),
            max_delay=timedelta(milliseconds=100)
        )
        protocol = HPCProtocol(transport, retry_policy=retry_policy)
        
        # Send message - should succeed after retries
        message = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST"
        )
        
        await protocol.send_message("test://endpoint", message)
        
        # Should have been called 3 times (2 failures + 1 success)
        assert call_count == 3


class TestHTTPTransportIntegration:
    """Test HTTP transport integration."""
    
    def test_http_transport_creation(self):
        """Test HTTP transport creation."""
        # Basic HTTP transport
        transport = create_http_transport(
            ssl_enabled=False,
            timeout=timedelta(seconds=30)
        )
        
        assert transport.get_scheme() == "http"
        assert transport.config["timeout"] == 30.0
        assert not transport.config["ssl"]["enabled"]
        
        # HTTPS transport with mock cert files (skip actual loading)
        try:
            transport_https = create_http_transport(
                ssl_enabled=True,
                cert_file="/path/to/cert.pem",
                key_file="/path/to/key.pem"
            )
            assert transport_https.get_scheme() == "https"
            assert transport_https.config["ssl"]["enabled"]
        except FileNotFoundError:
            # Skip if cert files don't exist
            pass


if __name__ == "__main__":
    pytest.main([__file__])
