"""Tests for node registry and capabilities management."""

import asyncio
import pytest
from datetime import datetime, timedelta

from hpc_comms.core.registry import NodeRegistry, LoadBalancer, WorkTracker
from hpc_comms.core.messages import (
    NodeInfo, NodeCapabilities, NodeStatus, ResourceRequirements
)
from hpc_comms.core.errors import NodeError, ResourceError


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestNodeRegistry:
    """Test node registry functionality."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return NodeRegistry(heartbeat_timeout=timedelta(seconds=1))
    
    @pytest.fixture
    def sample_node(self):
        """Create a sample node for testing."""
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
    async def test_register_node(self, registry, sample_node):
        """Test node registration."""
        await registry.register_node(sample_node)
        
        retrieved = await registry.get_node("node1")
        assert retrieved is not None
        assert retrieved.node_id == "node1"
        assert retrieved.status == NodeStatus.ONLINE
        assert retrieved.capabilities.backend_type == "rocm"
    
    @pytest.mark.asyncio
    async def test_update_heartbeat(self, registry, sample_node):
        """Test heartbeat update."""
        await registry.register_node(sample_node)
        
        # Store original heartbeat
        original_heartbeat = (await registry.get_node("node1")).last_heartbeat
        
        # Small delay to ensure timestamp difference
        await asyncio.sleep(0.01)
        
        # Update heartbeat
        updated_node = NodeInfo(
            node_id="node1",
            endpoint="http://node1:8080",
            capabilities=sample_node.capabilities,
            status=NodeStatus.BUSY,
            metadata={"load": 0.8}
        )
        await registry.update_heartbeat("node1", updated_node)
        
        retrieved = await registry.get_node("node1")
        assert retrieved.status == NodeStatus.BUSY
        assert retrieved.metadata["load"] == 0.8
        assert retrieved.last_heartbeat >= original_heartbeat  # Should be updated
    
    @pytest.mark.asyncio
    async def test_deregister_node(self, registry, sample_node):
        """Test node deregistration."""
        await registry.register_node(sample_node)
        
        assert await registry.get_node("node1") is not None
        
        await registry.deregister_node("node1")
        
        assert await registry.get_node("node1") is None
    
    @pytest.mark.asyncio
    async def test_get_all_nodes(self, registry, sample_node):
        """Test getting all nodes."""
        await registry.register_node(sample_node)
        
        # Add another node
        caps2 = NodeCapabilities(
            backend_type="cpu",
            device_count=1,
            memory_gb=32.0
        )
        node2 = NodeInfo(
            node_id="node2",
            endpoint="http://node2:8080",
            capabilities=caps2,
            status=NodeStatus.ONLINE
        )
        await registry.register_node(node2)
        
        all_nodes = await registry.get_all_nodes()
        assert len(all_nodes) == 2
        
        node_ids = {node.node_id for node in all_nodes}
        assert node_ids == {"node1", "node2"}
    
    @pytest.mark.asyncio
    async def test_get_online_nodes(self, registry, sample_node):
        """Test getting online nodes."""
        await registry.register_node(sample_node)
        
        # Add another node
        caps2 = NodeCapabilities(
            backend_type="cpu",
            device_count=1,
            memory_gb=32.0
        )
        node2 = NodeInfo(
            node_id="node2",
            endpoint="http://node2:8080",
            capabilities=caps2,
            status=NodeStatus.OFFLINE  # This will be set to ONLINE by register_node
        )
        await registry.register_node(node2)
        
        # Manually set node2 to OFFLINE after registration
        async with registry._lock:
            registry.nodes["node2"].status = NodeStatus.OFFLINE
        
        online_nodes = await registry.get_online_nodes()
        assert len(online_nodes) == 1
        assert online_nodes[0].node_id == "node1"
    
    @pytest.mark.asyncio
    async def test_find_nodes_for_work(self, registry, sample_node):
        """Test finding nodes for specific work requirements."""
        await registry.register_node(sample_node)
        
        # Add CPU-only node
        caps2 = NodeCapabilities(
            backend_type="cpu",
            device_count=1,
            memory_gb=16.0,
            max_concurrent_tasks=2,
            performance_profile={"score": 50}
        )
        node2 = NodeInfo(
            node_id="node2",
            endpoint="http://node2:8080",
            capabilities=caps2,
            status=NodeStatus.ONLINE
        )
        await registry.register_node(node2)
        
        # Requirements that both nodes can handle
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        suitable_nodes = await registry.find_nodes_for_work(requirements)
        assert len(suitable_nodes) == 2
        
        # Should be sorted by performance (node1 first)
        assert suitable_nodes[0].node_id == "node1"
        assert suitable_nodes[1].node_id == "node2"
        
        # Requirements that only node1 can handle (more memory)
        requirements_memory = ResourceRequirements(
            min_memory_mb=20000,  # 20GB
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        suitable_nodes = await registry.find_nodes_for_work(requirements_memory)
        assert len(suitable_nodes) == 1
        assert suitable_nodes[0].node_id == "node1"
    
    @pytest.mark.asyncio
    async def test_get_best_node(self, registry, sample_node):
        """Test getting best node for work."""
        await registry.register_node(sample_node)
        
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        best_node = await registry.get_best_node(requirements)
        assert best_node is not None
        assert best_node.node_id == "node1"
        
        # No suitable nodes
        requirements_impossible = ResourceRequirements(
            min_memory_mb=100000,  # 100GB
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        best_node = await registry.get_best_node(requirements_impossible)
        assert best_node is None
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_nodes(self, registry, sample_node):
        """Test cleanup of expired nodes."""
        await registry.register_node(sample_node)
        
        # Manually set old heartbeat by directly accessing registry state
        old_heartbeat = datetime.utcnow() - timedelta(seconds=10)
        async with registry._lock:
            registry.nodes["node1"].last_heartbeat = old_heartbeat
        
        # Should have 1 node initially
        all_nodes = await registry.get_all_nodes()
        assert len(all_nodes) == 1
        
        # Cleanup expired nodes
        removed_count = await registry.cleanup_expired_nodes()
        assert removed_count == 1
        
        # Should have no nodes now
        all_nodes = await registry.get_all_nodes()
        assert len(all_nodes) == 0
    
    @pytest.mark.asyncio
    async def test_registry_stats(self, registry, sample_node):
        """Test registry statistics."""
        await registry.register_node(sample_node)
        
        # Add another node
        caps2 = NodeCapabilities(
            backend_type="cpu",
            device_count=1,
            memory_gb=32.0
        )
        node2 = NodeInfo(
            node_id="node2",
            endpoint="http://node2:8080",
            capabilities=caps2,
            status=NodeStatus.OFFLINE  # Will be set to ONLINE by register_node
        )
        await registry.register_node(node2)
        
        # Manually set node2 to OFFLINE after registration
        async with registry._lock:
            registry.nodes["node2"].status = NodeStatus.OFFLINE
        
        stats = await registry.get_registry_stats()
        
        assert stats["total_nodes"] == 2
        assert stats["online_nodes"] == 1
        assert stats["offline_nodes"] == 1
        assert stats["backend_counts"]["rocm"] == 1
        assert stats["backend_counts"]["cpu"] == 1
    
    @pytest.mark.asyncio
    async def test_registry_shutdown(self, registry):
        """Test registry shutdown."""
        # Should not raise any errors
        await registry.shutdown()


class TestLoadBalancer:
    """Test load balancer functionality."""
    
    @pytest.fixture
    def nodes(self):
        """Create test nodes."""
        caps1 = NodeCapabilities(
            backend_type="rocm",
            device_count=2,
            memory_gb=64.0,
            max_concurrent_tasks=4,
            performance_profile={"score": 100}
        )
        node1 = NodeInfo(
            node_id="node1",
            endpoint="http://node1:8080",
            capabilities=caps1,
            status=NodeStatus.ONLINE
        )
        
        caps2 = NodeCapabilities(
            backend_type="cpu",
            device_count=1,
            memory_gb=32.0,
            max_concurrent_tasks=2,
            performance_profile={"score": 50}
        )
        node2 = NodeInfo(
            node_id="node2",
            endpoint="http://node2:8080",
            capabilities=caps2,
            status=NodeStatus.ONLINE
        )
        
        return [node1, node2]
    
    def test_round_robin_strategy(self, nodes):
        """Test round-robin load balancing."""
        balancer = LoadBalancer(strategy="round_robin")
        
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        # Should cycle through nodes
        selected1 = asyncio.run(balancer.select_node(nodes, requirements))
        selected2 = asyncio.run(balancer.select_node(nodes, requirements))
        selected3 = asyncio.run(balancer.select_node(nodes, requirements))
        
        assert selected1.node_id == "node1"
        assert selected2.node_id == "node2"
        assert selected3.node_id == "node1"  # Back to first
    
    def test_least_loaded_strategy(self, nodes):
        """Test least-loaded load balancing."""
        balancer = LoadBalancer(strategy="least_loaded")
        
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        # Increment load on node1
        balancer.increment_load("node1")
        balancer.increment_load("node1")
        
        # Should select node2 (least loaded)
        selected = asyncio.run(balancer.select_node(nodes, requirements))
        assert selected.node_id == "node2"
        
        # Increment load on node2
        balancer.increment_load("node2")
        
        # Should still select node2 (less load than node1)
        selected = asyncio.run(balancer.select_node(nodes, requirements))
        assert selected.node_id == "node2"
    
    def test_performance_based_strategy(self, nodes):
        """Test performance-based load balancing."""
        balancer = LoadBalancer(strategy="performance_based")
        
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        # Should select node1 (higher performance score)
        selected = asyncio.run(balancer.select_node(nodes, requirements))
        assert selected.node_id == "node1"
        
        # Add load to node1
        balancer.increment_load("node1")
        balancer.increment_load("node1")  # Max out node1
        balancer.increment_load("node1")
        balancer.increment_load("node1")
        
        # Should now select node2 due to load penalty
        selected = asyncio.run(balancer.select_node(nodes, requirements))
        assert selected.node_id == "node2"
    
    def test_load_tracking(self, nodes):
        """Test load tracking."""
        balancer = LoadBalancer()
        
        # Increment loads
        balancer.increment_load("node1")
        balancer.increment_load("node1")
        balancer.increment_load("node2")
        
        loads = balancer.get_loads()
        assert loads["node1"] == 2
        assert loads["node2"] == 1
        
        # Decrement loads
        balancer.decrement_load("node1")
        
        loads = balancer.get_loads()
        assert loads["node1"] == 1
        assert loads["node2"] == 1
        
        # Decrement below zero should not happen
        balancer.decrement_load("node1")
        balancer.decrement_load("node1")
        
        loads = balancer.get_loads()
        assert loads["node1"] == 0
        assert loads["node2"] == 1
    
    def test_empty_nodes_list(self):
        """Test load balancer with empty nodes list."""
        balancer = LoadBalancer()
        
        requirements = ResourceRequirements(
            min_memory_mb=1024,
            min_cpu_cores=1,
            max_execution_time=timedelta(minutes=10)
        )
        
        selected = asyncio.run(balancer.select_node([], requirements))
        assert selected is None


class TestWorkTracker:
    """Test work tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_work_assignment(self):
        """Test work assignment tracking."""
        tracker = WorkTracker()
        
        await tracker.assign_work("work1", "node1")
        await tracker.assign_work("work2", "node1")
        await tracker.assign_work("work3", "node2")
        
        # Check work assignments
        assert await tracker.get_node_for_work("work1") == "node1"
        assert await tracker.get_node_for_work("work2") == "node1"
        assert await tracker.get_node_for_work("work3") == "node2"
        assert await tracker.get_node_for_work("work4") is None
        
        # Check node work
        node1_work = await tracker.get_work_for_node("node1")
        assert node1_work == {"work1", "work2"}
        
        node2_work = await tracker.get_work_for_node("node2")
        assert node2_work == {"work3"}
        
        # Check work counts
        assert await tracker.get_node_work_count("node1") == 2
        assert await tracker.get_node_work_count("node2") == 1
        assert await tracker.get_node_work_count("node3") == 0
    
    @pytest.mark.asyncio
    async def test_work_completion(self):
        """Test work completion tracking."""
        tracker = WorkTracker()
        
        await tracker.assign_work("work1", "node1")
        await tracker.assign_work("work2", "node1")
        
        # Complete work1
        node_id = await tracker.complete_work("work1")
        assert node_id == "node1"
        
        # Work1 should no longer be assigned
        assert await tracker.get_node_for_work("work1") is None
        assert await tracker.get_node_work_count("node1") == 1
        
        # Complete work2
        node_id = await tracker.complete_work("work2")
        assert node_id == "node1"
        
        # All work should be completed
        assert await tracker.get_node_work_count("node1") == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_completed_work(self):
        """Test cleanup of completed work."""
        tracker = WorkTracker()
        
        await tracker.assign_work("work1", "node1")
        await tracker.assign_work("work2", "node2")
        await tracker.assign_work("work3", "node1")
        
        # Mark some work as completed (without calling complete_work)
        completed_work = {"work1", "work3"}
        
        cleaned = await tracker.cleanup_completed_work(completed_work)
        assert cleaned == 2
        
        # Only work2 should remain
        assert await tracker.get_node_for_work("work1") is None
        assert await tracker.get_node_for_work("work2") == "node2"
        assert await tracker.get_node_for_work("work3") is None


if __name__ == "__main__":
    pytest.main([__file__])
