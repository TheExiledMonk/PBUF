"""Tests for message queue implementation."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

# Required for async fixtures
pytest_plugins = ('pytest_asyncio',)

from hpc_comms.queue.memory_queue import (
    MemoryQueue, QueueMessage, QueuePriority, QueueStats, QueueFullError
)
from hpc_comms.queue.distributed_queue import QueueManager, DistributedQueue


class TestQueueMessage:
    """Test queue message operations."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = QueueMessage(
            payload={"data": "test"},
            priority=QueuePriority.HIGH
        )
        
        assert message.payload == {"data": "test"}
        assert message.priority == QueuePriority.HIGH
        assert message.retry_count == 0
        assert message.max_retries == 3
        assert not message.is_expired()
        assert message.is_ready()
        assert message.can_retry()
    
    def test_message_serialization(self):
        """Test message serialization/deserialization."""
        original = QueueMessage(
            payload={"data": "test"},
            priority=QueuePriority.HIGH,
            expires_at=datetime.utcnow() + timedelta(hours=1),
            retry_count=1,
            metadata={"source": "test"}
        )
        
        # Serialize
        data = original.to_bytes()
        assert isinstance(data, bytes)
        
        # Deserialize
        restored = QueueMessage.from_bytes(data)
        
        assert restored.id == original.id
        assert restored.payload == original.payload
        assert restored.priority == original.priority
        assert restored.retry_count == original.retry_count
        assert restored.metadata == original.metadata
    
    def test_message_expiration(self):
        """Test message expiration."""
        # Non-expired message
        message = QueueMessage(
            payload={"data": "test"},
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        assert not message.is_expired()
        
        # Expired message
        expired_message = QueueMessage(
            payload={"data": "test"},
            expires_at=datetime.utcnow() - timedelta(hours=1)
        )
        assert expired_message.is_expired()
    
    def test_message_delay(self):
        """Test message delay."""
        # Ready message
        message = QueueMessage(
            payload={"data": "test"},
            delay_until=datetime.utcnow() - timedelta(minutes=1)
        )
        assert message.is_ready()
        
        # Delayed message
        delayed_message = QueueMessage(
            payload={"data": "test"},
            delay_until=datetime.utcnow() + timedelta(minutes=1)
        )
        assert not delayed_message.is_ready()
    
    def test_message_retry(self):
        """Test message retry logic."""
        message = QueueMessage(
            payload={"data": "test"},
            max_retries=3
        )
        
        assert message.can_retry()
        
        message.retry_count = 2
        assert message.can_retry()
        
        message.retry_count = 3
        assert not message.can_retry()


class TestMemoryQueue:
    """Test memory queue operations."""
    
    @pytest.fixture
    async def queue(self):
        """Create a test queue."""
        q = MemoryQueue(max_size=100)
        await q.start()
        yield q
        await q.stop()
    
    @pytest.mark.asyncio
    async def test_put_and_get(self, queue):
        """Test basic put/get operations."""
        message_id = await queue.put({"data": "test"})
        
        assert message_id is not None
        assert await queue.size() == 1
        
        message = await queue.get(timeout=1.0)
        assert message is not None
        assert message.payload == {"data": "test"}
        assert message.id == message_id
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Test priority-based message ordering."""
        # Add messages in different priorities
        await queue.put({"priority": "low"}, priority=QueuePriority.LOW)
        await queue.put({"priority": "high"}, priority=QueuePriority.HIGH)
        await queue.put({"priority": "normal"}, priority=QueuePriority.NORMAL)
        await queue.put({"priority": "critical"}, priority=QueuePriority.CRITICAL)
        
        # Should get in priority order: CRITICAL, HIGH, NORMAL, LOW
        order = []
        for _ in range(4):
            message = await queue.get(timeout=1.0)
            order.append(message.payload["priority"])
        
        assert order == ["critical", "high", "normal", "low"]
    
    @pytest.mark.asyncio
    async def test_message_ack(self, queue):
        """Test message acknowledgment."""
        message_id = await queue.put({"data": "test"})
        
        message = await queue.get(timeout=1.0)
        assert await queue.size() == 1  # Message still in queue (processing)
        
        await queue.ack(message_id)
        assert await queue.size() == 0  # Message removed
    
    @pytest.mark.asyncio
    async def test_message_nack(self, queue):
        """Test message negative acknowledgment."""
        message_id = await queue.put({"data": "test"}, max_retries=2)
        
        message = await queue.get(timeout=1.0)
        await queue.nack(message_id, requeue=True)
        
        # Message should be requeued
        assert await queue.size() == 1
        
        # Get the retried message
        retried = await queue.get(timeout=1.0)
        assert retried.id == message_id
        assert retried.retry_count == 1
    
    @pytest.mark.asyncio
    async def test_message_nack_max_retries(self, queue):
        """Test message negative acknowledgment with max retries exceeded."""
        message_id = await queue.put({"data": "test"}, max_retries=1)
        
        # First failure
        message = await queue.get(timeout=1.0)
        await queue.nack(message_id, requeue=True)
        
        # Second failure (max retries exceeded)
        retried = await queue.get(timeout=1.0)
        await queue.nack(retried.id, requeue=True)
        
        # Message should be in dead letter queue
        assert await queue.size() == 0
        dead_messages = await queue.get_dead_letter_messages()
        assert len(dead_messages) == 1
        assert dead_messages[0].id == message_id
    
    @pytest.mark.asyncio
    async def test_delayed_message(self, queue):
        """Test delayed message delivery."""
        # Add delayed message
        await queue.put(
            {"data": "delayed"},
            delay=timedelta(milliseconds=100)
        )
        
        # Should not get delayed message immediately
        message = await queue.get(timeout=0.05)
        assert message is None
        
        # Should get it after delay
        message = await queue.get(timeout=0.1)
        assert message is not None
        assert message.payload == {"data": "delayed"}
    
    @pytest.mark.asyncio
    async def test_expired_message(self, queue):
        """Test expired message cleanup."""
        # Add expired message
        await queue.put(
            {"data": "expired"},
            ttl=timedelta(milliseconds=50)
        )
        
        # Wait for expiration
        await asyncio.sleep(0.1)
        
        # Should not get expired message
        message = await queue.get(timeout=0.1)
        assert message is None
    
    @pytest.mark.asyncio
    async def test_queue_full(self, queue):
        """Test queue full behavior."""
        # Fill the queue
        for i in range(queue.max_size):
            await queue.put({"data": f"message_{i}"})
        
        # Should raise QueueFullError
        with pytest.raises(QueueFullError):
            await queue.put({"data": "too_many"})
    
    @pytest.mark.asyncio
    async def test_queue_stats(self, queue):
        """Test queue statistics."""
        stats = await queue.get_stats()
        assert stats.total_messages == 0
        assert stats.pending_messages == 0
        
        # Add some messages
        await queue.put({"data": "test1"})
        await queue.put({"data": "test2"})
        
        stats = await queue.get_stats()
        assert stats.total_messages == 2
        assert stats.pending_messages == 2
        assert stats.queue_size == 2
        
        # Get a message
        message = await queue.get(timeout=1.0)
        stats = await queue.get_stats()
        assert stats.pending_messages == 1
        assert stats.processing_messages == 1
        
        # Ack the message
        await queue.ack(message.id)
        stats = await queue.get_stats()
        assert stats.completed_messages == 1
        assert stats.processing_messages == 0
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, queue):
        """Test clearing the queue."""
        # Add messages
        await queue.put({"data": "test1"})
        await queue.put({"data": "test2"})
        
        assert await queue.size() == 2
        
        # Clear queue
        await queue.clear()
        
        assert await queue.size() == 0
        
        # Stats should be reset
        stats = await queue.get_stats()
        assert stats.pending_messages == 0
        assert stats.processing_messages == 0


class TestDistributedQueue:
    """Test distributed queue operations."""
    
    @pytest.fixture
    def mock_transport(self):
        """Create a mock transport."""
        from unittest.mock import AsyncMock
        
        transport = AsyncMock()
        transport.send_message = AsyncMock()
        return transport
    
    @pytest.fixture
    async def queue_manager(self, mock_transport):
        """Create a queue manager."""
        manager = QueueManager(
            transport=mock_transport,
            node_id="test_node",
            replication_factor=1
        )
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.fixture
    def distributed_queue(self, queue_manager):
        """Create a distributed queue."""
        return DistributedQueue(queue_manager)
    
    @pytest.mark.asyncio
    async def test_distributed_put_get(self, distributed_queue):
        """Test distributed put/get operations."""
        message_id = await distributed_queue.put({"data": "test"})
        
        assert message_id is not None
        
        message = await distributed_queue.get(timeout=1.0)
        assert message is not None
        assert message.payload == {"data": "test"}
    
    @pytest.mark.asyncio
    async def test_node_management(self, queue_manager):
        """Test node management."""
        # Add node
        await queue_manager.add_node("node2", "http://node2:8080")
        
        assert "node2" in queue_manager.known_nodes
        assert queue_manager.node_endpoints["node2"] == "http://node2:8080"
        
        # Remove node
        await queue_manager.remove_node("node2")
        
        assert "node2" not in queue_manager.known_nodes
        assert "node2" not in queue_manager.node_endpoints
    
    @pytest.mark.asyncio
    async def test_leader_election(self, queue_manager):
        """Test leader election."""
        # Add some nodes
        await queue_manager.add_node("node_a", "http://node_a:8080")
        await queue_manager.add_node("node_b", "http://node_b:8080")
        await queue_manager.add_node("node_c", "http://node_c:8080")
        
        # Trigger election
        await queue_manager._elect_leader()
        
        # Should have a leader
        assert queue_manager.leader_node is not None
        
        # Leader should be the node with lowest ID
        expected_leader = min(["test_node", "node_a", "node_b", "node_c"])
        assert queue_manager.leader_node == expected_leader
    
    @pytest.mark.asyncio
    async def test_message_replication(self, queue_manager, mock_transport):
        """Test message replication."""
        # Set this node as leader
        queue_manager.leader_node = queue_manager.node_id
        
        # Add another node for replication
        await queue_manager.add_node("node2", "http://node2:8080")
        
        # Put a message (should replicate)
        message_id = await queue_manager.put({"data": "test"})
        
        # Should have sent replication message
        mock_transport.send_message.assert_called()
        
        # Check call arguments
        call_args = mock_transport.send_message.call_args
        assert call_args[0][0] == "http://node2:8080"
        assert call_args[0][1].payload["action"] == "replicate"
        assert call_args[0][1].payload["message_id"] == message_id
    
    @pytest.mark.asyncio
    async def test_handle_replication(self, queue_manager):
        """Test handling incoming replication."""
        replication_data = {
            "action": "replicate",
            "message_id": "test_message",
            "payload": {"data": "replicated"},
            "priority": QueuePriority.NORMAL.value,
            "delay_seconds": None,
            "ttl_seconds": None,
            "max_retries": 3,
            "metadata": {}
        }
        
        await queue_manager._handle_replication(replication_data)
        
        # Should have message in local queue
        message = await queue_manager.local_queue.get(timeout=1.0)
        assert message is not None
        assert message.payload == {"data": "replicated"}
        assert message.id == "test_message"
    
    @pytest.mark.asyncio
    async def test_distributed_stats(self, distributed_queue):
        """Test distributed queue statistics."""
        stats = await distributed_queue.get_stats()
        
        # Should have cluster-specific stats
        assert hasattr(stats, 'to_dict')
        stats_dict = stats.to_dict()
        
        # Check basic stats exist
        assert 'total_messages' in stats_dict
        assert 'queue_size' in stats_dict


if __name__ == "__main__":
    pytest.main([__file__])
