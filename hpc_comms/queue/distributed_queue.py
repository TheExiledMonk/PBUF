"""Distributed message queue implementation using HTTP transport."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

from ..core.transport import Transport
from .memory_queue import MemoryQueue, QueueMessage, QueuePriority, QueueStats, QueueFullError


logger = logging.getLogger(__name__)


class QueueManager:
    """Manages distributed queue operations."""
    
    def __init__(
        self,
        transport: Transport,
        node_id: str,
        replication_factor: int = 2,
        sync_interval: float = 30.0
    ):
        self.transport = transport
        self.node_id = node_id
        self.replication_factor = replication_factor
        self.sync_interval = sync_interval
        
        # Local queue instance
        self.local_queue = MemoryQueue()
        
        # Cluster management
        self.known_nodes: Set[str] = set()
        self.node_endpoints: Dict[str, str] = {}
        self.leader_node: Optional[str] = None
        
        # Replication tracking
        self.replicated_messages: Set[str] = set()
        self.pending_replications: Dict[str, List[str]] = {}
        
        # Background tasks
        self.sync_task: Optional[asyncio.Task] = None
        self.leader_election_task: Optional[asyncio.Task] = None
        
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the distributed queue manager."""
        await self.local_queue.start()
        
        # Start background tasks
        self.sync_task = asyncio.create_task(self._sync_loop())
        self.leader_election_task = asyncio.create_task(self._leader_election_loop())
        
        logger.info(f"Queue manager started for node {self.node_id}")
    
    async def stop(self) -> None:
        """Stop the distributed queue manager."""
        await self.local_queue.stop()
        
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        
        if self.leader_election_task:
            self.leader_election_task.cancel()
            try:
                await self.leader_election_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Queue manager stopped for node {self.node_id}")
    
    async def add_node(self, node_id: str, endpoint: str) -> None:
        """Add a node to the cluster."""
        async with self._lock:
            self.known_nodes.add(node_id)
            self.node_endpoints[node_id] = endpoint
            logger.info(f"Added node {node_id} at {endpoint}")
    
    async def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster."""
        async with self._lock:
            self.known_nodes.discard(node_id)
            self.node_endpoints.pop(node_id, None)
            
            if self.leader_node == node_id:
                self.leader_node = None
            
            logger.info(f"Removed node {node_id}")
    
    async def put(
        self,
        payload: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL,
        delay: Optional[timedelta] = None,
        ttl: Optional[timedelta] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to the distributed queue."""
        # Add to local queue first
        message_id = await self.local_queue.put(
            payload=payload,
            priority=priority,
            delay=delay,
            ttl=ttl,
            max_retries=max_retries,
            metadata=metadata
        )
        
        # If we're the leader, replicate to other nodes
        if self.leader_node == self.node_id:
            await self._replicate_message(message_id, payload, priority, delay, ttl, max_retries, metadata)
        
        return message_id
    
    async def get(self, timeout: Optional[float] = None) -> Optional[QueueMessage]:
        """Get a message from the distributed queue."""
        return await self.local_queue.get(timeout)
    
    async def ack(self, message_id: str) -> None:
        """Acknowledge successful processing of a message."""
        await self.local_queue.ack(message_id)
        
        # Notify other nodes to remove the replicated message
        if self.leader_node == self.node_id:
            await self._notify_ack(message_id)
    
    async def nack(self, message_id: str, requeue: bool = True) -> None:
        """Negative acknowledge - message processing failed."""
        await self.local_queue.nack(message_id, requeue)
    
    async def get_stats(self) -> QueueStats:
        """Get distributed queue statistics."""
        local_stats = await self.local_queue.get_stats()
        
        # Add cluster stats
        cluster_stats = local_stats.to_dict()
        cluster_stats.update({
            'cluster_nodes': len(self.known_nodes),
            'leader_node': self.leader_node,
            'replicated_messages': len(self.replicated_messages),
            'is_leader': self.leader_node == self.node_id
        })
        
        return QueueStats(**{k: v for k, v in cluster_stats.items() if hasattr(QueueStats, '__annotations__') and k in QueueStats.__annotations__})
    
    async def _replicate_message(
        self,
        message_id: str,
        payload: Dict[str, Any],
        priority: QueuePriority,
        delay: Optional[timedelta],
        ttl: Optional[timedelta],
        max_retries: int,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Replicate a message to other nodes."""
        if len(self.known_nodes) < 2:
            return  # No other nodes to replicate to
        
        # Select replication targets
        replication_targets = list(self.known_nodes - {self.node_id})
        if len(replication_targets) > self.replication_factor:
            # Simple selection - could be improved with consistent hashing
            replication_targets = replication_targets[:self.replication_factor]
        
        replication_data = {
            'action': 'replicate',
            'message_id': message_id,
            'payload': payload,
            'priority': priority.value,
            'delay_seconds': delay.total_seconds() if delay else None,
            'ttl_seconds': ttl.total_seconds() if ttl else None,
            'max_retries': max_retries,
            'metadata': metadata
        }
        
        # Send to replication targets
        tasks = []
        for target_node in replication_targets:
            endpoint = self.node_endpoints[target_node]
            task = asyncio.create_task(
                self._send_replication(endpoint, replication_data)
            )
            tasks.append(task)
        
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.replicated_messages.add(message_id)
                self.pending_replications[message_id] = replication_targets
            except Exception as e:
                logger.error(f"Replication failed for message {message_id}: {e}")
    
    async def _send_replication(self, endpoint: str, data: Dict[str, Any]) -> None:
        """Send replication data to a node."""
        try:
            from ..core.messages import Message
            
            message = Message(
                source_node=self.node_id,
                target_node="queue_manager",
                message_type="QUEUE_REPLICATION",
                payload=data
            )
            
            await self.transport.send_message(endpoint, message)
        except Exception as e:
            logger.error(f"Failed to send replication to {endpoint}: {e}")
            raise
    
    async def _handle_replication(self, data: Dict[str, Any]) -> None:
        """Handle incoming replication data."""
        try:
            delay = None
            if data.get('delay_seconds'):
                delay = timedelta(seconds=data['delay_seconds'])
            
            ttl = None
            if data.get('ttl_seconds'):
                ttl = timedelta(seconds=data['ttl_seconds'])
            
            await self.local_queue.put(
                payload=data['payload'],
                priority=QueuePriority(data['priority']),
                delay=delay,
                ttl=ttl,
                max_retries=data['max_retries'],
                metadata=data['metadata']
            )
            
            logger.debug(f"Replicated message {data['message_id']}")
        except Exception as e:
            logger.error(f"Failed to handle replication: {e}")
    
    async def _notify_ack(self, message_id: str) -> None:
        """Notify other nodes of message acknowledgment."""
        if message_id not in self.pending_replications:
            return
        
        ack_data = {
            'action': 'ack',
            'message_id': message_id
        }
        
        tasks = []
        for target_node in self.pending_replications[message_id]:
            endpoint = self.node_endpoints[target_node]
            task = asyncio.create_task(self._send_ack(endpoint, ack_data))
            tasks.append(task)
        
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.replicated_messages.discard(message_id)
                self.pending_replications.pop(message_id, None)
            except Exception as e:
                logger.error(f"Failed to notify ack for message {message_id}: {e}")
    
    async def _send_ack(self, endpoint: str, data: Dict[str, Any]) -> None:
        """Send acknowledgment notification to a node."""
        try:
            from ..core.messages import Message
            
            message = Message(
                source_node=self.node_id,
                target_node="queue_manager",
                message_type="QUEUE_ACK",
                payload=data
            )
            
            await self.transport.send_message(endpoint, message)
        except Exception as e:
            logger.error(f"Failed to send ack notification to {endpoint}: {e}")
            raise
    
    async def _handle_ack(self, data: Dict[str, Any]) -> None:
        """Handle incoming acknowledgment notification."""
        message_id = data['message_id']
        
        # Remove from local queue if it exists
        try:
            await self.local_queue.ack(message_id)
            logger.debug(f"Acknowledged replicated message {message_id}")
        except KeyError:
            # Message not found, already processed
            pass
    
    async def _sync_loop(self) -> None:
        """Background task to synchronize with other nodes."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._sync_with_cluster()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
    
    async def _sync_with_cluster(self) -> None:
        """Synchronize queue state with cluster."""
        if not self.known_nodes:
            return
        
        # Get local stats
        local_stats = await self.local_queue.get_stats()
        
        sync_data = {
            'action': 'sync',
            'node_id': self.node_id,
            'stats': local_stats.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send to all nodes
        tasks = []
        for node_id, endpoint in self.node_endpoints.items():
            if node_id != self.node_id:
                task = asyncio.create_task(self._send_sync(endpoint, sync_data))
                tasks.append(task)
        
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Failed to sync with cluster: {e}")
    
    async def _send_sync(self, endpoint: str, data: Dict[str, Any]) -> None:
        """Send sync data to a node."""
        try:
            from ..core.messages import Message
            
            message = Message(
                source_node=self.node_id,
                target_node="queue_manager",
                message_type="QUEUE_SYNC",
                payload=data
            )
            
            await self.transport.send_message(endpoint, message)
        except Exception as e:
            logger.error(f"Failed to send sync to {endpoint}: {e}")
            raise
    
    async def _handle_sync(self, data: Dict[str, Any]) -> None:
        """Handle incoming sync data."""
        node_id = data['node_id']
        stats = data['stats']
        
        # Update node info
        if node_id not in self.known_nodes:
            logger.info(f"Discovered new node {node_id} through sync")
        
        # Could implement more sophisticated sync logic here
        logger.debug(f"Received sync from {node_id}: {stats}")
    
    async def _leader_election_loop(self) -> None:
        """Background task for leader election."""
        while True:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._elect_leader()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Leader election error: {e}")
    
    async def _elect_leader(self) -> None:
        """Simple leader election based on node ID (lowest wins)."""
        if not self.known_nodes:
            return
        
        # Add self to known nodes if not present
        self.known_nodes.add(self.node_id)
        
        # Find lowest node ID (simple approach)
        all_nodes = sorted(self.known_nodes)
        new_leader = all_nodes[0]
        
        if self.leader_node != new_leader:
            old_leader = self.leader_node
            self.leader_node = new_leader
            
            if new_leader == self.node_id:
                logger.info(f"Became cluster leader (was {old_leader})")
            else:
                logger.info(f"Leader changed to {new_leader} (was {old_leader})")


class DistributedQueue:
    """Distributed queue interface."""
    
    def __init__(self, manager: QueueManager):
        self.manager = manager
    
    async def put(
        self,
        payload: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL,
        delay: Optional[timedelta] = None,
        ttl: Optional[timedelta] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to the distributed queue."""
        return await self.manager.put(payload, priority, delay, ttl, max_retries, metadata)
    
    async def get(self, timeout: Optional[float] = None) -> Optional[QueueMessage]:
        """Get a message from the distributed queue."""
        return await self.manager.get(timeout)
    
    async def ack(self, message_id: str) -> None:
        """Acknowledge successful processing of a message."""
        await self.manager.ack(message_id)
    
    async def nack(self, message_id: str, requeue: bool = True) -> None:
        """Negative acknowledge - message processing failed."""
        await self.manager.nack(message_id, requeue)
    
    async def get_stats(self) -> QueueStats:
        """Get distributed queue statistics."""
        return await self.manager.get_stats()
    
    async def size(self) -> int:
        """Get current queue size."""
        return await self.manager.local_queue.size()
