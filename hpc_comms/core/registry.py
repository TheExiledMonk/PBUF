"""Node registry and capabilities management."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict

from .messages import NodeInfo, NodeStatus, NodeCapabilities, ResourceRequirements
from .errors import NodeError, ResourceError


class NodeRegistry:
    """Registry for managing compute nodes and their capabilities."""
    
    def __init__(self, heartbeat_timeout: timedelta = timedelta(minutes=2)):
        self.nodes: Dict[str, NodeInfo] = {}
        self.heartbeat_timeout = heartbeat_timeout
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        # Don't start cleanup task automatically - let user call start_cleanup()
    
    def start_cleanup(self) -> None:
        """Start the background cleanup task."""
        if self._cleanup_task is None:
            try:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            except RuntimeError:
                # No event loop running - cleanup task will be started manually
                pass
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired nodes."""
        while True:
            try:
                await self.cleanup_expired_nodes()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
                await asyncio.sleep(30)
    
    async def register_node(self, node_info: NodeInfo) -> None:
        """Register a new compute node."""
        async with self._lock:
            node_info.last_heartbeat = datetime.utcnow()
            node_info.status = NodeStatus.ONLINE
            self.nodes[node_info.node_id] = node_info
        print(f"Registered node {node_info.node_id} at {node_info.endpoint}")
    
    async def update_heartbeat(self, node_id: str, node_info: NodeInfo) -> None:
        """Update heartbeat for a node."""
        async with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = datetime.utcnow()
                self.nodes[node_id].status = node_info.status
                self.nodes[node_id].capabilities = node_info.capabilities
                self.nodes[node_id].metadata.update(node_info.metadata)
            else:
                # Node not registered, register it
                await self.register_node(node_info)
    
    async def deregister_node(self, node_id: str) -> None:
        """Deregister a compute node."""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                print(f"Deregistered node {node_id}")
    
    async def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Get node information by ID."""
        async with self._lock:
            return self.nodes.get(node_id)
    
    async def get_all_nodes(self) -> List[NodeInfo]:
        """Get all registered nodes."""
        async with self._lock:
            return list(self.nodes.values())
    
    async def get_online_nodes(self) -> List[NodeInfo]:
        """Get all online nodes."""
        async with self._lock:
            return [
                node for node in self.nodes.values()
                if node.is_healthy(self.heartbeat_timeout)
            ]
    
    async def find_nodes_for_work(self, requirements: ResourceRequirements) -> List[NodeInfo]:
        """Find nodes that can handle the given work requirements."""
        online_nodes = await self.get_online_nodes()
        suitable_nodes = []
        
        for node in online_nodes:
            if node.can_handle_work(requirements):
                suitable_nodes.append(node)
        
        # Sort by performance (best first)
        suitable_nodes.sort(
            key=lambda n: n.capabilities.performance_profile.get("score", 0),
            reverse=True
        )
        
        return suitable_nodes
    
    async def get_best_node(self, requirements: ResourceRequirements) -> Optional[NodeInfo]:
        """Get the best node for the given work requirements."""
        suitable_nodes = await self.find_nodes_for_work(requirements)
        return suitable_nodes[0] if suitable_nodes else None
    
    async def update_node_status(self, node_id: str, status: NodeStatus) -> None:
        """Update node status."""
        async with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                self.nodes[node_id].last_heartbeat = datetime.utcnow()
    
    async def cleanup_expired_nodes(self) -> int:
        """Remove expired nodes and return count of removed nodes."""
        removed_count = 0
        current_time = datetime.utcnow()
        expired_nodes = []
        
        async with self._lock:
            for node_id, node in self.nodes.items():
                if not node.is_healthy(self.heartbeat_timeout):
                    expired_nodes.append(node_id)
            
            for node_id in expired_nodes:
                del self.nodes[node_id]
                removed_count += 1
        
        if removed_count > 0:
            print(f"Cleaned up {removed_count} expired nodes")
        
        return removed_count
    
    async def get_registry_stats(self) -> Dict[str, int]:
        """Get registry statistics."""
        async with self._lock:
            total_nodes = len(self.nodes)
            online_nodes = sum(
                1 for node in self.nodes.values()
                if node.is_healthy(self.heartbeat_timeout)
            )
            offline_nodes = total_nodes - online_nodes
            
            # Count by backend type
            backend_counts = defaultdict(int)
            for node in self.nodes.values():
                backend_counts[node.capabilities.backend_type] += 1
            
            return {
                "total_nodes": total_nodes,
                "online_nodes": online_nodes,
                "offline_nodes": offline_nodes,
                "backend_counts": dict(backend_counts)
            }
    
    async def shutdown(self) -> None:
        """Shutdown the registry and cleanup tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class LoadBalancer:
    """Load balancer for distributing work across nodes."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self._round_robin_index = 0
        self._node_loads: Dict[str, int] = defaultdict(int)
    
    async def select_node(
        self, 
        available_nodes: List[NodeInfo], 
        requirements: ResourceRequirements
    ) -> Optional[NodeInfo]:
        """Select a node based on the load balancing strategy."""
        if not available_nodes:
            return None
        
        if self.strategy == "round_robin":
            return self._select_round_robin(available_nodes)
        elif self.strategy == "least_loaded":
            return self._select_least_loaded(available_nodes)
        elif self.strategy == "performance_based":
            return self._select_performance_based(available_nodes, requirements)
        else:
            return available_nodes[0]
    
    def _select_round_robin(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node using round-robin strategy."""
        node = nodes[self._round_robin_index % len(nodes)]
        self._round_robin_index += 1
        return node
    
    def _select_least_loaded(self, nodes: List[NodeInfo]) -> NodeInfo:
        """Select node with least current load."""
        return min(nodes, key=lambda n: self._node_loads.get(n.node_id, 0))
    
    def _select_performance_based(
        self, 
        nodes: List[NodeInfo], 
        requirements: ResourceRequirements
    ) -> NodeInfo:
        """Select node based on performance and current load."""
        def score(node: NodeInfo) -> float:
            # Base performance score
            perf_score = node.capabilities.performance_profile.get("score", 0)
            
            # Load penalty
            current_load = self._node_loads.get(node.node_id, 0)
            max_concurrent = node.capabilities.max_concurrent_tasks
            load_ratio = current_load / max_concurrent if max_concurrent > 0 else 1
            load_penalty = load_ratio * 100
            
            return perf_score - load_penalty
        
        return max(nodes, key=score)
    
    def increment_load(self, node_id: str) -> None:
        """Increment load counter for a node."""
        self._node_loads[node_id] += 1
    
    def decrement_load(self, node_id: str) -> None:
        """Decrement load counter for a node."""
        self._node_loads[node_id] = max(0, self._node_loads[node_id] - 1)
    
    def get_loads(self) -> Dict[str, int]:
        """Get current load for all nodes."""
        return dict(self._node_loads)


class WorkTracker:
    """Track work assignments and completion."""
    
    def __init__(self):
        self.work_assignments: Dict[str, str] = {}  # work_id -> node_id
        self.node_work: Dict[str, Set[str]] = defaultdict(set)  # node_id -> set of work_ids
        self._lock = asyncio.Lock()
    
    async def assign_work(self, work_id: str, node_id: str) -> None:
        """Assign work to a node."""
        async with self._lock:
            self.work_assignments[work_id] = node_id
            self.node_work[node_id].add(work_id)
    
    async def complete_work(self, work_id: str) -> Optional[str]:
        """Mark work as completed and return the node that handled it."""
        async with self._lock:
            node_id = self.work_assignments.pop(work_id, None)
            if node_id:
                self.node_work[node_id].discard(work_id)
            return node_id
    
    async def get_node_for_work(self, work_id: str) -> Optional[str]:
        """Get the node handling a specific work item."""
        async with self._lock:
            return self.work_assignments.get(work_id)
    
    async def get_work_for_node(self, node_id: str) -> Set[str]:
        """Get all work assigned to a node."""
        async with self._lock:
            return set(self.node_work.get(node_id, set()))
    
    async def get_node_work_count(self, node_id: str) -> int:
        """Get count of active work for a node."""
        async with self._lock:
            return len(self.node_work.get(node_id, set()))
    
    async def cleanup_completed_work(self, work_ids: Set[str]) -> int:
        """Clean up completed work assignments."""
        cleaned = 0
        async with self._lock:
            for work_id in work_ids:
                node_id = self.work_assignments.pop(work_id, None)
                if node_id:
                    self.node_work[node_id].discard(work_id)
                    cleaned += 1
        return cleaned
