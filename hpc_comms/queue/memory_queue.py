"""In-memory message queue implementation."""

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from enum import Enum

import msgpack


class QueuePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class QueueMessage:
    """Message in the queue."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: QueuePriority = QueuePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    delay_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        data = {
            'id': self.id,
            'payload': self.payload,
            'priority': self.priority.value,
            'created_at': self.created_at.timestamp(),
            'expires_at': self.expires_at.timestamp() if self.expires_at else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'delay_until': self.delay_until.timestamp() if self.delay_until else None,
            'metadata': self.metadata
        }
        return msgpack.packb(data)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'QueueMessage':
        """Deserialize message from bytes."""
        unpacked = msgpack.unpackb(data, raw=False)
        return cls(
            id=unpacked['id'],
            payload=unpacked['payload'],
            priority=QueuePriority(unpacked['priority']),
            created_at=datetime.fromtimestamp(unpacked['created_at']),
            expires_at=datetime.fromtimestamp(unpacked['expires_at']) if unpacked['expires_at'] else None,
            retry_count=unpacked['retry_count'],
            max_retries=unpacked['max_retries'],
            delay_until=datetime.fromtimestamp(unpacked['delay_until']) if unpacked['delay_until'] else None,
            metadata=unpacked['metadata']
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def is_ready(self) -> bool:
        """Check if message is ready for processing."""
        if self.delay_until:
            return datetime.utcnow() >= self.delay_until
        return True
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries


@dataclass
class QueueStats:
    """Queue statistics."""
    total_messages: int = 0
    pending_messages: int = 0
    processing_messages: int = 0
    completed_messages: int = 0
    failed_messages: int = 0
    expired_messages: int = 0
    queue_size: int = 0
    avg_processing_time: float = 0.0
    messages_per_second: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_messages': self.total_messages,
            'pending_messages': self.pending_messages,
            'processing_messages': self.processing_messages,
            'completed_messages': self.completed_messages,
            'failed_messages': self.failed_messages,
            'expired_messages': self.expired_messages,
            'queue_size': self.queue_size,
            'avg_processing_time': self.avg_processing_time,
            'messages_per_second': self.messages_per_second
        }


class MemoryQueue:
    """High-performance in-memory message queue."""
    
    def __init__(
        self,
        max_size: int = 10000,
        cleanup_interval: float = 60.0,
        default_ttl: Optional[timedelta] = None
    ):
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.default_ttl = default_ttl
        
        # Multiple queues by priority
        self._queues: Dict[QueuePriority, deque] = {
            priority: deque() for priority in QueuePriority
        }
        
        # Message tracking
        self._messages: Dict[str, QueueMessage] = {}
        self._processing: Set[str] = set()
        self._dead_letter: deque = deque()
        
        # Statistics
        self._stats = QueueStats()
        self._processing_times: List[float] = []
        self._last_cleanup = time.time()
        self._start_time = time.time()
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the queue background tasks."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop(self) -> None:
        """Stop the queue background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def put(
        self,
        payload: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL,
        delay: Optional[timedelta] = None,
        ttl: Optional[timedelta] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to the queue."""
        async with self._lock:
            # Check queue size
            if len(self._messages) >= self.max_size:
                raise QueueFullError(f"Queue is full (max size: {self.max_size})")
            
            # Create message
            message = QueueMessage(
                payload=payload,
                priority=priority,
                delay_until=datetime.utcnow() + delay if delay else None,
                expires_at=datetime.utcnow() + (ttl or self.default_ttl) if (ttl or self.default_ttl) else None,
                max_retries=max_retries,
                metadata=metadata or {}
            )
            
            # Store message
            self._messages[message.id] = message
            self._queues[priority].append(message)
            
            # Update stats
            async with self._stats_lock:
                self._stats.total_messages += 1
                self._stats.pending_messages += 1
                self._stats.queue_size = len(self._messages)
            
            return message.id
    
    async def get(self, timeout: Optional[float] = None) -> Optional[QueueMessage]:
        """Get a message from the queue."""
        start_time = time.time()
        
        while True:
            async with self._lock:
                # Check for ready messages by priority
                for priority in sorted(QueuePriority, key=lambda p: p.value, reverse=True):
                    queue = self._queues[priority]
                    while queue:
                        message = queue.popleft()
                        
                        # Skip expired messages
                        if message.is_expired():
                            del self._messages[message.id]
                            async with self._stats_lock:
                                self._stats.expired_messages += 1
                                self._stats.pending_messages -= 1
                                self._stats.queue_size = len(self._messages)
                            continue
                        
                        # Skip delayed messages
                        if not message.is_ready():
                            queue.append(message)  # Put back at end
                            continue
                        
                        # Mark as processing
                        self._processing.add(message.id)
                        message.processing_start = time.time()
                        
                        # Update stats
                        async with self._stats_lock:
                            self._stats.pending_messages -= 1
                            self._stats.processing_messages += 1
                        
                        return message
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return None
            
            # Wait a bit before retrying
            await asyncio.sleep(0.01)
    
    async def ack(self, message_id: str) -> None:
        """Acknowledge successful processing of a message."""
        async with self._lock:
            if message_id in self._processing:
                self._processing.remove(message_id)
                
                if message_id in self._messages:
                    message = self._messages[message_id]
                    
                    # Record processing time
                    if hasattr(message, 'processing_start'):
                        processing_time = time.time() - message.processing_start
                        self._processing_times.append(processing_time)
                        # Keep only last 1000 processing times
                        if len(self._processing_times) > 1000:
                            self._processing_times.pop(0)
                    
                    # Remove message
                    del self._messages[message_id]
                    
                    # Update stats
                    async with self._stats_lock:
                        self._stats.processing_messages -= 1
                        self._stats.completed_messages += 1
                        self._stats.queue_size = len(self._messages)
                        
                        # Calculate average processing time
                        if self._processing_times:
                            self._stats.avg_processing_time = sum(self._processing_times) / len(self._processing_times)
                        
                        # Calculate messages per second
                        elapsed = time.time() - self._start_time
                        if elapsed > 0:
                            self._stats.messages_per_second = self._stats.completed_messages / elapsed
    
    async def nack(self, message_id: str, requeue: bool = True) -> None:
        """Negative acknowledge - message processing failed."""
        async with self._lock:
            if message_id in self._processing:
                self._processing.remove(message_id)
                
                if message_id in self._messages:
                    message = self._messages[message_id]
                    message.retry_count += 1
                    
                    if message.can_retry() and requeue:
                        # Add delay for retry (exponential backoff)
                        delay = timedelta(seconds=2 ** message.retry_count)
                        message.delay_until = datetime.utcnow() + delay
                        
                        # Requeue with same priority
                        self._queues[message.priority].append(message)
                        
                        async with self._stats_lock:
                            self._stats.pending_messages += 1
                    else:
                        # Move to dead letter queue
                        self._dead_letter.append(message)
                        del self._messages[message_id]
                        
                        async with self._stats_lock:
                            self._stats.failed_messages += 1
                    
                    async with self._stats_lock:
                        self._stats.processing_messages -= 1
                        self._stats.queue_size = len(self._messages)
    
    async def size(self) -> int:
        """Get current queue size."""
        return len(self._messages)
    
    async def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        async with self._stats_lock:
            self._stats.queue_size = len(self._messages)
            return QueueStats(**self._stats.to_dict())
    
    async def clear(self) -> None:
        """Clear all messages from the queue."""
        async with self._lock:
            self._messages.clear()
            self._processing.clear()
            for queue in self._queues.values():
                queue.clear()
            self._dead_letter.clear()
            
            async with self._stats_lock:
                self._stats.pending_messages = 0
                self._stats.processing_messages = 0
                self._stats.queue_size = 0
    
    async def get_dead_letter_messages(self, limit: int = 100) -> List[QueueMessage]:
        """Get messages from the dead letter queue."""
        messages = []
        while len(messages) < limit and self._dead_letter:
            messages.append(self._dead_letter.popleft())
        return messages
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired messages."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Queue cleanup error: {e}")
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired messages."""
        current_time = datetime.utcnow()
        expired_count = 0
        
        async with self._lock:
            expired_messages = []
            
            for message_id, message in self._messages.items():
                if message.is_expired():
                    expired_messages.append(message_id)
            
            for message_id in expired_messages:
                if message_id in self._processing:
                    self._processing.remove(message_id)
                
                del self._messages[message_id]
                expired_count += 1
            
            # Remove from queues
            for queue in self._queues.values():
                valid_messages = deque()
                for message in queue:
                    if not message.is_expired():
                        valid_messages.append(message)
                    else:
                        expired_count += 1
                queue.clear()
                queue.extend(valid_messages)
            
            # Update stats
            if expired_count > 0:
                async with self._stats_lock:
                    self._stats.expired_messages += expired_count
                    self._stats.queue_size = len(self._messages)


class QueueFullError(Exception):
    """Raised when queue is full."""
    pass
