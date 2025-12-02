"""High-level HPC protocol implementation with retry logic and error handling."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, AsyncIterator

from .messages import Message, WorkRequest, WorkResponse, NodeInfo, serialize_message, deserialize_message
from .transport import Transport, Connection, Server, Router, MessageHandler, CircuitBreaker
from .errors import HPCError, ConnectionError, TimeoutError, is_retryable_error


logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """Retry configuration for failed operations."""
    max_attempts: int = 3
    base_delay: timedelta = timedelta(seconds=1)
    max_delay: timedelta = timedelta(seconds=30)
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> timedelta:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


DEFAULT_RETRY_POLICY = RetryPolicy()


class HPCProtocol:
    """High-level HPC protocol implementation."""
    
    def __init__(
        self, 
        transport: Transport,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.transport = transport
        self.retry_policy = retry_policy or DEFAULT_RETRY_POLICY
        self.circuit_breaker = circuit_breaker
        self.router = Router()
        self.message_handlers: Dict[str, List[Callable]] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default message handlers."""
        self.router.register_handler("HEALTH_CHECK", self._handle_health_check)
        self.router.register_handler("ERROR_REPORT", self._handle_error_report)
    
    async def send_message(
        self, 
        endpoint: str, 
        message: Message, 
        timeout: Optional[timedelta] = None,
        retry_policy: Optional[RetryPolicy] = None
    ) -> None:
        """Send a message with retry logic."""
        policy = retry_policy or self.retry_policy
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                if self.circuit_breaker:
                    await self.circuit_breaker.call(
                        self.transport.send_message, endpoint, message, timeout
                    )
                else:
                    await self.transport.send_message(endpoint, message, timeout)
                return
                
            except Exception as e:
                if attempt == policy.max_attempts:
                    raise e
                
                if not is_retryable_error(e) if hasattr(e, 'error_code') else False:
                    raise e
                
                delay = policy.get_delay(attempt)
                logger.warning(f"Send failed (attempt {attempt}/{policy.max_attempts}), retrying in {delay}: {e}")
                await asyncio.sleep(delay.total_seconds())
    
    async def request_response(
        self, 
        endpoint: str, 
        message: Message, 
        timeout: Optional[timedelta] = None,
        retry_policy: Optional[RetryPolicy] = None
    ) -> Message:
        """Send a message and wait for response with retry logic."""
        policy = retry_policy or self.retry_policy
        
        for attempt in range(1, policy.max_attempts + 1):
            try:
                if self.circuit_breaker:
                    response = await self.circuit_breaker.call(
                        self.transport.request_response, endpoint, message, timeout
                    )
                else:
                    response = await self.transport.request_response(endpoint, message, timeout)
                
                if response is None:
                    raise TimeoutError("No response received")
                
                return response
                
            except Exception as e:
                if attempt == policy.max_attempts:
                    raise e
                
                if not is_retryable_error(e) if hasattr(e, 'error_code') else False:
                    raise e
                
                delay = policy.get_delay(attempt)
                logger.warning(f"Request failed (attempt {attempt}/{policy.max_attempts}), retrying in {delay}: {e}")
                await asyncio.sleep(delay.total_seconds())
    
    async def register_node(
        self, 
        controller_endpoint: str, 
        node_info: NodeInfo,
        timeout: Optional[timedelta] = None
    ) -> None:
        """Register a compute node with the controller."""
        message = Message(
            source_node=node_info.node_id,
            target_node="controller",
            message_type="NODE_REGISTER",
            payload=node_info.model_dump()
        )
        
        await self.send_message(controller_endpoint, message, timeout)
        logger.info(f"Registered node {node_info.node_id} with controller")
    
    async def send_heartbeat(
        self, 
        controller_endpoint: str, 
        node_info: NodeInfo,
        timeout: Optional[timedelta] = None
    ) -> None:
        """Send heartbeat to controller."""
        message = Message(
            source_node=node_info.node_id,
            target_node="controller",
            message_type="NODE_HEARTBEAT",
            payload=node_info.model_dump()
        )
        
        await self.send_message(controller_endpoint, message, timeout)
    
    async def request_work(
        self, 
        controller_endpoint: str, 
        node_id: str,
        capabilities: Dict[str, Any],
        timeout: Optional[timedelta] = None
    ) -> Optional[WorkRequest]:
        """Request work from controller."""
        message = Message(
            source_node=node_id,
            target_node="controller",
            message_type="WORK_REQUEST_QUERY",
            payload={"capabilities": capabilities}
        )
        
        try:
            response = await self.request_response(controller_endpoint, message, timeout)
            if response.message_type == "WORK_REQUEST":
                work_request = WorkRequest.from_dict(response.to_dict())
                return work_request
            return None
        except TimeoutError:
            return None
    
    async def submit_work(
        self, 
        node_endpoint: str, 
        work_request: WorkRequest,
        timeout: Optional[timedelta] = None
    ) -> WorkResponse:
        """Submit work to a compute node."""
        response = await self.request_response(node_endpoint, work_request, timeout)
        
        if response.message_type != "WORK_RESPONSE":
            raise HPCError(f"Expected WORK_RESPONSE, got {response.message_type}")
        
        return WorkResponse.from_dict(response.to_dict())
    
    async def submit_work_response(
        self, 
        controller_endpoint: str, 
        work_response: WorkResponse,
        timeout: Optional[timedelta] = None
    ) -> None:
        """Submit work response to controller."""
        await self.send_message(controller_endpoint, work_response, timeout)
    
    def register_handler(
        self, 
        message_type: str, 
        handler: Callable[[Connection, Message], Optional[Message]]
    ) -> None:
        """Register a message handler."""
        self.router.register_handler(message_type, handler)
    
    async def start_server(self, endpoint: str) -> Server:
        """Start the protocol server."""
        server = await self.transport.listen(endpoint)
        
        # Start message handling loop
        asyncio.create_task(self._handle_connections(server))
        
        return server
    
    async def _handle_connections(self, server: Server) -> None:
        """Handle incoming connections."""
        while server.is_running:
            try:
                connection = await server.accept()
                if connection:
                    asyncio.create_task(self._handle_connection(connection))
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
                await asyncio.sleep(1)
    
    async def _handle_connection(self, connection: Connection) -> None:
        """Handle messages from a single connection."""
        try:
            while not connection.is_closed:
                message = await connection.receive()
                if message:
                    response = await self.router.route_message(connection, message)
                    if response:
                        await connection.send(response)
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            await connection.close()
    
    async def _handle_health_check(self, connection: Connection, message: Message) -> Message:
        """Handle health check messages."""
        return Message(
            source_node="controller",
            target_node=message.source_node,
            message_type="HEALTH_CHECK_RESPONSE",
            payload={"status": "healthy", "timestamp": datetime.utcnow().isoformat()},
            correlation_id=message.message_id
        )
    
    async def _handle_error_report(self, connection: Connection, message: Message) -> None:
        """Handle error report messages."""
        error_details = message.payload
        logger.error(f"Error report from {message.source_node}: {error_details}")


class WorkQueue:
    """Queue for managing work distribution."""
    
    def __init__(self, max_size: int = 1000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._work_items: Dict[str, WorkRequest] = {}
        self._lock = asyncio.Lock()
    
    async def put(self, work_request: WorkRequest) -> None:
        """Add work request to queue."""
        async with self._lock:
            self._work_items[work_request.work_id] = work_request
        await self._queue.put(work_request)
    
    async def get(self) -> Optional[WorkRequest]:
        """Get next work request from queue."""
        try:
            work_request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            async with self._lock:
                self._work_items.pop(work_request.work_id, None)
            return work_request
        except asyncio.TimeoutError:
            return None
    
    async def get_by_id(self, work_id: str) -> Optional[WorkRequest]:
        """Get work request by ID."""
        async with self._lock:
            return self._work_items.get(work_id)
    
    async def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()
    
    async def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
