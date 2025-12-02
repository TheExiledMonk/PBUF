"""Abstract transport interface for HPC communication."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, List, Optional, Callable, AsyncIterator

from .messages import Message
from .errors import HPCError, ConnectionError


class Connection(ABC):
    """Abstract connection interface for sending/receiving messages."""
    
    def __init__(self, endpoint: str, timeout: Optional[timedelta] = None):
        self.endpoint = endpoint
        self.timeout = timeout or timedelta(seconds=30)
        self.is_closed = False
    
    @abstractmethod
    async def send(self, message: Message) -> None:
        """Send a message through this connection."""
        pass
    
    @abstractmethod
    async def receive(self, timeout: Optional[timedelta] = None) -> Optional[Message]:
        """Receive a message from this connection."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class Server(ABC):
    """Abstract server interface for listening for incoming connections."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.is_running = False
    
    @abstractmethod
    async def start(self) -> None:
        """Start the server."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the server."""
        pass
    
    @abstractmethod
    async def accept(self) -> Optional[Connection]:
        """Accept an incoming connection."""
        pass
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class Transport(ABC):
    """Abstract transport interface for different communication backends."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_timeout = timedelta(seconds=self.config.get('timeout', 30))
    
    @abstractmethod
    async def connect(self, endpoint: str, timeout: Optional[timedelta] = None) -> Connection:
        """Establish connection to remote endpoint."""
        pass
    
    @abstractmethod
    async def listen(self, endpoint: str) -> Server:
        """Start listening for incoming connections."""
        pass
    
    @abstractmethod
    def get_scheme(self) -> str:
        """Get the URL scheme for this transport (e.g., 'http', 'grpc')."""
        pass
    
    async def send_message(
        self, 
        endpoint: str, 
        message: Message, 
        timeout: Optional[timedelta] = None
    ) -> None:
        """Send a message to an endpoint."""
        async with await self.connect(endpoint, timeout) as conn:
            await conn.send(message)
    
    async def request_response(
        self, 
        endpoint: str, 
        message: Message, 
        timeout: Optional[timedelta] = None
    ) -> Optional[Message]:
        """Send a message and wait for response."""
        async with await self.connect(endpoint, timeout) as conn:
            await conn.send(message)
            return await conn.receive(timeout)


class MessageHandler(ABC):
    """Abstract message handler interface."""
    
    @abstractmethod
    async def handle_message(self, connection: Connection, message: Message) -> Optional[Message]:
        """Handle an incoming message and optionally return a response."""
        pass


class Router:
    """Message router that dispatches to appropriate handlers."""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}
        self.default_handler: Optional[Callable] = None
    
    def register_handler(
        self, 
        message_type: str, 
        handler: Callable[[Connection, Message], Optional[Message]]
    ) -> None:
        """Register a handler for a specific message type."""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        self.handlers[message_type].append(handler)
    
    def register_default_handler(
        self, 
        handler: Callable[[Connection, Message], Optional[Message]]
    ) -> None:
        """Register a default handler for unknown message types."""
        self.default_handler = handler
    
    async def route_message(
        self, 
        connection: Connection, 
        message: Message
    ) -> Optional[Message]:
        """Route message to appropriate handler."""
        handlers = self.handlers.get(message.message_type, [])
        
        if handlers:
            # Call all handlers for this message type
            for handler in handlers:
                try:
                    response = await handler(connection, message)
                    if response:
                        return response
                except Exception as e:
                    # Log error but continue with other handlers
                    print(f"Handler error for {message.message_type}: {e}")
        
        elif self.default_handler:
            return await self.default_handler(connection, message)
        
        return None


class ConnectionPool:
    """Pool of reusable connections."""
    
    def __init__(self, transport: Transport, max_size: int = 10):
        self.transport = transport
        self.max_size = max_size
        self._pool: List[Connection] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_size)
    
    async def get_connection(self, endpoint: str) -> Connection:
        """Get a connection from the pool or create a new one."""
        async with self._semaphore:
            async with self._lock:
                # Try to find an existing connection for this endpoint
                for i, conn in enumerate(self._pool):
                    if conn.endpoint == endpoint and not conn.is_closed:
                        self._pool.pop(i)
                        return conn
                
                # Create new connection
                return await self.transport.connect(endpoint)
    
    async def return_connection(self, connection: Connection) -> None:
        """Return a connection to the pool."""
        if connection.is_closed:
            return
        
        async with self._lock:
            if len(self._pool) < self.max_size:
                self._pool.append(connection)
            else:
                await connection.close()
    
    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            for conn in self._pool:
                await conn.close()
            self._pool.clear()


class CircuitBreaker:
    """Circuit breaker for handling failures."""
    
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_timeout: timedelta = timedelta(seconds=60),
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
            else:
                raise ConnectionError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        return (
            self.last_failure_time and
            (datetime.now() - self.last_failure_time) >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


# Import datetime for circuit breaker
from datetime import datetime
