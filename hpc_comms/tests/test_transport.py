"""Tests for transport layer."""

import pytest
import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

from hpc_comms.core.transport import Transport, Connection, Server, Router, ConnectionPool, CircuitBreaker
from hpc_comms.core.messages import Message
from hpc_comms.core.errors import ConnectionError, TimeoutError


class MockConnection(Connection):
    """Mock connection for testing."""
    
    def __init__(self, endpoint: str, timeout: timedelta = None):
        super().__init__(endpoint, timeout)
        self.sent_messages = []
        self.received_messages = asyncio.Queue()
        self.closed = False
    
    async def send(self, message: Message) -> None:
        if self.closed:
            raise ConnectionError("Connection is closed")
        self.sent_messages.append(message)
    
    async def receive(self, timeout: timedelta = None) -> Message:
        if self.closed:
            raise ConnectionError("Connection is closed")
        try:
            return await asyncio.wait_for(
                self.received_messages.get(), 
                timeout=timeout.total_seconds() if timeout else 1.0
            )
        except asyncio.TimeoutError:
            raise TimeoutError("Receive timeout")
    
    async def close(self) -> None:
        self.closed = True
        self.is_closed = True
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    def add_message(self, message: Message):
        """Add a message to be received."""
        self.received_messages.put_nowait(message)


class NonReusingMockTransport(Transport):
    """Mock transport that never reuses connections (for pool testing)."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.connections = {}
        self.servers = {}
    
    async def connect(self, endpoint: str, timeout: timedelta = None) -> MockConnection:
        # Always create a new connection
        conn = MockConnection(endpoint, timeout or self.default_timeout)
        self.connections[endpoint] = conn
        return conn
    
    async def listen(self, endpoint: str) -> Server:
        server = MockServer(endpoint)
        self.servers[endpoint] = server
        return server
    
    def get_scheme(self) -> str:
        return "mock"


class MockTransport(Transport):
    """Mock transport for testing."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.connections = {}
        self.servers = {}
    
    async def connect(self, endpoint: str, timeout: timedelta = None) -> MockConnection:
        if endpoint in self.connections and not self.connections[endpoint].is_closed:
            return self.connections[endpoint]
        
        conn = MockConnection(endpoint, timeout or self.default_timeout)
        self.connections[endpoint] = conn
        return conn
    
    async def listen(self, endpoint: str) -> Server:
        server = MockServer(endpoint)
        self.servers[endpoint] = server
        return server
    
    def get_scheme(self) -> str:
        return "mock"


class MockServer(Server):
    """Mock server for testing."""
    
    def __init__(self, endpoint: str):
        super().__init__(endpoint)
        self.connection_queue = asyncio.Queue()
        self.connections = []
    
    async def start(self) -> None:
        self.is_running = True
    
    async def stop(self) -> None:
        self.is_running = False
        for conn in self.connections:
            await conn.close()
    
    async def accept(self) -> Connection:
        if not self.is_running:
            return None
        try:
            return await asyncio.wait_for(self.connection_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def add_connection(self, connection: Connection):
        """Add a connection for testing."""
        self.connections.append(connection)
        self.connection_queue.put_nowait(connection)


class TestConnection:
    """Test connection functionality."""
    
    @pytest.mark.asyncio
    async def test_connection_send_receive(self):
        """Test basic send/receive functionality."""
        conn = MockConnection("test://endpoint")
        
        # Send message
        message = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST",
            payload={"data": "test"}
        )
        await conn.send(message)
        
        assert len(conn.sent_messages) == 1
        assert conn.sent_messages[0].message_type == "TEST"
        
        # Receive message
        conn.add_message(message)
        received = await conn.receive()
        
        assert received.message_type == "TEST"
        assert received.payload == {"data": "test"}
    
    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test connection timeout."""
        conn = MockConnection("test://endpoint", timeout=timedelta(seconds=0.1))
        
        with pytest.raises(TimeoutError):
            await conn.receive()
    
    @pytest.mark.asyncio
    async def test_connection_close(self):
        """Test connection closing."""
        conn = MockConnection("test://endpoint")
        
        # Normal operation
        message = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST"
        )
        await conn.send(message)
        
        # Close connection
        await conn.close()
        assert conn.is_closed
        
        # Operations on closed connection should fail
        with pytest.raises(ConnectionError):
            await conn.send(message)
        
        with pytest.raises(ConnectionError):
            await conn.receive()


class TestTransport:
    """Test transport functionality."""
    
    @pytest.mark.asyncio
    async def test_transport_connect(self):
        """Test transport connection."""
        transport = MockTransport()
        
        conn1 = await transport.connect("test://endpoint1")
        conn2 = await transport.connect("test://endpoint1")  # Should reuse
        
        assert conn1 is conn2
        assert conn1.endpoint == "test://endpoint1"
        
        # Different endpoint should create new connection
        conn3 = await transport.connect("test://endpoint2")
        assert conn3 is not conn1
        assert conn3.endpoint == "test://endpoint2"
    
    @pytest.mark.asyncio
    async def test_transport_listen(self):
        """Test transport listening."""
        transport = MockTransport()
        
        server = await transport.listen("test://endpoint")
        assert isinstance(server, Server)
        assert server.endpoint == "test://endpoint"
    
    @pytest.mark.asyncio
    async def test_transport_send_message(self):
        """Test sending message through transport."""
        transport = MockTransport()
        
        message = Message(
            source_node="node1",
            target_node="node2",
            message_type="TEST"
        )
        
        # Get connection first, then send message
        conn = await transport.connect("test://endpoint")
        await transport.send_message("test://endpoint", message)
        
        # Check that message was sent (connection should still have the message even after being closed)
        assert len(conn.sent_messages) == 1
        assert conn.sent_messages[0].message_type == "TEST"


class TestRouter:
    """Test message router."""
    
    @pytest.mark.asyncio
    async def test_router_registration(self):
        """Test handler registration."""
        router = Router()
        
        handler_called = False
        
        async def test_handler(conn, message):
            nonlocal handler_called
            handler_called = True
            return Message(
                source_node="router",
                target_node=message.source_node,
                message_type="RESPONSE"
            )
        
        router.register_handler("TEST", test_handler)
        
        conn = MockConnection("test://endpoint")
        message = Message(
            source_node="node1",
            target_node="router",
            message_type="TEST"
        )
        
        response = await router.route_message(conn, message)
        
        assert handler_called
        assert response.message_type == "RESPONSE"
    
    @pytest.mark.asyncio
    async def test_router_default_handler(self):
        """Test default handler."""
        router = Router()
        
        default_called = False
        
        async def default_handler(conn, message):
            nonlocal default_called
            default_called = True
            return Message(
                source_node="router",
                target_node=message.source_node,
                message_type="DEFAULT_RESPONSE"
            )
        
        router.register_default_handler(default_handler)
        
        conn = MockConnection("test://endpoint")
        message = Message(
            source_node="node1",
            target_node="router",
            message_type="UNKNOWN"
        )
        
        response = await router.route_message(conn, message)
        
        assert default_called
        assert response.message_type == "DEFAULT_RESPONSE"
    
    @pytest.mark.asyncio
    async def test_router_no_handler(self):
        """Test router with no handler."""
        router = Router()
        
        conn = MockConnection("test://endpoint")
        message = Message(
            source_node="node1",
            target_node="router",
            message_type="UNKNOWN"
        )
        
        response = await router.route_message(conn, message)
        assert response is None


class TestConnectionPool:
    """Test connection pool."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_reuse(self):
        """Test connection reuse in pool."""
        transport = MockTransport()
        pool = ConnectionPool(transport, max_size=2)
        
        # Get first connection
        conn1 = await pool.get_connection("test://endpoint")
        
        # Return connection to pool
        await pool.return_connection(conn1)
        
        # Get connection again - should reuse
        conn2 = await pool.get_connection("test://endpoint")
        
        assert conn1 is conn2
    
    @pytest.mark.asyncio
    async def test_connection_pool_max_size(self):
        """Test connection pool max size limit."""
        transport = NonReusingMockTransport()
        pool = ConnectionPool(transport, max_size=1)
        
        # Get connection and return to pool
        conn1 = await pool.get_connection("test://endpoint")
        await pool.return_connection(conn1)
        
        # Get connection again - should reuse from pool
        conn2 = await pool.get_connection("test://endpoint")
        assert conn2 is conn1  # Should be the same connection
        
        # Get another connection while conn2 is "in use"
        conn3 = await pool.get_connection("test://endpoint")  # This will be new since pool is empty
        assert conn3 is not conn1  # Should be a new connection
        
        # Return both connections - only one should be kept in pool
        await pool.return_connection(conn1)  # This should be accepted (pool is empty)
        await pool.return_connection(conn3)  # This should be closed due to size limit
        
        # Verify the pool behavior
        assert not conn1.is_closed  # conn1 should still be open (in pool)
        assert conn3.is_closed  # conn3 should be closed (discarded)
    
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self):
        """Test connection pool cleanup."""
        transport = MockTransport()
        pool = ConnectionPool(transport, max_size=2)
        
        conn1 = await pool.get_connection("test://endpoint")
        conn2 = await pool.get_connection("test://endpoint2")
        
        await pool.return_connection(conn1)
        await pool.return_connection(conn2)
        
        await pool.close_all()
        
        assert conn1.is_closed
        assert conn2.is_closed


class TestCircuitBreaker:
    """Test circuit breaker."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure(self):
        """Test circuit breaker with failures."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        async def fail_func():
            raise ValueError("Test error")
        
        # First failure
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        
        assert breaker.state == "OPEN"
        assert breaker.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test circuit breaker in open state."""
        breaker = CircuitBreaker(failure_threshold=1)
        
        async def fail_func():
            raise ValueError("Test error")
        
        # Trigger open state
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        
        assert breaker.state == "OPEN"
        
        # Calls should fail immediately when open
        with pytest.raises(ConnectionError):
            await breaker.call(fail_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open(self):
        """Test circuit breaker half-open state."""
        from datetime import datetime
        
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=timedelta(milliseconds=100)
        )
        
        async def fail_func():
            raise ValueError("Test error")
        
        async def success_func():
            return "success"
        
        # Trigger open state
        with pytest.raises(ValueError):
            await breaker.call(fail_func)
        
        assert breaker.state == "OPEN"
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Next call should be half-open
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0


if __name__ == "__main__":
    pytest.main([__file__])
