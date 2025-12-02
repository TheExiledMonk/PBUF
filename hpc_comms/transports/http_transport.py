"""HTTP transport implementation for HPC communication."""

from __future__ import annotations

import asyncio
import json
import ssl
from datetime import timedelta
from typing import Any, Dict, Optional, AsyncIterator

import aiohttp
from aiohttp import web, ClientSession, ClientTimeout

from ..core.transport import Transport, Connection, Server
from ..core.messages import Message, serialize_message, deserialize_message
from ..core.errors import HPCError, ConnectionError, TimeoutError, error_from_http_status


class HTTPConnection(Connection):
    """HTTP-based connection implementation."""
    
    def __init__(
        self, 
        endpoint: str, 
        session: ClientSession,
        timeout: Optional[timedelta] = None
    ):
        super().__init__(endpoint, timeout)
        self.session = session
        self._response_queue: asyncio.Queue = asyncio.Queue()
    
    async def send(self, message: Message) -> None:
        """Send a message via HTTP POST."""
        url = f"{self.endpoint}/api/v1/message"
        data = serialize_message(message)
        
        try:
            timeout = ClientTimeout(total=self.timeout.total_seconds())
            async with self.session.post(
                url, 
                data=data,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            ) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise error_from_http_status(
                        response.status, 
                        f"HTTP {response.status}: {error_text}",
                        {"url": url, "message_id": message.message_id}
                    )
        except asyncio.TimeoutError:
            raise TimeoutError("HTTP request timeout", operation="send", timeout_seconds=self.timeout.total_seconds())
        except aiohttp.ClientError as e:
            raise ConnectionError(f"HTTP connection error: {e}", endpoint=self.endpoint)
    
    async def receive(self, timeout: Optional[timedelta] = None) -> Optional[Message]:
        """Receive a message via HTTP GET (for server-sent events style)."""
        try:
            receive_timeout = timeout or self.timeout
            message_data = await asyncio.wait_for(
                self._response_queue.get(), 
                timeout=receive_timeout.total_seconds()
            )
            return deserialize_message(message_data)
        except asyncio.TimeoutError:
            raise TimeoutError("HTTP receive timeout", operation="receive")
        except asyncio.QueueEmpty:
            return None
    
    async def close(self) -> None:
        """Close the HTTP connection."""
        self.is_closed = True
        # Note: Session is managed by the transport, not individual connections


class HTTPServer(Server):
    """HTTP server implementation."""
    
    def __init__(self, endpoint: str, ssl_context: Optional[ssl.SSLContext] = None):
        super().__init__(endpoint)
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.ssl_context = ssl_context
        self._connection_queue: asyncio.Queue = asyncio.Queue()
        
        # Setup routes
        self.app.router.add_post("/api/v1/message", self._handle_message)
        self.app.router.add_get("/api/v1/message", self._handle_receive)
        self.app.router.add_get("/api/v1/health", self._handle_health)
        self.app.router.add_post("/api/v1/work/request", self._handle_work_request)
        self.app.router.add_post("/api/v1/work/{work_id}/response", self._handle_work_response)
        self.app.router.add_post("/api/v1/nodes/register", self._handle_node_register)
        self.app.router.add_post("/api/v1/nodes/{node_id}/heartbeat", self._handle_node_heartbeat)
        self.app.router.add_get("/api/v1/nodes/{node_id}/capabilities", self._handle_node_capabilities)
    
    async def start(self) -> None:
        """Start the HTTP server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        # Parse endpoint
        if "://" in self.endpoint:
            scheme, rest = self.endpoint.split("://", 1)
        else:
            scheme, rest = "http", self.endpoint
        
        if ":" in rest:
            host, port_str = rest.split(":", 1)
            port = int(port_str)
        else:
            host, port = rest, 8080
        
        self.site = web.TCPSite(
            self.runner, 
            host, 
            port, 
            ssl_context=self.ssl_context
        )
        await self.site.start()
        self.is_running = True
        
        print(f"HTTP server started on {self.endpoint}")
    
    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self.runner:
            await self.runner.cleanup()
        self.is_running = False
    
    async def accept(self) -> Optional[Connection]:
        """Accept an incoming connection (HTTP style)."""
        try:
            connection_data = await asyncio.wait_for(
                self._connection_queue.get(), 
                timeout=1.0
            )
            return connection_data
        except asyncio.TimeoutError:
            return None
    
    async def _handle_message(self, request: web.Request) -> web.Response:
        """Handle incoming message POST."""
        try:
            data = await request.text()
            message = deserialize_message(data)
            
            # Create a virtual connection for this request
            connection = HTTPConnection(
                endpoint=str(request.url),
                session=None,  # Not needed for server-side
                timeout=timedelta(seconds=30)
            )
            
            # Put connection in queue for accept()
            await self._connection_queue.put(connection)
            
            return web.json_response({"status": "accepted", "message_id": message.message_id})
            
        except Exception as e:
            return web.json_response(
                {"error": str(e)}, 
                status=400
            )
    
    async def _handle_receive(self, request: web.Request) -> web.Response:
        """Handle message GET request."""
        # For HTTP, this would typically be implemented with Server-Sent Events
        # For now, return empty response
        return web.json_response({"status": "no_messages"})
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Handle health check."""
        return web.json_response({
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time()
        })
    
    async def _handle_work_request(self, request: web.Request) -> web.Response:
        """Handle work request."""
        try:
            data = await request.text()
            message = deserialize_message(data)
            
            # This would be handled by the application layer
            # For now, just acknowledge
            return web.json_response({"status": "accepted", "work_id": message.payload.get("work_id")})
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
    
    async def _handle_work_response(self, request: web.Request) -> web.Response:
        """Handle work response."""
        try:
            work_id = request.match_info["work_id"]
            data = await request.text()
            message = deserialize_message(data)
            
            return web.json_response({"status": "accepted", "work_id": work_id})
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
    
    async def _handle_node_register(self, request: web.Request) -> web.Response:
        """Handle node registration."""
        try:
            data = await request.text()
            node_info = json.loads(data)
            
            return web.json_response({"status": "registered", "node_id": node_info.get("node_id")})
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
    
    async def _handle_node_heartbeat(self, request: web.Request) -> web.Response:
        """Handle node heartbeat."""
        try:
            node_id = request.match_info["node_id"]
            data = await request.text()
            heartbeat_data = json.loads(data)
            
            return web.json_response({"status": "received", "node_id": node_id})
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)
    
    async def _handle_node_capabilities(self, request: web.Request) -> web.Response:
        """Handle node capabilities query."""
        try:
            node_id = request.match_info["node_id"]
            
            # This would typically look up node capabilities
            return web.json_response({
                "node_id": node_id,
                "capabilities": {
                    "backend_type": "cpu",
                    "device_count": 1,
                    "memory_gb": 16.0
                }
            })
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)


class HTTPTransport(Transport):
    """HTTP transport implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.session: Optional[ClientSession] = None
        self.ssl_context: Optional[ssl.SSLContext] = None
        
        # Setup SSL if configured
        if self.config.get("ssl", {}).get("enabled", False):
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            cert_file = self.config["ssl"].get("cert_file")
            key_file = self.config["ssl"].get("key_file")
            if cert_file and key_file:
                self.ssl_context.load_cert_chain(cert_file, key_file)
    
    async def connect(self, endpoint: str, timeout: Optional[timedelta] = None) -> HTTPConnection:
        """Establish HTTP connection."""
        if not self.session:
            timeout_obj = ClientTimeout(total=self.default_timeout.total_seconds())
            connector = aiohttp.TCPConnector(ssl=self.ssl_context)
            self.session = ClientSession(timeout=timeout_obj, connector=connector)
        
        return HTTPConnection(endpoint, self.session, timeout or self.default_timeout)
    
    async def listen(self, endpoint: str) -> HTTPServer:
        """Start HTTP server."""
        return HTTPServer(endpoint, self.ssl_context)
    
    def get_scheme(self) -> str:
        """Get URL scheme."""
        return "https" if self.ssl_context else "http"
    
    async def close(self) -> None:
        """Close transport and cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None


# Helper functions for creating HTTP transport
def create_http_transport(
    ssl_enabled: bool = False,
    cert_file: Optional[str] = None,
    key_file: Optional[str] = None,
    timeout: timedelta = timedelta(seconds=30)
) -> HTTPTransport:
    """Create HTTP transport with configuration."""
    config = {
        "timeout": timeout.total_seconds(),
        "ssl": {
            "enabled": ssl_enabled,
            "cert_file": cert_file,
            "key_file": key_file
        }
    }
    return HTTPTransport(config)
