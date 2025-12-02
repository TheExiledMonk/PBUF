"""HPC Communication Module - Pure communication layer for distributed cosmos2."""

from .core.messages import Message, WorkRequest, WorkResponse, NodeInfo, NodeCapabilities
from .core.transport import Transport, Connection, Server
from .core.errors import HPCError, ConnectionError, AuthenticationError
from .core.protocol import HPCProtocol

__version__ = "0.1.0"
__all__ = [
    "Message", "WorkRequest", "WorkResponse", 
    "NodeInfo", "NodeCapabilities",
    "Transport", "Connection", "Server",
    "HPCError", "ConnectionError", "AuthenticationError",
    "HPCProtocol"
]
