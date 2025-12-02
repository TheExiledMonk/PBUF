"""Authentication providers for HPC communication."""

from __future__ import annotations

import ssl
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Dict, Optional

from .tokens import TokenManager, SimpleTokenManager, JWTTokenManager
from ..core.errors import AuthenticationError


class AuthProvider(ABC):
    """Authentication provider interface."""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[str]:
        """Authenticate credentials and return token or None."""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token and return payload."""
        pass


class TokenAuthProvider(AuthProvider):
    """Token-based authentication provider."""
    
    def __init__(
        self, 
        token_manager: Optional[TokenManager] = None,
        secret_key: Optional[str] = None,
        use_jwt: bool = False
    ):
        if token_manager:
            self.token_manager = token_manager
        else:
            if not secret_key:
                raise ValueError("secret_key is required when token_manager is not provided")
            
            if use_jwt:
                self.token_manager = JWTTokenManager(secret_key)
            else:
                self.token_manager = SimpleTokenManager(secret_key)
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[str]:
        """Authenticate credentials and generate token."""
        # For this simple implementation, we just check for required fields
        node_id = credentials.get("node_id")
        if not node_id:
            raise AuthenticationError("node_id is required")
        
        # In a real implementation, you would validate against a database
        # For now, we accept any node_id and generate a token
        payload = {
            "node_id": node_id,
            "type": "compute_node" if credentials.get("capabilities") else "controller"
        }
        
        try:
            return await self.token_manager.generate_token(payload)
        except Exception as e:
            raise AuthenticationError(f"Failed to generate token: {e}")
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token."""
        try:
            payload = await self.token_manager.validate_token(token)
            if payload is None:
                raise AuthenticationError("Invalid token")
            return payload
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {e}")


class CertificateAuthProvider(AuthProvider):
    """Certificate-based authentication provider."""
    
    def __init__(self, ca_cert_file: str, require_client_cert: bool = True):
        self.ca_cert_file = ca_cert_file
        self.require_client_cert = require_client_cert
        self.authorized_certs: Dict[str, Dict[str, Any]] = {}
        
        # Load CA certificates
        self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.ssl_context.load_verify_locations(ca_cert_file)
        
        if require_client_cert:
            self.ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    def add_authorized_certificate(self, cert_fingerprint: str, node_info: Dict[str, Any]) -> None:
        """Add an authorized certificate fingerprint."""
        self.authorized_certs[cert_fingerprint] = node_info
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[str]:
        """Authenticate using certificate."""
        cert_data = credentials.get("certificate")
        if not cert_data:
            raise AuthenticationError("Certificate is required")
        
        # Extract certificate fingerprint
        cert_fingerprint = self._extract_fingerprint(cert_data)
        
        # Check if certificate is authorized
        if cert_fingerprint not in self.authorized_certs:
            raise AuthenticationError("Certificate not authorized")
        
        # Generate a simple token for the session
        node_info = self.authorized_certs[cert_fingerprint]
        payload = {
            "node_id": node_info["node_id"],
            "type": "compute_node",
            "cert_fingerprint": cert_fingerprint
        }
        
        # Use simple token manager for certificate auth
        token_manager = SimpleTokenManager(secrets.token_hex(32))
        return await token_manager.generate_token(payload)
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a certificate-based token."""
        # For certificate auth, we need to validate the token and certificate
        token_manager = SimpleTokenManager(secrets.token_hex(32))
        payload = await token_manager.validate_token(token)
        
        if payload and "cert_fingerprint" in payload:
            cert_fingerprint = payload["cert_fingerprint"]
            if cert_fingerprint in self.authorized_certs:
                return payload
        
        return None
    
    def _extract_fingerprint(self, cert_data: str) -> str:
        """Extract SHA-256 fingerprint from certificate data."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.serialization import load_pem_x509_certificate
        from cryptography.hazmat.backends import default_backend
        
        try:
            cert = load_pem_x509_certificate(cert_data.encode(), default_backend())
            fingerprint = cert.fingerprint(hashes.SHA256())
            return fingerprint.hex()
        except Exception as e:
            raise AuthenticationError(f"Failed to extract certificate fingerprint: {e}")


class NoAuthProvider(AuthProvider):
    """No authentication (for testing only)."""
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[str]:
        """No authentication - return dummy token."""
        return "no-auth-token"
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Always validate tokens."""
        return {"node_id": "anonymous", "type": "unauthenticated"}


# Utility functions
def create_token_auth_provider(
    secret_key: str,
    use_jwt: bool = False,
    token_expiry: timedelta = timedelta(hours=24)
) -> TokenAuthProvider:
    """Create a token-based authentication provider."""
    if use_jwt:
        token_manager = JWTTokenManager(secret_key, default_expiry=token_expiry)
    else:
        token_manager = SimpleTokenManager(secret_key, default_expiry=token_expiry)
    
    return TokenAuthProvider(token_manager)


def create_certificate_auth_provider(
    ca_cert_file: str,
    authorized_certs_file: Optional[str] = None
) -> CertificateAuthProvider:
    """Create a certificate-based authentication provider."""
    provider = CertificateAuthProvider(ca_cert_file)
    
    if authorized_certs_file:
        import json
        with open(authorized_certs_file, 'r') as f:
            authorized_certs = json.load(f)
        
        for fingerprint, node_info in authorized_certs.items():
            provider.add_authorized_certificate(fingerprint, node_info)
    
    return provider


# Required imports
import secrets
