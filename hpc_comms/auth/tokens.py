"""Token management for authentication."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False


class TokenManager(ABC):
    """Abstract token manager interface."""
    
    @abstractmethod
    async def generate_token(self, payload: Dict[str, Any], expiry: Optional[timedelta] = None) -> str:
        """Generate a new token."""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token and return payload."""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> None:
        """Revoke a token."""
        pass


class SimpleTokenManager(TokenManager):
    """Simple in-memory token manager using HMAC signatures."""
    
    def __init__(self, secret_key: str, default_expiry: timedelta = timedelta(hours=24)):
        self.secret_key = secret_key.encode()
        self.default_expiry = default_expiry
        self.tokens: Dict[str, Dict[str, Any]] = {}
        self.revoked_tokens: set[str] = set()
    
    async def generate_token(self, payload: Dict[str, Any], expiry: Optional[timedelta] = None) -> str:
        """Generate a new token with HMAC signature."""
        expiry_time = datetime.utcnow() + (expiry or self.default_expiry)
        
        # Create token payload
        token_payload = {
            **payload,
            "exp": expiry_time.timestamp(),
            "iat": datetime.utcnow().timestamp(),
            "jti": secrets.token_hex(16)  # JWT ID
        }
        
        # Generate signature
        payload_str = json.dumps(token_payload, sort_keys=True, separators=(',', ':'))
        signature = hmac.new(self.secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
        
        # Create token
        token_data = {
            "payload": token_payload,
            "signature": signature
        }
        token = base64.b64encode(json.dumps(token_data).encode()).decode()
        
        # Store token
        self.tokens[token_payload["jti"]] = {
            "token": token,
            "payload": token_payload,
            "created_at": datetime.utcnow()
        }
        
        return token
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a token and return payload."""
        if token in self.revoked_tokens:
            return None
        
        try:
            # Decode token
            token_bytes = base64.b64decode(token, validate=True)
            token_data = json.loads(token_bytes.decode())
            payload = token_data["payload"]
            signature = token_data["signature"]
            
            # Verify signature
            payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            expected_signature = hmac.new(self.secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
            
            if not secrets.compare_digest(signature, expected_signature):
                return None
            
            # Check expiry
            if datetime.utcnow().timestamp() > payload["exp"]:
                await self.revoke_token(token)
                return None
            
            return payload
            
        except (json.JSONDecodeError, KeyError, binascii.Error, ValueError, UnicodeDecodeError):
            return None
    
    async def revoke_token(self, token: str) -> None:
        """Revoke a token."""
        self.revoked_tokens.add(token)
        
        # Remove from active tokens
        try:
            token_data = json.loads(base64.b64decode(token).decode())
            jti = token_data["payload"]["jti"]
            self.tokens.pop(jti, None)
        except (json.JSONDecodeError, KeyError, base64.binascii.Error):
            pass
    
    async def cleanup_expired_tokens(self) -> None:
        """Clean up expired tokens."""
        current_time = datetime.utcnow().timestamp()
        expired_tokens = []
        
        for jti, token_info in self.tokens.items():
            if token_info["payload"]["exp"] < current_time:
                expired_tokens.append(jti)
        
        for jti in expired_tokens:
            token_info = self.tokens.pop(jti, {})
            self.revoked_tokens.add(token_info.get("token", ""))


class JWTTokenManager(TokenManager):
    """JWT token manager using PyJWT."""
    
    def __init__(
        self, 
        secret_key: str, 
        algorithm: str = "HS256",
        default_expiry: timedelta = timedelta(hours=24)
    ):
        if not JWT_AVAILABLE:
            raise ImportError("PyJWT is required for JWTTokenManager. Install with: pip install pyjwt")
        
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.default_expiry = default_expiry
        self.revoked_tokens: set[str] = set()
    
    async def generate_token(self, payload: Dict[str, Any], expiry: Optional[timedelta] = None) -> str:
        """Generate a new JWT token."""
        expiry_time = datetime.utcnow() + (expiry or self.default_expiry)
        
        token_payload = {
            **payload,
            "exp": expiry_time,
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)
        }
        
        return jwt.encode(token_payload, self.secret_key, algorithm=self.algorithm)
    
    async def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate a JWT token and return payload."""
        if token in self.revoked_tokens:
            return None
        
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            await self.revoke_token(token)
            return None
        except jwt.InvalidTokenError:
            return None
    
    async def revoke_token(self, token: str) -> None:
        """Revoke a JWT token."""
        self.revoked_tokens.add(token)


# Required imports
import base64
import hmac
import json
import secrets
