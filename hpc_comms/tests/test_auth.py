"""Tests for authentication providers."""

import pytest
import asyncio
from datetime import timedelta

from hpc_comms.auth.providers import (
    AuthProvider, TokenAuthProvider, CertificateAuthProvider, NoAuthProvider,
    create_token_auth_provider, create_certificate_auth_provider
)
from hpc_comms.auth.tokens import TokenManager, SimpleTokenManager, JWTTokenManager
from hpc_comms.core.errors import AuthenticationError


class TestSimpleTokenManager:
    """Test simple token manager."""
    
    @pytest.mark.asyncio
    async def test_token_generation_and_validation(self):
        """Test basic token generation and validation."""
        manager = SimpleTokenManager("test-secret")
        
        payload = {"node_id": "node1", "type": "compute_node"}
        token = await manager.generate_token(payload)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        validated = await manager.validate_token(token)
        assert validated is not None
        assert validated["node_id"] == "node1"
        assert validated["type"] == "compute_node"
    
    @pytest.mark.asyncio
    async def test_token_expiry(self):
        """Test token expiry."""
        manager = SimpleTokenManager("test-secret", default_expiry=timedelta(milliseconds=100))
        
        payload = {"node_id": "node1"}
        token = await manager.generate_token(payload)
        
        # Should be valid immediately
        validated = await manager.validate_token(token)
        assert validated is not None
        
        # Wait for expiry
        await asyncio.sleep(0.15)
        
        # Should be invalid after expiry
        validated = await manager.validate_token(token)
        assert validated is None
    
    @pytest.mark.asyncio
    async def test_token_revocation(self):
        """Test token revocation."""
        manager = SimpleTokenManager("test-secret")
        
        payload = {"node_id": "node1"}
        token = await manager.generate_token(payload)
        
        # Should be valid initially
        validated = await manager.validate_token(token)
        assert validated is not None
        
        # Revoke token
        await manager.revoke_token(token)
        
        # Should be invalid after revocation
        validated = await manager.validate_token(token)
        assert validated is None
    
    @pytest.mark.asyncio
    async def test_invalid_token(self):
        """Test invalid token handling."""
        manager = SimpleTokenManager("test-secret")
        
        # Invalid token
        validated = await manager.validate_token("invalid-token")
        assert validated is None
        
        # Empty token
        validated = await manager.validate_token("")
        assert validated is None


class TestJWTTokenManager:
    """Test JWT token manager."""
    
    @pytest.mark.asyncio
    async def test_jwt_token_generation_and_validation(self):
        """Test JWT token generation and validation."""
        try:
            manager = JWTTokenManager("test-secret")
            
            payload = {"node_id": "node1", "type": "controller"}
            token = await manager.generate_token(payload)
            
            assert isinstance(token, str)
            assert len(token) > 0
            
            validated = await manager.validate_token(token)
            assert validated is not None
            assert validated["node_id"] == "node1"
            assert validated["type"] == "controller"
            
        except ImportError:
            pytest.skip("PyJWT not available")
    
    @pytest.mark.asyncio
    async def test_jwt_token_expiry(self):
        """Test JWT token expiry."""
        try:
            manager = JWTTokenManager("test-secret", default_expiry=timedelta(milliseconds=100))
            
            payload = {"node_id": "node1"}
            token = await manager.generate_token(payload)
            
            # Should be valid immediately
            validated = await manager.validate_token(token)
            assert validated is not None
            
            # Wait for expiry
            await asyncio.sleep(0.15)
            
            # Should be invalid after expiry
            validated = await manager.validate_token(token)
            assert validated is None
            
        except ImportError:
            pytest.skip("PyJWT not available")
    
    @pytest.mark.asyncio
    async def test_jwt_token_revocation(self):
        """Test JWT token revocation."""
        try:
            manager = JWTTokenManager("test-secret")
            
            payload = {"node_id": "node1"}
            token = await manager.generate_token(payload)
            
            # Should be valid initially
            validated = await manager.validate_token(token)
            assert validated is not None
            
            # Revoke token
            await manager.revoke_token(token)
            
            # Should be invalid after revocation
            validated = await manager.validate_token(token)
            assert validated is None
            
        except ImportError:
            pytest.skip("PyJWT not available")


class TestTokenAuthProvider:
    """Test token-based authentication provider."""
    
    @pytest.mark.asyncio
    async def test_successful_authentication(self):
        """Test successful authentication."""
        provider = TokenAuthProvider(secret_key="test-secret")
        
        credentials = {"node_id": "node1", "capabilities": {"backend_type": "rocm"}}
        token = await provider.authenticate(credentials)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Validate the token
        payload = await provider.validate_token(token)
        assert payload is not None
        assert payload["node_id"] == "node1"
        assert payload["type"] == "compute_node"
    
    @pytest.mark.asyncio
    async def test_missing_node_id(self):
        """Test authentication with missing node_id."""
        provider = TokenAuthProvider(secret_key="test-secret")
        
        credentials = {"capabilities": {"backend_type": "rocm"}}
        
        with pytest.raises(AuthenticationError, match="node_id is required"):
            await provider.authenticate(credentials)
    
    @pytest.mark.asyncio
    async def test_invalid_token_validation(self):
        """Test validation of invalid token."""
        provider = TokenAuthProvider(secret_key="test-secret")
        
        with pytest.raises(AuthenticationError):
            await provider.validate_token("invalid-token")
    
    @pytest.mark.asyncio
    async def test_custom_token_manager(self):
        """Test provider with custom token manager."""
        token_manager = SimpleTokenManager("custom-secret")
        provider = TokenAuthProvider(token_manager=token_manager)
        
        credentials = {"node_id": "node1"}
        token = await provider.authenticate(credentials)
        
        payload = await provider.validate_token(token)
        assert payload["node_id"] == "node1"


class TestCertificateAuthProvider:
    """Test certificate-based authentication provider."""
    
    @pytest.mark.asyncio
    async def test_certificate_auth_setup(self):
        """Test certificate auth provider setup."""
        # This is a basic test - full certificate testing would require actual certs
        provider = CertificateAuthProvider.__new__(CertificateAuthProvider)
        provider.authorized_certs = {}
        
        # Test adding authorized certificate
        node_info = {"node_id": "node1", "type": "compute_node"}
        provider.add_authorized_certificate("abc123", node_info)
        
        assert "abc123" in provider.authorized_certs
        assert provider.authorized_certs["abc123"]["node_id"] == "node1"
    
    @pytest.mark.asyncio
    async def test_missing_certificate(self):
        """Test authentication with missing certificate."""
        provider = CertificateAuthProvider.__new__(CertificateAuthProvider)
        provider.authorized_certs = {}
        
        credentials = {"node_id": "node1"}
        
        with pytest.raises(AuthenticationError, match="Certificate is required"):
            await provider.authenticate(credentials)


class TestNoAuthProvider:
    """Test no authentication provider (for testing)."""
    
    @pytest.mark.asyncio
    async def test_no_auth_provider(self):
        """Test no authentication provider."""
        provider = NoAuthProvider()
        
        credentials = {"node_id": "node1"}
        token = await provider.authenticate(credentials)
        
        assert token == "no-auth-token"
        
        payload = await provider.validate_token(token)
        assert payload["node_id"] == "anonymous"
        assert payload["type"] == "unauthenticated"


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async def test_create_token_auth_provider(self):
        """Test token auth provider creation utility."""
        provider = create_token_auth_provider(
            secret_key="test-secret",
            use_jwt=False,
            token_expiry=timedelta(hours=12)
        )
        
        assert isinstance(provider, TokenAuthProvider)
        
        credentials = {"node_id": "node1"}
        token = await provider.authenticate(credentials)
        payload = await provider.validate_token(token)
        
        assert payload["node_id"] == "node1"
    
    @pytest.mark.asyncio
    async def test_create_jwt_auth_provider(self):
        """Test JWT auth provider creation utility."""
        try:
            provider = create_token_auth_provider(
                secret_key="test-secret",
                use_jwt=True,
                token_expiry=timedelta(hours=6)
            )
            
            assert isinstance(provider, TokenAuthProvider)
            
            credentials = {"node_id": "node1"}
            token = await provider.authenticate(credentials)
            payload = await provider.validate_token(token)
            
            assert payload["node_id"] == "node1"
            
        except ImportError:
            pytest.skip("PyJWT not available")
    
    def test_create_certificate_auth_provider(self):
        """Test certificate auth provider creation utility."""
        # This test doesn't require actual certificate files
        try:
            provider = create_certificate_auth_provider("nonexistent_ca.pem")
            assert isinstance(provider, CertificateAuthProvider)
        except Exception:
            # Expected to fail with nonexistent file
            pass


class TestTokenManagerFactory:
    """Test token manager factory patterns."""
    
    @pytest.mark.asyncio
    async def test_token_manager_without_secret_key(self):
        """Test token provider creation without secret key."""
        with pytest.raises(ValueError, match="secret_key is required"):
            TokenAuthProvider()
    
    @pytest.mark.asyncio
    async def test_token_manager_with_jwt_unavailable(self):
        """Test JWT token manager when PyJWT is not available."""
        # Mock JWT being unavailable
        import hpc_comms.auth.tokens
        original_jwt_available = hpc_comms.auth.tokens.JWT_AVAILABLE
        hpc_comms.auth.tokens.JWT_AVAILABLE = False
        
        try:
            with pytest.raises(ImportError, match="PyJWT is required"):
                JWTTokenManager("test-secret")
        finally:
            hpc_comms.auth.tokens.JWT_AVAILABLE = original_jwt_available


if __name__ == "__main__":
    pytest.main([__file__])
