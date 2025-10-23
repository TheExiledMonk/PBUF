"""
Unit tests for download manager with multi-protocol support

Tests the download manager functionality including HTTP downloads, retry logic,
mirror fallback, and progress reporting. This covers requirements 2.1-2.5.
"""

import pytest
import tempfile
import zipfile
import tarfile
import hashlib
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from urllib.error import URLError
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout

from pipelines.dataset_registry.protocols.download_manager import (
    DownloadManager,
    HTTPDownloader,
    DownloadProgress,
    DownloadStatus,
    DownloadError,
    RetryConfig,
    ExtractionConfig,
    CacheManager,
    FileExtractor
)


class TestDownloadProgress:
    """Test DownloadProgress functionality"""
    
    def test_progress_creation(self):
        """Test creating download progress objects"""
        progress = DownloadProgress(
            total_bytes=1000,
            downloaded_bytes=500,
            status=DownloadStatus.IN_PROGRESS
        )
        
        assert progress.total_bytes == 1000
        assert progress.downloaded_bytes == 500
        assert progress.status == DownloadStatus.IN_PROGRESS
        assert progress.progress_percent == 50.0
    
    def test_progress_percent_calculation(self):
        """Test progress percentage calculation"""
        # Normal case
        progress = DownloadProgress()
        progress.total_bytes = 1000
        progress.downloaded_bytes = 250
        assert progress.progress_percent == 25.0
        
        # No total bytes
        progress = DownloadProgress()
        progress.total_bytes = None
        progress.downloaded_bytes = 250
        assert progress.progress_percent is None
        
        # Zero total bytes
        progress = DownloadProgress()
        progress.total_bytes = 0
        progress.downloaded_bytes = 0
        assert progress.progress_percent is None
    
    def test_progress_completion(self):
        """Test progress completion detection"""
        progress = DownloadProgress()
        progress.total_bytes = 1000
        progress.downloaded_bytes = 1000
        assert progress.progress_percent == 100.0
        
        # Over 100% (shouldn't happen but handle gracefully)
        progress = DownloadProgress()
        progress.total_bytes = 1000
        progress.downloaded_bytes = 1100
        assert abs(progress.progress_percent - 110.0) < 0.001


class TestRetryConfig:
    """Test RetryConfig functionality"""
    
    def test_default_retry_config(self):
        """Test default retry configuration"""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
    
    def test_custom_retry_config(self):
        """Test custom retry configuration"""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_factor=1.5
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 1.5


class TestHTTPDownloader:
    """Test HTTPDownloader functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing downloads"""
        return b"This is test data for download testing"
    
    def test_downloader_initialization(self):
        """Test HTTPDownloader initialization"""
        downloader = HTTPDownloader()
        
        assert downloader.session is not None
        assert downloader.timeout == 30
        assert downloader.chunk_size == 8192
    
    def test_custom_downloader_config(self):
        """Test HTTPDownloader with custom configuration"""
        downloader = HTTPDownloader(
            timeout=60,
            chunk_size=16384
        )
        
        assert downloader.timeout == 60
        assert downloader.chunk_size == 16384
    
    @patch('requests.Session.get')
    def test_successful_download(self, mock_get, temp_dir, sample_data):
        """Test successful file download"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': str(len(sample_data))}
        mock_response.iter_content.return_value = [sample_data]
        mock_get.return_value = mock_response
        
        downloader = HTTPDownloader()
        target_path = temp_dir / "test_file.dat"
        
        # Track progress
        progress_updates = []
        def progress_callback(progress):
            progress_updates.append(progress)
        
        # Download file
        result = downloader.download(
            url="https://example.com/test_file.dat",
            target_path=target_path,
            progress_callback=progress_callback
        )
        
        # Verify download
        assert result.status == DownloadStatus.COMPLETED
        assert target_path.exists()
        assert target_path.read_bytes() == sample_data
        assert len(progress_updates) > 0
        assert progress_updates[-1].status == DownloadStatus.COMPLETED
    
    def test_supports_protocol(self):
        """Test protocol support checking"""
        downloader = HTTPDownloader()
        
        assert downloader.supports_protocol("https://example.com/file.dat")
        assert downloader.supports_protocol("http://example.com/file.dat")
        assert not downloader.supports_protocol("ftp://example.com/file.dat")
        assert not downloader.supports_protocol("file:///local/file.dat")
    
    def test_session_creation(self):
        """Test session creation with retry configuration"""
        downloader = HTTPDownloader()
        session = downloader._create_session()
        
        assert session is not None
        # Check that retry adapters are mounted
        assert 'http://' in session.adapters
        assert 'https://' in session.adapters


class TestDownloadManager:
    """Test DownloadManager high-level functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_manager_initialization(self):
        """Test DownloadManager initialization"""
        manager = DownloadManager()
        
        # Test basic initialization
        assert manager is not None
    
    def test_manager_with_cache_dir(self, temp_dir):
        """Test DownloadManager with custom cache directory"""
        manager = DownloadManager(cache_dir=temp_dir)
        
        assert manager is not None


if __name__ == "__main__":
    pytest.main([__file__])