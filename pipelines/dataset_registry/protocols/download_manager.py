"""
Download manager with multi-protocol support

This module provides the core download infrastructure for the dataset registry,
supporting multiple protocols with retry logic, progress reporting, and cancellation.
"""

import hashlib
import time
import zipfile
import tarfile
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..core.logging_integration import get_logging_integration, log_registry_operation


class DownloadStatus(Enum):
    """Download operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    """Progress information for download operations"""
    total_bytes: Optional[int] = None
    downloaded_bytes: int = 0
    status: DownloadStatus = DownloadStatus.PENDING
    error_message: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_url: Optional[str] = None
    attempt_number: int = 1
    sources_tried: List[str] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> Optional[float]:
        """Calculate download progress percentage"""
        if self.total_bytes and self.total_bytes > 0:
            return (self.downloaded_bytes / self.total_bytes) * 100
        return None
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Calculate elapsed time in seconds"""
        if self.start_time:
            end = self.end_time or time.time()
            return end - self.start_time
        return None


class DownloadError(Exception):
    """Base exception for download operations"""
    def __init__(self, message: str, source_url: str, status_code: Optional[int] = None):
        self.source_url = source_url
        self.status_code = status_code
        super().__init__(message)


class NetworkError(DownloadError):
    """Network-related download errors"""
    pass


class AuthenticationError(DownloadError):
    """Authentication-related download errors"""
    pass


class FileNotFoundError(DownloadError):
    """File not found errors"""
    pass


class ProtocolDownloader(ABC):
    """Abstract base class for protocol-specific downloaders"""
    
    @abstractmethod
    def download(
        self,
        url: str,
        target_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        cancellation_token: Optional[Dict[str, bool]] = None
    ) -> DownloadProgress:
        """
        Download a file from the given URL to the target path
        
        Args:
            url: Source URL to download from
            target_path: Local path to save the file
            progress_callback: Optional callback for progress updates
            cancellation_token: Optional dict with 'cancelled' key for cancellation
            
        Returns:
            DownloadProgress object with final status
            
        Raises:
            DownloadError: If download fails
        """
        pass
    
    @abstractmethod
    def supports_protocol(self, url: str) -> bool:
        """Check if this downloader supports the given URL protocol"""
        pass


class HTTPDownloader(ProtocolDownloader):
    """HTTP/HTTPS protocol downloader with requests"""
    
    def __init__(self, timeout: int = 30, chunk_size: int = 8192):
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry configuration"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def supports_protocol(self, url: str) -> bool:
        """Check if URL uses HTTP or HTTPS protocol"""
        parsed = urlparse(url)
        return parsed.scheme.lower() in ['http', 'https']
    
    def download(
        self,
        url: str,
        target_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        cancellation_token: Optional[Dict[str, bool]] = None
    ) -> DownloadProgress:
        """
        Download file via HTTP/HTTPS with progress tracking
        
        Args:
            url: HTTP/HTTPS URL to download
            target_path: Local file path to save to
            progress_callback: Optional progress callback function
            cancellation_token: Optional cancellation token dict
            
        Returns:
            DownloadProgress with final status
            
        Raises:
            DownloadError: If download fails
        """
        progress = DownloadProgress(status=DownloadStatus.PENDING)
        
        if cancellation_token is None:
            cancellation_token = {'cancelled': False}
        
        try:
            progress.start_time = time.time()
            progress.status = DownloadStatus.IN_PROGRESS
            
            if progress_callback:
                progress_callback(progress)
            
            # Make HEAD request to get content length
            try:
                head_response = self.session.head(url, timeout=self.timeout)
                if head_response.status_code == 200:
                    progress.total_bytes = int(head_response.headers.get('content-length', 0)) or None
            except requests.RequestException:
                # HEAD request failed, continue without content length
                pass
            
            # Start the actual download
            response = self.session.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            # Update total bytes from actual response if not set
            if progress.total_bytes is None:
                progress.total_bytes = int(response.headers.get('content-length', 0)) or None
            
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress tracking
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    # Check for cancellation
                    if cancellation_token.get('cancelled', False):
                        progress.status = DownloadStatus.CANCELLED
                        progress.end_time = time.time()
                        if target_path.exists():
                            target_path.unlink()  # Clean up partial file
                        return progress
                    
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
                        progress.downloaded_bytes += len(chunk)
                        
                        if progress_callback:
                            progress_callback(progress)
            
            progress.status = DownloadStatus.COMPLETED
            progress.end_time = time.time()
            
            if progress_callback:
                progress_callback(progress)
                
            return progress
            
        except requests.exceptions.HTTPError as e:
            progress.status = DownloadStatus.FAILED
            progress.end_time = time.time()
            progress.error_message = f"HTTP error: {e}"
            
            if e.response.status_code == 404:
                raise FileNotFoundError(f"File not found at {url}", url, e.response.status_code)
            elif e.response.status_code in [401, 403]:
                raise AuthenticationError(f"Authentication failed for {url}", url, e.response.status_code)
            else:
                raise DownloadError(f"HTTP error {e.response.status_code}: {e}", url, e.response.status_code)
                
        except requests.exceptions.ConnectionError as e:
            progress.status = DownloadStatus.FAILED
            progress.end_time = time.time()
            progress.error_message = f"Connection error: {e}"
            raise NetworkError(f"Connection failed for {url}: {e}", url)
            
        except requests.exceptions.Timeout as e:
            progress.status = DownloadStatus.FAILED
            progress.end_time = time.time()
            progress.error_message = f"Timeout error: {e}"
            raise NetworkError(f"Timeout downloading from {url}: {e}", url)
            
        except requests.exceptions.SSLError as e:
            progress.status = DownloadStatus.FAILED
            progress.end_time = time.time()
            progress.error_message = f"SSL error: {e}"
            # SSL errors are retryable network errors
            raise NetworkError(f"SSL error downloading from {url}: {e}", url)
            
        except Exception as e:
            # Check if it's a MAC verification error (common SSL/TLS issue)
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['mac', 'ssl', 'tls', 'certificate', 'handshake']):
                progress.status = DownloadStatus.FAILED
                progress.end_time = time.time()
                progress.error_message = f"SSL/TLS error: {e}"
                # Make SSL/TLS errors retryable
                raise NetworkError(f"SSL/TLS error downloading from {url}: {e}", url)
            else:
                progress.status = DownloadStatus.FAILED
                progress.end_time = time.time()
                progress.error_message = f"Unexpected error: {e}"
                raise DownloadError(f"Unexpected error downloading from {url}: {e}", url)


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to delays


@dataclass
class SourceConfig:
    """Configuration for a download source"""
    url: str
    protocol: str
    priority: int = 0  # Lower numbers = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionConfig:
    """Configuration for file extraction"""
    format: str  # 'zip', 'tar', 'tar.gz', 'tar.bz2', etc.
    target_files: Optional[List[str]] = None  # Specific files to extract
    extract_all: bool = False  # Extract all files
    target_directory: Optional[Path] = None  # Directory to extract to


class CacheManager:
    """Manages local file caching to avoid redundant downloads"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, url: str) -> Path:
        """Generate cache file path for a URL"""
        # Create a safe filename from URL
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        parsed = urlparse(url)
        filename = Path(parsed.path).name or "download"
        
        # Combine filename with hash to avoid collisions
        cache_filename = f"{filename}_{url_hash}"
        return self.cache_dir / cache_filename
    
    def is_cached(self, url: str, expected_size: Optional[int] = None) -> bool:
        """Check if file is already cached and valid"""
        cache_path = self.get_cache_path(url)
        
        if not cache_path.exists():
            return False
        
        # Check file size if provided
        if expected_size is not None:
            actual_size = cache_path.stat().st_size
            if actual_size != expected_size:
                return False
        
        return True
    
    def get_cached_file(self, url: str) -> Optional[Path]:
        """Get cached file path if it exists"""
        cache_path = self.get_cache_path(url)
        return cache_path if cache_path.exists() else None
    
    def verify_cached_file(self, url: str, expected_checksum: Optional[str] = None) -> bool:
        """Verify cached file integrity"""
        cache_path = self.get_cache_path(url)
        
        if not cache_path.exists():
            return False
        
        if expected_checksum:
            actual_checksum = self._calculate_checksum(cache_path)
            return actual_checksum == expected_checksum
        
        return True
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class FileExtractor:
    """Handles file extraction from archives"""
    
    @staticmethod
    def extract_file(
        archive_path: Path,
        extraction_config: ExtractionConfig,
        target_path: Path
    ) -> List[Path]:
        """
        Extract files from an archive
        
        Args:
            archive_path: Path to the archive file
            extraction_config: Configuration for extraction
            target_path: Target path for extracted files
            
        Returns:
            List of extracted file paths
            
        Raises:
            DownloadError: If extraction fails
        """
        extracted_files = []
        
        try:
            if extraction_config.format.lower() == 'zip':
                extracted_files = FileExtractor._extract_zip(
                    archive_path, extraction_config, target_path
                )
            elif extraction_config.format.lower().startswith('tar'):
                extracted_files = FileExtractor._extract_tar(
                    archive_path, extraction_config, target_path
                )
            else:
                raise DownloadError(
                    f"Unsupported archive format: {extraction_config.format}",
                    str(archive_path)
                )
                
        except Exception as e:
            raise DownloadError(f"Extraction failed: {e}", str(archive_path))
        
        return extracted_files
    
    @staticmethod
    def _extract_zip(
        archive_path: Path,
        extraction_config: ExtractionConfig,
        target_path: Path
    ) -> List[Path]:
        """Extract files from ZIP archive"""
        extracted_files = []
        
        with zipfile.ZipFile(archive_path, 'r') as zip_file:
            if extraction_config.extract_all:
                # Extract all files
                zip_file.extractall(target_path.parent)
                extracted_files = [target_path.parent / name for name in zip_file.namelist()]
            elif extraction_config.target_files:
                # Extract specific files
                for target_file in extraction_config.target_files:
                    try:
                        zip_file.extract(target_file, target_path.parent)
                        extracted_path = target_path.parent / target_file
                        
                        # Move to final target path if it's the only file
                        if len(extraction_config.target_files) == 1:
                            extracted_path.rename(target_path)
                            extracted_files.append(target_path)
                        else:
                            extracted_files.append(extracted_path)
                            
                    except KeyError:
                        raise DownloadError(
                            f"File '{target_file}' not found in ZIP archive",
                            str(archive_path)
                        )
            else:
                # Extract first file by default
                names = zip_file.namelist()
                if names:
                    zip_file.extract(names[0], target_path.parent)
                    extracted_path = target_path.parent / names[0]
                    extracted_path.rename(target_path)
                    extracted_files.append(target_path)
        
        return extracted_files
    
    @staticmethod
    def _extract_tar(
        archive_path: Path,
        extraction_config: ExtractionConfig,
        target_path: Path
    ) -> List[Path]:
        """Extract files from TAR archive"""
        extracted_files = []
        
        # Determine compression mode
        mode = 'r'
        if extraction_config.format.endswith('.gz'):
            mode = 'r:gz'
        elif extraction_config.format.endswith('.bz2'):
            mode = 'r:bz2'
        
        with tarfile.open(archive_path, mode) as tar_file:
            if extraction_config.extract_all:
                # Extract all files
                tar_file.extractall(target_path.parent)
                extracted_files = [target_path.parent / member.name for member in tar_file.getmembers()]
            elif extraction_config.target_files:
                # Extract specific files
                for target_file in extraction_config.target_files:
                    try:
                        member = tar_file.getmember(target_file)
                        tar_file.extract(member, target_path.parent)
                        extracted_path = target_path.parent / target_file
                        
                        # Move to final target path if it's the only file
                        if len(extraction_config.target_files) == 1:
                            extracted_path.rename(target_path)
                            extracted_files.append(target_path)
                        else:
                            extracted_files.append(extracted_path)
                            
                    except KeyError:
                        raise DownloadError(
                            f"File '{target_file}' not found in TAR archive",
                            str(archive_path)
                        )
            else:
                # Extract first file by default
                members = tar_file.getmembers()
                if members:
                    tar_file.extract(members[0], target_path.parent)
                    extracted_path = target_path.parent / members[0].name
                    extracted_path.rename(target_path)
                    extracted_files.append(target_path)
        
        return extracted_files


class DownloadManager:
    """
    Main download manager that coordinates multiple protocol downloaders
    with retry logic, fallback mechanisms, caching, and extraction support
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        cache_dir: Optional[Path] = None,
        enable_caching: bool = True
    ):
        self.downloaders: List[ProtocolDownloader] = []
        self.retry_config = retry_config or RetryConfig()
        self.enable_caching = enable_caching
        
        if enable_caching:
            cache_path = cache_dir or Path.cwd() / "data" / "cache"
            self.cache_manager = CacheManager(cache_path)
        else:
            self.cache_manager = None
            
        self._register_default_downloaders()
    
    def _register_default_downloaders(self):
        """Register default protocol downloaders"""
        self.downloaders.append(HTTPDownloader())
    
    def register_downloader(self, downloader: ProtocolDownloader):
        """Register a new protocol downloader"""
        self.downloaders.append(downloader)
    
    def get_downloader(self, url: str) -> Optional[ProtocolDownloader]:
        """Get appropriate downloader for the given URL"""
        for downloader in self.downloaders:
            if downloader.supports_protocol(url):
                return downloader
        return None
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry using exponential backoff"""
        delay = self.retry_config.base_delay * (self.retry_config.backoff_factor ** (attempt - 1))
        delay = min(delay, self.retry_config.max_delay)
        
        if self.retry_config.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def download_with_extraction(
        self,
        sources: List[Union[str, SourceConfig]],
        target_path: Path,
        extraction_config: Optional[ExtractionConfig] = None,
        expected_checksum: Optional[str] = None,
        expected_size: Optional[int] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        cancellation_token: Optional[Dict[str, bool]] = None
    ) -> DownloadProgress:
        """
        Download and optionally extract files with caching support
        
        Args:
            sources: List of URLs or SourceConfig objects to try
            target_path: Local path to save the final file
            extraction_config: Optional extraction configuration for archives
            expected_checksum: Expected SHA256 checksum for validation
            expected_size: Expected file size for validation
            progress_callback: Optional progress callback function
            cancellation_token: Optional cancellation token dict
            
        Returns:
            DownloadProgress with final status
        """
        if not sources:
            raise DownloadError("No sources provided for download", "")
        
        # Convert strings to SourceConfig objects and sort by priority
        source_configs = []
        for i, source in enumerate(sources):
            if isinstance(source, str):
                source_configs.append(SourceConfig(url=source, protocol="auto", priority=i))
            else:
                source_configs.append(source)
        
        source_configs.sort(key=lambda x: x.priority)
        
        # Check cache first if enabled
        if self.enable_caching and self.cache_manager:
            for source_config in source_configs:
                if self.cache_manager.is_cached(source_config.url, expected_size):
                    if self.cache_manager.verify_cached_file(source_config.url, expected_checksum):
                        cached_path = self.cache_manager.get_cached_file(source_config.url)
                        
                        if cached_path:
                            # Handle extraction if needed
                            if extraction_config:
                                try:
                                    FileExtractor.extract_file(cached_path, extraction_config, target_path)
                                    progress = DownloadProgress(
                                        status=DownloadStatus.COMPLETED,
                                        total_bytes=cached_path.stat().st_size,
                                        downloaded_bytes=cached_path.stat().st_size,
                                        current_url=source_config.url
                                    )
                                    return progress
                                except Exception as e:
                                    # Cache file might be corrupted, continue to download
                                    continue
                            else:
                                # Copy cached file to target
                                target_path.parent.mkdir(parents=True, exist_ok=True)
                                import shutil
                                shutil.copy2(cached_path, target_path)
                                
                                progress = DownloadProgress(
                                    status=DownloadStatus.COMPLETED,
                                    total_bytes=cached_path.stat().st_size,
                                    downloaded_bytes=cached_path.stat().st_size,
                                    current_url=source_config.url
                                )
                                return progress
        
        # Download from sources
        return self.download_with_fallback(
            sources, target_path, extraction_config, expected_checksum,
            progress_callback, cancellation_token
        )
    
    def download_with_fallback(
        self,
        sources: List[Union[str, SourceConfig]],
        target_path: Path,
        extraction_config: Optional[ExtractionConfig] = None,
        expected_checksum: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        cancellation_token: Optional[Dict[str, bool]] = None,
        dataset_name: Optional[str] = None
    ) -> DownloadProgress:
        """
        Download from multiple sources with automatic fallback
        
        Args:
            sources: List of URLs or SourceConfig objects to try
            target_path: Local path to save the file
            extraction_config: Optional extraction configuration for archives
            expected_checksum: Expected SHA256 checksum for validation
            progress_callback: Optional progress callback function
            cancellation_token: Optional cancellation token dict
            dataset_name: Optional dataset name for logging
            
        Returns:
            DownloadProgress with final status
            
        Raises:
            DownloadError: If all sources fail
        """
        # Convert strings to SourceConfig objects and sort by priority
        source_configs = []
        for i, source in enumerate(sources):
            if isinstance(source, str):
                source_configs.append(SourceConfig(url=source, protocol="auto", priority=i))
            else:
                source_configs.append(source)
        
        source_configs.sort(key=lambda x: x.priority)
        
        progress = DownloadProgress(status=DownloadStatus.PENDING)
        last_error = None
        
        if cancellation_token is None:
            cancellation_token = {'cancelled': False}
        
        # Log download start
        start_time = time.time()
        if dataset_name:
            log_registry_operation(
                "download",
                dataset_name,
                status="started",
                metadata={
                    "sources_count": len(source_configs),
                    "primary_source": source_configs[0].url if source_configs else None,
                    "target_path": str(target_path),
                    "has_checksum": expected_checksum is not None,
                    "has_extraction": extraction_config is not None
                }
            )
        
        for source_config in source_configs:
            if cancellation_token.get('cancelled', False):
                progress.status = DownloadStatus.CANCELLED
                return progress
            
            progress.current_url = source_config.url
            progress.sources_tried.append(source_config.url)
            
            try:
                # Determine download path (cache or direct)
                if self.enable_caching and self.cache_manager:
                    download_path = self.cache_manager.get_cache_path(source_config.url)
                else:
                    download_path = target_path
                
                # Try this source with retry logic
                result = self._download_with_retry(
                    source_config.url,
                    download_path,
                    progress_callback,
                    cancellation_token
                )
                
                if result.status == DownloadStatus.COMPLETED:
                    # Handle extraction if needed FIRST
                    if extraction_config:
                        try:
                            FileExtractor.extract_file(download_path, extraction_config, target_path)
                        except Exception as e:
                            result.status = DownloadStatus.FAILED
                            result.error_message = f"EXTRACTION_FAILED: {e}"
                            last_error = result.error_message
                            continue
                    elif download_path != target_path:
                        # Copy from cache to target if not extracted
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(download_path, target_path)
                    
                    # Verify checksum of the FINAL file (extracted or copied)
                    if expected_checksum:
                        actual_checksum = self._calculate_file_checksum(target_path)
                        if actual_checksum != expected_checksum:
                            result.status = DownloadStatus.FAILED
                            result.error_message = f"CHECKSUM_MISMATCH: expected {expected_checksum}, got {actual_checksum} for file {target_path}"
                            last_error = result.error_message
                            continue
                    
                    # Log successful completion
                    if dataset_name:
                        duration_ms = (time.time() - start_time) * 1000
                        log_registry_operation(
                            "download",
                            dataset_name,
                            status="success",
                            duration_ms=duration_ms,
                            metadata={
                                "source_used": source_config.url,
                                "sources_tried": len(progress.sources_tried),
                                "file_size": target_path.stat().st_size if target_path.exists() else None,
                                "checksum_verified": expected_checksum is not None,
                                "extraction_performed": extraction_config is not None
                            }
                        )
                    
                    return result
                elif result.status == DownloadStatus.CANCELLED:
                    return result
                else:
                    last_error = result.error_message or f"Download failed with status: {result.status}"
                    
            except DownloadError as e:
                last_error = str(e)
                # Continue to next source
                continue
        
        # All sources failed
        progress.status = DownloadStatus.FAILED
        progress.error_message = f"All {len(source_configs)} sources failed. Last error: {last_error or 'Unknown'}"
        
        # Log download failure
        if dataset_name:
            duration_ms = (time.time() - start_time) * 1000
            log_registry_operation(
                "download",
                dataset_name,
                status="failed",
                duration_ms=duration_ms,
                error=progress.error_message,
                metadata={
                    "sources_tried": len(source_configs),
                    "sources_attempted": [s.url for s in source_configs]
                }
            )
        
        raise DownloadError(
            f"Failed to download from all sources: {[s.url for s in source_configs]}. Last error: {last_error or 'Unknown'}",
            source_configs[-1].url if source_configs else ""
        )
    
    def _download_with_retry(
        self,
        url: str,
        target_path: Path,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        cancellation_token: Optional[Dict[str, bool]] = None
    ) -> DownloadProgress:
        """
        Download from a single URL with retry logic
        
        Args:
            url: Source URL to download
            target_path: Local path to save the file
            progress_callback: Optional progress callback function
            cancellation_token: Optional cancellation token dict
            
        Returns:
            DownloadProgress with final status
        """
        downloader = self.get_downloader(url)
        if not downloader:
            raise DownloadError(f"No downloader available for URL: {url}", url)
        
        last_error = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            if cancellation_token and cancellation_token.get('cancelled', False):
                progress = DownloadProgress(status=DownloadStatus.CANCELLED)
                return progress
            
            try:
                # Clean up any partial file from previous attempt
                if attempt > 1 and target_path.exists():
                    target_path.unlink()
                
                progress = downloader.download(url, target_path, progress_callback, cancellation_token)
                progress.attempt_number = attempt
                
                if progress.status == DownloadStatus.COMPLETED:
                    return progress
                elif progress.status == DownloadStatus.CANCELLED:
                    return progress
                else:
                    last_error = progress.error_message
                    
            except (NetworkError, FileNotFoundError, AuthenticationError) as e:
                last_error = str(e)
                
                # Don't retry for certain error types
                if isinstance(e, (FileNotFoundError, AuthenticationError)):
                    progress = DownloadProgress(
                        status=DownloadStatus.FAILED,
                        error_message=str(e),
                        attempt_number=attempt
                    )
                    return progress
                
                # For network errors, retry with backoff
                if attempt < self.retry_config.max_attempts:
                    delay = self._calculate_retry_delay(attempt)
                    
                    # For SSL/MAC errors, ensure minimum 1 second delay
                    if isinstance(e, NetworkError) and any(keyword in str(e).lower() for keyword in ['ssl', 'tls', 'mac', 'certificate', 'handshake']):
                        delay = max(delay, 1.0)
                    
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    progress = DownloadProgress(
                        status=DownloadStatus.FAILED,
                        error_message=str(e),
                        attempt_number=attempt
                    )
                    return progress
            
            except Exception as e:
                # Unexpected error, don't retry
                progress = DownloadProgress(
                    status=DownloadStatus.FAILED,
                    error_message=f"Unexpected error: {e}",
                    attempt_number=attempt
                )
                return progress
        
        # Should not reach here, but just in case
        progress = DownloadProgress(
            status=DownloadStatus.FAILED,
            error_message=f"Max attempts ({self.retry_config.max_attempts}) exceeded. Last error: {last_error or 'Unknown'}",
            attempt_number=self.retry_config.max_attempts
        )
        return progress
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def download(
        self,
        url: str,
        target_path: Path,
        extraction_config: Optional[ExtractionConfig] = None,
        expected_checksum: Optional[str] = None,
        expected_size: Optional[int] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        cancellation_token: Optional[Dict[str, bool]] = None
    ) -> DownloadProgress:
        """
        Download a file using the appropriate protocol downloader
        
        Args:
            url: Source URL to download
            target_path: Local path to save the file
            extraction_config: Optional extraction configuration for archives
            expected_checksum: Expected SHA256 checksum for validation
            expected_size: Expected file size for validation
            progress_callback: Optional progress callback function
            cancellation_token: Optional cancellation token dict
            
        Returns:
            DownloadProgress with final status
            
        Raises:
            DownloadError: If no suitable downloader found or download fails
        """
        return self.download_with_extraction(
            [url], target_path, extraction_config, expected_checksum, expected_size,
            progress_callback, cancellation_token
        )