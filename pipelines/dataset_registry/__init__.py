"""
Centralized Dataset Downloader & Verification Registry

This module provides a unified system for acquiring, verifying, and documenting
all external datasets used in PBUF cosmological analyses. It ensures complete
reproducibility by maintaining structured provenance records.

Main components:
- core: Manifest management and schema validation
- protocols: Multi-protocol download support (HTTPS, Zenodo, arXiv)
- verification: Cryptographic and structural validation
- integration: Seamless pipeline integration
"""

__version__ = "1.0.0"

# Core exports
from .core.manifest_schema import DatasetManifest, DatasetInfo
try:
    from .core.registry_manager import RegistryManager
except ImportError:
    RegistryManager = None

from .verification.verification_engine import (
    VerificationEngine, 
    VerificationResult, 
    VerificationStatus,
    VerificationError,
    ChecksumError,
    SizeError,
    SchemaError
)

try:
    from .protocols.download_manager import DownloadManager
except ImportError:
    DownloadManager = None

try:
    from .integration.dataset_integration import load_dataset, verify_all_datasets
except ImportError:
    load_dataset = None
    verify_all_datasets = None

__all__ = [
    "DatasetManifest",
    "DatasetInfo", 
    "RegistryManager",
    "VerificationEngine",
    "VerificationResult",
    "VerificationStatus",
    "VerificationError",
    "ChecksumError", 
    "SizeError",
    "SchemaError",
    "DownloadManager",
    "load_dataset",
    "verify_all_datasets"
]