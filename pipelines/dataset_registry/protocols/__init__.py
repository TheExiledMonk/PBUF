"""
Multi-protocol download support

This module provides download managers for various protocols including
HTTPS, Zenodo API, arXiv, and local mirrors.
"""

from .download_manager import DownloadManager

__all__ = ["DownloadManager"]