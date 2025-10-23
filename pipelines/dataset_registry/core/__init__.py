"""
Core dataset registry components

This module contains the fundamental components for dataset manifest management,
schema validation, and registry operations.
"""

from .manifest_schema import DatasetManifest, DatasetInfo
from .registry_manager import RegistryManager

__all__ = ["DatasetManifest", "DatasetInfo", "RegistryManager"]