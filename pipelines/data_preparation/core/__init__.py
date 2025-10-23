"""
Core components for the data preparation framework.

This module contains the fundamental interfaces, schemas, and base classes
that define the framework's architecture.
"""

from .schema import StandardDataset
from .interfaces import DerivationModule
from .validation import ValidationEngine

__all__ = [
    "StandardDataset",
    "DerivationModule",
    "ValidationEngine"
]