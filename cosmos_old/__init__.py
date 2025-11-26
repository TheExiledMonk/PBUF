"""
Top-level Cosmos package.

This module exposes a tiny surface so callers can import the shared model
interfaces and the factory without reaching into individual implementations.
"""

from cosmos.factory.model_factory import create_model
from cosmos.interfaces import CMBOutput, CosmologyModel

__all__ = ["create_model", "CMBOutput", "CosmologyModel"]
