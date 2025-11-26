"""Standardized dataset loaders for cosmos2 (no dependency on cosmos/)."""

from .registry import get_dataset, Dataset

__all__ = ["get_dataset", "Dataset"]
