"""
Pipeline integration components

This module provides seamless integration with existing PBUF pipelines,
including drop-in replacements for dataset loading functions.
"""

from .dataset_integration import load_dataset, verify_all_datasets

__all__ = ["load_dataset", "verify_all_datasets"]