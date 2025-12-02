"""
Independent Reporting System for Cosmos2 Science Runs

This is a standalone module that analyzes and reports on cosmos2 science run results.
It's completely separate from the cosmos2 codebase and can be used independently.
"""

from .core.report_generator import ReportGenerator
from .data.data_loader import DataLoader

__version__ = "1.0.0"
__all__ = ["ReportGenerator", "DataLoader"]
