"""
PBUF Reports Module
==================

Scientific reporting utilities for cosmological model comparisons.
"""

from .markdown_writer import write_markdown_summary
from .basin_plotter import generate_basin_plots

__all__ = ["write_markdown_summary", "generate_basin_plots"]
