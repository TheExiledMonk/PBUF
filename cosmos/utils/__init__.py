"""Utility helpers shared across Cosmos optimisation modules."""

from .io import (
    strip_json_comments,
    read_json,
    atomic_write_json,
    merge_dict_with_defaults,
)

__all__ = [
    "strip_json_comments",
    "read_json",
    "atomic_write_json",
    "merge_dict_with_defaults",
]
