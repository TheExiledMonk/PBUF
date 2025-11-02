"""
Utility helpers for reading and writing JSON configuration files.

This consolidates the comment-stripping/atomic-write logic that previously lived
in the standalone science runner script so it can be reused by the CLI modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping


def strip_json_comments(text: str) -> str:
    """
    Remove // and /* */ style comments from JSON-like text.

    This mirrors the permissive parser that existing configurations relied on.
    """
    result: list[str] = []
    in_string = False
    string_char = ""
    in_single_comment = False
    in_block_comment = False
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < length else ""

        if in_single_comment:
            if ch == "\n":
                in_single_comment = False
                result.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_string:
            result.append(ch)
            if ch == "\\":
                if i + 1 < length:
                    result.append(text[i + 1])
                    i += 2
                    continue
            elif ch == string_char:
                in_string = False
            i += 1
            continue

        if ch == "/" and nxt == "/":
            in_single_comment = True
            i += 2
            continue

        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue

        if ch in {'"', "'"}:
            in_string = True
            string_char = ch
            result.append(ch)
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def read_json(path: Path, *, allow_comments: bool = False) -> Any:
    """
    Load JSON data from *path*, optionally tolerating // and /* */ comments.
    """
    text = path.read_text(encoding="utf-8")
    if allow_comments:
        text = strip_json_comments(text)
    return json.loads(text)


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    """
    Write JSON data atomically by staging to a temporary sibling file first.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def merge_dict_with_defaults(data: Mapping[str, Any], defaults: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Recursively merge *defaults* into *data*, returning a new dictionary.
    """
    merged: MutableMapping[str, Any] = {}
    for key, default_value in defaults.items():
        if key not in data:
            merged[key] = default_value
            continue
        value = data[key]
        if isinstance(value, Mapping) and isinstance(default_value, Mapping):
            merged[key] = merge_dict_with_defaults(value, default_value)
        else:
            merged[key] = value
    for key, value in data.items():
        if key not in merged:
            merged[key] = value
    return merged


__all__ = [
    "strip_json_comments",
    "read_json",
    "atomic_write_json",
    "merge_dict_with_defaults",
]
