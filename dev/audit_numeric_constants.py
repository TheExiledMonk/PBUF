#!/usr/bin/env python3
"""
Scan PBUF-related Cosmos2 modules for numeric literals.

Usage:
    python dev/audit_numeric_constants.py [paths...]

When no paths are supplied, the default PBUF surface (models + kernels +
shared helpers) is scanned. Results are grouped by value with file/line/snippet
context to speed up manual classification.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_TARGETS = [
    Path("cosmos2/models/pbuf/model.py"),
    Path("cosmos2/models/pbuf/distances.py"),
    Path("cosmos2/models/pbuf/elastic.py"),
    Path("cosmos2/models/pbuf/growth.py"),
    Path("cosmos2/models/pbuf/normalization.py"),
    Path("cosmos2/models/pbuf/cmb.py"),
    Path("cosmos2/models/pbuf/fits.py"),
    Path("cosmos2/models/pbuf/phase6a.py"),
    Path("cosmos2/models/pbuf/phase7a.py"),
    Path("cosmos2/models/pbuf/thermal_table.py"),
    Path("cosmos2/models/pbuf/temperature.py"),
    Path("cosmos2/models/pbuf/utils.py"),
]


def _load_number_nodes(tree: ast.AST) -> List[Tuple[float, ast.AST]]:
    """Return [(value, node)] for every numeric literal (ints/floats, signed included)."""

    values: List[Tuple[float, ast.AST]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Constant(self, node: ast.Constant) -> None:  # type: ignore[override]
            if isinstance(node.value, bool):
                return
            if isinstance(node.value, (int, float)):
                values.append((float(node.value), node))

        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:  # type: ignore[override]
            if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
                val = node.operand.value
                if isinstance(val, (int, float)):
                    sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
                    values.append((sign * float(val), node))
                    return
            self.generic_visit(node)

    Visitor().visit(tree)
    return values


def _read_paths(argv: list[str]) -> List[Path]:
    if argv:
        return [Path(p) for p in argv]
    return list(DEFAULT_TARGETS)


def _scan_path(path: Path) -> List[Tuple[float, str, int, str]]:
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    entries: List[Tuple[float, str, int, str]] = []
    for val, node in _load_number_nodes(tree):
        lineno = getattr(node, "lineno", -1)
        snippet = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""
        entries.append((val, str(path), lineno, snippet))
    return entries


def scan(paths: Iterable[Path]) -> dict[float, List[Tuple[str, int, str]]]:
    grouped: dict[float, List[Tuple[str, int, str]]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for val, filename, lineno, snippet in _scan_path(path):
            grouped[val].append((filename, lineno, snippet))
    return grouped


def main(argv: list[str]) -> int:
    paths = _read_paths(argv)
    grouped = scan(paths)
    for value in sorted(grouped.keys()):
        print(f"value: {value}")
        for filename, lineno, snippet in grouped[value]:
            print(f"  {filename}:{lineno}: {snippet}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
