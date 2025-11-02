#!/usr/bin/env python3
"""
discover_json_layout.py
Quick structure discovery tool for PBUF grid-run JSONs.

Usage:
    python discover_json_layout.py path/to/results.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def summarize_value(value, depth=0, key_path="root"):
    """Recursively describe JSON structure."""
    indent = "  " * depth
    dtype = type(value).__name__

    # Print high-level info
    if isinstance(value, dict):
        print(f"{indent}{key_path}: dict ({len(value)} keys)")
        for k, v in value.items():
            summarize_value(v, depth + 1, f"{key_path}.{k}")
    elif isinstance(value, list):
        print(f"{indent}{key_path}: list ({len(value)} items)")
        # Inspect first few items to sample
        for i, v in enumerate(value[:3]):
            summarize_value(v, depth + 1, f"{key_path}[{i}]")
        if len(value) > 3:
            print(f"{indent}  ... ({len(value)-3} more items)")
    else:
        # For numeric types, show min/max if scalar lists come later
        preview = str(value)
        if len(preview) > 60:
            preview = preview[:60] + "..."
        print(f"{indent}{key_path}: {dtype} = {preview}")


def main(json_path):
    path = Path(json_path)
    if not path.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        data = json.load(f)

    print(f"\n=== JSON Structure for {path.name} ===")
    summarize_value(data)
    print("\n✅ Structure discovery complete.\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python discover_json_layout.py path/to/file.json")
        sys.exit(1)
    main(sys.argv[1])
