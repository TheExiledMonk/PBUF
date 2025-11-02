#!/usr/bin/env python3
"""
analyze_basins_lcdm.py
Flatten LCDM grid JSON, summarize chi² and parameters, and export to CSV.
Usage:
    python analyze_basins_lcdm.py path/to/grid_lcdm_*.json [chi2_threshold]
"""

import json
import sys
import pandas as pd
from pathlib import Path

def load_grid(path):
    with open(path, "r") as f:
        data = json.load(f)
    evals = []
    for e in data.get("evaluations", []):
        if e.get("status") != "valid":
            continue
        if not e.get("passes_phase6a", False):
            continue

        row = {
            "id": e.get("id"),
            "chi2_total": e.get("chi2_total"),
        }
        # Add per-dataset chi²
        for k, v in e.get("chi2_breakdown", {}).items():
            row[f"chi2_{k}"] = v
        # Add parameters
        for k, v in e.get("params", {}).items():
            row[k] = v
        evals.append(row)
    return pd.DataFrame(evals)

def main(json_path, chi2_threshold=None):
    df = load_grid(json_path)
    print(f"Loaded {len(df)} valid evaluations")

    if chi2_threshold is not None:
        df = df[df["chi2_total"] < chi2_threshold]
        print(f"→ {len(df)} below χ²_total < {chi2_threshold}")

    if df.empty:
        print("⚠️ No valid entries found.")
        return

    # Select numeric columns safely
    numeric_cols = df.select_dtypes(include=["number"]).columns
    subset_cols = [c for c in ["chi2_total","H0","Om0","Or0","Ok0"] if c in numeric_cols]

    print("\n=== Summary of surviving models ===")
    print(df[subset_cols].describe())

    out_csv = Path(json_path).with_suffix(".csv")
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Exported flattened data to {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_basins_lcdm.py file.json [chi2_threshold]")
        sys.exit(1)
    json_file = sys.argv[1]
    chi_cut = float(sys.argv[2]) if len(sys.argv) > 2 else None
    main(json_file, chi_cut)
