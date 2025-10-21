#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Cosmic Chronometer H(z) data into canonical derived artefacts
====================================================================

Reads the raw University of Bologna chronometer dataset (Moresco et al. 2022),
normalises column names, constructs the covariance matrix following the
recipe from Moresco et al. (2020, JCAP 05 (2020) 005), and writes:

    chronometers_index.csv
    chronometers_index.cov.npy
    chronometers_index.meta.json

Measured values are never altered; only metadata and formatting are standardised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from utils.io import write_json_atomic
from utils.logging import info

TRANSFORM_VERSION = "chronometers_prepare_v1"
CANONICAL_COLUMNS = ["z", "H", "sigma_H", "reference"]


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class MetaRecord:
    path: Path
    sha256: str
    size: int
    kind: str
    def as_dict(self) -> dict:
        return {
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
            "size": self.size,
            "kind": self.kind,
        }


# ---------------------------------------------------------------------
# Data loading and standardisation
# ---------------------------------------------------------------------
def discover_catalog(raw_dir: Path) -> Path:
    candidates = sorted(
        list(raw_dir.glob("*.dat"))
        + list(raw_dir.glob("*.txt"))
        + list(raw_dir.glob("*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(f"No raw chronometer catalog found in {raw_dir}")
    return candidates[0]


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", header=None)
    # Expect 3–4 columns: z, H, sigma, [reference]
    ncols = df.shape[1]
    if ncols < 3:
        raise ValueError(f"Unexpected column count ({ncols}) in {path}")
    if ncols == 3:
        df.columns = ["z", "H", "sigma_H"]
        df["reference"] = ""
    else:
        df.columns = ["z", "H", "sigma_H", "reference"]
    return df


def standardise_dataframe(df: pd.DataFrame, release_tag: str) -> pd.DataFrame:
    data = df.copy()
    # Coerce numeric types
    data["z"] = pd.to_numeric(data["z"], errors="coerce")
    data["H"] = pd.to_numeric(data["H"], errors="coerce")
    data["sigma_H"] = pd.to_numeric(data["sigma_H"], errors="coerce")
    data["source"] = "moresco_2022"
    data["release_tag"] = release_tag
    data["row_index"] = np.arange(len(data), dtype=np.int32)
    return data[CANONICAL_COLUMNS + ["source", "release_tag", "row_index"]]


# ---------------------------------------------------------------------
# Covariance construction (Moresco et al. 2020)
# ---------------------------------------------------------------------
def build_covariance(df: pd.DataFrame, frac_sys: float = 0.03) -> np.ndarray:
    """
    Construct covariance matrix:
        C_ij = σ_i^2 δ_ij + (frac_sys * H_i)(frac_sys * H_j)
    """
    H = df["H"].to_numpy()
    sigma = df["sigma_H"].to_numpy()
    n = len(H)
    C = np.diag(sigma**2)
    sys_term = frac_sys * H
    C += np.outer(sys_term, sys_term)
    return C


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", newline="") as tmp:
        df.to_csv(tmp.name, index=False)
    shutil.move(tmp.name, path)


# ---------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------
def build_manifest(
    raw_dir: Path,
    derived_dir: Path,
    raw_files: Iterable[Path],
    derived_files: Dict[str, Path],
    frac_sys: float,
    release_tag: str,
    row_count: int,
    columns: List[str],
) -> dict:
    records = []
    for file_path in raw_files:
        records.append(MetaRecord(file_path, sha256sum(file_path), file_path.stat().st_size, "raw"))
    for file_path in derived_files.values():
        records.append(MetaRecord(file_path, sha256sum(file_path), file_path.stat().st_size, "derived"))

    manifest = {
        "prepared_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "transform_version": TRANSFORM_VERSION,
        "raw_directory": str(raw_dir.resolve()),
        "derived_directory": str(derived_dir.resolve()),
        "records": [r.as_dict() for r in records],
        "covariance_recipe": f"C_ij = σ_i² δ_ij + ({frac_sys:.3f}·H_i)({frac_sys:.3f}·H_j)",
        "frac_sys": frac_sys,
        "release_tag": release_tag,
        "row_count": row_count,
        "columns": columns,
        "citation": "Moresco et al. (2022), data_CC.dat; covariance per Moresco et al. (2020, JCAP 05 005).",
        "notes": "Measured H(z) values unchanged; only formatting and covariance standardised.",
    }
    return manifest


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare cosmic chronometer H(z) data.")
    p.add_argument("--raw", required=True, help="Path to raw data directory (contains data_CC.dat)")
    p.add_argument("--derived", required=True, help="Output directory for derived artefacts")
    p.add_argument("--frac-sys", type=float, default=0.03,
                   help="Fractional systematic uncertainty for covariance (default 0.03)")
    p.add_argument("--release-tag", default="moresco2022", help="Release tag identifier")
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    raw_dir = Path(args.raw).expanduser()
    derived_dir = Path(args.derived).expanduser()

    catalog_path = discover_catalog(raw_dir)
    info(f"Preparing chronometer catalog {catalog_path.name}")
    df_raw = load_raw_dataframe(catalog_path)
    df_standard = standardise_dataframe(df_raw, args.release_tag)

    derived_dir.mkdir(parents=True, exist_ok=True)
    table_path = derived_dir / "chronometers_index.csv"
    write_csv(df_standard, table_path)

    cov = build_covariance(df_standard, frac_sys=args.frac_sys)
    cov_path = derived_dir / "chronometers_index.cov.npy"
    np.save(cov_path, cov)

    manifest_path = derived_dir / "chronometers_index.meta.json"
    raw_files = [p for p in raw_dir.glob("*") if p.is_file()]
    derived_files = {"table": table_path, "covariance": cov_path}
    manifest = build_manifest(
        raw_dir, derived_dir, raw_files, derived_files,
        frac_sys=args.frac_sys,
        release_tag=args.release_tag,
        row_count=len(df_standard),
        columns=list(df_standard.columns),
    )
    write_json_atomic(manifest_path, manifest)

    info(f"Wrote derived artefacts to {derived_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
