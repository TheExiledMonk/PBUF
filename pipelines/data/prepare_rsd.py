#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare growth-rate fσ8 data into canonical derived artefacts
=============================================================

Reads the raw growth-rate compilation dataset, normalises column names, builds
the diagonal covariance matrix, and writes:

    rsd_index.csv
    rsd_index.cov.npy
    rsd_index.meta.json

Measured values are never altered; only metadata and formatting are standardised.
"""

from __future__ import annotations

import argparse
import hashlib
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

TRANSFORM_VERSION = "rsd_prepare_v1"
CANONICAL_COLUMNS = [
    "z",
    "fs8",
    "sigma_fs8",
    "dataset",
    "reference",
    "omega_m_fid",
]


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
        list(raw_dir.glob("*.csv"))
        + list(raw_dir.glob("*.dat"))
        + list(raw_dir.glob("*.txt"))
    )
    if not candidates:
        raise FileNotFoundError(f"No raw growth-rate catalog found in {raw_dir}")
    return candidates[0]


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    rename_map = {
        "ref": "reference",
        "fsigma8": "fs8",
        "sigma_fsigma8": "sigma_fs8",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required_columns = CANONICAL_COLUMNS
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")
    return df[required_columns]


def standardise_dataframe(df: pd.DataFrame, release_tag: str) -> pd.DataFrame:
    data = df.copy()
    for column in ["z", "fs8", "sigma_fs8", "omega_m_fid"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["source"] = "nesseris_2017"
    data["release_tag"] = release_tag
    data["row_index"] = np.arange(len(data), dtype=np.int32)
    return data[CANONICAL_COLUMNS + ["source", "release_tag", "row_index"]]


# ---------------------------------------------------------------------
# Covariance construction
# ---------------------------------------------------------------------
def build_covariance(df: pd.DataFrame) -> np.ndarray:
    """
    Construct diagonal covariance matrix:
        C_ij = σ_i^2 δ_ij
    """
    sigma = df["sigma_fs8"].to_numpy()
    return np.diag(sigma**2)


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", newline=""
    ) as tmp:
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
    release_tag: str,
    row_count: int,
    columns: List[str],
) -> dict:
    records = []
    for file_path in raw_files:
        records.append(
            MetaRecord(
                file_path, sha256sum(file_path), file_path.stat().st_size, "raw"
            )
        )
    for file_path in derived_files.values():
        records.append(
            MetaRecord(
                file_path, sha256sum(file_path), file_path.stat().st_size, "derived"
            )
        )

    manifest = {
        "prepared_at": datetime.now(timezone.utc).astimezone().isoformat(
            timespec="seconds"
        ),
        "transform_version": TRANSFORM_VERSION,
        "raw_directory": str(raw_dir.resolve()),
        "derived_directory": str(derived_dir.resolve()),
        "records": [r.as_dict() for r in records],
        "covariance_recipe": "C_ij = σ_i² δ_ij (diagonal)",
        "release_tag": release_tag,
        "row_count": row_count,
        "columns": columns,
        "citation": "Nesseris et al. (2017); Basilakos & Nesseris (2022) growth-rate compilation.",
        "notes": "No data modified; only formatting and covariance standardisation performed.",
    }
    return manifest


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare growth-rate fσ8 data.")
    p.add_argument(
        "--raw",
        required=True,
        help="Path to raw data directory (contains fsigma8 compilation CSV)",
    )
    p.add_argument(
        "--derived",
        required=True,
        help="Output directory for derived artefacts",
    )
    p.add_argument(
        "--release-tag",
        default="nesseris2017",
        help="Release tag identifier",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    raw_dir = Path(args.raw).expanduser()
    derived_dir = Path(args.derived).expanduser()

    catalog_path = discover_catalog(raw_dir)
    info(f"Preparing growth-rate catalog {catalog_path.name}")
    df_raw = load_raw_dataframe(catalog_path)
    df_standard = standardise_dataframe(df_raw, args.release_tag)

    derived_dir.mkdir(parents=True, exist_ok=True)
    table_path = derived_dir / "rsd_index.csv"
    write_csv(df_standard, table_path)

    cov = build_covariance(df_standard)
    cov_path = derived_dir / "rsd_index.cov.npy"
    np.save(cov_path, cov)

    manifest_path = derived_dir / "rsd_index.meta.json"
    raw_files = [p for p in raw_dir.glob("*") if p.is_file()]
    derived_files = {"table": table_path, "covariance": cov_path}
    manifest = build_manifest(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        raw_files=raw_files,
        derived_files=derived_files,
        release_tag=args.release_tag,
        row_count=len(df_standard),
        columns=list(df_standard.columns),
    )
    write_json_atomic(manifest_path, manifest)

    info(f"Wrote derived artefacts to {derived_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
