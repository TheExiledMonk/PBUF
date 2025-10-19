#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare BAO data into canonical derived artefacts (DESI, BOSS, eBOSS).

Automatically detects mean/cov pairs (e.g., desi_gaussian_bao_..._mean.txt)
from the CobayaSampler/bao_data repository and produces canonical CSV,
Parquet, and metadata outputs.
"""

from __future__ import annotations
import argparse, hashlib, json, shutil, sys, tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
from utils.io import write_json_atomic
from utils.logging import info

TRANSFORM_VERSION = "bao_prepare_v2"

CANONICAL_COLUMNS = [
    "z",
    "Dv_over_rs",
    "DM_over_rs",
    "H_times_rs",
    "type",
    "survey",
    "release_tag",
    "row_index",
]

RAW_COLUMN_MAP = {
    "zeff": "z",
    "z": "z",
    "DV/rd": "Dv_over_rs",
    "Dv/rd": "Dv_over_rs",
    "DM/rd": "DM_over_rs",
    "H*rd": "H_times_rs",
    "DV/rs": "Dv_over_rs",
    "DM/rs": "DM_over_rs",
    "H*rs": "H_times_rs",
}


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_catalog(raw_dir: Path) -> List[Path]:
    """
    Find all BAO mean measurement files (.txt, .dat, .csv) excluding covariances.
    """
    patterns = ["*.txt", "*.dat", "*.csv"]
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(raw_dir.glob(pattern))
    # Only keep mean files, not covariance
    candidates = [p for p in candidates if "cov" not in p.stem.lower()]
    if not candidates:
        raise FileNotFoundError(f"No raw BAO mean data files found in {raw_dir}")
    return sorted(candidates)


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    """
    Load BAO mean data files like DESI DR2, which list one measurement per line:
        z  value  quantity
    and pivot them into a wide format with columns DM_over_rs, Dv_over_rs, H_times_rs.
    """
    df = pd.read_csv(
        path,
        comment="#",
        delim_whitespace=True,
        names=["z", "value", "quantity"],
        engine="python",
    )

    # Normalize quantity labels
    df["quantity"] = (
        df["quantity"]
        .str.strip()
        .str.replace("DH_over_rs", "H_times_rs")
        .str.replace("DV_over_rs", "Dv_over_rs")
    )

    # Pivot so each z becomes one row
    df_pivot = df.pivot_table(
        index="z",
        columns="quantity",
        values="value",
        aggfunc="first"
    ).reset_index()

    # Rename columns safely
    rename_map = {
        "DM_over_rs": "DM_over_rs",
        "Dv_over_rs": "Dv_over_rs",
        "H_times_rs": "H_times_rs",
    }
    df_pivot = df_pivot.rename(columns=rename_map)

    return df_pivot



def infer_metadata_from_filename(path: Path) -> Tuple[str, str]:
    """Infer survey and type (anisotropic vs isotropic) from filename."""
    name = path.stem.lower()
    survey = "desi" if "desi" in name else "boss" if "boss" in name else "unknown"
    dataset_type = "anisotropic" if any(x in name for x in ["dm", "hz", "gccomb"]) else "isotropic"
    return survey, dataset_type


def standardise_dataframe(df: pd.DataFrame, release_tag: str, dataset_type: str, survey: str) -> pd.DataFrame:
    data = df.copy()

    if "z" not in data.columns:
        # Try first numeric column as z
        numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
        if numeric_cols:
            data["z"] = data[numeric_cols[0]]
        else:
            raise ValueError("No usable redshift column found.")

    optional_defaults = {
        "Dv_over_rs": np.nan,
        "DM_over_rs": np.nan,
        "H_times_rs": np.nan,
        "type": dataset_type,
        "survey": survey,
    }
    for k, v in optional_defaults.items():
        if k not in data.columns:
            data[k] = v

    numeric_columns = ["z", "Dv_over_rs", "DM_over_rs", "H_times_rs"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["release_tag"] = release_tag
    data["row_index"] = np.arange(len(data), dtype=np.int32)
    return data[CANONICAL_COLUMNS]


def load_covariance_if_present(path: Path) -> Tuple[np.ndarray | None, Path | None]:
    """Look for covariance file with same prefix as mean file."""
    base = path.name.replace("_mean", "_cov")
    cov_path = path.parent / base
    if not cov_path.exists():
        return None, None
    try:
        C = np.loadtxt(cov_path)
        if C.ndim == 1:
            n = int(np.sqrt(C.size))
            C = C.reshape((n, n))
        return C, cov_path
    except Exception as e:
        info(f"Warning: Failed to load covariance for {path.name}: {e}")
        return None, cov_path


# ---------------------------------------------------------------------
# Manifest + metadata
# ---------------------------------------------------------------------
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


def build_manifest(
    raw_dir: Path,
    derived_dir: Path,
    raw_files: Iterable[Path],
    derived_files: Dict[str, Path],
    release_tag: str,
    dataset_types: List[str],
    row_count: int,
    columns: List[str],
    cov_info: Dict[str, str] | None,
) -> dict:
    records = []
    for f in raw_files:
        records.append(MetaRecord(f, sha256sum(f), f.stat().st_size, "raw"))
    for f in derived_files.values():
        records.append(MetaRecord(f, sha256sum(f), f.stat().st_size, "derived"))

    return {
        "prepared_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "transform_version": TRANSFORM_VERSION,
        "raw_directory": str(raw_dir.resolve()),
        "derived_directory": str(derived_dir.resolve()),
        "records": [r.as_dict() for r in records],
        "release_tag": release_tag,
        "dataset_types": dataset_types,
        "row_count": row_count,
        "columns": columns,
        "covariance_file": cov_info,
        "notes": "BAO distances standardized; no numerical alteration beyond normalization.",
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BAO datasets into canonical artefacts.")
    parser.add_argument("--raw", required=True, help="Path to raw BAO data directory")
    parser.add_argument("--derived", required=True, help="Output directory for derived artefacts")
    parser.add_argument("--release-tag", default="main", help="Source tag or version from bao_data repo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_dir = Path(args.raw).expanduser()
    derived_dir = Path(args.derived).expanduser()
    derived_dir.mkdir(parents=True, exist_ok=True)


    raw_files = discover_catalog(raw_dir)
    frames = []
    dataset_types = []
    cov_info = None

    for file in raw_files:
        survey, dataset_type = infer_metadata_from_filename(file)
        info(f"Preparing BAO file {file.name} ({survey}, {dataset_type})")
        df_raw = load_raw_dataframe(file)
        df_standard = standardise_dataframe(df_raw, args.release_tag, dataset_type, survey)
        frames.append(df_standard)
        dataset_types.append(dataset_type)
        # Try loading corresponding covariance
        C, cov_path = load_covariance_if_present(file)
        if C is not None:
            np.save(derived_dir / f"{file.stem}_cov.npy", C)
            cov_info = {"path": str(cov_path), "shape": list(C.shape)}

    df_all = pd.concat(frames, ignore_index=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    csv_path = derived_dir / "bao_index.csv"
    parquet_path = derived_dir / "bao_index.parquet"
    manifest_path = derived_dir / "bao_index.meta.json"

    tmp = tempfile.NamedTemporaryFile("w", dir=derived_dir, delete=False, encoding="utf-8", newline="")
    df_all.to_csv(tmp.name, index=False)
    shutil.move(tmp.name, csv_path)

    try:
        df_all.to_parquet(parquet_path, index=False)
    except Exception as exc:
        info(f"Skipping Parquet export ({exc})")

    derived_files = {"table": csv_path}
    if parquet_path.exists():
        derived_files["parquet"] = parquet_path

    manifest = build_manifest(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        raw_files=raw_files,
        derived_files=derived_files,
        release_tag=args.release_tag,
        dataset_types=dataset_types,
        row_count=len(df_all),
        columns=list(df_all.columns),
        cov_info=cov_info,
    )
    write_json_atomic(manifest_path, manifest)
    info(f"Wrote derived BAO artefacts to {derived_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
