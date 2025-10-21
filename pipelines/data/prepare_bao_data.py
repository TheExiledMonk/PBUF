#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare BAO data into canonical derived artefacts (DESI, BOSS, eBOSS).

Automatically detects mean/cov pairs (e.g., desi_gaussian_bao_..._mean.txt)
from the CobayaSampler/bao_data repository and produces canonical CSV,
Parquet, and metadata outputs.
"""

from __future__ import annotations
import argparse, hashlib, itertools, shutil, sys, tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from utils.io import write_json_atomic
from utils.logging import info

TRANSFORM_VERSION = "bao_prepare_v2"
RSD_TRANSFORM_VERSION = "bao_rsd_extract_v1"

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

RSD_CANONICAL_COLUMNS = [
    "z",
    "fs8",
    "sigma_fs8",
    "dataset",
    "reference",
    "release_tag",
    "source",
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


@dataclass
class RsdDatasetExtraction:
    dataframe: pd.DataFrame
    covariance: np.ndarray
    raw_files: List[Path]
    method: str


def _read_table(path: Path) -> pd.DataFrame:
    for kwargs in (
        {"delim_whitespace": True},
        {},
    ):
        try:
            df = pd.read_csv(path, comment="#", **kwargs)
            if df.empty:
                continue
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _choose_column(columns: List[str], patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        for column in columns:
            if pattern in column.lower():
                return column
    return None


def _normalise_rsd_dataframe(df: pd.DataFrame, dataset: str, reference: str, release_tag: str) -> pd.DataFrame:
    columns = list(df.columns)
    z_column = _choose_column(columns, ["z_eff", "zeff", "z"])
    fs8_column = _choose_column(columns, ["fsigma8", "f_sigma8", "fs8", "f8"])
    sigma_column = _choose_column(columns, ["sigma_f", "err_f", "sigmafs", "errfs", "sigma_fs", "sigmafs8", "err_fsig", "errfs8"])
    if not (z_column and fs8_column and sigma_column):
        return pd.DataFrame()

    data = pd.DataFrame(
        {
            "z": pd.to_numeric(df[z_column], errors="coerce"),
            "fs8": pd.to_numeric(df[fs8_column], errors="coerce"),
            "sigma_fs8": pd.to_numeric(df[sigma_column], errors="coerce"),
            "dataset": dataset,
            "reference": reference,
            "release_tag": release_tag,
        }
    )
    data = data.dropna(subset=["z", "fs8", "sigma_fs8"])
    return data


def _correlation_from_columns(df: pd.DataFrame, n: int) -> Optional[np.ndarray]:
    corr_columns = [c for c in df.columns if "corr" in c.lower() or "rho" in c.lower()]
    if not corr_columns:
        return None
    values: List[float] = []
    for column in corr_columns:
        vals = pd.to_numeric(df[column], errors="coerce")
        values.extend([float(v) for v in vals.dropna().tolist()])
    combos = list(itertools.combinations(range(n), 2))
    if len(values) < len(combos):
        return None
    corr_matrix = np.eye(n)
    for (i, j), value in zip(combos, values):
        corr_matrix[i, j] = corr_matrix[j, i] = value
    return corr_matrix


def _extract_covariance_block(matrix: np.ndarray, size: int) -> Optional[np.ndarray]:
    if matrix.ndim != 2:
        return None
    if matrix.shape == (size, size):
        return matrix
    if matrix.shape[0] >= size and matrix.shape[1] >= size:
        start_row = matrix.shape[0] - size
        start_col = matrix.shape[1] - size
        return matrix[start_row:, start_col:]
    return None


def _load_covariance_matrix(cov_path: Path, size: int) -> Optional[np.ndarray]:
    try:
        raw = np.loadtxt(cov_path)
    except Exception:
        return None
    raw = np.atleast_2d(raw)
    if raw.shape == (size, size):
        return raw
    if raw.size == size:
        diag = raw.reshape(size)
        return np.diag(np.asarray(diag, dtype=float) ** 2)
    block = _extract_covariance_block(raw, size)
    if block is not None:
        return block
    return None


def _build_covariance(sigmas: np.ndarray, corr_matrix: Optional[np.ndarray], cov_matrix: Optional[np.ndarray]) -> tuple[np.ndarray, str]:
    if cov_matrix is not None:
        return cov_matrix, "provided_covariance_matrix"
    if corr_matrix is not None:
        cov = corr_matrix * np.outer(sigmas, sigmas)
        return cov, "correlation_sigma_product"
    return np.diag(sigmas ** 2), "diagonal_sigma"


def _parse_rsd_file(
    path: Path,
    release_tag: str,
    dataset: str,
    reference: str,
    cov_candidates: List[Path],
) -> Optional[RsdDatasetExtraction]:
    df_raw = _read_table(path)
    if df_raw.empty:
        return None
    df_norm = _normalise_rsd_dataframe(df_raw, dataset, reference, release_tag)
    if df_norm.empty:
        return None

    size = len(df_norm)
    corr_matrix = _correlation_from_columns(df_raw, size)
    cov_matrix = None
    method = "diagonal_sigma"
    for cov_path in cov_candidates:
        if cov_path.exists():
            cov_matrix = _load_covariance_matrix(cov_path, size)
            if cov_matrix is not None:
                method = "provided_covariance_matrix"
                break
    cov, method = _build_covariance(df_norm["sigma_fs8"].to_numpy(), corr_matrix, cov_matrix)
    df_norm["source"] = "bao_rsd_extract"
    df_norm["row_index"] = np.arange(len(df_norm), dtype=np.int32)
    raw_files = [path] + [c for c in cov_candidates if c.exists()]
    return RsdDatasetExtraction(df_norm, cov, raw_files, method)


def _collect_desi_rsd(raw_dir: Path, release_tag: str) -> List[RsdDatasetExtraction]:
    sources: List[RsdDatasetExtraction] = []
    candidate_dir = raw_dir / "desi_bao_dr2"
    search_roots = [raw_dir]
    if candidate_dir.exists():
        search_roots.append(candidate_dir)

    dataset_map = {
        "lrg": ("DESI DR2 LRG", "DESI Collaboration 2025"),
        "elg": ("DESI DR2 ELG", "DESI Collaboration 2025"),
        "qso": ("DESI DR2 QSO", "DESI Collaboration 2025"),
        "bgs": ("DESI DR2 BGS", "DESI Collaboration 2025"),
    }

    seen_files: set[Path] = set()
    for root in search_roots:
        for slug, (dataset, reference) in dataset_map.items():
            for path in root.glob(f"*{slug}*fs*8*"):
                if path.suffix.lower() not in {".txt", ".dat", ".csv"}:
                    continue
                if path in seen_files:
                    continue
                seen_files.add(path)
                cov_candidates = [
                    root / f"cov_{slug}.txt",
                    root / f"{slug}_cov.txt",
                    root / f"{path.stem}_cov.txt",
                ]
                extraction = _parse_rsd_file(path, release_tag, dataset, reference, cov_candidates)
                if extraction:
                    sources.append(extraction)
    return sources


def _collect_boss_rsd(raw_dir: Path, release_tag: str) -> List[RsdDatasetExtraction]:
    sources: List[RsdDatasetExtraction] = []
    possible_files = [
        raw_dir / "boss_dr12" / "fsigma8_boss.txt",
        raw_dir / "boss_dr12" / "fsigma8_consensus.txt",
        raw_dir / "fsigma8_boss.txt",
    ]
    cov_candidates = [
        raw_dir / "boss_dr12" / "bao_consensus_full_covariance.dat",
        raw_dir / "FS_consensus_covtot_dM_Hz_fsig.txt",
        raw_dir / "boss_dr12" / "fsigma8_cov.txt",
    ]
    for path in possible_files:
        if not path.exists():
            continue
        extraction = _parse_rsd_file(
            path,
            release_tag,
            "BOSS DR12",
            "Alam et al. 2017",
            cov_candidates,
        )
        if extraction:
            sources.append(extraction)
    return sources


def _collect_eboss_rsd(raw_dir: Path, release_tag: str) -> List[RsdDatasetExtraction]:
    sources: List[RsdDatasetExtraction] = []
    possible_files = [
        raw_dir / "eboss_dr16" / "bao+rsd_consensus_table.txt",
        raw_dir / "eboss_dr16" / "fsigma8_eboss.txt",
        raw_dir / "bao+rsd_consensus_table.txt",
    ]
    cov_candidates = [
        raw_dir / "eboss_dr16" / "bao+rsd_consensus_cov.txt",
        raw_dir / "eboss_dr16" / "fsigma8_cov.txt",
        raw_dir / "bao+rsd_consensus_cov.txt",
    ]
    for path in possible_files:
        if not path.exists():
            continue
        extraction = _parse_rsd_file(
            path,
            release_tag,
            "eBOSS DR16",
            "Alam et al. 2021",
            cov_candidates,
        )
        if extraction:
            sources.append(extraction)
    return sources


def collect_rsd_datasets(raw_dir: Path, release_tag: str) -> List[RsdDatasetExtraction]:
    datasets: List[RsdDatasetExtraction] = []
    datasets.extend(_collect_desi_rsd(raw_dir, release_tag))
    datasets.extend(_collect_boss_rsd(raw_dir, release_tag))
    datasets.extend(_collect_eboss_rsd(raw_dir, release_tag))
    return datasets


def _block_diagonal(matrices: List[np.ndarray]) -> np.ndarray:
    if not matrices:
        return np.zeros((0, 0))
    size = sum(mat.shape[0] for mat in matrices)
    result = np.zeros((size, size))
    offset = 0
    for mat in matrices:
        span = mat.shape[0]
        result[offset : offset + span, offset : offset + span] = mat
        offset += span
    return result


def _unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = []
    seen_keys = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen_keys:
            continue
        seen.append(path)
        seen_keys.add(key)
    return seen


def resolve_rsd_output_dir(derived_dir: Path) -> Path:
    if derived_dir.name == "derived" and derived_dir.parent.name == "bao":
        return derived_dir.parent.parent / "rsd" / "derived"
    return derived_dir / "rsd"


def build_rsd_manifest(
    raw_files: Iterable[Path],
    derived_dir: Path,
    derived_files: Dict[str, Path],
    release_tag: str,
    row_count: int,
    columns: List[str],
    methods: List[str],
    datasets: List[str],
) -> dict:
    records = []
    for file_path in raw_files:
        if file_path.exists():
            records.append(MetaRecord(file_path, sha256sum(file_path), file_path.stat().st_size, "raw"))
    for file_path in derived_files.values():
        if file_path.exists():
            records.append(MetaRecord(file_path, sha256sum(file_path), file_path.stat().st_size, "derived"))

    return {
        "prepared_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "transform_version": RSD_TRANSFORM_VERSION,
        "derived_directory": str(derived_dir.resolve()),
        "records": [r.as_dict() for r in records],
        "release_tag": release_tag,
        "row_count": row_count,
        "columns": columns,
        "covariance_recipe": ", ".join(sorted(set(methods))) if methods else "diagonal_sigma",
        "datasets": sorted(set(datasets)),
        "notes": "RSD growth-rate values extracted from BAO raw data bundle.",
    }


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
    rsd_points: int,
    rsd_extracted_from: str | None,
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
        "rsd_points": rsd_points,
        "rsd_extracted_from": rsd_extracted_from,
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

    rsd_datasets = collect_rsd_datasets(raw_dir, args.release_tag)
    rsd_points = sum(len(dataset.dataframe) for dataset in rsd_datasets)
    rsd_methods = [dataset.method for dataset in rsd_datasets]
    rsd_raw_files = _unique_paths(
        itertools.chain.from_iterable(dataset.raw_files for dataset in rsd_datasets)
    )
    rsd_extracted_from = ", ".join(
        sorted(
            {
                row.dataset
                for dataset in rsd_datasets
                for _, row in dataset.dataframe.iterrows()
            }
        )
    ) if rsd_points else None
    rsd_derived_files: Dict[str, Path] = {}
    if rsd_points:
        rsd_dir = resolve_rsd_output_dir(derived_dir)
        rsd_dir.mkdir(parents=True, exist_ok=True)
        rsd_df = pd.concat([dataset.dataframe for dataset in rsd_datasets], ignore_index=True)
        rsd_df["row_index"] = np.arange(len(rsd_df), dtype=np.int32)
        rsd_df = rsd_df[RSD_CANONICAL_COLUMNS]

        rsd_csv = rsd_dir / "rsd_index.csv"
        with tempfile.NamedTemporaryFile("w", dir=rsd_dir, delete=False, encoding="utf-8", newline="") as tmp_rsd:
            rsd_df.to_csv(tmp_rsd.name, index=False)
        shutil.move(tmp_rsd.name, rsd_csv)

        rsd_cov = _block_diagonal([dataset.covariance for dataset in rsd_datasets])
        rsd_cov_path = rsd_dir / "rsd_index.cov.npy"
        np.save(rsd_cov_path, rsd_cov)

        rsd_manifest_path = rsd_dir / "rsd_index.meta.json"
        rsd_manifest = build_rsd_manifest(
            raw_files=rsd_raw_files,
            derived_dir=rsd_dir,
            derived_files={"table": rsd_csv, "covariance": rsd_cov_path},
            release_tag=args.release_tag,
            row_count=len(rsd_df),
            columns=list(rsd_df.columns),
            methods=rsd_methods,
            datasets=list(rsd_df["dataset"].unique()),
        )
        write_json_atomic(rsd_manifest_path, rsd_manifest)

        info(f"Extracted {rsd_points} RSD points from BAO data into {rsd_dir.resolve()}")
        rsd_derived_files["rsd_table"] = rsd_csv
        rsd_derived_files["rsd_covariance"] = rsd_cov_path
        rsd_derived_files["rsd_manifest"] = rsd_manifest_path

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
    derived_files.update(rsd_derived_files)

    raw_files_all = raw_files[:]
    for candidate in rsd_raw_files:
        if candidate not in raw_files_all:
            raw_files_all.append(candidate)

    manifest = build_manifest(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        raw_files=raw_files_all,
        derived_files=derived_files,
        release_tag=args.release_tag,
        dataset_types=dataset_types,
        row_count=len(df_all),
        columns=list(df_all.columns),
        cov_info=cov_info,
        rsd_points=rsd_points,
        rsd_extracted_from=rsd_extracted_from,
    )
    write_json_atomic(manifest_path, manifest)
    info(f"Wrote derived BAO artefacts to {derived_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
