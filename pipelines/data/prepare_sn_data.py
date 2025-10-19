"""
Prepare Pantheon+ supernova data into canonical derived artifacts.

This script reads the raw catalog and covariance components, normalises
column names, ensures deterministic ordering, and writes the prepared
CSV/Parquet/NPY artefacts together with a provenance manifest. Measured
values are never altered — only metadata and formatting are standardised.
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
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.io import write_json_atomic
from utils.logging import info

TRANSFORM_VERSION = "sn_prepare_v1"

CANONICAL_COLUMNS = [
    "sn_id",
    "z_helio",
    "z_cmb",
    "mu",
    "sigma_mu",
    "sample_flags",
    "host_mstellar",
    "ra_deg",
    "dec_deg",
    "source",
    "release_tag",
    "row_index",
]


RAW_COLUMN_MAP = {
    "CID": "sn_id",
    "SNID": "sn_id",
    "SN": "sn_id",
    "name": "sn_id",
    "zHEL": "z_helio",
    "zHELIO": "z_helio",
    "zHD": "z_hd",
    "zCMB": "z_cmb_raw",
    "MU": "mu",
    "MU_SH0ES": "mu",
    "MU_SH0ES_ERR_DIAG": "sigma_mu",
    "SIGMA_MB": "sigma_mu",
    "HOST_LOGMASS": "host_mstellar",
    "HOST_LOGMASS_ERR": "host_mstellar_err",
    "RA": "ra_deg",
    "DEC": "dec_deg",
}


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_catalog(raw_dir: Path) -> Path:
    candidates = sorted(
        list(raw_dir.glob("*.dat"))
        + list(raw_dir.glob("*.txt"))
        + list(raw_dir.glob("*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(f"No raw catalog found in {raw_dir}")
    return candidates[0]


def load_raw_dataframe(catalog_path: Path) -> pd.DataFrame:
    if catalog_path.suffix.lower() in {".dat", ".txt", ".out"}:
        df = pd.read_csv(catalog_path, sep=r"\s+", engine="python", comment="#")
    else:
        df = pd.read_csv(catalog_path)
    rename_map = {col: RAW_COLUMN_MAP.get(col, col) for col in df.columns}
    df = df.rename(columns=rename_map)
    return df


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def standardise_dataframe(df: pd.DataFrame, release_tag: str) -> pd.DataFrame:
    data = df.copy()
    for required_column in ("sn_id",):
        if required_column not in data.columns:
            raise ValueError(f"Required column '{required_column}' missing from raw data")

    if "mu" not in data.columns and "MU_SH0ES" in df.columns:
        data["mu"] = df["MU_SH0ES"]
    if "sigma_mu" not in data.columns and "MU_SH0ES_ERR_DIAG" in df.columns:
        data["sigma_mu"] = df["MU_SH0ES_ERR_DIAG"]

    if "z_cmb" not in data.columns:
        if "z_hd" in data.columns:
            data["z_cmb"] = data["z_hd"]
        elif "z_cmb_raw" in data.columns:
            data["z_cmb"] = data["z_cmb_raw"]
    if "z_helio" not in data.columns and "zHEL" in df.columns:
        data["z_helio"] = df["zHEL"]

    # Ensure optional columns exist with NaNs where absent.
    optional_defaults = {
        "z_helio": np.nan,
        "z_cmb": np.nan,
        "sigma_mu": np.nan,
        "sample_flags": "",
        "host_mstellar": np.nan,
        "ra_deg": np.nan,
        "dec_deg": np.nan,
    }
    for column, default in optional_defaults.items():
        if column not in data.columns:
            data[column] = default

    numeric_columns = ["z_helio", "z_cmb", "mu", "sigma_mu", "host_mstellar", "ra_deg", "dec_deg"]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = _coerce_numeric(data[column])

    flag_columns = [col for col in ("USED_IN_SH0ES_HF", "IS_CALIBRATOR", "IDSURVEY") if col in df.columns]
    if flag_columns:
        data["sample_flags"] = (
            df[flag_columns]
            .astype(str)
            .agg(lambda row: ";".join(f"{col}={val}" for col, val in row.items()), axis=1)
        )
    else:
        data["sample_flags"] = ""

    data["source"] = "pantheon+"
    data["release_tag"] = release_tag
    data["row_index"] = np.arange(len(data), dtype=np.int32)

    # Reorder columns to canonical schema.
    data = data[CANONICAL_COLUMNS]
    return data


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", newline="") as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    shutil.move(tmp_path, path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - optional dependency
        info(f"Skipping Parquet export ({exc})")


def compose_covariance(
    raw_dir: Path,
    compose_labels: Sequence[str],
    n_rows: int,
) -> Tuple[np.ndarray, List[Mapping[str, str]], Dict[str, np.ndarray]]:
    compose_labels = list(dict.fromkeys(compose_labels))
    lower_labels = {label.lower() for label in compose_labels}
    if {"stat+sys", "stat"}.issubset(lower_labels) or {"stat+sys", "sys"}.issubset(lower_labels):
        raise ValueError("Invalid covariance selection: choose either 'stat+sys' or component labels ('stat','sys'), not both.")
    if not compose_labels:
        compose_labels = ["stat+sys"]

    file_map = {path.name.lower(): path for path in raw_dir.glob("*") if path.is_file()}

    def find_path(keyword: str) -> Path | None:
        keyword = keyword.lower()
        for name, path in file_map.items():
            if keyword in name:
                return path
        return None

    stat_path = find_path("statonly")
    stat_sys_path = find_path("stat+sys")

    cache: Dict[Path, np.ndarray] = {}

    def load_cached(path: Path) -> np.ndarray:
        if path not in cache:
            cache[path] = load_matrix(path, expected_size=n_rows)
        return cache[path]

    matrices: List[np.ndarray] = []
    used_files: List[Mapping[str, str]] = []
    components: Dict[str, np.ndarray] = {}

    for label in compose_labels:
        key = label.lower()
        if key in {"stat+sys", "full", "total"}:
            if stat_sys_path is None:
                raise FileNotFoundError("Full covariance file (STAT+SYS) not found in raw directory")
            mat = load_cached(stat_sys_path)
            matrices.append(mat)
            used_files.append({"label": label, "source": stat_sys_path.name})
            components.setdefault("stat+sys", mat)
        elif key == "stat":
            if stat_path is None:
                raise FileNotFoundError("STATONLY covariance file not found in raw directory")
            mat = load_cached(stat_path)
            matrices.append(mat)
            used_files.append({"label": label, "source": stat_path.name})
            components.setdefault("stat", mat)
        elif key == "sys":
            if stat_path is None or stat_sys_path is None:
                raise FileNotFoundError("Cannot derive systematics covariance without STATONLY and STAT+SYS files")
            stat = load_cached(stat_path)
            full = load_cached(stat_sys_path)
            mat = full - stat
            matrices.append(mat)
            used_files.append({"label": label, "source": f"{stat_sys_path.name} - {stat_path.name}"})
            components.setdefault("sys", mat)
        else:
            match = find_path(label)
            if match is None:
                raise FileNotFoundError(f"No covariance component matching '{label}' in {raw_dir}")
            mat = load_cached(match)
            matrices.append(mat)
            used_files.append({"label": label, "source": match.name})
            components.setdefault(label, mat)

    cov = np.zeros((n_rows, n_rows))
    for mat in matrices:
        if mat.shape != (n_rows, n_rows):
            raise ValueError(f"Covariance shape {mat.shape} does not match row count {n_rows}")
        cov += mat

    return cov, used_files, components


def load_matrix(path: Path, expected_size: int | None = None) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    data = np.loadtxt(path)
    if data.ndim == 2:
        return data
    if expected_size is not None:
        total = expected_size * expected_size
        if data.size == total:
            return data.reshape((expected_size, expected_size))
        if data.size == total + 1:
            if abs(data[0] - expected_size) < 1e-8:
                data = data[1:]
            elif abs(data[-1]) < 1e-12:
                data = data[:-1]
            else:
                raise ValueError(
                    f"Unexpected extra element in covariance file {path}; cannot reshape into square matrix"
                )
            return data.reshape((expected_size, expected_size))
    raise ValueError(f"Unable to reshape covariance data from {path} into ({expected_size}, {expected_size})")


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
    compose_components: List[str],
    cov_used_files: List[Mapping[str, str]],
    z_prefer: str,
    symmetrized: bool,
    release_tag: str,
    row_count: int,
    columns: List[str],
) -> dict:
    records = []
    for file_path in raw_files:
        records.append(
            MetaRecord(path=file_path, sha256=sha256sum(file_path), size=file_path.stat().st_size, kind="raw")
        )
    for file_path in derived_files.values():
        records.append(
            MetaRecord(path=file_path, sha256=sha256sum(file_path), size=file_path.stat().st_size, kind="derived")
        )

    manifest = {
        "prepared_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "transform_version": TRANSFORM_VERSION,
        "raw_directory": str(raw_dir.resolve()),
        "derived_directory": str(derived_dir.resolve()),
        "records": [record.as_dict() for record in records],
        "compose_cov": compose_components,
        "covariance_components_used": cov_used_files,
        "symmetrized": symmetrized,
        "z_prefer": z_prefer,
        "release_tag": release_tag,
        "row_count": row_count,
        "columns": columns,
        "notes": "No manipulation of μ, redshifts, or covariance beyond symmetrization.",
    }
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare supernova datasets into canonical artefacts.")
    parser.add_argument("--raw", required=True, help="Path to raw data directory")
    parser.add_argument("--derived", required=True, help="Output directory for derived artefacts")
    parser.add_argument("--z-prefer", default="z_cmb", help="Preferred redshift column (fallback to z_helio)")
    parser.add_argument("--compose-cov", default="", help="Comma-separated covariance component labels (e.g., stat,sys)")
    parser.add_argument("--release-tag", default="unknown", help="Release tag or version identifier from the source")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_dir = Path(args.raw).expanduser()
    derived_dir = Path(args.derived).expanduser()
    compose_components = [item.strip() for item in args.compose_cov.split(",") if item.strip()]

    catalog_path = discover_catalog(raw_dir)
    info(f"Preparing catalog {catalog_path.name}")
    df_raw = load_raw_dataframe(catalog_path)
    df_standard = standardise_dataframe(df_raw, args.release_tag)

    derived_dir.mkdir(parents=True, exist_ok=True)
    table_path = derived_dir / "supernova_index.csv"
    write_csv(df_standard, table_path)

    parquet_path = derived_dir / "supernova_index.parquet"
    write_parquet(df_standard, parquet_path)

    cov_matrix, cov_used_files, cov_components = compose_covariance(raw_dir, compose_components, len(df_standard))
    if cov_matrix.size == 0:
        raise RuntimeError("No covariance matrix constructed; provide --compose-cov components or raw .npy file.")

    symmetric = np.allclose(cov_matrix, cov_matrix.T, atol=1e-10)
    if not symmetric:
        info("Covariance nearly symmetric; applying (C + C^T) / 2 symmetrization.")
        cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
        symmetrized = True
    else:
        symmetrized = False

    cov_path = derived_dir / "supernova_index.cov.npy"
    np.save(cov_path, cov_matrix)

    cov_components_dir = derived_dir / "cov_components"
    derived_files = {
        "table": table_path,
        "covariance": cov_path,
    }
    if parquet_path.exists():
        derived_files["parquet"] = parquet_path

    if cov_components:
        cov_components_dir.mkdir(parents=True, exist_ok=True)
        for label, matrix in cov_components.items():
            component_path = cov_components_dir / f"{label.replace('+', '_')}.npy"
            np.save(component_path, matrix)
            derived_files[f"cov_component_{label}"] = component_path

    manifest_path = derived_dir / "supernova_index.meta.json"
    raw_files = [path for path in raw_dir.glob("*") if path.is_file()]
    manifest = build_manifest(
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        raw_files=raw_files,
        derived_files=derived_files,
        compose_components=compose_components,
        cov_used_files=cov_used_files,
        z_prefer=args.z_prefer,
        symmetrized=symmetrized,
        release_tag=args.release_tag,
        row_count=len(df_standard),
        columns=list(df_standard.columns),
    )
    write_json_atomic(manifest_path, manifest)

    info(f"Wrote derived artefacts to {derived_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
