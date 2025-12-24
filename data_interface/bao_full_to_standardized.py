"""
Build full BAO datasets (ISO + ANISO) from raw `data/raw/bao_data` sources.

Goal:
  - Prefer latest DESI release (DR2) when the same measurement is present in DR1.
  - Keep additional DR1-only measurements (e.g. QSO DV) that are not present in DR2.
  - Emit standardized NPZ caches under `data/standardized/`:
      - bao_iso_full.npz
      - bao_aniso_full.npz

The raw directory is expected to mirror the `data/raw/bao_data` layout included in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    from data_interface.standardize import ensure_standard_dataset
except ModuleNotFoundError:  # pragma: no cover
    # Allow running as `python data_interface/bao_full_to_standardized.py` without importing the
    # full data_interface package (which may pull optional deps like pandas).
    from standardize import ensure_standard_dataset  # type: ignore


RAW_DIR_CANDIDATES = (
    Path("data/raw/bao_data"),
    Path("data/raw/bao/bao_data"),
)
OUT_DIR = Path("data/standardized")


@dataclass(frozen=True)
class BaoRawSource:
    release: str
    mean_path: Path
    cov_path: Path


def _resolve_raw_dir() -> Path:
    for candidate in RAW_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(p) for p in RAW_DIR_CANDIDATES)
    raise FileNotFoundError(f"Could not find BAO raw directory (tried {tried})")


def _read_gaussian_mean(path: Path) -> list[tuple[float, float, str]]:
    rows: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            raise ValueError(f"Expected 3 columns [z value quantity] in {path} (got: {stripped!r})")
        z = float(parts[0])
        value = float(parts[1])
        quantity = str(parts[2]).strip()
        rows.append((z, value, quantity))
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def _read_cov(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=float)
    if raw.ndim == 0:
        return np.asarray([[float(raw)]], dtype=float)
    if raw.ndim == 1:
        if raw.size == 1:
            return np.asarray([[float(raw.reshape(-1)[0])]], dtype=float)
        raise ValueError(f"Unexpected 1D covariance payload in {path} (len={raw.size})")
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(f"Covariance in {path} must be square (got {raw.shape})")
    return np.asarray(raw, dtype=float)


def _select_indices(rows: list[tuple[float, float, str]], *, quantities: set[str]) -> list[int]:
    selected: list[int] = []
    for idx, (_, _, qty) in enumerate(rows):
        if qty in quantities:
            selected.append(idx)
    if not selected:
        raise ValueError(f"No rows found for quantities {sorted(quantities)}")
    return selected


def _block_diag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]), dtype=float)
    out[: a.shape[0], : a.shape[1]] = a
    out[a.shape[0] :, a.shape[1] :] = b
    return out


def _append_independent(
    *,
    z: np.ndarray,
    obs: np.ndarray,
    cov: np.ndarray,
    labels: np.ndarray,
    new_z: np.ndarray,
    new_obs: np.ndarray,
    new_cov: np.ndarray,
    new_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if new_obs.size == 0:
        return z, obs, cov, labels
    if new_cov.shape != (new_obs.size, new_obs.size):
        raise ValueError(f"New covariance shape mismatch: {new_cov.shape} vs {new_obs.size}")
    if new_z.size != new_obs.size or new_labels.size != new_obs.size:
        raise ValueError("New z/labels must match new obs length.")
    return (
        np.concatenate([z, new_z], dtype=float),
        np.concatenate([obs, new_obs], dtype=float),
        _block_diag(cov, new_cov),
        np.concatenate([labels, new_labels]),
    )


def _desi_sources(raw_dir: Path) -> tuple[BaoRawSource, BaoRawSource, BaoRawSource]:
    dr2 = BaoRawSource(
        release="DESI_DR2",
        mean_path=raw_dir / "desi_bao_dr2" / "desi_gaussian_bao_ALL_GCcomb_mean.txt",
        cov_path=raw_dir / "desi_bao_dr2" / "desi_gaussian_bao_ALL_GCcomb_cov.txt",
    )
    dr1 = BaoRawSource(
        release="DESI_DR1",
        mean_path=raw_dir / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt",
        cov_path=raw_dir / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt",
    )
    dr1_qso_dv = BaoRawSource(
        release="DESI_DR1",
        mean_path=raw_dir / "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",
        cov_path=raw_dir / "desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt",
    )
    return dr2, dr1, dr1_qso_dv


def build_bao_iso_full(*, raw_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or _resolve_raw_dir()
    dr2, dr1, dr1_qso_dv = _desi_sources(raw_dir)

    dr2_rows = _read_gaussian_mean(dr2.mean_path)
    dr2_cov = _read_cov(dr2.cov_path)
    if dr2_cov.shape[0] != len(dr2_rows):
        raise ValueError(f"DR2 mean/cov length mismatch: {len(dr2_rows)} vs {dr2_cov.shape}")

    dr2_idx = _select_indices(dr2_rows, quantities={"DV_over_rs"})
    z = np.asarray([dr2_rows[i][0] for i in dr2_idx], dtype=float)
    obs = np.asarray([dr2_rows[i][1] for i in dr2_idx], dtype=float)
    cov = dr2_cov[np.ix_(dr2_idx, dr2_idx)]
    labels = np.asarray(["D_V/rd"] * obs.size, dtype=object)

    # Add DR1-only DV points not already present in DR2.
    dr1_qso_rows = _read_gaussian_mean(dr1_qso_dv.mean_path)
    dr1_qso_cov = _read_cov(dr1_qso_dv.cov_path)
    if dr1_qso_cov.shape[0] != len(dr1_qso_rows):
        raise ValueError(f"DR1 QSO mean/cov length mismatch: {len(dr1_qso_rows)} vs {dr1_qso_cov.shape}")
    dr1_qso_idx = _select_indices(dr1_qso_rows, quantities={"DV_over_rs"})
    dr1_qso_z = np.asarray([dr1_qso_rows[i][0] for i in dr1_qso_idx], dtype=float)
    dr1_qso_obs = np.asarray([dr1_qso_rows[i][1] for i in dr1_qso_idx], dtype=float)
    dr1_qso_cov = dr1_qso_cov[np.ix_(dr1_qso_idx, dr1_qso_idx)]
    dr1_qso_labels = np.asarray(["D_V/rd"] * dr1_qso_obs.size, dtype=object)

    # De-dup: treat same-redshift DV points as duplicates across DESI releases, keep DR2.
    existing = set(round(float(val), 6) for val in z.tolist())
    keep = [i for i, z_val in enumerate(dr1_qso_z.tolist()) if round(float(z_val), 6) not in existing]
    if keep:
        dr1_qso_z = dr1_qso_z[keep]
        dr1_qso_obs = dr1_qso_obs[keep]
        dr1_qso_cov = dr1_qso_cov[np.ix_(keep, keep)]
        dr1_qso_labels = dr1_qso_labels[keep]
        z, obs, cov, labels = _append_independent(
            z=z,
            obs=obs,
            cov=cov,
            labels=labels,
            new_z=dr1_qso_z,
            new_obs=dr1_qso_obs,
            new_cov=dr1_qso_cov,
            new_labels=dr1_qso_labels,
        )

    meta = {
        "survey": "DESI",
        "dataset_type": "BAO_ISO",
        "observable": "D_V(z)/r_d isotropic",
        "reference": "DESI Collaboration 2024 (DR1) + DESI Collaboration 2025 (DR2)",
        "release_priority": ["DESI_DR2", "DESI_DR1"],
        "components": [
            {"release": dr2.release, "mean": str(dr2.mean_path), "cov": str(dr2.cov_path)},
            {"release": dr1_qso_dv.release, "mean": str(dr1_qso_dv.mean_path), "cov": str(dr1_qso_dv.cov_path)},
        ],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    data = {
        "name": "BAO_ISO_full",
        "type": "BAO_ISO",
        "z": z,
        "obs": obs,
        "cov": cov,
        "err": np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        "meta": meta,
        "labels": labels,
    }
    return ensure_standard_dataset(data, "BAO_ISO")


def build_bao_aniso_full(*, raw_dir: Path | None = None) -> dict:
    raw_dir = raw_dir or _resolve_raw_dir()
    dr2, dr1, _dr1_qso_dv = _desi_sources(raw_dir)

    dr2_rows = _read_gaussian_mean(dr2.mean_path)
    dr2_cov = _read_cov(dr2.cov_path)
    if dr2_cov.shape[0] != len(dr2_rows):
        raise ValueError(f"DR2 mean/cov length mismatch: {len(dr2_rows)} vs {dr2_cov.shape}")

    # Use DR2 DM/DH as the primary anisotropic dataset.
    dr2_idx = _select_indices(dr2_rows, quantities={"DM_over_rs", "DH_over_rs"})
    # Sort indices by (z, quantity order) to keep stable interleaving.
    qty_order = {"DM_over_rs": 0, "DH_over_rs": 1}
    dr2_idx = sorted(dr2_idx, key=lambda i: (dr2_rows[i][0], qty_order.get(dr2_rows[i][2], 99)))

    z = np.asarray([dr2_rows[i][0] for i in dr2_idx], dtype=float)
    obs = np.asarray([dr2_rows[i][1] for i in dr2_idx], dtype=float)
    cov = dr2_cov[np.ix_(dr2_idx, dr2_idx)]
    labels = np.asarray(
        ["D_M/rd" if dr2_rows[i][2] == "DM_over_rs" else "D_H/rd" for i in dr2_idx],
        dtype=object,
    )

    def _load_sdss_dmdh(mean_path: Path, cov_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = _read_gaussian_mean(mean_path)
        cov_mat = _read_cov(cov_path)
        if cov_mat.shape[0] != len(rows):
            raise ValueError(f"Mean/cov length mismatch for {mean_path} vs {cov_path}")
        allowed = {"DM_over_rs", "DH_over_rs"}
        idx = _select_indices(rows, quantities=allowed)
        idx = sorted(idx, key=lambda i: (rows[i][0], qty_order.get(rows[i][2], 99)))
        z_vals = np.asarray([rows[i][0] for i in idx], dtype=float)
        obs_vals = np.asarray([rows[i][1] for i in idx], dtype=float)
        cov_vals = cov_mat[np.ix_(idx, idx)]
        labels_vals = np.asarray(
            ["D_M/rd" if rows[i][2] == "DM_over_rs" else "D_H/rd" for i in idx],
            dtype=object,
        )
        return z_vals, obs_vals, cov_vals, labels_vals

    # Add SDSS/eBOSS-era measurements as independent blocks (no DESI de-duplication).
    sdss_blocks: list[tuple[str, Path, Path]] = [
        ("SDSS_DR16_LRG", raw_dir / "sdss_DR16_LRG_BAO_DMDH.dat", raw_dir / "sdss_DR16_LRG_BAO_DMDH_covtot.txt"),
        ("SDSS_DR16_QSO", raw_dir / "sdss_DR16_QSO_BAO_DMDH.txt", raw_dir / "sdss_DR16_QSO_BAO_DMDH_covtot.txt"),
        ("SDSS_DR12_LRG", raw_dir / "sdss_DR12_LRG_BAO_DMDH.dat", raw_dir / "sdss_DR12_LRG_BAO_DMDH_covtot.txt"),
    ]
    components_extra: list[dict] = []
    for tag, mean_path, cov_path in sdss_blocks:
        if not mean_path.exists() or not cov_path.exists():
            continue
        z_new, obs_new, cov_new, labels_new = _load_sdss_dmdh(mean_path, cov_path)
        z, obs, cov, labels = _append_independent(
            z=z,
            obs=obs,
            cov=cov,
            labels=labels,
            new_z=z_new,
            new_obs=obs_new,
            new_cov=cov_new,
            new_labels=labels_new,
        )
        components_extra.append({"release": tag, "mean": str(mean_path), "cov": str(cov_path)})

    meta = {
        "survey": "DESI+SDSS",
        "dataset_type": "BAO_ANISO",
        "observable": "D_M(z)/r_d and D_H(z)/r_d",
        "reference": "DESI Collaboration 2025 (DR2) + SDSS DR16/DR12 BAO",
        "release_priority": ["DESI_DR2", "DESI_DR1", "SDSS_DR16", "SDSS_DR12"],
        "components": [
            {"release": dr2.release, "mean": str(dr2.mean_path), "cov": str(dr2.cov_path)},
            {"release": dr1.release, "mean": str(dr1.mean_path), "cov": str(dr1.cov_path), "note": "DR1 superseded where overlapping"},
            *components_extra,
        ],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    data = {
        "name": "BAO_ANISO_full",
        "type": "BAO_ANISO",
        "z": z,
        "obs": obs,
        "cov": cov,
        "err": np.sqrt(np.clip(np.diag(cov), 0.0, None)),
        "meta": meta,
        "labels": labels,
    }
    return ensure_standard_dataset(data, "BAO_ANISO")


def write_bao_full_npz(*, raw_dir: Path | None = None, out_dir: Path | None = None) -> tuple[Path, Path]:
    raw_dir = raw_dir or _resolve_raw_dir()
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    iso = build_bao_iso_full(raw_dir=raw_dir)
    aniso = build_bao_aniso_full(raw_dir=raw_dir)

    iso_path = out_dir / "bao_iso_full.npz"
    aniso_path = out_dir / "bao_aniso_full.npz"

    np.savez(
        iso_path,
        name=np.asarray("BAO_ISO_full", dtype=object),
        labels=np.asarray(iso.get("labels", []), dtype=object),
        n_data=np.asarray(int(iso.get("n_data", iso["obs"].size)), dtype=int),
        meta=np.asarray([iso.get("meta", {})], dtype=object),
        obs=np.asarray(iso["obs"], dtype=float),
        z=np.asarray(iso["z"], dtype=float),
        cov=np.asarray(iso["cov"], dtype=float),
    )
    np.savez(
        aniso_path,
        name=np.asarray("BAO_ANISO_full", dtype=object),
        labels=np.asarray(aniso.get("labels", []), dtype=object),
        n_data=np.asarray(int(aniso.get("n_data", aniso["obs"].size)), dtype=int),
        meta=np.asarray([aniso.get("meta", {})], dtype=object),
        obs=np.asarray(aniso["obs"], dtype=float),
        z=np.asarray(aniso["z"], dtype=float),
        cov=np.asarray(aniso["cov"], dtype=float),
    )
    return iso_path, aniso_path


def main() -> None:
    iso_path, aniso_path = write_bao_full_npz()
    print(f"[ok] Wrote {iso_path}")
    print(f"[ok] Wrote {aniso_path}")


if __name__ == "__main__":
    main()
