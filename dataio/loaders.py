"""
Dataset loaders with schema validation for the PBUF pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from dataio.registry import DatasetManifest, load_manifest
from dataio.transforms import to_numpy
from dataio.validators import ValidationError, validate_covariance, validate_lengths, validate_vector
from utils import logging as log
from utils.io import read_json, read_yaml


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_supernovae(config: Mapping[str, object]) -> Dict[str, object]:
    """
    Load a supernova distance modulus dataset.

    Returns
    -------
    dict
        Standardised dataset dictionary documented in the project spec.
    """

    data_path = _resolve_path(str(config["data_path"]))
    table = np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    z = to_numpy(table["z"])
    mu = to_numpy(table["mu"])
    sigma = to_numpy(table["sigma_mu"]) if "sigma_mu" in table.dtype.names else None

    cov_path = config.get("covariance_path")
    cov = None
    if cov_path:
        cov_abs = _resolve_path(str(cov_path))
        cov = np.loadtxt(cov_abs, delimiter=",")
        validate_covariance(cov)
    validate_vector(z, name="z")
    validate_vector(mu, name="mu")
    if sigma is not None:
        validate_vector(sigma, name="sigma_mu")
        validate_lengths(z, mu, sigma)
    else:
        validate_lengths(z, mu)

    meta = {
        "name": config.get("name", "Unknown"),
        "path": str(data_path),
        "covariance": str(_resolve_path(str(cov_path))) if cov_path else None,
        "notes": config.get("description"),
    }

    dataset = {
        "z": z,
        "y": mu,
        "sigma": sigma,
        "cov": cov,
        "tags": config.get("tags", []),
        "meta": meta,
    }
    return dataset


LOADERS = {
    "mock_supernovae": load_supernovae,
}


def load_sn_pantheon(entry: Mapping[str, object]) -> Dict[str, object]:
    prepared = entry.get("prepared", {})
    table_path = _resolve_path(str(prepared["table"]))
    cov_path = _resolve_path(str(prepared["cov"]))
    meta_path = _resolve_path(str(prepared["meta"]))
    manifest: DatasetManifest = load_manifest(meta_path)

    if pd is None:
        raise RuntimeError("pandas is required to load prepared Pantheon+ datasets")

    df = pd.read_csv(table_path)
    if "row_index" not in df.columns:
        raise ValueError("Prepared table missing 'row_index' column")
    z_prefer = entry.get("z_prefer", manifest.get("z_prefer", "z_cmb"))
    z_column = z_prefer if z_prefer in df.columns else "z_helio"
    if z_column not in df.columns:
        raise ValueError(f"Preferred redshift column '{z_column}' not found in prepared table")

    z = df[z_column].to_numpy(dtype=float)
    mu = df["mu"].to_numpy(dtype=float)
    sigma = df["sigma_mu"].to_numpy(dtype=float) if "sigma_mu" in df.columns else None
    if sigma is not None and np.isnan(sigma).all():
        sigma = None

    cov = np.load(cov_path)
    validate_covariance(cov)
    if cov.shape[0] != len(df):
        raise ValueError(f"Covariance shape {cov.shape} does not match table length {len(df)}")
    index_map = df["row_index"].to_numpy(dtype=int)

    validate_vector(z, name="z")
    validate_vector(mu, name="mu")
    if sigma is not None:
        validate_vector(sigma, name="sigma_mu")

    tags = df["sample_flags"].fillna("").astype(str).tolist() if "sample_flags" in df.columns else []

    records = manifest.get("records", [])
    raw_files = [record["path"] for record in records if record.get("kind") == "raw"]
    derived_files = [record["path"] for record in records if record.get("kind") == "derived"]
    meta = {
        "name": entry.get("name", "Pantheon+ Supernovae"),
        "N": int(manifest.get("row_count", len(df))),
        "source": "pantheon+",
        "z_prefer": manifest.get("z_prefer", z_column),
        "z_column_used": z_column,
        "release_tag": manifest.get("release_tag", entry.get("release_tag")),
        "path": str(table_path),
        "covariance": str(cov_path),
        "raw_files": raw_files,
        "hashes": records,
        "prepared_at": manifest.prepared_at,
        "transform_version": manifest.transform_version,
        "notes": manifest.get("notes", ""),
        "covariance_components_used": manifest.get("covariance_components_used", []),
        "table_path": str(table_path),
        "covariance_path": str(cov_path),
        "meta_path": str(meta_path),
        "derived_files": derived_files,
    }

    dataset = {
        "z": z,
        "y": mu,
        "mu": mu,
        "sigma": sigma,
        "sigma_mu": sigma,
        "cov": cov,
        "tags": tags,
        "index_map": index_map,
        "meta": meta,
    }
    return dataset


def _find_dataset_entry(name: str, config: Mapping[str, Mapping[str, object]]) -> Tuple[str, Mapping[str, object]]:
    datasets = config["datasets"]
    if name in datasets:
        return name, datasets[name]
    for group_name, group in datasets.items():
        if isinstance(group, Mapping) and name in group:
            return group_name, group[name]
    raise KeyError(f"Dataset '{name}' not found in configuration.")


def load_dataset(name: str, config: Mapping[str, Mapping[str, object]]) -> Dict[str, object]:
    """
    Dispatch to the appropriate loader by dataset key.

    Parameters
    ----------
    name : str
        Dataset identifier (see config/datasets.yml).
    config : mapping
        Parsed datasets configuration.
    """

    group, entry = _find_dataset_entry(name, config)
    if "prepared" in entry:
        return load_sn_pantheon(entry)

    loader = LOADERS.get(name, load_supernovae)
    return loader(entry)


def _select_cmb_prior_entry(name: str) -> Mapping[str, object]:
    datasets_cfg = read_yaml("config/datasets.yml")
    _, entry = _find_dataset_entry(name, datasets_cfg)
    return entry


def load_cmb_priors(name: str) -> Dict[str, object]:
    """
    Load compressed CMB distance priors defined in `config/datasets.yml`.

    Parameters
    ----------
    name : str
        Identifier under the `cmb_priors` section.
    """

    entry = _select_cmb_prior_entry(name)
    declared_paths = []
    if "path" in entry:
        declared_paths.append(("primary", _resolve_path(str(entry["path"]))))
    if "sample" in entry:
        declared_paths.append(("sample", _resolve_path(str(entry["sample"]))))
    if not declared_paths:
        raise ValidationError(f"CMB prior '{name}' has no declared data paths")

    payload = None
    resolved_path = None
    source_tag = None
    for tag, candidate in declared_paths:
        if candidate.exists():
            payload = read_json(candidate)
            resolved_path = candidate
            source_tag = tag
            break
    if payload is None or resolved_path is None:
        paths = ", ".join(str(path) for _, path in declared_paths)
        raise FileNotFoundError(f"CMB prior '{name}' not found at {paths}")
    if not isinstance(payload, dict):
        raise ValidationError("CMB prior file must contain a JSON object")

    labels: list[str]
    means: Dict[str, float]
    cov_matrix: np.ndarray

    if "parameters" in payload:
        params_block = payload["parameters"]
        if not isinstance(params_block, Mapping):
            raise ValidationError("'parameters' block must be a mapping")
        labels = list(params_block.keys())
        means = {}
        sigmas: list[float] = []
        for label, spec in params_block.items():
            if not isinstance(spec, Mapping):
                raise ValidationError(f"Parameter '{label}' must map to a dict with mean/sigma")
            try:
                mean_val = float(spec["mean"])
                sigma_val = float(spec["sigma"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(f"Parameter '{label}' entries must include numeric 'mean' and 'sigma'") from exc
            if sigma_val <= 0.0:
                raise ValidationError(f"Parameter '{label}' must have positive uncertainty")
            means[label] = mean_val
            sigmas.append(sigma_val)
        corr = payload.get("correlation")
        if corr is None:
            raise ValidationError("CMB prior file missing 'correlation' matrix for covariance reconstruction")
        corr_matrix = np.asarray(corr, dtype=float)
        validate_covariance(corr_matrix)
        sigmas_arr = np.asarray(sigmas, dtype=float)
        cov_matrix = np.outer(sigmas_arr, sigmas_arr) * corr_matrix
    else:
        means_payload = payload.get("means")
        labels_payload = payload.get("labels")
        cov = payload.get("cov")
        if not isinstance(means_payload, Mapping):
            raise ValidationError("CMB prior file missing 'means' dictionary")
        if not isinstance(labels_payload, list) or not all(isinstance(label, str) for label in labels_payload):
            raise ValidationError("CMB prior file must provide an ordered 'labels' list")
        if cov is None:
            raise ValidationError("CMB prior file missing 'cov' field")
        labels = list(labels_payload)
        means = {key: float(value) for key, value in means_payload.items()}
        cov_matrix = np.asarray(cov, dtype=float)

    

    validate_covariance(cov_matrix)
    try:
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ValidationError("Failed to diagonalise covariance matrix") from exc
    if np.any(eigenvalues <= 0.0):
        raise ValidationError("Covariance matrix must be positive definite")
    if cov_matrix.shape[0] != len(labels):
        raise ValidationError("Covariance dimension does not match label count")
    meta = {key: value for key, value in entry.items() if key not in {"path", "sample"}}
    file_meta = {key: value for key, value in payload.items() if key not in {"parameters", "correlation", "inverse_covariance_example"}}
    meta.update(file_meta)
    meta["resolved_path"] = str(resolved_path)
    meta["source_tag"] = source_tag

    return {
        "means": means,
        "cov": cov_matrix,
        "labels": labels,
        "meta": meta,
    }

# ---------------------------------------------------------------------
#  BAO Loaders (Isotropic and Anisotropic)
# ---------------------------------------------------------------------

def load_bao_priors(name: str) -> Dict[str, object]:
    """
    Load isotropic BAO distance prior dataset from config/datasets.yml.

    Expected JSON structure (example):
    {
        "labels": ["Dv_rs_0.106", "Dv_rs_0.35", "Dv_rs_0.57"],
        "means": {"Dv_rs_0.106": 3.047, "Dv_rs_0.35": 8.88, "Dv_rs_0.57": 13.67},
        "cov": [[...]],
        "meta": {"release_tag": "BOSS_DR12", "source": "6dF+BOSS+SDSS"}
    }
    """
    entry = _select_cmb_prior_entry(name)
    declared_paths = []
    if "path" in entry:
        declared_paths.append(("primary", _resolve_path(str(entry["path"]))))
    if "sample" in entry:
        declared_paths.append(("sample", _resolve_path(str(entry["sample"]))))
    if not declared_paths:
        raise ValidationError(f"BAO prior '{name}' has no declared paths")

    payload, resolved_path, source_tag = None, None, None
    for tag, candidate in declared_paths:
        if candidate.exists():
            payload = read_json(candidate)
            resolved_path = candidate
            source_tag = tag
            break
    if payload is None or resolved_path is None:
        paths = ", ".join(str(p) for _, p in declared_paths)
        raise FileNotFoundError(f"BAO prior '{name}' not found at {paths}")

    if not isinstance(payload, dict):
        raise ValidationError("BAO prior file must contain a JSON object")

    # Validate structure
    labels = payload.get("labels")
    means = payload.get("means")
    cov = payload.get("cov")

    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise ValidationError("BAO prior must define a 'labels' list of strings")
    if not isinstance(means, Mapping):
        raise ValidationError("BAO prior must define a 'means' mapping")
    if cov is None:
        raise ValidationError("BAO prior must define a 'cov' matrix")

    cov_matrix = np.asarray(cov, dtype=float)
    validate_covariance(cov_matrix)
    if cov_matrix.shape[0] != len(labels):
        raise ValidationError("Covariance dimension does not match labels")

    meta = {k: v for k, v in entry.items() if k not in {"path", "sample"}}
    meta.update(payload.get("meta", {}))
    meta["resolved_path"] = str(resolved_path)
    meta["source_tag"] = source_tag

    return {
        "means": {k: float(v) for k, v in means.items()},
        "cov": cov_matrix,
        "labels": labels,
        "meta": meta,
    }


def load_bao_ani_priors(name: str) -> Dict[str, object]:
    """
    Load anisotropic BAO priors (D_M/r_s and H(z)*r_s constraints).

    Expected JSON structure:
    {
        "labels": ["DM_rs_0.38", "DM_rs_0.51", "DM_rs_0.61", "Hrs_0.38", "Hrs_0.51", "Hrs_0.61"],
        "means": {"DM_rs_0.38": 10.3, "Hrs_0.38": 0.045, ...},
        "cov": [[...]],
        "meta": {"release_tag": "BOSS_DR12_ANI", "source": "BOSS/eBOSS anisotropic BAO"}
    }
    """
    entry = _select_cmb_prior_entry(name)
    declared_paths = []
    if "path" in entry:
        declared_paths.append(("primary", _resolve_path(str(entry["path"]))))
    if "sample" in entry:
        declared_paths.append(("sample", _resolve_path(str(entry["sample"]))))
    if not declared_paths:
        raise ValidationError(f"BAO anisotropic prior '{name}' has no declared paths")

    payload, resolved_path, source_tag = None, None, None
    for tag, candidate in declared_paths:
        if candidate.exists():
            payload = read_json(candidate)
            resolved_path = candidate
            source_tag = tag
            break
    if payload is None or resolved_path is None:
        paths = ", ".join(str(p) for _, p in declared_paths)
        raise FileNotFoundError(f"BAO anisotropic prior '{name}' not found at {paths}")

    if not isinstance(payload, dict):
        raise ValidationError("BAO anisotropic prior must be a JSON object")

    labels = payload.get("labels")
    means = payload.get("means")
    cov = payload.get("cov")

    if not isinstance(labels, list) or not all(isinstance(l, str) for l in labels):
        raise ValidationError("BAO anisotropic prior must define a 'labels' list of strings")
    if not isinstance(means, Mapping):
        raise ValidationError("BAO anisotropic prior must define a 'means' mapping")
    if cov is None:
        raise ValidationError("BAO anisotropic prior must define a 'cov' matrix")

    cov_matrix = np.asarray(cov, dtype=float)
    validate_covariance(cov_matrix)
    if cov_matrix.shape[0] != len(labels):
        raise ValidationError("Covariance dimension does not match labels")

    meta = {k: v for k, v in entry.items() if k not in {"path", "sample"}}
    meta.update(payload.get("meta", {}))
    meta["resolved_path"] = str(resolved_path)
    meta["source_tag"] = source_tag

    return {
        "means": {k: float(v) for k, v in means.items()},
        "cov": cov_matrix,
        "labels": labels,
        "meta": meta,
    }
def load_bao_real_data(
    csv_path: str | Path = "data/bao/derived/bao_index.csv",
    cov_path: str | Path = "data/bao/derived/bao_index.cov.npy",
    meta_path: str | Path = "data/bao/derived/bao_index.meta.json",
) -> Dict[str, object]:
    """
    Load prepared BAO data (from prepare_bao_data.py output).

    Supports isotropic (D_V/r_s) and anisotropic (D_M/r_s, H*r_s) together.
    Returns a dataset dictionary compatible with BAO fit pipelines.
    """
    csv_path = _resolve_path(str(csv_path))
    cov_path = _resolve_path(str(cov_path))
    meta_path = _resolve_path(str(meta_path))

    if not csv_path.exists():
        raise FileNotFoundError(f"BAO data table not found at {csv_path}")
    if not pd:
        raise RuntimeError("pandas is required to load BAO derived datasets")

    df = pd.read_csv(csv_path)
    cov = np.load(cov_path) if cov_path.exists() else None
    meta = read_json(meta_path) if meta_path.exists() else {}

    # Build label list dynamically
    labels = []
    for _, row in df.iterrows():
        z = row["z"]
        if not np.isnan(row.get("Dv_over_rs", np.nan)):
            labels.append(f"Dv_rs_{z:.3f}")
        if not np.isnan(row.get("DM_over_rs", np.nan)):
            labels.append(f"DM_rs_{z:.3f}")
        if not np.isnan(row.get("H_times_rs", np.nan)):
            labels.append(f"Hrs_{z:.3f}")

    # Build mean mapping
    means = {}
    for _, row in df.iterrows():
        z = row["z"]
        if not np.isnan(row.get("Dv_over_rs", np.nan)):
            means[f"Dv_rs_{z:.3f}"] = float(row["Dv_over_rs"])
        if not np.isnan(row.get("DM_over_rs", np.nan)):
            means[f"DM_rs_{z:.3f}"] = float(row["DM_over_rs"])
        if not np.isnan(row.get("H_times_rs", np.nan)):
            means[f"Hrs_{z:.3f}"] = float(row["H_times_rs"])

    dataset = {
        "means": means,
        "labels": labels,
        "cov": cov,
        "meta": {
            "release_tag": meta.get("release_tag", "DESI_DR2"),
            "source": meta.get("raw_directory", "CobayaSampler/bao_data v2.6"),
            "prepared_at": meta.get("prepared_at"),
            "transform_version": meta.get("transform_version"),
            "resolved_path": str(csv_path),
        },
    }

    log.info(f"Loaded real BAO dataset with {len(labels)} measurements from {csv_path}")
    return dataset

# ---------------------------------------------------------------------
#  Register additional dataset loaders
# ---------------------------------------------------------------------

LOADERS.update({
    "bao_real": load_bao_real_data,
    "bao_priors": load_bao_priors,
    "bao_ani_priors": load_bao_ani_priors,
})