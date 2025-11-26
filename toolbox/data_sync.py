"""Simple wrappers that orchestrate the Cosmos raw data download → conversion workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .downloader import available_datasets as _available_datasets, download_dataset
from .converter import convert_dataset


DEFAULT_STANDARDIZED_DIR = Path("data/standardized")


def available_datasets() -> Sequence[str]:
    """Return the dataset names that Cosmos knows how to download."""
    return tuple(_available_datasets())


def ensure_standardized_dir() -> Path:
    DEFAULT_STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_STANDARDIZED_DIR


def sync_dataset(
    name: str,
    dataset_type: str | None = None,
    cosmology_config: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """
    Download a raw dataset and convert it to the standardized NPZ.

    Parameters
    ----------
    name : str
        Dataset key defined in config/downloader/datasets.yaml.
    dataset_type : optional str
        Explicit converter hint ("CMB", "SN", etc.). If not provided,
        the converter will attempt to auto-detect.
    cosmology_config : optional Mapping
        Optional cosmology metadata which some conversions may require.

    Returns
    -------
    Mapping[str, object]
        Summary dict with metadata and output file path.
    """
    metadata = download_dataset(name)
    standardized_dir = ensure_standardized_dir()
    output_path = standardized_dir / f"{name}.npz"
    convert_dataset(
        name,
        str(output_path),
        dataset_type=dataset_type,
        cosmology_config=cosmology_config or {},
    )
    return {
        "name": name,
        "metadata": metadata,
        "output": str(output_path),
    }


def sync_all(
    *,
    dataset_names: Iterable[str] | None = None,
    dataset_type_map: Mapping[str, str] | None = None,
    cosmology_map: Mapping[str, Mapping[str, object]] | None = None,
) -> list[Mapping[str, object]]:
    """
    Sequentially download/convert all requested datasets.
    """
    results: list[Mapping[str, object]] = []
    names = dataset_names or available_datasets()
    for name in names:
        result = sync_dataset(
            name,
            dataset_type=dataset_type_map.get(name) if dataset_type_map else None,
            cosmology_config=cosmology_map.get(name) if cosmology_map else None,
        )
        results.append(result)
    return results
