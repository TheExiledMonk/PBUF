"""
Data Downloader - Fetch public cosmology datasets.

This module downloads known public datasets into data/raw/ with proper
metadata tracking for reproducibility.

Supported datasets are configured under config/downloader/datasets.yaml.
"""

import json
from pathlib import Path
from datetime import datetime
import shutil
import urllib.request
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DOWNLOADER_CONFIG_DIR = REPO_ROOT / "config" / "downloader"
DATASET_CONFIG_PATH = DOWNLOADER_CONFIG_DIR / "datasets.yaml"


def _load_dataset_sources() -> dict[str, dict[str, object]]:
    if not DATASET_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration missing at {DATASET_CONFIG_PATH}."
            " Create the file or install the datasets package."
        )
    with DATASET_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("Dataset configuration must define a 'datasets' mapping.")
    return datasets


DATASET_SOURCES = _load_dataset_sources()


def available_datasets() -> tuple[str, ...]:
    """Return dataset keys currently configured for download."""
    return tuple(DATASET_SOURCES.keys())


def download_dataset(name: str) -> dict:
    """
    Download the requested dataset into data/raw/<name>/ and return metadata.

    Parameters
    ----------
    name : str
        Dataset name (must be in DATASET_SOURCES)

    Returns
    -------
    dict
        Metadata dictionary with source info, local paths, etc.

    Raises
    ------
    ValueError
        If dataset name is not supported
    """
    if name not in DATASET_SOURCES:
        available = ", ".join(DATASET_SOURCES.keys())
        print(f"📝 Dataset '{name}' not implemented yet.")
        print(f"   Available datasets: {available}")
        print(f"   Creating placeholder directory for manual placement...")
        return _create_placeholder_dataset(name)

    source_info = DATASET_SOURCES[name]

    # Create output directory (where metadata lives)
    output_dir = Path(f"data/raw/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_hint = source_info.get("raw_dir")
    raw_path = _resolve_raw_path(raw_hint) if raw_hint else output_dir
    raw_ready = raw_path.exists()

    print(f"📥 Downloading {name}...")
    print(f"   Source: {source_info['url']}")
    print(f"   Raw data expected at: {raw_path}")

    url = source_info["url"]
    placeholder = url.lower().startswith("placeholder://")
    local_file = None
    status = "downloaded"

    if placeholder:
        print("ℹ️ Placeholder source detected – no automatic download.")
        if raw_ready:
            print("   Raw data already present.")
        else:
            print("   Raw data not found; create the bundle at the location above.")
        _write_manual_planck_hint(output_dir, raw_path, name, source_info)
        status = "manual-ready" if raw_ready else "manual"
    else:
        # Download file
        local_file = output_dir / source_info["filename"]
        try:
            urllib.request.urlretrieve(source_info["url"], local_file)
            print(f"✅ Downloaded to {local_file}")
        except Exception as e:
            print(f"⚠️ Download failed: {e}")
            print(f"   Creating placeholder for manual download...")
            _create_download_placeholder(output_dir, source_info)
            local_file = None
            status = "placeholder"
        if local_file and source_info.get("extract"):
            extracted = _maybe_extract_archive(local_file, output_dir, raw_path)
            raw_ready = raw_path.exists()
            if extracted:
                print("   Raw data extracted and ready.")

    metadata = {
        "dataset": name,
        "downloaded_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": source_info["url"],
        "notes": source_info["notes"],
        "raw_path": str(raw_path),
        "raw_ready": raw_ready,
        "local_path": str(local_file) if local_file else None,
        "status": status
    }

    metadata_file = output_dir / "source.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Created metadata: {metadata_file}")

    return metadata


def _create_placeholder_dataset(name: str) -> dict:
    """Create a placeholder directory structure for unimplemented datasets."""
    output_dir = Path(f"data/raw/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create README with instructions
    readme_content = f"""# Placeholder for {name}

This dataset is not yet automatically downloadable.

## Manual Instructions

Please manually download the dataset and place files in this directory.

Dataset: {name}
Expected files: [consult the original data source]

## Sources

The typical source for this dataset would be:
- Academic paper: [DOI or arXiv link]
- Data repository: [URL]
- Survey website: [URL]

## After Manual Placement

Run the conversion command:
```
python -m toolbox.cli data-sync --datasets {name}
```

## Metadata

This placeholder was created by PBUF CLI on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    readme_file = output_dir / "README.txt"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    metadata = {
        "dataset": name,
        "downloaded_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": "manual_placement_required",
        "notes": f"Placeholder created - manual download required",
        "local_path": None,
        "status": "placeholder"
    }

    metadata_file = output_dir / "source.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _create_download_placeholder(output_dir: Path, source_info: dict):
    """Create placeholder files when download fails."""
    readme_content = f"""# Download Placeholder for {output_dir.name}

Automatic download failed. Please manually download from:

{source_info['url']}

## Expected Files

{chr(10).join(f"- {f}" for f in source_info.get('expected_files', []))}

    ## Instructions

    1. Download the data from the URL above
    2. Extract/place files in this directory
    3. Run conversion: python -m toolbox.cli data-sync --datasets {output_dir.name}

## Notes

{source_info['notes']}

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    readme_file = output_dir / "README_download_failed.txt"
    with open(readme_file, "w") as f:
        f.write(readme_content)


def _maybe_extract_archive(archive_path: Path, download_dir: Path, raw_path: Path) -> bool:
    """Extract downloaded archives if requested by the dataset configuration."""
    if archive_path is None or not archive_path.exists():
        print("   ⚠️ Archive missing; skipping extraction.")
        return False
    extract_root = raw_path.parent
    extract_root.mkdir(parents=True, exist_ok=True)
    try:
        print(f"   Extracting archive contents to {extract_root}...")
        shutil.unpack_archive(str(archive_path), str(extract_root))
        if raw_path.exists():
            return True
        print("   ⚠️ Expected raw directory still missing after extraction.")
        return False
    except shutil.ReadError as exc:
        print(f"   ⚠️ Failed to unpack archive: {exc}")
    except Exception as exc:
        print(f"   ⚠️ Extraction error: {exc}")
    return False


def _resolve_raw_path(raw_hint: str) -> Path:
    """Resolve the raw data directory, allowing relative hints."""
    candidate = Path(raw_hint)
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / "data/raw" / candidate).resolve()


def _write_manual_planck_hint(output_dir: Path, raw_path: Path, dataset_name: str, source_info: dict):
    """Write context-specific instructions for manually provided Planck data."""
    hint_file = output_dir / "README_planck_manual.md"
    expected_files = source_info.get("expected_files") or []
    expected_section = "\n".join(f"- {entry}" for entry in expected_files) if expected_files else "- (none specified)"

    content = f"""# Manual placeholder for {dataset_name}

Automatic download is disabled for Planck 2018 raw products.

Place the extracted Planck bundle so that the top-level directory matches:
{raw_path}

Expected files within the archive:
{expected_section}

Notes:
{source_info['notes']}

After the files are in place, rerun:
```
python -m toolbox.cli data-sync --datasets {dataset_name}
```

The final download URL will be provided later; update the configuration once it is available.
"""

    hint_file.write_text(content)


def list_available_datasets():
    """List all available datasets for download."""
    print("📊 Available datasets for download:")
    for name, info in DATASET_SOURCES.items():
        print(f"\n  {name}")
        print(f"    URL: {info['url']}")
        print(f"    Notes: {info['notes']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python downloader.py <dataset_name>")
        print("\nAvailable datasets:")
        list_available_datasets()
        sys.exit(1)

    dataset_name = sys.argv[1]
    metadata = download_dataset(dataset_name)

    print(f"\n✅ Download operation completed for {dataset_name}")
    print(f"   Status: {metadata['status']}")
    if metadata['local_path']:
        print(f"   Local path: {metadata['local_path']}")
