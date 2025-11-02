"""
Data Downloader - Fetch public cosmology datasets.

This module downloads known public datasets into data/raw/ with proper
metadata tracking for reproducibility.

Supported datasets:
- planck2018_distance_priors
- pantheon_sn
- bao_boss_dr12
- bao_eBOSS
- cc_cosmic_chronometers_compilation
- rsd_fsigma8_compilation
"""

import json
import os
from pathlib import Path
from datetime import datetime
import urllib.request


# Dataset source information
DATASET_SOURCES = {
    "planck2018_distance_priors": {
        "url": "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_CosmoParams_R3.00.zip",
        "notes": "Planck 2018 compressed distance priors R, l_A, theta_* from Table 2 of Planck 2018 paper VI",
        "filename": "COM_CosmoParams_R3.00.zip",
        "expected_files": ["COM_PowerSpect_CMB_R3.00.zip", "COM_CosmoParams_R3.00.pdf"]
    },
    "pantheon_sn": {
        "url": "https://github.com/dscolnic/Pantheon/archive/refs/heads/master.zip",
        "notes": "Pantheon+ supernovae compilation with distance moduli and covariance",
        "filename": "Pantheon-master.zip",
        "expected_files": ["Pantheon/data/Pantheon.dat", "Pantheon/data/sys_full_long.txt"]
    },
    "bao_boss_dr12": {
        "url": "https://data.sdss.org/sas/dr12/boss/papers/clustering/BAO_consensus_boss_dr12.json",
        "notes": "BOSS DR12 BAO consensus measurements including isotropic and anisotropic",
        "filename": "BAO_consensus_boss_dr12.json",
        "expected_files": ["BAO_consensus_boss_dr12.json"]
    },
    "bao_eBOSS": {
        "url": "https://data.sdss.org/sas/dr16/eboss/papers/clustering/eBOSS_LRG_bao_consensus.json",
        "notes": "eBOSS LRG BAO measurements",
        "filename": "eBOSS_LRG_bao_consensus.json",
        "expected_files": ["eBOSS_LRG_bao_consensus.json"]
    },
    "cc_cosmic_chronometers_compilation": {
        "url": "https://arxiv.org/src/1807.06209v3/anc/cosmic_chronometers.txt",
        "notes": "Compilation of cosmic chronometer H(z) measurements",
        "filename": "cosmic_chronometers.txt",
        "expected_files": ["cosmic_chronometers.txt"]
    },
    "rsd_fsigma8_compilation": {
        "url": "https://arxiv.org/src/1801.02656v3/anc/fsigma8_compilation.txt",
        "notes": "Compilation of redshift-space distortion fσ8 measurements",
        "filename": "fsigma8_compilation.txt",
        "expected_files": ["fsigma8_compilation.txt"]
    }
}


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

    # Create output directory
    output_dir = Path(f"data/raw/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {name}...")
    print(f"   Source: {source_info['url']}")

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

    # Create metadata file
    metadata = {
        "dataset": name,
        "downloaded_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": source_info["url"],
        "notes": source_info["notes"],
        "local_path": str(local_file) if local_file else None,
        "status": "downloaded" if local_file else "placeholder"
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
python cli.py dataset convert --source {name} --output data/standardized/{name}.npz
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
3. Run conversion: python cli.py dataset convert --source {output_dir.name} --output data/standardized/{output_dir.name}.npz

## Notes

{source_info['notes']}

Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    readme_file = output_dir / "README_download_failed.txt"
    with open(readme_file, "w") as f:
        f.write(readme_content)


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
