#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Cosmic Chronometer H(z) dataset (Moresco et al. 2022)
===========================================================

Downloads the latest official chronometer compilation from the
University of Bologna site:

    https://cluster.difa.unibo.it/astro/CC_data/data_CC.dat

This file lists 32 H(z) measurements (z, H, σH, reference).

The script only downloads and logs provenance; data preparation
(e.g. building the covariance matrix following Moresco+2020)
is handled in a separate `prepare_chronometers.py` pipeline.

Usage:
    python fetch_chronometers.py --out data/chronometers/raw
"""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List
import requests
from dataclasses import dataclass

# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------
DATA_URL = "https://cluster.difa.unibo.it/astro/CC_data/data_CC.dat"
FILENAME = "OHD_Moresco2022.dat"
SAMPLE_FILE = Path("pipelines/data/samples/chronometers/ohd_sample.csv")

# ---------------------------------------------------------------------
#  Utility and data model
# ---------------------------------------------------------------------
@dataclass
class FetchRecord:
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


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------
#  Download logic
# ---------------------------------------------------------------------
def fetch_online(out_dir: Path) -> List[FetchRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / FILENAME

    print(f"[INFO] Downloading cosmic chronometer data from {DATA_URL}")
    r = requests.get(DATA_URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)

    digest = sha256sum(dest)
    record = FetchRecord(dest, digest, dest.stat().st_size, "raw")
    print(f"[INFO] ✅ Downloaded {dest.name} ({dest.stat().st_size/1024:.1f} kB)")
    return [record]


def copy_sample(out_dir: Path) -> List[FetchRecord]:
    """Offline fallback: use bundled sample data."""
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Sample file missing: {SAMPLE_FILE}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / SAMPLE_FILE.name
    shutil.copy2(SAMPLE_FILE, dest)
    digest = sha256sum(dest)
    print(f"[WARN] Using sample chronometer data instead.")
    return [FetchRecord(dest, digest, dest.stat().st_size, "sample")]


# ---------------------------------------------------------------------
#  Provenance log
# ---------------------------------------------------------------------
def write_log(out_dir: Path, records: Iterable[FetchRecord], used_sample: bool):
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": DATA_URL,
        "used_sample": used_sample,
        "citation": "Moresco et al. (2022), University of Bologna CC dataset, "
                    "https://cluster.difa.unibo.it/astro/CC_data/data_CC.dat",
        "note": "Covariance must be added following Moresco et al. (2020, JCAP 05, 005).",
        "records": [r.as_dict() for r in records],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False, encoding="utf-8")
    json.dump(payload, tmp, indent=2)
    tmp.close()
    shutil.move(tmp.name, out_dir / "fetched.json")
    print(f"[INFO] Logged provenance to fetched.json")


# ---------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------
def fetch_source(out_dir: Path, use_sample: bool) -> List[FetchRecord]:
    try:
        if use_sample:
            records = copy_sample(out_dir)
        else:
            records = fetch_online(out_dir)
    except Exception as e:
        print(f"[ERROR] Fetch failed: {e}")
        print("[WARN] Falling back to sample data.")
        records = copy_sample(out_dir)
    write_log(out_dir, records, use_sample)
    return records


# ---------------------------------------------------------------------
#  CLI entrypoint
# ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch cosmic chronometer H(z) dataset.")
    p.add_argument("--out", required=True, help="Destination directory for raw chronometer data")
    p.add_argument("--use-sample", action="store_true", help="Copy bundled sample instead of downloading")
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser()
    records = fetch_source(out_dir, args.use_sample)
    print(f"[INFO] ✅ Fetched {len(records)} file(s) into {out_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
