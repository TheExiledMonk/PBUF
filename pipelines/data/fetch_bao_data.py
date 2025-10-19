#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch BAO datasets (DESI DR2, BOSS, eBOSS) from CobayaSampler/bao_data.

Instead of downloading single files, this script clones the official
GitHub repository and copies the relevant folder (e.g. `desi_bao_dr2`)
into the project's raw data directory. This ensures robust retrieval
even if URLs or folder structures change.

Offline users can provide --use-sample to copy mock data from
`pipelines/data/samples/bao/`.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import io
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

REPO_URL = "https://github.com/CobayaSampler/bao_data.git"
DATA_SUBDIR = "desi_bao_dr2"

SAMPLE_DATA = {
    "bao_data": [
        (Path("pipelines/data/samples/bao/bao_mock_iso.csv"), "bao_iso.csv"),
        (Path("pipelines/data/samples/bao/bao_mock_aniso.csv"), "bao_aniso.csv"),
    ]
}


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
#  Git-based fetch
# ---------------------------------------------------------------------
def clone_repo(temp_dir: Path, release_tag: str) -> Path:
    """Attempt to clone via git; if fails, return None so we can fallback."""
    repo_dir = temp_dir / "bao_data"
    cmd = ["git", "clone", "--depth", "1", "--branch", release_tag, REPO_URL, str(repo_dir)]
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Cloning {REPO_URL} (attempt {attempt}/{max_retries}) ...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return repo_dir
        except subprocess.CalledProcessError:
            wait = 2 ** attempt
            print(f"[WARN] Git clone failed, retrying in {wait}s ...")
            time.sleep(wait)
    print("[ERROR] Git clone failed after multiple attempts.")
    return None


def download_zip_fallback(temp_dir: Path, release_tag: str) -> Path:
    """Download repo as ZIP if git fails."""
    zip_url = f"https://github.com/CobayaSampler/bao_data/archive/refs/tags/v2.6.zip"
    print(f"[INFO] Attempting ZIP download from {zip_url}")
    r = requests.get(zip_url, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(temp_dir)
    extracted_dir = next(temp_dir.glob("bao_data-*"))
    print(f"[INFO] Extracted ZIP to {extracted_dir}")
    return extracted_dir


def copy_desi_data(repo_dir: Path, out_dir: Path) -> List[FetchRecord]:
    src_dir = repo_dir / DATA_SUBDIR
    if not src_dir.exists():
        raise FileNotFoundError(f"Expected data directory '{DATA_SUBDIR}' not found in {repo_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[FetchRecord] = []
    for file_path in src_dir.glob("*"):
        if not file_path.is_file():
            continue
        dest = out_dir / file_path.name
        shutil.copy2(file_path, dest)
        digest = sha256sum(dest)
        records.append(FetchRecord(dest, digest, dest.stat().st_size, "raw"))
    return records


# ---------------------------------------------------------------------
#  Offline mode
# ---------------------------------------------------------------------
def copy_sample(out_dir: Path) -> List[FetchRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[FetchRecord] = []
    for src, name in SAMPLE_DATA["bao_data"]:
        if not src.exists():
            raise FileNotFoundError(f"Sample file missing: {src}")
        dest = out_dir / name
        shutil.copy2(src, dest)
        digest = sha256sum(dest)
        records.append(FetchRecord(dest, digest, dest.stat().st_size, "sample"))
    return records


# ---------------------------------------------------------------------
#  Provenance log
# ---------------------------------------------------------------------
def write_log(out_dir: Path, records: Iterable[FetchRecord], release_tag: str, used_sample: bool, commit_hash: str | None = None):
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": REPO_URL,
        "release_tag": release_tag,
        "commit": commit_hash or "unknown",
        "used_sample": used_sample,
        "records": [r.as_dict() for r in records],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False, encoding="utf-8")
    json.dump(payload, tmp, indent=2)
    tmp.close()
    shutil.move(tmp.name, out_dir / "fetched.json")


# ---------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------
def fetch_source(out_dir: Path, use_sample: bool, release_tag: str) -> List[FetchRecord]:
    if use_sample:
        print("[INFO] Using bundled BAO sample data.")
        return copy_sample(out_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        repo_dir = clone_repo(tmpdir, release_tag)
        if repo_dir is None:
            try:
                repo_dir = download_zip_fallback(tmpdir, release_tag)
            except Exception as e:
                print(f"[ERROR] ZIP fallback failed: {e}")
                print("[WARN] Using sample data instead.")
                return copy_sample(out_dir)

        commit_hash = "unknown"
        git_dir = repo_dir / ".git"
        if git_dir.exists():
            try:
                commit_hash = subprocess.check_output(
                    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
            except Exception:
                pass

        records = copy_desi_data(repo_dir, out_dir)
        write_log(out_dir, records, release_tag, False, commit_hash)
        print(f"[INFO] ✅ Copied {len(records)} BAO files to {out_dir.resolve()}")
        return records


# ---------------------------------------------------------------------
#  CLI entrypoint
# ---------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch BAO data via Git clone from CobayaSampler/bao_data.")
    parser.add_argument("--out", required=True, help="Destination directory for BAO raw files")
    parser.add_argument("--release-tag", default="main", help="Git branch or tag (default: main)")
    parser.add_argument("--use-sample", action="store_true", help="Copy bundled mock BAO data instead of downloading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser()
    records = fetch_source(out_dir, args.use_sample, args.release_tag)
    print(f"[INFO] ✅ Fetched {len(records)} BAO file(s) into {out_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
