"""
Fetch supernova datasets (Pantheon+, SH0ES) into the raw data directory.

The fetcher records SHA256 checksums and retains a provenance log so that
downstream preparation steps can verify source integrity. For unit tests
and offline development the `--use-sample` flag copies deterministic sample
files bundled with the repository instead of performing network downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    requests = None


SAMPLE_DATA = {
    "pantheon_plus": [
        (
            Path("pipelines/data/samples/pantheon_plus/Pantheon+SH0ES_mock.dat"),
            "Pantheon+SH0ES.dat",
        ),
        (
            Path("pipelines/data/samples/pantheon_plus/Pantheon+SH0ES_mock_STATONLY.cov"),
            "Pantheon+SH0ES_STATONLY.cov",
        ),
        (
            Path("pipelines/data/samples/pantheon_plus/Pantheon+SH0ES_mock_STAT+SYS.cov"),
            "Pantheon+SH0ES_STAT+SYS.cov",
        ),
    ],
    "shoes": [],
}


@dataclass
class FetchRecord:
    path: Path
    url: str
    sha256: str
    size: int

    def as_dict(self) -> dict:
        return {
            "path": str(self.path.resolve()),
            "url": self.url,
            "sha256": self.sha256,
            "size": self.size,
        }


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_requests():
    if requests is None:
        raise RuntimeError(
            "The requests package is required for network downloads. "
            "Install it or provide --use-sample to copy bundled fixtures."
        )


def download_file(url: str, destination: Path) -> FetchRecord:
    """
    Download a remote file with retry and exponential backoff.
    Handles SSL and transient connection errors gracefully.
    """
    ensure_requests()
    max_retries = 5
    backoff_base = 2
    for attempt in range(1, max_retries + 1):
        tmp_path: Path | None = None
        try:
            response = requests.get(url, stream=True, timeout=(10, 120))
            response.raise_for_status()

            with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = Path(tmp.name)

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_path, destination)
            digest = sha256sum(destination)
            return FetchRecord(
                path=destination,
                url=url,
                sha256=digest,
                size=destination.stat().st_size,
            )

        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            # transient network or SSL failure
            wait = backoff_base ** attempt
            print(f"[WARN] Attempt {attempt}/{max_retries} failed for {url}: {e}")
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < max_retries:
                print(f"[INFO] Retrying in {wait}s ...")
                time.sleep(wait)
                continue
            else:
                print(f"[ERROR] Giving up after {max_retries} attempts.")
                raise

        except Exception as e:
            # catch-all for other unexpected failures
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            print(f"[ERROR] Unexpected failure on {url}: {e}")
            raise



def copy_sample(source: Path, destination: Path) -> FetchRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256sum(destination)
    return FetchRecord(path=destination, url=f"file://{source.resolve()}", sha256=digest, size=destination.stat().st_size)


def write_log(out_dir: Path, records: Iterable[FetchRecord], source: str, release_tag: str, used_sample: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "fetched.json"
    payload = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": source,
        "release_tag": release_tag,
        "used_sample": used_sample,
        "records": [record.as_dict() for record in records],
    }
    with tempfile.NamedTemporaryFile("w", dir=out_dir, delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2)
        temp_name = tmp.name
    shutil.move(temp_name, log_path)


def pantheon_plus_files(release_tag: str) -> List[dict]:
    base = f"https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/{release_tag}/Pantheon+_Data/4_DISTANCES_AND_COVAR"
    return [
        {"url": f"{base}/Pantheon+SH0ES.dat", "filename": "Pantheon+SH0ES.dat"},
        {"url": f"{base}/Pantheon+SH0ES_STATONLY.cov", "filename": "Pantheon+SH0ES_STATONLY.cov"},
        {"url": f"{base}/Pantheon+SH0ES_STAT+SYS.cov", "filename": "Pantheon+SH0ES_STAT+SYS.cov"},
        {"url": f"{base}/README", "filename": "README_DISTANCES_AND_COVAR"},
    ]


def shoes_files(release_tag: str) -> List[dict]:
    base = f"https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/{release_tag}/SH0ES_Data"
    return [
        {"url": f"{base}/table2.tex", "filename": "table2.tex"},
        {"url": f"{base}/table2.README", "filename": "table2.README"},
    ]


def resolve_source_files(source: str, release_tag: str) -> List[dict]:
    if source == "pantheon_plus":
        return pantheon_plus_files(release_tag)
    if source == "shoes":
        return shoes_files(release_tag)
    raise ValueError(f"Unknown source '{source}'")


def fetch_source(source: str, out_dir: Path, use_sample: bool, release_tag: str) -> List[FetchRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[FetchRecord] = []
    if use_sample:
        samples = SAMPLE_DATA.get(source, [])
        if not samples:
            raise RuntimeError(f"No bundled sample data available for source '{source}'")
        for sample_path, target_name in samples:
            src = sample_path
            if not src.exists():
                raise FileNotFoundError(f"Sample file missing: {src}")
            records.append(copy_sample(src, out_dir / target_name))
        return records

    file_specs = resolve_source_files(source, release_tag)
    for file_spec in file_specs:
        url = file_spec["url"]
        destination = out_dir / file_spec["filename"]
        records.append(download_file(url, destination))
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch supernova raw datasets.")
    parser.add_argument("--source", choices=["pantheon_plus", "shoes"], required=True)
    parser.add_argument("--out", required=True, help="Destination directory for raw files")
    parser.add_argument("--release-tag", default="main", help="Git tag or branch in the Pantheon+SH0ES/DataRelease repo")
    parser.add_argument("--use-sample", action="store_true", help="Copy bundled sample data instead of downloading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser()
    records = fetch_source(args.source, out_dir, args.use_sample, args.release_tag)
    write_log(out_dir, records, args.source, args.release_tag, args.use_sample)
    print(f"[INFO] Fetched {len(records)} file(s) for {args.source} into {out_dir.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1:])
