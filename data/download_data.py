#!/usr/bin/env python3
"""
Unified GWOSC + GCN + Fermi data downloader with resume support.
-----------------------------------------------------------------

Features:
- Downloads and extracts the full GCN circular archive TAR bundle locally
- Fetches the GWOSC event-versions CSV directly from the public API
- Mirrors IceCube’s TXS 0506+056 samples, alert catalog, the T2K/Super-K joint fit release, and the Super-K neutron data release
- Converts GW GPS → UTC
- Finds all real Fermi GBM triggers for that calendar date from HEASARC
- Recursively checks `current/`, `quicklook/`, and `previous/` for data files
- Reads trigger time directly from FITS headers (TRIGTIME or fallback)
- Selects the trigger whose time is closest to the GW event
- Downloads all FITS files from that trigger (recursively)
- Outputs summary JSON to logs/downloads_summary.json
- RESUME SUPPORT: Detects last downloaded event, deletes it (assumes incomplete), and resumes

Usage:
    python download_data.py [--debug] [--force-downloads] [--skip-fermi]

Dependencies:
    pip install astropy
"""

import argparse
import os
import re
import sys
import csv
import json
import shutil
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError, HTTPError
from datetime import datetime, timedelta, timezone
from astropy.io import fits

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------

BASE_URL = "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/"
FERMI_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
TARGET_SUFFIXES = (".fit", ".fits")
GCN_ARCHIVE_URL = "https://gcn.nasa.gov/circulars/archive.json.tar.gz"
GWOSC_EVENT_VERSIONS_URL = (
    "https://gwosc.org/api/v2/event-versions?include-default-parameters=true&format=csv"
)
ICECUBE_TXS_URL = (
    "https://icecube.wisc.edu/data-releases/"
    "20180712_IceCube_data_from_2008_to_2017_related_to_analysis_of_TXS_0506+056.zip"
)
ICECUBE_ALERTS_URL = (
    "https://icecube.wisc.edu/data-releases/"
    "20180712_IceCube_catalog_of_alert_events_up_through_IceCube-170922A.zip"
)
T2K_SUPERK_URL = (
    "https://zenodo.org/records/12702685/files/DataRelease_JointFit.zip?download=1"
)
SUPERK_NEUTRON_URL = (
    "https://zenodo.org/records/15392411/files/sk_neutron_data.tar.gz?download=1"
)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
LOCAL_DATA_ROOT = REPO_ROOT / "data" / "quantum"
GCN_CIRCULARS_DIR = LOCAL_DATA_ROOT / "gcn"
ICECUBE_ROOT_DIR = LOCAL_DATA_ROOT / "icecube"
T2K_ROOT_DIR = LOCAL_DATA_ROOT / "t2k_superk"
SUPERK_NEUTRON_DIR = LOCAL_DATA_ROOT / "superk_neutron"
DEFAULT_EVENT_VERSIONS_PATH = LOCAL_DATA_ROOT / "gwosc" / "event-versions.csv"
GCN_ARCHIVE_TAR_NAME = GCN_ARCHIVE_URL.rsplit("/", 1)[-1]

DEBUG = False
SUMMARY = {}

CHUNK_SIZE = 1 << 15  # 32 KB
STALL_WINDOW = 90.0  # seconds of observation before considering stall
STALL_RATIO = 75.0   # current rate must not drop below avg/STALL_RATIO
MIN_RATE_BPS = 512.0  # absolute minimum rate before we call it stalled
STALL_SLEEP = 60  # seconds to wait before retrying stalled download
DOWNLOAD_STATS = {"avg_rate": None}

# ---------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------

def log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")


class SlowDownloadError(Exception):
    """Raised when a download appears to be stalled for too long."""

# ---------------------------------------------------------------
# TIME CONVERSIONS
# ---------------------------------------------------------------

def gw_gps_to_utc(gps_value: str) -> datetime:
    """Convert GW GPS seconds to UTC."""
    gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
    return gps_epoch + timedelta(seconds=float(gps_value))

def met_to_utc(met: float) -> datetime:
    """Convert Fermi MET seconds → UTC."""
    return FERMI_EPOCH + timedelta(seconds=met)

# ---------------------------------------------------------------
# HTTP HELPERS
# ---------------------------------------------------------------

def http_get(url: str):
    try:
        log(f"GET {url}")
        return urllib.request.urlopen(url, timeout=15)
    except (HTTPError, URLError) as e:
        log(f"HTTP error for {url}: {e}")
        return None
    except Exception as e:
        log(f"Error for {url}: {e}")
        return None

# ---------------------------------------------------------------
# DIRECTORY LISTINGS
# ---------------------------------------------------------------

def list_year_triggers(year: int) -> list[str]:
    """List all Fermi GBM trigger directories for a given year."""
    url = f"{BASE_URL}{year}/"
    resp = http_get(url)
    if not resp:
        return []
    html = resp.read().decode(errors="ignore")
    return sorted(set(re.findall(r'href="(bn\d{9})/"', html)))

def list_date_triggers(utc_dt: datetime) -> tuple[int, list[str]]:
    """Return (year, [bnYYMMDDxxx]) triggers for that date."""
    year = utc_dt.year
    all_trigs = list_year_triggers(year)
    prefix = f"bn{utc_dt.strftime('%y%m%d')}"
    matches = [t for t in all_trigs if t.startswith(prefix)]
    if matches:
        print(f"[FOUND] {len(matches)} trigger(s) on {utc_dt.date()}: {', '.join(matches)}")
    else:
        print(f"[MISS] No triggers found on {utc_dt.date()}")
    return year, matches

def list_fits_recursive(base_url: str) -> list[str]:
    """Recursively collect all .fit/.fits URLs under a base directory."""
    resp = http_get(base_url)
    if not resp:
        return []
    html = resp.read().decode(errors="ignore")

    urls = [base_url + f for f in re.findall(r'href="([^"]+\.(?:fit|fits))"', html)]
    subdirs = [d for d in re.findall(r'href="([^"/]+/)"', html) if not d.startswith("../")]

    for sub in subdirs:
        urls += list_fits_recursive(base_url + sub)
    # remove duplicates while preserving order
    seen = {}
    for u in urls:
        if u not in seen:
            seen[u] = True
    return list(seen.keys())

# ---------------------------------------------------------------
# FITS UTILITIES
# ---------------------------------------------------------------

def download_temp_fits(url: str) -> str | None:
    """Download a small FITS file to a temp location."""
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fit")
        tmp.close()
        urllib.request.urlretrieve(url, tmp.name)
        return tmp.name
    except Exception as e:
        log(f"Failed to download {url}: {e}")
        return None

def extract_trigger_met_from_fits(path: str) -> float | None:
    """
    Extract trigger MET from FITS file.
    Prefer TRIGTIME keyword, else use TIMEZERO/TSTART + min(TIME).
    """
    try:
        with fits.open(path) as hdul:
            # look for TRIGTIME
            for hdu in hdul:
                if "TRIGTIME" in hdu.header:
                    return float(hdu.header["TRIGTIME"])
            # fallback via TIME column
            for hdu in hdul:
                if hasattr(hdu, "columns") and "TIME" in hdu.columns.names:
                    data = hdu.data
                    if data is None or len(data) == 0:
                        continue
                    times = data["TIME"]
                    if len(times) == 0:
                        continue
                    t0 = hdu.header.get("TIMEZERO", 0.0)
                    tstart = hdu.header.get("TSTART", 0.0)
                    return float(t0 + tstart + float(times.min()))
    except Exception as e:
        log(f"Error reading {path}: {e}")
    return None

# ---------------------------------------------------------------
# TRIGGER TIME ESTIMATION (with subdirectories)
# ---------------------------------------------------------------

def estimate_trigger_utc_for_bn(year: int, trig_id: str) -> datetime | None:
    """
    Estimate trigger UTC for given bn... by checking current/, quicklook/, previous/.
    """
    base_root = f"{BASE_URL}{year}/{trig_id}/"
    subdirs = ["current/", "quicklook/", "previous/", ""]

    for sub in subdirs:
        base = base_root + sub
        resp = http_get(base)
        if not resp:
            continue
        html = resp.read().decode(errors="ignore")

        candidates = re.findall(r'href="([^"]+trigdat[^"]*\.(?:fit|fits))"', html)
        if not candidates:
            candidates = re.findall(r'href="([^"]+tte[^"]*\.(?:fit|fits))"', html)
        if not candidates:
            continue

        fits_url = base + candidates[0]
        log(f"Checking {fits_url}")
        tmp = download_temp_fits(fits_url)
        if not tmp:
            continue
        met = extract_trigger_met_from_fits(tmp)
        try:
            os.unlink(tmp)
        except Exception:
            pass
        if met is not None:
            return met_to_utc(met)
    return None

# ---------------------------------------------------------------
# SELECT BEST TRIGGER
# ---------------------------------------------------------------

def select_best_trigger(utc_event: datetime, year: int, trig_ids: list[str]) -> tuple[str, datetime] | None:
    """Pick trigger whose trigger time (UTC) is closest to the GW event UTC."""
    best_id, best_utc, best_diff = None, None, float("inf")

    for trig in trig_ids:
        trig_utc = estimate_trigger_utc_for_bn(year, trig)
        if trig_utc is None:
            log(f"No usable trigger time for {trig}")
            continue
        diff = abs((trig_utc - utc_event).total_seconds())
        log(f"{trig}: TRIG_UTC={trig_utc.isoformat()} Δt={diff:.2f}s")
        if diff < best_diff:
            best_id, best_utc, best_diff = trig, trig_utc, diff

    if best_id is None:
        print("[MISS] No valid trigger times found.")
        return None
    print(f"[MATCH] {best_id} → TRIGTIME={best_utc.isoformat()} Δt≈{best_diff:.2f}s")
    return best_id, best_utc

# ---------------------------------------------------------------
# DOWNLOAD FILES (WITH SUBDIRS)
# ---------------------------------------------------------------

def _download_with_monitor(url: str, dest: str) -> None:
    """Stream download while monitoring throughput to detect stalls."""
    bytes_downloaded = 0
    start = time.time()

    with urllib.request.urlopen(url, timeout=30) as response, open(dest, "wb") as outfile:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            outfile.write(chunk)
            bytes_downloaded += len(chunk)
            now = time.time()

            elapsed = now - start
            if elapsed >= STALL_WINDOW:
                instant_rate = bytes_downloaded / max(elapsed, 1e-6)
                avg_rate = DOWNLOAD_STATS["avg_rate"]
                threshold = max(
                    (avg_rate / STALL_RATIO) if avg_rate else 0.0,
                    MIN_RATE_BPS,
                )
                if instant_rate < threshold:
                    raise SlowDownloadError(
                        f"rate {instant_rate:.1f} B/s below threshold {threshold:.1f} B/s"
                    )

    if bytes_downloaded == 0:
        raise SlowDownloadError("empty file downloaded")

    elapsed_total = max(time.time() - start, 1e-6)
    rate = bytes_downloaded / elapsed_total
    avg = DOWNLOAD_STATS["avg_rate"]
    DOWNLOAD_STATS["avg_rate"] = rate if avg is None else (0.8 * avg + 0.2 * rate)


def download_file_with_retry(url: str, dest: str) -> bool:
    """
    Download a file with retry logic for bad internet connections.
    - For connection errors (reset, timeout, etc.): wait 60s and retry indefinitely
    - For HTTP errors (404, 503, etc.): give up immediately
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            # Remove partial file if it exists
            if os.path.exists(dest):
                os.unlink(dest)
            
            _download_with_monitor(url, dest)
            
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                if attempt > 1:
                    print(f"[SUCCESS] Downloaded after {attempt} attempts")
                return True
            else:
                print(f"[RETRY] Attempt {attempt}: File incomplete or empty, waiting {STALL_SLEEP}s...")
                time.sleep(STALL_SLEEP)
                
        except HTTPError as e:
            # HTTP errors like 404, 503 - give up immediately
            print(f"[ERROR] HTTP {e.code}: {e.reason} - skipping file")
            if os.path.exists(dest):
                try:
                    os.unlink(dest)
                except Exception:
                    pass
            return False
            
        except URLError as e:
            # Connection errors - retry with wait
            error_msg = str(e.reason)
            # Check for common network issues
            if "nodename nor servname provided" in error_msg or "Name or service not known" in error_msg:
                print(f"[RETRY] Attempt {attempt}: Remote end not found (DNS issue), waiting {STALL_SLEEP}s...")
            elif "Connection reset" in error_msg:
                print(f"[RETRY] Attempt {attempt}: Connection reset by peer, waiting {STALL_SLEEP}s...")
            elif "timed out" in error_msg.lower():
                print(f"[RETRY] Attempt {attempt}: Connection timed out, waiting {STALL_SLEEP}s...")
            else:
                print(f"[RETRY] Attempt {attempt}: Connection error ({e.reason}), waiting {STALL_SLEEP}s...")
            
            if os.path.exists(dest):
                try:
                    os.unlink(dest)
                except Exception:
                    pass
            time.sleep(STALL_SLEEP)
        
        except SlowDownloadError as e:
            print(f"[RETRY] Attempt {attempt}: {e}, waiting {STALL_SLEEP}s before retrying...")
            if os.path.exists(dest):
                try:
                    os.unlink(dest)
                except Exception:
                    pass
            time.sleep(STALL_SLEEP)
            
        except Exception as e:
            # Other errors - retry with wait
            print(f"[RETRY] Attempt {attempt}: {e}, waiting {STALL_SLEEP}s...")
            if os.path.exists(dest):
                try:
                    os.unlink(dest)
                except Exception:
                    pass
            time.sleep(STALL_SLEEP)

# ---------------------------------------------------------------
# UPSTREAM DATA DOWNLOADERS
# ---------------------------------------------------------------

def download_gwosc_event_versions_csv(dest_path: Path, force: bool = False) -> Path:
    """Fetch the GWOSC event catalog CSV directly from the public API."""
    dest_path = dest_path.resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        print(f"[SKIP] GWOSC event catalog already present at {dest_path}")
        return dest_path

    print(f"[DOWNLOAD] Fetching GWOSC event catalog → {dest_path}")
    success = download_file_with_retry(GWOSC_EVENT_VERSIONS_URL, str(dest_path))
    if not success:
        raise RuntimeError("Failed to download GWOSC event catalog.")
    return dest_path


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract tar contents while preventing path traversal."""
    dest = dest.resolve()
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path detected in tar archive: {member.name}")
    tar.extractall(path=dest)


def _safe_extract_zip(zip_file: zipfile.ZipFile, dest: Path) -> None:
    """Extract zip contents while preventing path traversal."""
    dest = dest.resolve()
    for member in zip_file.infolist():
        member_path = (dest / member.filename).resolve()
        if not str(member_path).startswith(str(dest)):
            raise RuntimeError(f"Unsafe path detected in zip archive: {member.filename}")
    zip_file.extractall(path=dest)


def download_and_extract_gcn_archive(dest_dir: Path, force: bool = False) -> Path:
    """Download the NASA GCN archive tarball and extract the JSON payloads."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    tar_path = dest_dir / GCN_ARCHIVE_TAR_NAME
    needs_download = force or not tar_path.exists()

    if needs_download:
        print(f"[DOWNLOAD] Fetching GCN circular archive → {tar_path}")
        success = download_file_with_retry(GCN_ARCHIVE_URL, str(tar_path))
        if not success:
            raise RuntimeError("Failed to download GCN circular archive.")
    else:
        print(f"[SKIP] GCN archive already downloaded at {tar_path}")

    marker = dest_dir / ".archive_extracted"
    needs_extract = force or not marker.exists()
    if marker.exists() and tar_path.exists() and not force:
        needs_extract = tar_path.stat().st_mtime > marker.stat().st_mtime

    if not tar_path.exists():
        raise FileNotFoundError(f"Missing archive at {tar_path}")

    if needs_extract:
        print(f"[EXTRACT] Unpacking {tar_path.name} into {dest_dir}")
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_extract_tar(tar, dest_dir)
        marker.write_text(f"Extracted {datetime.now(timezone.utc).isoformat()}\n")
    else:
        print(f"[SKIP] Existing GCN circular files assumed up-to-date in {dest_dir}")

    return dest_dir


def _download_tar_archive(url: str, dest_dir: Path, label: str, force: bool = False) -> Path:
    """Download and extract a tar.gz archive safely into dest_dir."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_name = url.split("/")[-1].split("?")[0] or "archive.tar.gz"
    tar_path = dest_dir / tar_name
    needs_download = force or not tar_path.exists()

    if needs_download:
        print(f"[DOWNLOAD] Fetching {label} → {tar_path}")
        success = download_file_with_retry(url, str(tar_path))
        if not success:
            raise RuntimeError(f"Failed to download {label}.")
    else:
        print(f"[SKIP] {label} already downloaded at {tar_path}")

    marker = dest_dir / ".archive_extracted"
    needs_extract = force or not marker.exists()
    if marker.exists() and tar_path.exists() and not force:
        needs_extract = tar_path.stat().st_mtime > marker.stat().st_mtime

    if not tar_path.exists():
        raise FileNotFoundError(f"Missing archive at {tar_path}")

    if needs_extract:
        print(f"[EXTRACT] Unpacking {tar_name} into {dest_dir}")
        with tarfile.open(tar_path, "r:*") as tar:
            _safe_extract_tar(tar, dest_dir)
        marker.write_text(f"Extracted {datetime.now(timezone.utc).isoformat()}\n")
    else:
        print(f"[SKIP] Existing files assumed up-to-date in {dest_dir}")

    return dest_dir


def _download_zip_archive(url: str, dest_dir: Path, label: str, force: bool = False) -> Path:
    """Download and extract a zip archive safely into dest_dir."""
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_name = url.rsplit("/", 1)[-1]
    zip_path = dest_dir / zip_name
    needs_download = force or not zip_path.exists()

    if needs_download:
        print(f"[DOWNLOAD] Fetching {label} → {zip_path}")
        success = download_file_with_retry(url, str(zip_path))
        if not success:
            raise RuntimeError(f"Failed to download {label}.")
    else:
        print(f"[SKIP] {label} already downloaded at {zip_path}")

    marker = dest_dir / ".archive_extracted"
    needs_extract = force or not marker.exists()
    if marker.exists() and zip_path.exists() and not force:
        needs_extract = zip_path.stat().st_mtime > marker.stat().st_mtime

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing archive at {zip_path}")

    if needs_extract:
        print(f"[EXTRACT] Unpacking {zip_name} into {dest_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, dest_dir)
        marker.write_text(f"Extracted {datetime.now(timezone.utc).isoformat()}\n")
    else:
        print(f"[SKIP] Existing files assumed up-to-date in {dest_dir}")

    return dest_dir


def download_icecube_txs_dataset(dest_dir: Path, force: bool = False) -> Path:
    """Download the IceCube TXS 0506+056 per-sample dataset."""
    return _download_zip_archive(
        ICECUBE_TXS_URL,
        dest_dir,
        "IceCube TXS 0506+056 dataset",
        force=force,
    )


def download_icecube_alert_catalog(dest_dir: Path, force: bool = False) -> Path:
    """Download the IceCube alert catalog ZIP."""
    return _download_zip_archive(
        ICECUBE_ALERTS_URL,
        dest_dir,
        "IceCube alert catalog",
        force=force,
    )


def download_t2k_superk_dataset(dest_dir: Path, force: bool = False) -> Path:
    """Download the T2K + Super-K joint fit data release."""
    return _download_zip_archive(
        T2K_SUPERK_URL,
        dest_dir,
        "T2K/Super-K joint oscillation release",
        force=force,
    )


def download_superk_neutron_dataset(dest_dir: Path, force: bool = False) -> Path:
    """Download the Super-K neutron production data release (tarball)."""
    return _download_tar_archive(
        SUPERK_NEUTRON_URL,
        dest_dir,
        "Super-K neutron production dataset",
        force=force,
    )


def download_upstream_metadata(
    csv_path: Path,
    gcn_dir: Path,
    icecube_root: Path,
    t2k_dir: Path,
    superk_dir: Path,
    force: bool = False,
) -> Path:
    """Ensure external datasets (GCN, GWOSC, IceCube, T2K/Super-K) are cached locally."""
    download_and_extract_gcn_archive(gcn_dir, force=force)
    download_icecube_txs_dataset(icecube_root / "txs_0506+056", force=force)
    download_icecube_alert_catalog(icecube_root / "alerts", force=force)
    download_t2k_superk_dataset(t2k_dir, force=force)
    download_superk_neutron_dataset(superk_dir, force=force)
    return download_gwosc_event_versions_csv(csv_path, force=force)

def download_all_files_for_trigger(year: int, trig_id: str) -> str:
    """Download all .fit/.fits files from all subdirs of the trigger folder."""
    base = f"{BASE_URL}{year}/{trig_id}/"
    fits_urls = list_fits_recursive(base)
    if not fits_urls:
        print(f"[EMPTY] No FITS files under {trig_id}")
        return "empty"

    local_dir = os.path.join("data", "data", "fermi", trig_id)
    os.makedirs(local_dir, exist_ok=True)

    downloaded = 0
    failed = 0
    for url in fits_urls:
        fname = os.path.basename(url)
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest):
            log(f"Skip existing {fname}")
            continue
        
        print(f"[DL] {trig_id} → {fname}")
        if download_file_with_retry(url, dest):
            downloaded += 1
        else:
            print(f"[ERR] {fname}: Failed after all retry attempts")
            failed += 1
    
    if failed > 0:
        print(f"[WARN] {trig_id}: {downloaded} files downloaded, {failed} failed.")
    else:
        print(f"[OK] {trig_id}: {downloaded} files downloaded.")
    return "downloaded" if downloaded else "no_files"

# ---------------------------------------------------------------
# RESUME LOGIC
# ---------------------------------------------------------------

def find_resume_point(rows: list[dict], name_col: str, gps_col: str) -> int:
    """
    Find the index to resume from by checking existing fermi directories.
    Returns the index of the last downloaded event (to be re-downloaded).
    Returns 0 if no downloads found.
    """
    fermi_dir = os.path.join("data", "data", "fermi")
    if not os.path.exists(fermi_dir):
        print("[RESUME] No fermi directory found, starting from beginning.")
        return 0
    
    existing_dirs = set(os.listdir(fermi_dir))
    if not existing_dirs:
        print("[RESUME] No existing downloads found, starting from beginning.")
        return 0
    
    print(f"[RESUME] Found {len(existing_dirs)} existing fermi directories.")
    
    # Find the last event in CSV order that has a corresponding directory
    last_downloaded_idx = -1
    last_downloaded_name = None
    last_trigger_id = None
    
    for idx, row in enumerate(rows):
        name = (row.get(name_col) or "").strip()
        gps = (row.get(gps_col) or "").strip()
        if not name or not gps:
            continue
        
        utc_event = gw_gps_to_utc(gps)
        if utc_event.year < 2008:
            continue
        
        # Check if any directory matches this event's date pattern
        date_prefix = f"bn{utc_event.strftime('%y%m%d')}"
        matching_dirs = [d for d in existing_dirs if d.startswith(date_prefix)]
        
        if matching_dirs:
            last_downloaded_idx = idx
            last_downloaded_name = name
            last_trigger_id = matching_dirs[0]  # Take first match
    
    if last_downloaded_idx == -1:
        print("[RESUME] No matching downloads found in CSV order, starting from beginning.")
        return 0
    
    print(f"[RESUME] Last downloaded event: {last_downloaded_name} (index {last_downloaded_idx})")
    print(f"[RESUME] Corresponding trigger: {last_trigger_id}")
    
    # Delete the last downloaded directory (assume it's incomplete)
    last_dir_path = os.path.join(fermi_dir, last_trigger_id)
    if os.path.exists(last_dir_path):
        print(f"[RESUME] Deleting potentially incomplete directory: {last_trigger_id}")
        try:
            shutil.rmtree(last_dir_path)
            print(f"[RESUME] Deleted {last_trigger_id}")
        except Exception as e:
            print(f"[RESUME] Warning: Could not delete {last_trigger_id}: {e}")
    
    print(f"[RESUME] Resuming from event index {last_downloaded_idx}")
    return last_downloaded_idx

# ---------------------------------------------------------------
# CSV PROCESSOR
# ---------------------------------------------------------------

def process_csv(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(f, dialect=dialect)
        reader.fieldnames = [h.strip().replace("\ufeff", "") for h in reader.fieldnames]
        rows = list(reader)

    total_rows = len(rows)
    print(f"Loaded {total_rows} events from {csv_path}")

    header_map = {h.lower().replace(" ", ""): h for h in reader.fieldnames}
    name_col = header_map.get("name")
    gps_col = header_map.get("gps")

    if not name_col or not gps_col:
        print("[ERROR] CSV must have 'Name' and 'GPS' columns.")
        sys.exit(1)

    # Find resume point
    start_idx = find_resume_point(rows, name_col, gps_col)
    
    if start_idx > 0:
        print(f"[RESUME] Skipping first {start_idx} events\n")
    
    processed_count = 0

    for idx, row in enumerate(rows):
        if idx < start_idx:
            continue

        progress = f"[{idx + 1}/{total_rows}]"
        name = (row.get(name_col) or "").strip()
        gps = (row.get(gps_col) or "").strip()
        if not name or not gps:
            print(f"{progress} [SKIP] Missing name/GPS")
            continue
        utc_event = gw_gps_to_utc(gps)
        print(f"\n{progress} {name}: GPS={gps} → UTC={utc_event.isoformat()}")
        processed_count += 1
        if utc_event.year < 2008:
            print(f"{progress} [SKIP] Pre-Fermi epoch.")
            continue
        year, trig_ids = list_date_triggers(utc_event)
        if not trig_ids:
            SUMMARY[name] = {"utc_event": utc_event.isoformat(), "result": "no_triggers"}
            continue
        match = select_best_trigger(utc_event, year, trig_ids)
        if not match:
            SUMMARY[name] = {"utc_event": utc_event.isoformat(), "result": "no_match"}
            continue
        trig_id, trig_utc = match
        result = download_all_files_for_trigger(year, trig_id)
        SUMMARY[name] = {
            "trigger_id": trig_id,
            "trigtime_utc": trig_utc.isoformat(),
            "utc_event": utc_event.isoformat(),
            "result": result,
        }

    os.makedirs("logs", exist_ok=True)
    with open("logs/downloads_summary.json", "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, indent=2)
    print("\n[INFO] Summary written to logs/downloads_summary.json")
    print(f"[INFO] Processed {processed_count} event rows out of {total_rows}.")

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GCN archive, GWOSC catalog, and matching Fermi GBM data."
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_EVENT_VERSIONS_PATH),
        help="Path where the GWOSC event versions CSV should be saved (default: %(default)s).",
    )
    parser.add_argument(
        "--gcn-dir",
        default=str(GCN_CIRCULARS_DIR),
        help="Directory where the GCN archive tarball should be extracted (default: %(default)s).",
    )
    parser.add_argument(
        "--icecube-dir",
        default=str(ICECUBE_ROOT_DIR),
        help="Root directory for IceCube neutrino datasets (default: %(default)s).",
    )
    parser.add_argument(
        "--t2k-dir",
        default=str(T2K_ROOT_DIR),
        help="Directory for the T2K/Super-K joint oscillation release (default: %(default)s).",
    )
    parser.add_argument(
        "--superk-neutron-dir",
        default=str(SUPERK_NEUTRON_DIR),
        help="Directory for the Super-K neutron production release (default: %(default)s).",
    )
    parser.add_argument(
        "--force-downloads",
        action="store_true",
        help="Redownload and re-extract upstream data even if files already exist.",
    )
    parser.add_argument(
        "--skip-fermi",
        action="store_true",
        help="Only download metadata inputs (GCN + GWOSC) and skip the Fermi data pass.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global DEBUG
    DEBUG = args.debug

    csv_path = download_upstream_metadata(
        Path(args.csv_path),
        Path(args.gcn_dir),
        Path(args.icecube_dir),
        Path(args.t2k_dir),
        Path(args.superk_neutron_dir),
        force=args.force_downloads,
    )

    if args.skip_fermi:
        print("[INFO] --skip-fermi specified; skipping Fermi GBM downloads.")
        return

    process_csv(str(csv_path))
    print("Done.")


if __name__ == "__main__":
    main()
