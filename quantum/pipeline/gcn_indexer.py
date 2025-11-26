#!/usr/bin/env python3
"""
GCN JSON Indexer
----------------

Scans data/gcn/ for NASA GCN circular JSONs and builds an index file
that maps eventId → local path and metadata.

Usage:
    python gcn_indexer.py [--debug]

Output:
    data/gcn_index.json
    logs/gcn_index_report.txt
"""

import os, json, sys, datetime

DEBUG = "--debug" in sys.argv
GCN_DIR = "data/gcn"
OUTPUT_INDEX = "data/gcn_index.json"
REPORT_FILE = "logs/gcn_index_report.txt"

def log(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

def ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to load {path}: {e}")
        return None

def ts_to_iso(ts_ms):
    try:
        return datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.timezone.utc).isoformat()
    except Exception:
        return None

def scan_gcn_dir():
    log(f"Scanning {GCN_DIR}...")
    entries = []
    for root, _, files in os.walk(GCN_DIR):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            data = safe_load_json(fpath)
            if not data or "eventId" not in data:
                continue

            eid = data.get("eventId", os.path.splitext(fname)[0])
            subject = data.get("subject", "")
            bibcode = data.get("bibcode", "")
            created_on = data.get("createdOn")
            created_iso = ts_to_iso(created_on) if isinstance(created_on, (int, float)) else None

            entries.append({
                "eventId": eid,
                "path": fpath,
                "subject": subject,
                "bibcode": bibcode,
                "created": created_iso
            })
    return entries

def build_index(entries):
    index = {}
    for e in entries:
        key = e["eventId"].strip()
        if not key:
            continue
        index[key] = e
    return index

def write_index(index):
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"[OK] Indexed {len(index)} GCN circulars → {OUTPUT_INDEX}")

def write_report(entries):
    total = len(entries)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(f"GCN Index Report ({datetime.datetime.utcnow().isoformat()} UTC)\n")
        f.write(f"Total files scanned: {total}\n\n")
        years = {}
        for e in entries:
            if e["created"]:
                y = e["created"][:4]
                years[y] = years.get(y, 0) + 1
        for y, n in sorted(years.items()):
            f.write(f"{y}: {n} entries\n")
    print(f"[INFO] Report written to {REPORT_FILE}")

def main():
    ensure_dirs()
    entries = scan_gcn_dir()
    if not entries:
        print("[WARN] No GCN JSONs found.")
        sys.exit(1)
    index = build_index(entries)
    write_index(index)
    write_report(entries)
    print("Done.")

if __name__ == "__main__":
    main()
