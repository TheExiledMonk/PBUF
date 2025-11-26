#!/usr/bin/env python3
"""Automate ingestion + normalization for multi-messenger datasets.

This script inspects every raw dataset we download (GWOSC, Fermi/GCN,
IceCube releases, Super-K tables, etc.), infers the layout, normalizes
columns to a shared event schema, and writes a consolidated CSV that the
existing importer can consume without extra hand editing.

Output columns:
    event_id, messenger_type, utc_time, ra_deg, dec_deg,
    energy_or_band, sigma_time_s, sigma_angle_deg, source

Times are emitted as ISO-8601 strings in UTC. Energies are converted to
GeV when a numeric measurement exists; otherwise the original band/range
description is stored verbatim. Angular uncertainties are in degrees.

Usage example:
    python -m quantum.pipeline.multimessenger_ingest \
        --output outputs/multimessenger_events.csv \
        --summary outputs/multimessenger_ingest.log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .fermi_loader import load_fermi_directory


# ---------------------------------------------------------------------------
# Dataclasses & shared helpers
# ---------------------------------------------------------------------------


UTC = timezone.utc
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
FERMI_EPOCH = datetime(2001, 1, 1, tzinfo=UTC)
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)
EV_TO_GEV = 1e-9


@dataclass
class NormalizedRecord:
    event_id: str
    messenger_type: str
    utc_time: Optional[datetime]
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    energy_or_band: Optional[str]
    sigma_time_s: Optional[float]
    sigma_angle_deg: Optional[float]
    source: str

    def to_row(self) -> Dict[str, str]:
        def _fmt(value: Optional[float]) -> str:
            return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.6g}"

        return {
            "event_id": self.event_id,
            "messenger_type": self.messenger_type,
            "utc_time": self.utc_time.isoformat() if self.utc_time else "",
            "ra_deg": _fmt(self.ra_deg),
            "dec_deg": _fmt(self.dec_deg),
            "energy_or_band": self.energy_or_band or "",
            "sigma_time_s": _fmt(self.sigma_time_s),
            "sigma_angle_deg": _fmt(self.sigma_angle_deg),
            "source": self.source,
        }


@dataclass
class LayoutReport:
    dataset: str
    path: str
    columns: Sequence[str]
    samples: Sequence[Dict[str, Any]]
    unused_columns: Sequence[str]
    notes: Optional[str] = None


def gps_to_utc(gps_seconds: float) -> datetime:
    return GPS_EPOCH + timedelta(seconds=float(gps_seconds))


def met_to_utc(met_seconds: float) -> datetime:
    return FERMI_EPOCH + timedelta(seconds=float(met_seconds))


def mjd_to_utc(mjd_days: float) -> datetime:
    return MJD_EPOCH + timedelta(days=float(mjd_days))


def energy_ev_to_gev(value_eV: float | None) -> Optional[float]:
    if value_eV in (None, "", float("nan")):
        return None
    try:
        return float(value_eV) * EV_TO_GEV
    except (TypeError, ValueError):
        return None


def parse_error_token(token: str) -> Optional[float]:
    token = token.strip()
    if not token or token in {"(-)", "-", "()"}:
        return None
    token = token.strip("()")
    parts = [p.strip().replace("+", "") for p in token.replace("/", ",").split(",") if p.strip()]
    values: List[float] = []
    for part in parts:
        part_norm = part.replace("−", "-")
        try:
            values.append(abs(float(part_norm)))
        except ValueError:
            continue
    if not values:
        return None
    return statistics.mean(values)


def hms_to_deg(hours: float, minutes: float, seconds: float) -> float:
    sign = 1.0
    if hours < 0:
        sign = -1.0
        hours = abs(hours)
    return sign * (hours + minutes / 60.0 + seconds / 3600.0) * 15.0


def dms_to_deg(degrees: float, arcmin: float, arcsec: float) -> float:
    sign = -1.0 if degrees < 0 else 1.0
    degrees = abs(degrees)
    return sign * (degrees + arcmin / 60.0 + arcsec / 3600.0)


def extract_coords_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    if not text:
        return None, None
    pair = _PAIR_COORD_PATTERN.search(text)
    if pair:
        ra = hms_to_deg(float(pair.group("rah")), float(pair.group("ram")), float(pair.group("ras")))
        dec = dms_to_deg(float(pair.group("decd")), float(pair.group("decm")), float(pair.group("decs")))
        return ra, dec
    ra_match = _RA_DECIMAL_PATTERN.search(text)
    dec_match = _DEC_DECIMAL_PATTERN.search(text)
    ra = float(ra_match.group(1)) if ra_match else None
    dec = float(dec_match.group(1)) if dec_match else None
    return ra, dec


_PAIR_COORD_PATTERN = re.compile(
    r"(?P<rah>\d{1,2})h\s*(?P<ram>\d{1,2})m\s*(?P<ras>\d{1,2}(?:\.\d+)?)s\s*,\s*"
    r"(?P<decd>[+-]?\d{1,2})d\s*(?P<decm>\d{1,2})m\s*(?P<decs>\d{1,2}(?:\.\d+)?)s",
    re.IGNORECASE,
)

_RA_DECIMAL_PATTERN = re.compile(r"RA[^0-9+-]*([+-]?\d{1,3}(?:\.\d+)?)", re.IGNORECASE)
_DEC_DECIMAL_PATTERN = re.compile(r"DEC[^0-9+-]*([+-]?\d{1,2}(?:\.\d+)?)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dataset-specific ingestion
# ---------------------------------------------------------------------------


def ingest_gwosc(csv_path: Path) -> Tuple[List[NormalizedRecord], LayoutReport]:
    records: List[NormalizedRecord] = []
    sample_rows: List[Dict[str, Any]] = []
    used_columns = {"name", "shortName", "gps"}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        fieldnames = reader.fieldnames or []

        for row_idx, row in enumerate(reader):
            if row_idx < 3:
                sample_rows.append({key: row.get(key) for key in fieldnames})

            event_id = (row.get("name") or row.get("shortName") or "").strip()
            gps_raw = row.get("gps")
            if not event_id or gps_raw in (None, ""):
                continue
            try:
                gps_value = float(gps_raw)
            except ValueError:
                continue
            records.append(
                NormalizedRecord(
                    event_id=event_id,
                    messenger_type="gw",
                    utc_time=gps_to_utc(gps_value),
                    ra_deg=None,
                    dec_deg=None,
                    energy_or_band=None,
                    sigma_time_s=1e-3,
                    sigma_angle_deg=None,
                    source=f"GWOSC:{csv_path.name}:{row_idx+2}",
                )
            )

    unused = [col for col in fieldnames if col not in used_columns]
    report = LayoutReport(
        dataset="GWOSC Event Versions",
        path=str(csv_path),
        columns=fieldnames,
        samples=sample_rows,
        unused_columns=unused,
    )
    return records, report


def ingest_fermi(fermi_dir: Path) -> Tuple[List[NormalizedRecord], LayoutReport]:
    triggers = load_fermi_directory(fermi_dir)
    sample_rows = []
    records: List[NormalizedRecord] = []

    for idx, trig in enumerate(triggers):
        if idx < 3:
            sample_rows.append({k: trig.get(k) for k in sorted(trig.keys())})
        event_id = trig.get("trigger_name")
        if not event_id:
            continue
        energy_eV = trig.get("E_eV")
        energy_range = None
        if not energy_eV:
            bounds = trig.get("channel_energy_keV")
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                energy_range = f"{bounds[0]}-{bounds[1]} keV"

        records.append(
            NormalizedRecord(
                event_id=str(event_id),
                messenger_type="gamma",
                utc_time=met_to_utc(trig.get("trig_met", 0.0)),
                ra_deg=None,
                dec_deg=None,
                energy_or_band=(
                    f"{energy_ev_to_gev(energy_eV):.6g} GeV"
                    if energy_eV
                    else (energy_range or None)
                ),
                sigma_time_s=float(trig.get("sigma_t", 0.0) or 0.0),
                sigma_angle_deg=None,
                source=f"Fermi:{trig.get('source', {}).get('directory', fermi_dir)}",
            )
        )

    columns = sorted(set().union(*(sample.keys() for sample in sample_rows)) if sample_rows else [])
    unused_cols = [col for col in columns if col not in {"trigger_name", "trig_met", "sigma_t", "E_eV"}]
    report = LayoutReport(
        dataset="Fermi/GBM Triggers",
        path=str(fermi_dir),
        columns=columns,
        samples=sample_rows,
        unused_columns=unused_cols,
    )
    return records, report


def ingest_gcn(gcn_dir: Path, max_files: Optional[int]) -> Tuple[List[NormalizedRecord], LayoutReport]:
    json_files = sorted(gcn_dir.glob("**/*.json"))
    records: List[NormalizedRecord] = []
    sample_rows: List[Dict[str, Any]] = []
    used_keys = {"eventId", "circularId", "createdOn", "subject"}
    unused_keys: set[str] = set()

    for idx, path in enumerate(json_files):
        if max_files is not None and idx >= max_files:
            break
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if idx < 3:
            subset = {k: data.get(k) for k in list(data.keys())[:8]}
            sample_rows.append(subset)

        event_id = str(data.get("eventId") or data.get("circularId") or path.stem)
        created = data.get("createdOn")
        if created is None:
            continue
        try:
            utc_time = datetime.fromtimestamp(float(created) / 1000.0, tz=UTC)
        except (ValueError, TypeError):
            utc_time = None

        ra, dec = extract_coords_from_text(data.get("body", ""))

        records.append(
            NormalizedRecord(
                event_id=event_id,
                messenger_type="em",
                utc_time=utc_time,
                ra_deg=ra,
                dec_deg=dec,
                energy_or_band=None,
                sigma_time_s=None,
                sigma_angle_deg=None,
                source=f"GCN:{path.name}",
            )
        )

        unused_keys.update(set(data.keys()) - used_keys)

    report = LayoutReport(
        dataset="GCN Circulars",
        path=str(gcn_dir),
        columns=sorted(used_keys),
        samples=sample_rows,
        unused_columns=sorted(unused_keys),
        notes=(
            None
            if max_files is None
            else f"processed first {min(max_files, len(json_files))} / {len(json_files)} circulars"
        ),
    )
    return records, report


def ingest_icecube_alerts(catalog_path: Path) -> Tuple[List[NormalizedRecord], LayoutReport]:
    if not catalog_path.exists():
        return [], LayoutReport(
            dataset="IceCube Alerts",
            path=str(catalog_path),
            columns=[],
            samples=[],
            unused_columns=[],
            notes="catalog file missing",
        )

    records: List[NormalizedRecord] = []
    sample_rows: List[Dict[str, Any]] = []
    current_label: Optional[str] = None
    section: Optional[str] = None

    with catalog_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("##"):
                section = line.strip("# ")
                continue
            if line.startswith("#"):
                current_label = line.strip("# ").strip()
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                mjd = float(parts[0])
                ra = float(parts[1])
            except ValueError:
                continue
            ra_err = parse_error_token(parts[2])
            try:
                dec = float(parts[3])
            except ValueError:
                continue
            dec_err = parse_error_token(parts[4]) if len(parts) > 4 else None
            sigma_angle = statistics.mean(
                [value for value in (ra_err, dec_err) if value is not None]
            ) if any(value is not None for value in (ra_err, dec_err)) else None

            label = current_label or section or "IceCube alert"
            event_id = f"IC_{label.replace(' ', '_')}_{mjd:.5f}"
            current_label = None

            record = NormalizedRecord(
                event_id=event_id,
                messenger_type="neutrino",
                utc_time=mjd_to_utc(mjd),
                ra_deg=ra,
                dec_deg=dec,
                energy_or_band=None,
                sigma_time_s=1.0,
                sigma_angle_deg=sigma_angle,
                source=f"IceCubeAlerts:{catalog_path.name}:{line_no}",
            )
            if len(sample_rows) < 3:
                sample_rows.append(record.to_row())
            records.append(record)

    report = LayoutReport(
        dataset="IceCube Alert Catalog",
        path=str(catalog_path),
        columns=["MJD", "RA", "RA_ERR", "DEC", "DEC_ERR"],
        samples=sample_rows,
        unused_columns=[],
    )
    return records, report


def ingest_icecube_txs(txs_root: Path) -> Tuple[List[NormalizedRecord], LayoutReport]:
    data_dir = next((p for p in txs_root.iterdir() if p.is_dir()), None)
    if data_dir is None:
        return [], LayoutReport(
            dataset="IceCube TXS0506+056",
            path=str(txs_root),
            columns=[],
            samples=[],
            unused_columns=[],
            notes="extracted directory missing",
        )

    records: List[NormalizedRecord] = []
    sample_rows: List[Dict[str, Any]] = []
    event_files = sorted(data_dir.glob("events_*.txt"))
    for path in event_files:
        sample_name = path.stem.split("_")[-1].upper()
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip()
            columns = header.split()
            for row_idx, line in enumerate(handle):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                mjd, ra, dec, unc, logE = map(float, parts[:5])
                record = NormalizedRecord(
                    event_id=f"TXS0506+056_{sample_name}_{int(round(mjd * 1e5))}",
                    messenger_type="neutrino",
                    utc_time=mjd_to_utc(mjd),
                    ra_deg=ra,
                    dec_deg=dec,
                    energy_or_band=f"{10 ** logE:.6g} GeV",
                    sigma_time_s=1.0,
                    sigma_angle_deg=unc,
                    source=f"IceCubeTXS:{path.name}:{row_idx+2}",
                )
                if len(sample_rows) < 3:
                    sample_rows.append(record.to_row())
                records.append(record)

    report = LayoutReport(
        dataset="IceCube TXS0506+056 events",
        path=str(txs_root),
        columns=["MJD", "Ra_deg", "Dec_deg", "Unc_deg", "log10(Ereco)"],
        samples=sample_rows,
        unused_columns=[],
    )
    return records, report


def ingest_superk_neutron(csv_path: Path) -> Tuple[List[NormalizedRecord], LayoutReport]:
    if not csv_path.exists():
        return [], LayoutReport(
            dataset="Super-K Neutron",
            path=str(csv_path),
            columns=[],
            samples=[],
            unused_columns=[],
            notes="data_observations.csv missing",
        )

    records: List[NormalizedRecord] = []
    sample_rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if idx < 3:
                sample_rows.append(row)
            event_type = row.get("event_type")
            bin_id = row.get("x_bin_id")
            if not event_type or not bin_id:
                continue
            energy_mev = row.get("x_bin_center")
            try:
                energy_value = float(energy_mev) / 1000.0
                energy_repr = f"{energy_value:.6g} GeV"
            except (TypeError, ValueError):
                energy_repr = None

            record = NormalizedRecord(
                event_id=f"SK_NEUTRON_{event_type}_{bin_id}",
                messenger_type="neutron",
                utc_time=None,
                ra_deg=None,
                dec_deg=None,
                energy_or_band=energy_repr,
                sigma_time_s=None,
                sigma_angle_deg=None,
                source=f"SuperKNeutron:{csv_path.name}:{idx+2}",
            )
            records.append(record)

    unused_cols = [col for col in columns if col not in {"event_type", "x_bin_id", "x_bin_center"}]
    report = LayoutReport(
        dataset="Super-K neutron observations",
        path=str(csv_path),
        columns=columns,
        samples=sample_rows,
        unused_columns=unused_cols,
    )
    return records, report


def inspect_t2k_release(t2k_dir: Path) -> LayoutReport:
    root_files = sorted(p.name for p in t2k_dir.glob("*.root"))
    notes = "; ".join(root_files[:3])
    if len(root_files) > 3:
        notes += f" (+{len(root_files) - 3} more ROOT files)"
    return LayoutReport(
        dataset="T2K/Super-K Joint Fit",
        path=str(t2k_dir),
        columns=[],
        samples=[],
        unused_columns=[],
        notes=(notes or "ROOT files present – not auto-parsed. Use ROOT macro to export CSV."),
    )


# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------


def write_csv(records: Sequence[NormalizedRecord], output_path: Path) -> None:
    if not records:
        print("[WARN] No records to write; skipping CSV.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].to_row().keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    print(f"[OK] Wrote {len(records)} normalized records → {output_path}")


def print_reports(reports: Sequence[LayoutReport], summary_path: Optional[Path]) -> None:
    lines: List[str] = []
    for report in reports:
        lines.append(f"=== {report.dataset} ===")
        lines.append(f"Source: {report.path}")
        if report.columns:
            lines.append(f"Detected columns ({len(report.columns)}): {', '.join(report.columns)}")
        if report.unused_columns:
            lines.append(
                "Unused columns: " + ", ".join(report.unused_columns[:12]) + (
                    " ..." if len(report.unused_columns) > 12 else ""
                )
            )
        if report.samples:
            lines.append("Sample rows:")
            for sample in report.samples:
                lines.append(f"  {sample}")
        if report.notes:
            lines.append(f"Notes: {report.notes}")
        lines.append("")

    joined = "\n".join(lines)
    print(joined)
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(joined, encoding="utf-8")
        print(f"[INFO] Layout report written to {summary_path}")


def summarize_records(records: Sequence[NormalizedRecord]) -> None:
    from collections import Counter

    type_counts = Counter(record.messenger_type for record in records)
    source_counts = Counter(record.source.split(":", 1)[0] for record in records)
    print("Messenger counts:")
    for messenger, count in sorted(type_counts.items()):
        print(f"  {messenger:<10} {count}")
    print("Source counts:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src:<20} {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize multi-messenger datasets into a single CSV")
    parser.add_argument("--gwosc", default=str(Path("data") / "event-versions.csv"))
    parser.add_argument("--fermi", default=str(Path("data") / "data" / "fermi"))
    parser.add_argument("--gcn", default=str(Path("data") / "data" / "gcn"))
    parser.add_argument("--icecube-alerts", default=str(Path("data") / "data" / "icecube" / "alerts" / "catalog_of_alerts.txt"))
    parser.add_argument("--icecube-txs", default=str(Path("data") / "data" / "icecube" / "txs_0506+056"))
    parser.add_argument("--superk-neutron", default=str(Path("data") / "data" / "superk_neutron" / "data_release" / "data_observations.csv"))
    parser.add_argument("--t2k", default=str(Path("data") / "data" / "t2k_superk"))
    parser.add_argument("--max-gcn", type=int, default=None, help="Optional cap on number of GCN files to scan (for quick tests)")
    parser.add_argument("--output", default=str(Path("outputs") / "multimessenger_events.csv"))
    parser.add_argument("--summary", default=None, help="Optional path to write the layout log")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: List[LayoutReport] = []
    all_records: List[NormalizedRecord] = []

    gw_records, gw_report = ingest_gwosc(Path(args.gwosc))
    all_records.extend(gw_records)
    reports.append(gw_report)

    fermi_records, fermi_report = ingest_fermi(Path(args.fermi))
    all_records.extend(fermi_records)
    reports.append(fermi_report)

    gcn_records, gcn_report = ingest_gcn(Path(args.gcn), args.max_gcn)
    all_records.extend(gcn_records)
    reports.append(gcn_report)

    alerts_records, alerts_report = ingest_icecube_alerts(Path(args.icecube_alerts))
    all_records.extend(alerts_records)
    reports.append(alerts_report)

    txs_records, txs_report = ingest_icecube_txs(Path(args.icecube_txs))
    all_records.extend(txs_records)
    reports.append(txs_report)

    sk_records, sk_report = ingest_superk_neutron(Path(args.superk_neutron))
    all_records.extend(sk_records)
    reports.append(sk_report)

    reports.append(inspect_t2k_release(Path(args.t2k)))

    write_csv(all_records, Path(args.output))
    summarize_records(all_records)
    summary_path = Path(args.summary) if args.summary else None
    print_reports(reports, summary_path)


if __name__ == "__main__":
    main()
