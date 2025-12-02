"""Command-line entry points for the quantum toolbox."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from toolbox import data_sync, quantum_compact, quantum_ingest, quantum_downloader


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cosmos quantum toolbox commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("data-sync", help="Download/convert raw datasets")
    sync.add_argument("--datasets", nargs="+", help="Subset of dataset keys to sync")
    sync.add_argument(
        "--planck-components",
        help="Comma-separated planck_2018_raw component list (default: all)",
    )
    sync.add_argument(
        "--dataset-components",
        action="append",
        help="Dataset-specific component overrides (format: dataset=component1,component2)",
    )

    download = subparsers.add_parser("quantum-download", help="Mirror GWOSC/GCN/Fermi archives")
    download.add_argument("--force-downloads", action="store_true", help="Redownload even when data exists")
    download.add_argument("--skip-fermi", action="store_true", help="Skip the heavy Fermi fetch")
    download.add_argument("--debug", action="store_true", help="Enable verbose logging")

    ingest = subparsers.add_parser("quantum-ingest", help="Run the multimessenger ingestion pipeline")
    ingest.add_argument("--max-gcn", type=int, help="Limit number of GCN files")
    ingest.add_argument("--summary", type=Path, help="Where to write the layout summary")
    ingest.add_argument("--output", type=Path, help="Normalized output CSV path")

    compact = subparsers.add_parser("quantum-compact", help="Compact the normalized triggers/events into NPZ")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.command == "data-sync":
        dataset_names = args.datasets or data_sync.available_datasets()
        planck_components = None
        if args.planck_components:
            selection = [
                comp.strip() for comp in args.planck_components.split(",") if comp.strip()
            ]
            if selection:
                planck_components = {"planck_2018_raw": selection}
        dataset_component_map = None
        if args.dataset_components:
            dataset_component_map = {}
            for dataset_spec in args.dataset_components:
                if "=" not in dataset_spec:
                    continue
                dataset_name, values = dataset_spec.split("=", 1)
                components = [
                    comp.strip()
                    for comp in values.split(",")
                    if comp.strip()
                ]
                if components:
                    dataset_component_map[dataset_name.strip()] = components
        data_sync.sync_all(
            dataset_names=dataset_names,
            planck_component_map=planck_components,
            dataset_component_map=dataset_component_map,
        )
    elif args.command == "quantum-download":
        downloader_args: list[str] = []
        if args.force_downloads:
            downloader_args.append("--force-downloads")
        if args.skip_fermi:
            downloader_args.append("--skip-fermi")
        if args.debug:
            downloader_args.append("--debug")
        quantum_downloader.run_quantum_downloader(args=downloader_args)
    elif args.command == "quantum-ingest":
        extra = []
        if args.max_gcn is not None:
            extra.extend(["--max-gcn", str(args.max_gcn)])
        if args.summary:
            extra.extend(["--summary", str(args.summary)])
        if args.output:
            extra.extend(["--output", str(args.output)])
        quantum_ingest.run_ingest(args=extra)
    elif args.command == "quantum-compact":
        quantum_compact.run_compact()
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main(sys.argv[1:])
