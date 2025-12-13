"""Background worker process that continuously queries the controller for slices."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

from . import WorkerClient, WorkerHTTPTransport


def _parse_dataset_spec(specs: list[str]) -> dict[str, str]:
    datasets: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid dataset spec '{spec}', expected id=hash.")
        dataset_id, dataset_hash = spec.split("=", 1)
        datasets[dataset_id] = dataset_hash
    return datasets


def _load_dataset_file(path: Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset JSON must resolve to an object of id->hash.")
    return {str(k): str(v) for k, v in payload.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Cosmos worker in the background.")
    parser.add_argument("--endpoint", default="http://localhost:8080", help="Controller HTTP endpoint.")
    parser.add_argument("--worker-id", default=os.environ.get("HOSTNAME", "worker"), help="Unique id for this worker.")
    parser.add_argument("--cores", type=int, default=max(1, os.cpu_count() or 4), help="Reported core count.")
    parser.add_argument("--local", action="store_true", help="Treat this worker as a local node (50% slot usage).")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=[],
        help="dataset_id=hash entries describing cached datasets (repeated).",
    )
    parser.add_argument("--dataset-file", type=Path, help="JSON file with dataset_id->hash mapping.")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Seconds to sleep between idle polls.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the worker daemon.",
    )
    parser.add_argument("--auth-token", help="Optional bearer token for controller authentication.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    dataset_hashes = _load_dataset_file(args.dataset_file)
    dataset_hashes.update(_parse_dataset_spec(args.datasets))

    transport = WorkerHTTPTransport(
        args.endpoint,
        auth_token=args.auth_token,
    )
    worker = WorkerClient(
        transport=transport,
        worker_id=args.worker_id,
        cores=args.cores,
        local_node=args.local,
        datasets=list(dataset_hashes.items()),
    )

    stop = False

    def _shutdown(signum: int, frame: object | None) -> None:
        nonlocal stop
        logging.info("Worker daemon shutting down (signal %s)", signal.Signals(signum).name)
        stop = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)

    logging.info(
        "Worker daemon %s starting (%s cores, controller=%s, local=%s)",
        args.worker_id,
        args.cores,
        args.endpoint,
        args.local,
    )

    while not stop:
        try:
            worker.run()
        except Exception as exc:  # noqa: BLE001
            logging.exception("Worker %s crashed, retrying after backoff: %s", args.worker_id, exc)
            time.sleep(5.0)
            continue
        if stop:
            break
        time.sleep(args.poll_interval)

    transport.close()
    logging.info("Worker daemon %s exited", args.worker_id)


if __name__ == "__main__":
    main()
