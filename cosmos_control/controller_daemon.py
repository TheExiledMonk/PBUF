"""Background entrypoint that runs the Cosmos controller HTTP API."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

from . import Controller, run_controller_api


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Cosmos controller API as a daemon.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address for the controller API.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the controller HTTP server.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/science_runs"),
        help="Directory where controller run data is stored.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the controller daemon.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    controller = Controller(base_dir=args.base_dir)
    server = run_controller_api(controller, host=args.host, port=args.port)

    def _shutdown(signum: int, frame: object | None) -> None:
        logging.info("Shutting down controller (signal %s)", signal.Signals(signum).name)
        server.shutdown()
        server.server_close()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _shutdown)

    logging.info("Controller daemon listening on %s:%s (base dir %s)", args.host, args.port, args.base_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()
