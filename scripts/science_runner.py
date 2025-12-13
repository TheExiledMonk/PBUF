"""CLI entry point for Cosmos science runner."""

from __future__ import annotations

import argparse

from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.unified_runner import UnifiedScienceRunner
from science_runner import (
    _collect_override_items,
    _collect_paths,
    _interactive_confirm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cosmos science configurations.")
    parser.add_argument("--config", "-c", action="append", default=[], help="Path to a science config JSON/YAML file.")
    parser.add_argument("--config-dir", "-d", help="Directory containing science configs (JSON/YAML).")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt for confirmation before running each config.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare run outputs but skip execution.")
    parser.add_argument("--override-fits", action="append", help="Override joint fits (comma separated).")
    parser.add_argument("--override-models", action="append", help="Override models (comma separated).")
    parser.add_argument("--mode", choices=["fit", "scout"], help="Override the mode defined inside the config.")
    parser.add_argument("--controller-endpoint", help="Controller API endpoint (overrides COSMOS_CONTROLLER_ENDPOINT).")
    args = parser.parse_args()

    config_paths = _collect_paths(args.config, args.config_dir)
    if not config_paths:
        parser.error("No science config files were provided.")

    print("Cosmos2 Science Runner")
    print("Running threaded cosmos2 engine\n")
    for config_path in config_paths:
        config = ScienceRunConfig.from_path(config_path)
        if args.mode:
            config.mode = args.mode
        if args.controller_endpoint:
            config.engine_settings["controller_endpoint"] = args.controller_endpoint
        override_models = _collect_override_items(args.override_models)
        override_fits = _collect_override_items(args.override_fits)
        try:
            if override_models:
                config.set_models(override_models)
            if override_fits:
                config.set_fits(override_fits)
        except ValueError as exc:
            parser.error(str(exc))
        interactive = args.interactive or config.interactive
        if interactive and not _interactive_confirm(config):
            print(f"Skipping {config_path}")
            continue
        runner = UnifiedScienceRunner(config, dry_run=args.dry_run)
        runner.execute()


if __name__ == "__main__":
    main()
