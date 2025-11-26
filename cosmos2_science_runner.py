"""CLI entry point for the cosmos2 science runner (config parity with legacy runner)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

from cosmos2.science_runner.runner import Cosmos2ScienceRunner
from cosmos2.science_runner.config import ScienceRunConfig


def _collect_paths(config_files: Iterable[str], config_dir: str | None) -> List[Path]:
    paths: List[Path] = []
    for raw in config_files:
        expanded = Path(raw).expanduser().resolve()
        if expanded.exists():
            paths.append(expanded)
    if config_dir:
        directory = Path(config_dir).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Config directory '{directory}' does not exist.")
        for pattern in ("*.json", "*.yaml", "*.yml"):
            for match in sorted(directory.glob(pattern)):
                paths.append(match.resolve())
    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def _collect_override_items(raw_values: Sequence[str] | None) -> list[str]:
    if not raw_values:
        return []
    expanded: list[str] = []
    for entry in raw_values:
        for segment in entry.split(","):
            candidate = segment.strip()
            if candidate:
                expanded.append(candidate)
    return ScienceRunConfig._normalize_list(expanded)


def _make_progress_printer(run_name: str):
    def _callback(event: dict) -> None:
        event_type = event.get("type")
        if event_type == "model_complete":
            model = event.get("model", "model")
            best = event.get("best_chi2")
            weighted = event.get("weighted_chi2")
            print(f"[cosmos2][{run_name}] {model}: chi2={best} weighted={weighted}")
        if event_type == "collector_update":
            best = event.get("best_overall") or {}
            model = best.get("name") or best.get("model") or "model"
            score = best.get("weighted_chi2", best.get("best_chi2"))
            print(f"[cosmos2][{run_name}] best so far {model} -> {score}")

    return _callback


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cosmos2 science configurations.")
    parser.add_argument("--config", "-c", action="append", default=[], help="Path to a science config JSON/YAML file.")
    parser.add_argument("--config-dir", "-d", help="Directory containing science configs (JSON/YAML).")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt for confirmation before running each config.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare run outputs but skip execution.")
    parser.add_argument("--override-fits", action="append", help="Override joint fits (comma separated).")
    parser.add_argument("--override-models", action="append", help="Override models (comma separated).")
    parser.add_argument("--mode", choices=["fit", "scout"], help="Override the mode defined inside the config.")
    parser.add_argument("--engine", help="Override engine name in config (cosmos2_basin/basin/threaded).")
    parser.add_argument("--workers", type=int, help="Worker processes for batch evaluation.")
    parser.add_argument("--monitor", action="store_true", help="Enable console monitor updates during optimisation.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.json when available.")
    args = parser.parse_args()

    config_paths = _collect_paths(args.config, args.config_dir)
    if not config_paths:
        parser.error("No science config files were provided.")

    print("Cosmos2 Science Runner")
    for config_path in config_paths:
        config = ScienceRunConfig.from_path(config_path)
        if args.mode:
            config.mode = args.mode
        if args.engine:
            config.engine = args.engine
        if args.workers:
            config.engine_settings["workers"] = args.workers
        if args.monitor:
            config.engine_settings["monitor"] = True
        if args.resume:
            config.engine_settings["resume"] = True
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
        if interactive:
            try:
                from science_runner import _interactive_confirm  # type: ignore

                if not _interactive_confirm(config):
                    print(f"Skipping {config_path}")
                    continue
            except Exception:
                # Fallback: proceed without prompt if legacy helper is unavailable.
                pass

        runner = Cosmos2ScienceRunner(config, dry_run=args.dry_run)
        runner.execute(progress_callback=_make_progress_printer(config.run_name))


if __name__ == "__main__":
    main()
