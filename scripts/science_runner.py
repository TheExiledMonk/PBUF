"""CLI entry point for Cosmos science runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from cosmos2.fits.registry import FIT_REGISTRY
from cosmos2.science_runner.config import ScienceRunConfig
from cosmos2.science_runner.unified_runner import UnifiedScienceRunner


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


def _prompt_bool(prompt: str, current: bool) -> bool:
    default = "Y" if current else "N"
    response = input(f"{prompt} [{default}/{'n' if current else 'y'}]: ").strip().lower()
    if not response:
        return current
    if response in ("y", "yes"):
        return True
    if response in ("n", "no"):
        return False
    return current


def _parse_engine_setting_value(raw: str) -> str | float | int | bool:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "none":
        return ""
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def _print_menu_summary(config: ScienceRunConfig) -> None:
    separator = "=" * 58
    print(separator)
    print(" PBUF Science Runner - Interactive Configuration")
    print(separator)
    engine_threads = config.engine_settings.get("threads") or config.engine_settings.get("n_threads") or "default"
    reports = ", ".join(config.output.report_formats) if config.output.report_formats else "none"
    print(f"Run: {config.run_name}")
    print(f"Mode: {config.mode}")
    print(f"Models: {', '.join(config.models)}")
    print(f"Fits: {', '.join(config.fits_list)}")
    print(f"Engine: {config.engine}")
    print(f"Threads: {engine_threads}")
    print(f"Reports: {reports}")
    print(f"Plots: {'Enabled' if config.output.generate_plots else 'Disabled'}")
    print(f"Save-space: {config.output.save_space}")
    print(separator)
    print("Options:")
    print("  [1] Toggle models")
    print("  [2] Toggle fits")
    print("  [3] Engine settings")
    print("  [4] Plotting/reporting settings")
    print("  [5] Show full summary")
    print("  [6] Proceed")
    print("  [7] Cancel")
    print(separator)


def _toggle_models(config: ScienceRunConfig) -> None:
    print(f"Current models: {', '.join(config.models)}")
    selection = input("Enter models (comma separated, blank to keep): ").strip()
    if not selection:
        return
    items = _collect_override_items([selection])
    if not items:
        print("No valid model names provided.")
        return
    try:
        config.set_models(items)
    except ValueError as exc:
        print(f"Could not update models: {exc}")


def _toggle_fits(config: ScienceRunConfig) -> None:
    available = sorted(FIT_REGISTRY.keys())
    enabled = set(config.fits_list)
    for index, fit_name in enumerate(available, start=1):
        status = "on" if fit_name in enabled else "off"
        print(f"  [{index}] {fit_name} ({status})")
    selection = input("Toggle which indexes (comma separated, blank to skip): ").strip()
    if not selection:
        return
    toggled = set(enabled)
    for part in selection.split(","):
        candidate = part.strip()
        if not candidate:
            continue
        try:
            idx = int(candidate) - 1
        except ValueError:
            print(f"Ignoring invalid index '{candidate}'.")
            continue
        if 0 <= idx < len(available):
            fit_name = available[idx]
            if fit_name in toggled:
                toggled.remove(fit_name)
            else:
                toggled.add(fit_name)
    if not toggled:
        print("At least one fit must be selected.")
        return
    new_list = [fit for fit in available if fit in toggled]
    previous = list(config.fits_list)
    try:
        config.set_fits(new_list)
    except ValueError as exc:
        print(f"Could not update fits: {exc}")
        config.set_fits(previous)


def _engine_settings_menu(config: ScienceRunConfig) -> None:
    print(f"Current engine: {config.engine}")
    new_engine = input("Engine name (blank to keep current): ").strip()
    if new_engine:
        config.engine = new_engine
    print("Current engine settings:")
    print(json.dumps(config.engine_settings or {}, indent=2))
    settings_input = input("Update settings (key=value, comma separated, blank to keep): ").strip()
    if not settings_input:
        return
    updated = dict(config.engine_settings)
    for segment in settings_input.split(","):
        pair = segment.strip()
        if not pair or "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        updated[key.strip()] = _parse_engine_setting_value(value.strip())
    config.engine_settings = updated


def _reporting_menu(config: ScienceRunConfig) -> None:
    config.output.generate_plots = _prompt_bool("Generate plots", config.output.generate_plots)
    config.output.generate_reports = _prompt_bool("Generate reports", config.output.generate_reports)
    available = ["json", "html", "pdf"]
    current = set(config.output.report_formats)
    for index, fmt in enumerate(available, start=1):
        status = "enabled" if fmt in current else "disabled"
        print(f"  [{index}] {fmt.upper()} ({status})")
    selection = input("Toggle report formats by index (comma separated, blank to skip): ").strip()
    if selection:
        toggled = set(current)
        for part in selection.split(","):
            try:
                idx = int(part.strip()) - 1
            except ValueError:
                continue
            if 0 <= idx < len(available):
                fmt = available[idx]
                if fmt in toggled:
                    toggled.remove(fmt)
                else:
                    toggled.add(fmt)
        config.output.report_formats = [fmt for fmt in available if fmt in toggled]
    config.output.save_space = _prompt_bool("Save-space mode", config.output.save_space)


def _show_full_summary(config: ScienceRunConfig) -> None:
    print("Configuration summary:")
    print(json.dumps(config.to_dict(), indent=2))
    input("Press Enter to continue...")


def _interactive_confirm(config: ScienceRunConfig) -> bool:
    if not sys.stdin.isatty():
        print("Interactive prompts skipped because stdin is not a TTY.")
        return True
    while True:
        _print_menu_summary(config)
        try:
            response = input("Selection: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not response:
            continue
        if response == "1":
            _toggle_models(config)
            continue
        if response == "2":
            _toggle_fits(config)
            continue
        if response == "3":
            _engine_settings_menu(config)
            continue
        if response == "4":
            _reporting_menu(config)
            continue
        if response == "5":
            _show_full_summary(config)
            continue
        if response == "6":
            return True
        if response == "7":
            return False
        print("Unknown selection.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cosmos science configurations.")
    parser.add_argument("--config", "-c", action="append", default=[], help="Path to a science config JSON/YAML file.")
    parser.add_argument("--config-dir", "-d", help="Directory containing science configs (JSON/YAML).")
    parser.add_argument("--interactive", "-i", action="store_true", help="Prompt for confirmation before running each config.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare run outputs but skip execution.")
    parser.add_argument("--override-fits", action="append", help="Override joint fits (comma separated).")
    parser.add_argument("--override-models", action="append", help="Override models (comma separated).")
    parser.add_argument("--mode", choices=["fit", "scout"], help="Override the mode defined inside the config.")
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
