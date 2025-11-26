"""Monitoring thread rendering console dashboard."""

from __future__ import annotations

import os
import shutil
import time
from threading import Event, Lock, active_count
from typing import Dict, Iterable, Optional


def _clear_screen() -> None:
    # ANSI clear keeps the rendering lightweight and dependency-free.
    print("\033[2J\033[H", end="", flush=True)


def _format_progress(current: int, total: int | None) -> str:
    if total and total > 0:
        pct = min(100.0, max(0.0, (current / total) * 100.0))
        return f"{current}/{total} ({pct:5.1f}%)"
    return str(current)


def _safe_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _render_model(name: str, payload: Dict) -> str:
    best = _safe_float(payload.get("best_chi2"))
    last = _safe_float(payload.get("last_chi2"))
    best_so_far = _safe_float(payload.get("best_so_far", best))
    batch = int(payload.get("batch", 0))
    total_batches = payload.get("total_batches")
    recent = payload.get("recent_history") or []
    evals = int(payload.get("evals", 0))
    workers = payload.get("workers")
    started_at = payload.get("started_at")
    bar = _progress_bar(batch, total_batches, width=28)
    suffix = f" w={workers}" if workers else ""
    lines = [
        f"[{name}] {bar} {_format_progress(batch, total_batches)} evals={evals}{suffix}",
        f"  best χ²: {best:.4g}   last χ²: {last:.4g}   best-so-far: {best_so_far:.4g}",
    ]
    if recent:
        samples = [entry.get("chi2", float("inf")) for entry in recent if isinstance(entry, dict)]
        spark = _sparkline(samples, width=20)
        lines.append(f"  recent χ²: {spark}")
    eta = _eta(batch, total_batches, started_at)
    if eta:
        lines.append(f"  ETA: {eta}")
    return "\n".join(lines)


def _progress_bar(current: int, total: int | None, width: int = 24) -> str:
    if total and total > 0:
        frac = max(0.0, min(1.0, current / total))
    else:
        frac = 0.0
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _sparkline(values: Iterable[float], width: int = 20) -> str:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return "-" * width
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1.0
    levels = ".,:-=+*#%@"
    buckets = [int((v - lo) / span * (len(levels) - 1)) for v in vals]
    sampled = buckets[-width:]
    return "".join(levels[idx] for idx in sampled).ljust(width, ".")


def _eta(batch: int, total: int | None, started_at: float | None) -> str | None:
    if total is None or total <= 0 or batch <= 0 or started_at is None:
        return None
    elapsed = max(0.0, time.time() - float(started_at))
    per_batch = elapsed / max(batch, 1)
    remaining = max(0.0, (total - batch) * per_batch)
    return _format_eta(remaining)


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hrs:
        return f"{hrs:d}h {mins:02d}m"
    if mins:
        return f"{mins:d}m {secs:02d}s"
    return f"{secs:d}s"


def _system_stats() -> Dict[str, float | int]:
    cpu_load = None
    try:
        cpu_load = os.getloadavg()[0]
    except Exception:
        cpu_load = None
    mem_used = mem_total = None
    proc_children = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        mem_used = vm.used / (1024 ** 3)
        mem_total = vm.total / (1024 ** 3)
        proc_children = len(psutil.Process().children(recursive=False))
    except Exception:
        try:
            with open("/proc/meminfo") as fh:
                info = {line.split(":")[0]: float(line.split()[1]) for line in fh if ":" in line}
            mem_total = info.get("MemTotal", 0.0) / 1e6
            mem_free = info.get("MemAvailable", info.get("MemFree", 0.0)) / 1e6
            mem_used = mem_total - mem_free
        except Exception:
            pass
    return {
        "cpu_load": cpu_load,
        "cpus": os.cpu_count() or 1,
        "threads": active_count(),
        "children": proc_children,
        "mem_used": mem_used,
        "mem_total": mem_total,
    }


def run_monitor_thread(
    shared_state: Dict,
    refresh_hz: int = 0.2,
    iterations: Optional[int] = 1,
    stop_event: Optional[Event] = None,
    lock: Optional[Lock] = None,
) -> None:
    """
    Render a lightweight terminal dashboard while optimisation runs.

    - When iterations is None, loop until stop_event is set.
    - Uses ANSI clears to reuse the same screen space.
    """
    delay = 1.0 / max(refresh_hz, 1)
    loop_forever = iterations is None
    remaining = max(iterations or 1, 1)

    while loop_forever or remaining > 0:
        if stop_event is not None and stop_event.is_set():
            break
        if not loop_forever:
            remaining -= 1

        with lock or _nullcontext():
            best = shared_state.get("best_overall")
            models = shared_state.get("models") or {}
            chi2_history = shared_state.get("chi2_history") or []
            latest_batch = shared_state.get("latest_batch") or shared_state.get("last_batch")
        stats = _system_stats()

        _clear_screen()
        cols = shutil.get_terminal_size((100, 20)).columns
        header = "Cosmos2 Engine 2.0"
        max_workers = None
        if models:
            try:
                max_workers = max(payload.get("workers", 0) or 0 for payload in models.values())
            except Exception:
                max_workers = None
        if stats["cpu_load"] is not None:
            header += f" | CPU load {stats['cpu_load']:.2f}/{stats['cpus']}"
        if stats["mem_used"] is not None and stats["mem_total"] is not None:
            header += f" | Mem {stats['mem_used']:.1f}/{stats['mem_total']:.1f} GB"
        header += f" | Threads {stats['threads']}"
        if stats.get("children") is not None:
            header += f" | Procs {stats['children']}"
        if max_workers:
            header += f" | Workers {max_workers}"
        print(header.ljust(cols, " "))
        print("-" * min(cols, 120))
        if best:
            weighted = best.get("weighted_chi2", best.get("best_chi2"))
            print(f"Best overall: model={best.get('name', 'model')} χ²={best.get('best_chi2'):.4g} weighted={weighted:.4g}")
        else:
            print("Best overall: pending")

        if latest_batch and isinstance(latest_batch, dict):
            model = latest_batch.get("model", "model")
            print(f"Last batch: {model} χ²={latest_batch.get('best_chi2')} recent={latest_batch.get('chi2_history')}")

        if chi2_history:
            last = chi2_history[-1]
            print(
                f"History tail: len={len(chi2_history)} last χ²={last.get('chi2')} best_so_far={last.get('best_so_far')} model={last.get('model')}"
            )

        if models:
            print("\nModels:")
            for name, payload in sorted(models.items()):
                try:
                    print(_render_model(name, payload))
                except Exception:
                    print(f"[{name}] monitoring data unavailable")
        else:
            print("\nWaiting for model updates...")

        snapshot = {
            "timestamp": time.time(),
            "stats": stats,
            "best_overall": best,
            "models": models,
        }
        if lock is not None:
            try:
                with lock:
                    history = shared_state.setdefault("snapshots", [])
                    history.append(snapshot)
                    if len(history) > 50:
                        del history[: len(history) - 50]
            except Exception:
                pass

        time.sleep(delay)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
