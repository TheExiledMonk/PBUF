"""Collector thread for aggregating results."""

from __future__ import annotations

from queue import Empty
from typing import Dict


def run_collector_thread(queue, shared_state: Dict, *, poll_timeout: float | None = 0.1):
    """
    Drain results from the queue and keep track of the best chi2 across models,
    while aggregating per-model payloads and chi2 history snapshots for
    monitoring/reporting.

    Expects producers to send a dict with "best_params" and "best_chi2".
    Terminates when a None sentinel is received.
    """
    best_chi2 = float("inf")
    best_payload = None
    per_model: Dict[str, Dict] = {}
    chi2_history = []
    while True:
        try:
            payload = queue.get(timeout=poll_timeout)
        except Empty:
            continue
        if payload is None:
            break
        model_name = payload.get("name") or "model"
        per_model[model_name] = payload
        weighted_chi2 = float(payload.get("weighted_chi2", float("inf")))
        chi2 = float(payload.get("best_chi2", float("inf")))
        metric = weighted_chi2 if weighted_chi2 < float("inf") else chi2
        if metric < best_chi2:
            best_chi2 = metric
            best_payload = payload
        history_entries = payload.get("chi2_history") or []
        if history_entries:
            chi2_history.extend(
                [{**entry, "model": model_name} for entry in history_entries]
            )
        shared_state["best_overall"] = best_payload
        shared_state["per_model"] = dict(per_model)
        shared_state["chi2_history"] = list(chi2_history)
    shared_state["collector_done"] = True
