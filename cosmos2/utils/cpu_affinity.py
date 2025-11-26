"""CPU affinity helpers for cosmos2."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def set_process_affinity(cpu_list: Sequence[int]) -> bool:
    """
    Attempt to pin the current process to the supplied CPU list.

    Returns True if affinity was set, False if the platform or dependencies
    do not support it. No exceptions are raised to keep orchestration resilient.
    """
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        proc.cpu_affinity(list(cpu_list))
        return True
    except Exception:
        return False


def split_cores_for_models(total_cores: int, lcdm_share: float, pbuf_share: float) -> Tuple[List[int], List[int]]:
    """
    Split logical cores between LCDM and PBUF model workers.

    The shares are relative weights; results are rounded to at least one core per model
    when possible.
    """
    total = max(int(total_cores), 1)
    lcdm_weight = max(float(lcdm_share), 0.0)
    pbuf_weight = max(float(pbuf_share), 0.0)
    if lcdm_weight + pbuf_weight == 0.0:
        lcdm_weight = pbuf_weight = 1.0

    lcdm_count = max(int(round(total * lcdm_weight / (lcdm_weight + pbuf_weight))), 1)
    lcdm_count = min(lcdm_count, total - 1) if total > 1 else 1
    pbuf_count = max(total - lcdm_count, 1)

    cpus = list(range(total))
    return cpus[:lcdm_count], cpus[lcdm_count : lcdm_count + pbuf_count]
