"""
Parameter evolution policies for the PBUF model.

This module currently provides a lightweight scaffold that keeps the
API stable while logging how parameters would evolve with redshift.
As the physical parameterisations mature, this function can be swapped
out for more sophisticated behaviour without touching the pipelines.
"""

from __future__ import annotations

from typing import Dict, Mapping

import logging
import numpy as np


LOGGER = logging.getLogger(__name__)


def evolve_params(z: np.ndarray, base: Mapping[str, float], policy: Mapping[str, float] | None) -> Dict[str, np.ndarray]:
    """Return evolved parameter arrays evaluated at each redshift.

    The current scaffold applies a power-law modifier driven by
    ``policy['m_<param>']`` exponents while logging the request. This
    keeps the interface stable for downstream callers and provides an
    obvious hook for future extensions.
    """

    if policy is None:
        policy = {}

    z = np.asarray(z, dtype=float)
    if z.ndim == 0:
        z = z[None]

    LOGGER.debug("Evolving parameters for keys=%s with policy=%s", list(base.keys()), policy)

    evolved: Dict[str, np.ndarray] = {}
    modifier = 1.0 + z
    for key, value in base.items():
        exponent = float(policy.get(f"m_{key}", 0.0))
        if isinstance(value, (int, float, np.floating)):
            evolved[key] = float(value) * modifier ** exponent
        else:
            evolved[key] = value

    return evolved
