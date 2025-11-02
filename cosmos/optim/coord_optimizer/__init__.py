"""
Coordinate descent-style optimizer for isolating cosmological parameter basins.

The basin walker can be driven programmatically or through the CLI to
generate tight hyper-rectangles around viable regions for LCDM or PBUF
before launching more expensive local grids.
"""

from .basin_walker import (
    CoordinateBasinWalker,
    DEFAULT_REFERENCES,
    DEFAULT_PBUF_REFERENCE,
    DEFAULT_LCDM_REFERENCE,
    DEFAULT_SCAN_PRESETS,
    DEFAULT_PARAM_ORDER,
    DEFAULT_SECOND_PASS_PARAMS,
    SECOND_PASS_PARAMS,
)
from .coord_old import CoordinateBasinWalkerOld
from .observers import BasinWalkObserver, CompositeObserver, RecordingObserver

__all__ = [
    "CoordinateBasinWalker",
    "CoordinateBasinWalkerOld",
    "BasinWalkObserver",
    "CompositeObserver",
    "RecordingObserver",
    "DEFAULT_REFERENCES",
    "DEFAULT_PBUF_REFERENCE",
    "DEFAULT_LCDM_REFERENCE",
    "DEFAULT_SCAN_PRESETS",
    "DEFAULT_PARAM_ORDER",
    "DEFAULT_SECOND_PASS_PARAMS",
    "SECOND_PASS_PARAMS",
]
