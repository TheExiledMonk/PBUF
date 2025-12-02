"""Mode registry package for unified science runner."""

from __future__ import annotations

from cosmos2.science_runner.modes.base import (
    BaseModePlugin,
    ModeResult,
    RunContext,
    available_modes,
    get_mode,
    register_mode,
)

from cosmos2.science_runner.modes import joint  # noqa: F401
from cosmos2.science_runner.modes import jackknife  # noqa: F401
