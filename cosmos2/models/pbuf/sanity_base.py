"""Small shared result container for sanity checks (ported from cosmos_old)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SanityResult:
    """Aggregate success flag and reasons for failing guards."""

    ok: bool = True
    reasons: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.reasons.append(msg)

    def merge(self, other: "SanityResult") -> None:
        if not other.ok:
            self.ok = False
            self.reasons.extend(other.reasons)


__all__ = ["SanityResult"]
