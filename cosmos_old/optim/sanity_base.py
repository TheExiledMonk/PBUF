"""Small shared result returned by every sanity helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SanityResult:
    ok: bool = True
    reasons: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.ok = False
        self.reasons.append(msg)

    def merge(self, other: "SanityResult"):
        if not other.ok:
            self.ok = False
            self.reasons.extend(other.reasons)
