"""Template rendering utility for report generation."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Mapping


class ReportTemplateEngine:
    """Render reports from a configurable template file."""

    DEFAULT_TEMPLATE_NAME = "default_report.html"

    def __init__(self, template_path: Path | None = None) -> None:
        self.template_path = Path(template_path) if template_path else self._default_template_path()
        if not self.template_path.exists():
            raise FileNotFoundError(f"Report template not found: {self.template_path}")
        text = self.template_path.read_text()
        self._template = Template(text)

    @classmethod
    def _default_template_path(cls) -> Path:
        return Path(__file__).parent.parent / "templates" / cls.DEFAULT_TEMPLATE_NAME

    def render(self, context: Mapping[str, str]) -> str:
        """Render the configured template with the provided string context."""
        return self._template.safe_substitute(context)
